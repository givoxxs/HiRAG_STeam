import os
import logging
import numpy as np
import yaml
import json
import requests
import time
from hirag import HiRAG, QueryParam
from dataclasses import dataclass
from hirag.base import BaseKVStorage
from hirag._utils import compute_args_hash
from hirag._storage import Neo4jStorage, NetworkXStorage
from sentence_transformers import SentenceTransformer
import asyncio

# Load configuration from YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract configurations for custom LLM
CUSTOM_LLM_URL = config['custom_llm']['url']
CUSTOM_LLM_MODEL = config['custom_llm']['model']
CUSTOM_LLM_SYSTEM_PROMPT = config['custom_llm'].get('system_prompt', 'Bạn là trợ lý AI hữu ích. Luôn trả lời bằng tiếng Việt.')
CUSTOM_LLM_TIMEOUT = config['custom_llm'].get('timeout', 60)  # Default 60 seconds
CUSTOM_LLM_MAX_RETRIES = config['custom_llm'].get('max_retries', 2)  # Retry once if failed

# Extract configurations for Hugging Face embedding
HF_EMBEDDING_MODEL = config['huggingface']['embedding_model']
EMBEDDING_DIM = config['huggingface']['embedding_dim']
HF_DEVICE = config['huggingface'].get('device', 'cpu')

# Load Hugging Face model
print(f"Loading Hugging Face embedding model: {HF_EMBEDDING_MODEL} on {HF_DEVICE}...")
hf_model = SentenceTransformer(HF_EMBEDDING_MODEL, device=HF_DEVICE)
print("Model loaded successfully!")

@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro

@wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM, max_token_size=config['model_params']['max_token_size'])
async def HF_LOCAL_embedding(texts: list[str]) -> np.ndarray:
    """
    Generate embeddings using Hugging Face model running locally on CPU
    """
    # Run the synchronous encode in a thread pool to make it async
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    
    embeddings = await loop.run_in_executor(
        None, 
        lambda: hf_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    )
    return np.array(embeddings)

def _call_llm_api(data: dict, headers: dict) -> dict:
    """Simple API call with single retry"""
    for attempt in range(CUSTOM_LLM_MAX_RETRIES):
        try:
            response = requests.post(
                CUSTOM_LLM_URL, 
                headers=headers, 
                json=data,
                timeout=CUSTOM_LLM_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt == CUSTOM_LLM_MAX_RETRIES - 1:
                raise
            time.sleep(2)  # Wait 2s before retry

async def CUSTOM_LLM_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Call custom LLM API with caching"""
    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    elif CUSTOM_LLM_SYSTEM_PROMPT:
        messages.append({"role": "system", "content": CUSTOM_LLM_SYSTEM_PROMPT})
    
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Check cache
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    if hashing_kv is not None:
        args_hash = compute_args_hash(CUSTOM_LLM_MODEL, messages)
        cached = await hashing_kv.get_by_id(args_hash)
        if cached is not None:
            return cached["return"]

    # Call API
    data = {"model": CUSTOM_LLM_MODEL, "messages": messages}
    headers = {'Content-Type': 'application/json'}
    
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    
    try:
        response_json = await loop.run_in_executor(None, _call_llm_api, data, headers)
        content = response_json['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"LLM API error: {e}")
        return f"[ERROR: {str(e)}]"

    # Save to cache
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": content, "model": CUSTOM_LLM_MODEL}})
    
    return content


graph_func = HiRAG(working_dir=config['hirag']['working_dir'],
                      enable_llm_cache=config['hirag']['enable_llm_cache'],
                      embedding_func=HF_LOCAL_embedding,
                      best_model_func=CUSTOM_LLM_model_if_cache,
                      cheap_model_func=CUSTOM_LLM_model_if_cache,
                      enable_hierachical_mode=config['hirag']['enable_hierachical_mode'], 
                      embedding_batch_num=config['hirag']['embedding_batch_num'],
                      embedding_func_max_async=config['hirag']['embedding_func_max_async'],
                      enable_naive_rag=config['hirag']['enable_naive_rag'],
                      graph_storage_cls=Neo4jStorage,
                      addon_params={"neo4j_url": config['hirag']['neo4j_url'], "neo4j_auth": config['hirag']['neo4j_auth']}
                      )

# comment this if the working directory has already been indexed
# with open("nghiquyet18.txt", encoding="utf-16") as f:
#     graph_func.insert(f.read())


# print("Perform hi search:")
# print(graph_func.query("What are the top themes in this story?", param=QueryParam(mode="hi")))

# print("Nhập câu hỏi (gõ 'exit' để thoát):")
# while True:
#     user_query = input("> ").strip()
#     if user_query.lower() == "exit":
#         break
#     if not user_query:
#         continue
#     answer = graph_func.query(user_query, param=QueryParam(mode="hi"))
#     print(answer)

DATA_DIR = "Luat_txt"

def read_text_safe(file_path: str) -> str:
    encodings = ["utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1258"]

    for enc in encodings:
        try:
            with open(file_path, encoding=enc) as f:
                return f.read()
        except Exception:
            continue

    raise ValueError(f"Cannot decode file: {file_path}")

for filename in sorted(os.listdir(DATA_DIR)):
    if not filename.endswith(".txt"):
        continue

    file_path = os.path.join(DATA_DIR, filename)

    try:
        print(f"🚀 Processing {filename}")

        content = read_text_safe(file_path)
        content = f"[FILE_ID={filename}]\n" + content

        if len(content.strip()) < 100:
            print(f"⚠️ Skip empty file: {filename}")
            continue

        graph_func.insert(content)

        time.sleep(1)  # tránh overload LLM / embedding

    except Exception as e:
        logging.error(f"❌ Failed file {filename}: {e}")