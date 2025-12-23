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
CUSTOM_LLM_TIMEOUT = config['custom_llm'].get('timeout', 120)  # Default 120 seconds
CUSTOM_LLM_MAX_RETRIES = config['custom_llm'].get('max_retries', 3)  # Retry attempts
CUSTOM_LLM_MAX_TOKENS = config['custom_llm'].get('max_tokens', 4096)  # Max response tokens

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
    """Simple API call with retry logic"""
    for attempt in range(CUSTOM_LLM_MAX_RETRIES):
        try:
            logging.info(f"🔄 LLM API call attempt {attempt + 1}/{CUSTOM_LLM_MAX_RETRIES} (timeout={CUSTOM_LLM_TIMEOUT}s)")
            start_time = time.time()
            
            response = requests.post(
                CUSTOM_LLM_URL, 
                headers=headers, 
                json=data,
                timeout=CUSTOM_LLM_TIMEOUT
            )
            
            elapsed = time.time() - start_time
            logging.info(f"✅ LLM API responded in {elapsed:.2f}s")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout as e:
            elapsed = time.time() - start_time
            logging.warning(f"⏱️ Timeout after {elapsed:.2f}s on attempt {attempt + 1}: {e}")
            if attempt == CUSTOM_LLM_MAX_RETRIES - 1:
                raise
            time.sleep(3)  # Wait 3s before retry
            
        except requests.exceptions.RequestException as e:
            logging.warning(f"❌ Request error on attempt {attempt + 1}: {e}")
            if attempt == CUSTOM_LLM_MAX_RETRIES - 1:
                raise
            time.sleep(3)  # Wait 3s before retry
            
        except Exception as e:
            logging.error(f"💥 Unexpected error on attempt {attempt + 1}: {e}")
            if attempt == CUSTOM_LLM_MAX_RETRIES - 1:
                raise
            time.sleep(3)  # Wait 3s before retry

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
    data = {
        "model": CUSTOM_LLM_MODEL, 
        "messages": messages,
        "max_tokens": CUSTOM_LLM_MAX_TOKENS,
        "temperature": kwargs.get("temperature", 0.7)
    }
    headers = {'Content-Type': 'application/json'}
    
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    
    try:
        # Thêm timeout wrapper cho executor để đảm bảo timeout hoạt động
        total_timeout = CUSTOM_LLM_TIMEOUT * CUSTOM_LLM_MAX_RETRIES + 10  # Buffer time
        response_json = await asyncio.wait_for(
            loop.run_in_executor(None, _call_llm_api, data, headers),
            timeout=total_timeout
        )
        content = response_json['choices'][0]['message']['content']
        logging.info(f"✨ LLM response received: {len(content)} chars")
        
    except asyncio.TimeoutError:
        logging.error(f"⏱️ Total timeout exceeded ({total_timeout}s)")
        return "[ERROR: Request timeout - LLM took too long to respond]"
        
    except Exception as e:
        logging.error(f"❌ LLM API error: {type(e).__name__}: {e}")
        return f"[ERROR: {str(e)}]"

    # Save to cache
    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": content, "model": CUSTOM_LLM_MODEL}})
    
    return content


def check_neo4j_connection():
    """Kiểm tra kết nối Neo4j"""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            config['hirag']['neo4j_url'], 
            auth=tuple(config['hirag']['neo4j_auth'])
        )
        with driver.session() as session:
            result = session.run("RETURN 1")
            result.single()
        driver.close()
        return True
    except Exception as e:
        logging.error(f"Neo4j connection failed: {e}")
        return False

def create_graph_func_with_retry(max_retries=3):
    """Tạo HiRAG instance với retry"""
    for attempt in range(max_retries):
        try:
            logging.info(f"🔄 Attempting to initialize HiRAG (attempt {attempt + 1}/{max_retries})")
            
            if not check_neo4j_connection():
                logging.warning("⚠️ Neo4j connection check failed, waiting 5s...")
                time.sleep(5)
                continue
            
            graph = HiRAG(
                working_dir=config['hirag']['working_dir'],
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
            logging.info("✅ HiRAG initialized successfully")
            return graph
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize HiRAG: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(5)
    
    raise Exception("Failed to initialize HiRAG after retries")

graph_func = create_graph_func_with_retry()

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

# try:
#     file_name = "86_2015_QH13(12998).txt"
#     file_path = os.path.join(DATA_DIR, file_name)
#     print(f"🚀 Processing {file_name}")
#     content = read_text_safe(file_path)
#     graph_func.insert(content)
# except Exception as e:
#     logging.error(f"❌ Failed file {file_name}: {e}")

all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".txt")])
total_files = len(all_files)
processed = 0
failed = 0
start_time = time.time()

print(f"\n📚 Found {total_files} files to process\n")

# for idx, filename in enumerate(all_files, 1):
for idx, filename in enumerate(all_files[:5], 1): 
    file_path = os.path.join(DATA_DIR, filename)

    max_file_retries = 2  # Retry mỗi file tối đa 2 lần nếu bị lỗi connection
    
    for file_attempt in range(max_file_retries):
        try:
            elapsed = time.time() - start_time
            retry_msg = f" (retry {file_attempt + 1}/{max_file_retries})" if file_attempt > 0 else ""
            print(f"🚀 [{idx}/{total_files}] Processing {filename}{retry_msg} (⏱️ {elapsed/60:.1f} min elapsed)")

            content = read_text_safe(file_path)
            content = f"[FILE_ID={filename}]\n" + content

            if len(content.strip()) < 100:
                print(f"⚠️  Skip empty file: {filename}")
                break

            # Kiểm tra connection trước khi insert
            if not check_neo4j_connection():
                logging.warning("⚠️ Neo4j connection lost, reconnecting...")
                graph_func = create_graph_func_with_retry()
            
            graph_func.insert(content)
            processed += 1
            
            # Estimate time remaining
            if processed > 0:
                avg_time_per_file = elapsed / processed
                remaining = (total_files - idx) * avg_time_per_file
                print(f"✅ Done! (~{remaining/60:.1f} min remaining)")

            time.sleep(2)  # tăng lên 2s để giảm load Neo4j
            break  # Success, thoát retry loop

        except Exception as e:
            error_msg = str(e)
            is_connection_error = any(x in error_msg.lower() for x in ['routing', 'connection', 'failed to write'])
            
            if is_connection_error and file_attempt < max_file_retries - 1:
                logging.warning(f"⚠️ Connection error, will retry: {e}")
                print(f"⚠️ Connection lost, waiting 10s before retry...")
                time.sleep(10)
                
                # Thử reconnect
                try:
                    graph_func = create_graph_func_with_retry()
                except Exception as reconnect_error:
                    logging.error(f"Failed to reconnect: {reconnect_error}")
                continue
            else:
                # Không phải lỗi connection hoặc đã hết retry
                failed += 1
                logging.error(f"❌ Failed file {filename}: {e}")
                print(f"❌ Error (will continue): {e}")
                break

elapsed_total = time.time() - start_time
print(f"\n{'='*60}")
print(f"✨ Indexing completed!")
print(f"📊 Processed: {processed}/{total_files} files")
print(f"❌ Failed: {failed} files")
print(f"⏱️  Total time: {elapsed_total/60:.1f} minutes")
print(f"{'='*60}\n")