# Quick start

Hướng dẫn chạy `hi_Search_custom.py` từ đầu.

## Chuẩn bị
- Cài Conda (hoặc dùng môi trường ảo khác).
- Đảm bảo đã cài Neo4j và có thể khởi động thành công.

## Thiết lập môi trường
```bash
conda create -n hirag python=3.10 -y
conda activate hirag
```

## Lấy mã nguồn và cài đặt
```bash
git clone https://github.com/givoxxs/HiRAG_STeam.git
cd HiRAG
pip install -e .
```

## Tạo và chỉnh cấu hình
Sao chép file mẫu và chỉnh sửa:
```bash
cp config_example.yaml config.yaml
```

Mở `config.yaml` và điền/kiểm tra các trường tối thiểu:
- `custom_llm.url`, `custom_llm.model`, (tùy chọn `system_prompt`, `timeout`, `max_retries`).
- `huggingface.embedding_model`, `huggingface.embedding_dim`, `huggingface.device` (cpu/gpu).
- `hirag.working_dir`, `hirag.enable_llm_cache`, `hirag.embedding_batch_num`, `hirag.embedding_func_max_async`, `hirag.enable_naive_rag`, `hirag.enable_hierachical_mode`.
- Kết nối Neo4j: `hirag.neo4j_url`, `hirag.neo4j_auth` (user/password).

## Khởi động Neo4j
Khởi chạy Neo4j và đảm bảo URL/Auth khớp với cấu hình ở trên.

## Chạy tìm kiếm
```bash
python hi_Search_custom.py
```

Tại prompt, nhập câu hỏi (gõ `exit` để thoát).

