"""
worker
------
Package bổ sung — KHÔNG nằm trong bộ khung gốc của Agentic AI Audit System.

Gồm:
  * rate_limited_client.py — Rate Limiting (Token Bucket) + Retry (Exponential
    Backoff) + Batching Queue cho lệnh gọi Gemini API.
  * background_worker.py   — Autonomous Agent chạy ngầm bằng APScheduler,
    tự động quét CSDL và kiểm toán định kỳ.
  * state_store.py         — Theo dõi ID chứng từ đã xử lý.

Toàn bộ package này chỉ IMPORT và TÁI SỬ DỤNG các hàm/class công khai đã có
sẵn ở các file gốc (config.py, data_ingestion.py, audit_agent.py,
human_review.py, report_exporter.py) — không sửa file nào trong số đó.
"""
