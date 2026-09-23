"""
config.py
---------
Cấu hình trung tâm cho Agentic AI Audit System (v2 — RAG + Persistence).

THAY ĐỔI SO VỚI BẢN CŨ:
  * KHÔNG import streamlit ở tầng module nữa (trước đây khiến CLI bắt buộc
    phải cài Streamlit và nuốt exception của st.secrets). Việc đọc secret
    được đưa vào hàm _read_secret() với import trễ (lazy import).
  * Bổ sung cấu hình cho tầng RAG / Vector DB và kho luật cố định.
  * Bổ sung PROMPT_VERSION phục vụ audit trail (khả năng tái lập báo cáo).

CÁCH CẤU HÌNH API KEY (chọn 1 trong 3):
  1) Biến môi trường:        export GEMINI_API_KEY=...
  2) File .env cùng thư mục: GEMINI_API_KEY=...
  3) Streamlit secrets:      .streamlit/secrets.toml -> GEMINI_API_KEY = "..."

LƯU Ý BẢO MẬT: không commit file .env; không dán key thật vào chat/log/ảnh.
Nếu key từng bị lộ, hãy revoke tại https://aistudio.google.com/apikey.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 1) THƯ MỤC LÀM VIỆC (đường dẫn tuyệt đối, không phụ thuộc CWD)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# FIX (Vector DB rỗng — nguyên nhân gốc #2): nếu chỉ dùng Path(os.getenv(...))
# mà không .resolve(), một CHROMA_DB_DIR truyền vào dạng tương đối (ví dụ
# "./chroma_db") sẽ được hiểu tương đối theo CWD tại thời điểm chạy — không
# phải theo BASE_DIR. Khi tiến trình index (ví dụ script ingest chạy tay) và
# tiến trình phục vụ (Uvicorn/Streamlit) khởi động từ hai thư mục làm việc
# khác nhau, mỗi bên sẽ tự tạo một thư mục chroma_db/ rỗng của riêng mình ->
# Agent luôn thấy Vector DB rỗng dù đã index. .resolve() ép mọi đường dẫn,
# kể cả khi override qua biến môi trường, về dạng tuyệt đối và nhất quán.
LAW_STORAGE_DIR = Path(os.getenv("LAW_STORAGE_DIR", BASE_DIR / "downloaded_laws")).resolve()
CHROMA_DB_DIR = Path(os.getenv("CHROMA_DB_DIR", BASE_DIR / "chroma_db")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "audit_reports")).resolve()
TMP_DIR = Path(os.getenv("TMP_DIR", BASE_DIR / ".tmp_uploads")).resolve()
LOG_DIR = Path(os.getenv("LOG_DIR", BASE_DIR / "logs")).resolve()

for _directory in (LAW_STORAGE_DIR, CHROMA_DB_DIR, OUTPUT_DIR, TMP_DIR, LOG_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2) ĐỌC SECRET (env -> .env -> streamlit secrets)
# ============================================================
def _read_secret(key: str, default: str = "") -> str:
    """
    Đọc secret theo thứ tự ưu tiên: biến môi trường/.env trước, sau đó mới
    tới st.secrets. Import streamlit được đặt BÊN TRONG hàm để module này
    vẫn dùng được ở môi trường CLI thuần (không cài Streamlit).
    """
    value = os.getenv(key)
    if value:
        return value.strip()

    try:  # pragma: no cover - chỉ chạy khi ở trong runtime Streamlit
        import streamlit as st

        return str(st.secrets[key]).strip()
    except Exception:
        return default


GEMINI_API_KEY = _read_secret("GEMINI_API_KEY")

# Mật khẩu chặn vai trò Admin trên UI. Để trống = không chặn (CHỈ dùng khi demo
# cục bộ). Xem cảnh báo về phân quyền trong README.
ADMIN_PASSWORD = _read_secret("ADMIN_PASSWORD", "")


# ============================================================
# 3) CẤU HÌNH MODEL GEMINI
# ============================================================
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# Model embedding dùng cho Vector DB. Gemini embedding xử lý tiếng Việt tốt
# hơn đáng kể so với all-MiniLM mặc định của ChromaDB.
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# gemini-embedding-001 mặc định 3072 chiều; hạ xuống 768 để giảm dung lượng
# ChromaDB ~4 lần mà chất lượng truy hồi gần như không đổi.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
# FIX (429 RESOURCE_EXHAUSTED — nguyên nhân gốc #1): batch quá nhỏ (từng bị
# để =1) khiến một file 287 chunk bắn ra 287 request embed riêng lẻ trong
# chưa đầy 1 phút, chạm ngưỡng requests/phút của Free Tier. Giữ batch trong
# khoảng khuyến nghị 50-100. Việc throttling giữa các batch + xử lý retryDelay
# khi bị 429 nằm ở rag_engine.embed_texts().
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))

# temperature = 0.0 giúp GIẢM PHƯƠNG SAI đầu ra giữa các lần chạy.
# Lưu ý: điều này KHÔNG loại bỏ ảo giác và không đảm bảo tính xác định tuyệt
# đối (batching/routing ở tầng hạ tầng vẫn có thể gây khác biệt nhỏ).
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))

# Số lần thử lại khi gọi API lỗi tạm thời (429/500/503).
# FIX (503 UNAVAILABLE — nguyên nhân gốc #2): lịch nghỉ cũ 2s -> 4s quá ngắn
# khi hạ tầng Google đang tắc nghẽn ở giờ cao điểm và request bị dội lại gần
# như ngay lập tức. Đổi sang lịch nghỉ tăng dần rõ rệt: 5s -> 15s -> 30s.
# API_MAX_RETRIES = 4 để cả 3 mốc nghỉ trong API_RETRY_DELAYS đều thực sự
# được dùng hết trước khi báo lỗi (lần thử cuối cùng không cần nghỉ thêm).
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "4"))
API_RETRY_DELAYS = [5.0, 15.0, 30.0]

# Phiên bản prompt — ghi vào báo cáo để có thể tái lập/đối chiếu về sau.
PROMPT_VERSION = "2026.09-v2-rag"


# ============================================================
# 4) CẤU HÌNH RAG / VECTOR DB
# ============================================================
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "vietnam_law_corpus")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

# Số truy vấn con tối đa khi làm multi-query retrieval.
RAG_MAX_SUBQUERIES = int(os.getenv("RAG_MAX_SUBQUERIES", "4"))

# Bật/tắt RAG. Khi tắt (hoặc khi ChromaDB không khả dụng), hệ thống tự động
# quay về chế độ FULL-CONTEXT: gom toàn bộ văn bản trong downloaded_laws/.
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() in {"1", "true", "yes"}


# ============================================================
# 5) RÀNG BUỘC ĐẦU VÀO & KIỂM SOÁT CHẤT LƯỢNG
# ============================================================
ALLOWED_LAW_EXTENSIONS = {".pdf", ".docx", ".txt"}
ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".xlsx", ".xls"}

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Ngưỡng chặn "văn bản luật rỗng": PDF scan không có lớp text sẽ trả về chuỗi
# gần như rỗng. Nếu không chặn, Agent sẽ chạy trên kho luật rỗng và sinh ra
# kết luận vô căn cứ — đây là kiểu lỗi thầm lặng nguy hiểm nhất của hệ thống.
MIN_EXTRACTED_TEXT_LENGTH = int(os.getenv("MIN_EXTRACTED_TEXT_LENGTH", "200"))

# Ngưỡng khớp cho bước kiểm chứng trích dẫn (grounding verification).
GROUNDING_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", "0.85"))

# Cắt bớt tài liệu quá dài trước khi đưa vào prompt (ký tự).
MAX_DOCUMENT_CHARS = int(os.getenv("MAX_DOCUMENT_CHARS", "60000"))


# ============================================================
# 6) LOGGING
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_CONFIGURED = False


def setup_logging() -> logging.Logger:
    """Khởi tạo logging ghi đồng thời ra console và file logs/audit_system.log."""
    global _LOG_CONFIGURED
    logger = logging.getLogger("audit_system")

    if _LOG_CONFIGURED:
        return logger

    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    try:
        file_handler = logging.FileHandler(LOG_DIR / "audit_system.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Filesystem chỉ đọc (ví dụ một số môi trường cloud) — bỏ qua file log.
        pass

    logger.propagate = False
    _LOG_CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Lấy logger con, kế thừa cấu hình từ setup_logging()."""
    setup_logging()
    return logging.getLogger(f"audit_system.{name}")


# ============================================================
# 7) KIỂM TRA CẤU HÌNH
# ============================================================
class ConfigError(RuntimeError):
    """Lỗi cấu hình hệ thống (thiếu API key, thư mục không ghi được...)."""


def validate_config() -> None:
    """
    Kiểm tra cấu hình bắt buộc TRƯỚC khi khởi tạo Agent, để báo lỗi sớm và rõ
    ràng thay vì để lỗi bật ra giữa lúc đang gọi API.
    """
    if not GEMINI_API_KEY:
        raise ConfigError(
            "Chưa cấu hình GEMINI_API_KEY.\n"
            "  - Cách 1: tạo file .env với dòng  GEMINI_API_KEY=AIza...\n"
            "  - Cách 2: export GEMINI_API_KEY=AIza...\n"
            "  - Cách 3: .streamlit/secrets.toml -> GEMINI_API_KEY = \"AIza...\"\n"
            "Lấy key tại: https://aistudio.google.com/apikey"
        )

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ConfigError(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) phải nhỏ hơn CHUNK_SIZE ({CHUNK_SIZE})."
        )


def describe_runtime() -> dict:
    """Trả về snapshot cấu hình runtime để ghi vào audit trail của báo cáo."""
    return {
        "model": GEMINI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "temperature": TEMPERATURE,
        "prompt_version": PROMPT_VERSION,
        "rag_enabled": RAG_ENABLED,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": RAG_TOP_K,
    }
