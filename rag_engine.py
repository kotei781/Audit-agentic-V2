"""
rag_engine.py
--------------
TẦNG RAG & VECTOR DATABASE (ChromaDB persistent).

Giải quyết đúng lỗ hổng kiến trúc đã nêu: thay vì nhồi toàn bộ văn bản luật
vào prompt mỗi lần kiểm toán, hệ thống index kho luật MỘT LẦN vào ChromaDB
(lưu vĩnh viễn ở `./chroma_db`) rồi chỉ truy hồi các đoạn liên quan.

BA ĐIỂM THIẾT KẾ ĐÁNG CHÚ Ý:

1. CHUNKING THEO CẤU TRÚC PHÁP LÝ, KHÔNG CẮT MÙ THEO ĐỘ DÀI
   Văn bản luật Việt Nam có cấu trúc Chương > Điều > Khoản > Điểm. Cắt mù mỗi
   1000 ký tự sẽ xé đôi một Điều, khiến chunk truy hồi được mất phần định
   nghĩa hoặc mất phần chế tài. Engine này ưu tiên tách theo ranh giới "Điều
   N", chỉ khi một Điều dài hơn chunk_size mới cắt nhỏ tiếp (có overlap).
   Mỗi chunk đều mang metadata `article` để trích dẫn chính xác.

2. EMBEDDING BẰNG GEMINI, KHÔNG DÙNG MẶC ĐỊNH CỦA CHROMA
   Mặc định ChromaDB dùng all-MiniLM-L6-v2 (tiếng Anh, 384 chiều) — chất
   lượng trên văn bản pháp luật tiếng Việt rất kém. Ở đây ta tự tính embedding
   bằng Gemini và truyền thẳng vector vào Chroma. Cách này cũng tránh được
   rắc rối tương thích API của lớp custom EmbeddingFunction giữa các phiên
   bản ChromaDB.

3. MULTI-QUERY RETRIEVAL
   Một chứng từ thường chứa nhiều loại rủi ro (hóa đơn, thuế suất, mã số
   thuế...). Một truy vấn duy nhất từ toàn bộ chứng từ sẽ cho vector "trung
   bình hóa", truy hồi kém. Engine tách thành nhiều truy vấn con theo từ khóa
   nghiệp vụ rồi hợp nhất kết quả (dedupe theo chunk id, giữ khoảng cách tốt nhất).
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import config
import data_ingestion
from data_ingestion import IngestionError, compute_sha256, extract_text

logger = config.get_//logger("rag_engine")
if logger.hasHandlers():
    logger.handlers.clear()


# ============================================================
# NGOẠI LỆ
# ============================================================
class RAGError(Exception):
    """Lỗi chung của tầng RAG."""


class VectorDBUnavailableError(RAGError):
    """ChromaDB chưa được cài hoặc không khởi tạo được."""


class EmbeddingError(RAGError):
    """Lỗi khi gọi API embedding."""


# ============================================================
# KẾT QUẢ TRUY HỒI
# ============================================================
@dataclass
class RetrievedChunk:
    """Một đoạn luật được truy hồi từ Vector DB."""

    chunk_id: str
    text: str
    source: str
    article: str
    distance: float

    @property
    def similarity(self) -> float:
        """Chuyển cosine distance -> điểm tương đồng [0, 1] cho dễ đọc."""
        return max(0.0, min(1.0, 1.0 - self.distance))

    def to_context_block(self) -> str:
        header = f"[Nguồn: {self.source}"
        if self.article:
            header += f" | {self.article}"
        header += f" | tương đồng: {self.similarity:.2f}]"
        return f"{header}\n{self.text}"


# ============================================================
# 1) KẾT NỐI CHROMADB
# ============================================================
_client = None
_collection = None


def _import_chromadb():
    try:
        import chromadb
        from chromadb.config import Settings

        return chromadb, Settings
    except ImportError as exc:  # pragma: no cover
        raise VectorDBUnavailableError(
            "Chưa cài ChromaDB. Chạy:  pip install chromadb\n"
            "Hoặc đặt RAG_ENABLED=false trong .env để dùng chế độ full-context."
        ) from exc


def get_client():
    """Trả về ChromaDB PersistentClient (singleton trong tiến trình)."""
    global _client
    if _client is not None:
        return _client

    chromadb, Settings = _import_chromadb()
    try:
        _client = chromadb.PersistentClient(
            path=str(config.CHROMA_DB_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )
    except Exception as exc:  # noqa: BLE001
        raise VectorDBUnavailableError(
            f"Không khởi tạo được ChromaDB tại {config.CHROMA_DB_DIR}: {exc}"
        ) from exc

    return _client


def get_collection():
    """Trả về collection lưu kho luật, tạo mới nếu chưa có."""
    global _collection
    if _collection is not None:
        return _collection

    client = get_client()
    try:
        _collection = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            # cosine phù hợp với embedding đã chuẩn hóa; mặc định của Chroma là L2.
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:  # noqa: BLE001
        raise VectorDBUnavailableError(f"Không mở được collection: {exc}") from exc

    return _collection


def is_vector_db_ready() -> bool:
    """True nếu ChromaDB dùng được VÀ đã có ít nhất 1 chunk trong kho."""
    if not config.RAG_ENABLED:
        return False
    try:
        return get_collection().count() > 0
    except RAGError:
        return False
    except Exception:  # noqa: BLE001
        return False


# ============================================================
# 2) EMBEDDING BẰNG GEMINI
# ============================================================
_genai_client = None


def _get_genai_client():
    global _genai_client
    if _genai_client is not None:
        return _genai_client

    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover
        raise EmbeddingError("Chưa cài google-genai. Chạy: pip install google-genai") from exc

    config.validate_config()
    _genai_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _genai_client


def _l2_normalize(vector: Sequence[float]) -> List[float]:
    """
    Chuẩn hóa L2. Bắt buộc khi dùng output_dimensionality < mặc định:
    Google chỉ chuẩn hóa sẵn vector ở số chiều đầy đủ.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return list(vector)
    return [v / norm for v in vector]


_RETRY_DELAY_PATTERN = re.compile(r"retrydelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _extract_retry_delay(exc: Exception, default: float = 20.0) -> float:
    """
    Bóc số giây `retryDelay` mà Google trả về kèm lỗi 429 RESOURCE_EXHAUSTED
    (ví dụ '...retryDelay: 41s...'). Không tìm thấy thì dùng mặc định.
    """
    match = _RETRY_DELAY_PATTERN.search(str(exc))
    if match:
        try:
            return float(match.group(1))
        except (TypeError, ValueError):
            pass
    return default


def _is_rate_limited(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in ("429", "resource_exhausted", "quota", "rate limit"))


def embed_texts(texts: Sequence[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """
    Sinh embedding cho danh sách text bằng Gemini, chia lô để tránh quá tải.

    Args:
        texts: danh sách đoạn text.
        task_type: "RETRIEVAL_DOCUMENT" khi index, "RETRIEVAL_QUERY" khi truy vấn.
                   Dùng đúng task_type cải thiện đáng kể chất lượng truy hồi.

    FIX (429 RESOURCE_EXHAUSTED — nguyên nhân gốc #2 "thiếu throttling"):
      * Chủ động nghỉ 1s giữa các batch để không dồn dập vượt hạn ngạch
        requests/phút (đặc biệt ở Free Tier).
      * Nếu vẫn dính 429, đọc đúng `retryDelay` Google yêu cầu trong thông
        báo lỗi rồi nghỉ đúng khoảng đó trước khi thử lại batch đó (tối đa
        config.API_MAX_RETRIES lần) — thay vì thất bại ngay lập tức và bị
        ingest_all_persisted_laws() nuốt lỗi, bỏ qua cả file (nguyên nhân
        gốc #1 của lỗi "Vector DB rỗng").
    """
    if not texts:
        return []

    from google.genai import types

    client = _get_genai_client()
    vectors: List[List[float]] = []
    batch_size = max(1, config.EMBEDDING_BATCH_SIZE)
    total_batches = math.ceil(len(texts) / batch_size)

    for batch_index, start in enumerate(range(0, len(texts), batch_size)):
        batch = list(texts[start : start + batch_size])
        response = None
        last_error: Optional[Exception] = None

        for attempt in range(1, config.API_MAX_RETRIES + 1):
            try:
                response = client.models.embed_content(
                    model=config.EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=config.EMBEDDING_DIM,
                    ),
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if not _is_rate_limited(exc) or attempt == config.API_MAX_RETRIES:
                    raise EmbeddingError(
                        f"Lỗi khi gọi API embedding ({config.EMBEDDING_MODEL}): {exc}"
                    ) from exc

                delay = _extract_retry_delay(exc)
                logger.warning(
                    "Embedding batch %d/%d bị chặn hạn ngạch 429 (lần thử %d/%d) — "
                    "nghỉ %.1fs theo retryDelay rồi thử lại.",
                    batch_index + 1, total_batches, attempt, config.API_MAX_RETRIES, delay,
                )
                time.sleep(delay)

        if response is None:  # pragma: no cover - vòng lặp trên luôn raise hoặc break trước
            raise EmbeddingError(
                f"Lỗi khi gọi API embedding ({config.EMBEDDING_MODEL}): {last_error}"
            ) from last_error

        for item in response.embeddings:
            values = list(item.values or [])
            if not values:
                raise EmbeddingError("API embedding trả về vector rỗng.")
            vectors.append(_l2_normalize(values))

        # Throttling chủ động giữa các batch — tránh dồn dập vượt hạn ngạch
        # requests/phút ngay cả khi chưa bị Google trả về lỗi 429.
        if start + batch_size < len(texts):
            time.sleep(1)

    if len(vectors) != len(texts):
        raise EmbeddingError(
            f"Số vector trả về ({len(vectors)}) không khớp số đoạn text ({len(texts)})."
        )

    return vectors


# ============================================================
# 3) CHUNKING THEO CẤU TRÚC VĂN BẢN PHÁP LUẬT
# ============================================================
_ARTICLE_PATTERN = re.compile(r"(?im)^\s*(Điều\s+\d+[a-zA-Z]?\s*[.:\-–]?.*)$")
_CHAPTER_PATTERN = re.compile(r"(?im)^\s*(Chương\s+[IVXLCDM\d]+.*)$")
_PAGE_MARKER = re.compile(r"(?m)^--- Trang \d+ ---$")


@dataclass
class LawChunk:
    """Một chunk luật chuẩn bị ghi vào Vector DB."""

    chunk_id: str
    text: str
    source: str
    article: str
    chapter: str
    chunk_index: int

    def to_metadata(self) -> Dict[str, Any]:
        # ChromaDB chỉ chấp nhận metadata kiểu str/int/float/bool (không None, không list).
        return {
            "source": self.source,
            "article": self.article or "",
            "chapter": self.chapter or "",
            "chunk_index": self.chunk_index,
            "char_count": len(self.text),
        }


def _split_by_size(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Cắt text dài theo kích thước, ưu tiên ranh giới câu/xuống dòng."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    pieces: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Lùi về ranh giới tự nhiên gần nhất trong 20% cuối của cửa sổ.
            window_start = max(start + int(chunk_size * 0.8), start + 1)
            candidates = [
                text.rfind("\n", window_start, end),
                text.rfind(". ", window_start, end),
                text.rfind("; ", window_start, end),
            ]
            boundary = max(candidates)
            if boundary > window_start:
                end = boundary + 1

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= len(text):
            break
        start = max(end - overlap, start + step)

    return pieces


def chunk_legal_text(
    text: str,
    source: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[LawChunk]:
    """
    Cắt văn bản luật thành các chunk, ưu tiên ranh giới "Điều N".

    Mỗi chunk được gắn kèm tiêu đề Điều/Chương ngay trong nội dung để khi
    Agent trích dẫn `rule_citation` thì luôn nhìn thấy số Điều — đây là điều
    kiện cần để trích dẫn pháp lý dùng được.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    text = _PAGE_MARKER.sub("", text or "").strip()
    if not text:
        return []

    source_hash = compute_sha256(text.encode("utf-8"))[:10]

    # --- Bước 1: xác định vị trí các mốc "Điều N" ---
    markers = [(m.start(), m.group(1).strip()) for m in _ARTICLE_PATTERN.finditer(text)]

    segments: List[Tuple[str, str]] = []  # (tiêu đề Điều, nội dung)
    if not markers:
        segments.append(("", text))
    else:
        if markers[0][0] > 0:
            segments.append(("", text[: markers[0][0]]))
        for index, (position, title) in enumerate(markers):
            end = markers[index + 1][0] if index + 1 < len(markers) else len(text)
            segments.append((title, text[position:end]))

    # --- Bước 2: xác định Chương hiện hành cho từng vị trí ---
    chapter_marks = [(m.start(), m.group(1).strip()) for m in _CHAPTER_PATTERN.finditer(text)]

    def _chapter_at(offset: int) -> str:
        current = ""
        for position, title in chapter_marks:
            if position <= offset:
                current = title
            else:
                break
        return current

    # --- Bước 3: cắt nhỏ từng segment nếu vượt chunk_size ---
    chunks: List[LawChunk] = []
    cursor = 0
    for article_title, body in segments:
        body = body.strip()
        if not body:
            continue

        chapter = _chapter_at(cursor)
        cursor += len(body)

        for piece in _split_by_size(body, chunk_size, overlap):
            # Gắn lại tiêu đề Điều vào các mảnh con để chunk luôn tự mô tả được.
            if article_title and not piece.lstrip().lower().startswith("điều"):
                piece = f"[{article_title}]\n{piece}"

            index = len(chunks)
            chunks.append(
                LawChunk(
                    chunk_id=f"{source_hash}-{index:04d}",
                    text=piece,
                    source=source,
                    article=article_title,
                    chapter=chapter,
                    chunk_index=index,
                )
            )

    logger.info("Đã cắt '%s' thành %d chunk (%d Điều).", source, len(chunks), len(markers))
    return chunks


# ============================================================
# 4) NẠP VĂN BẢN LUẬT VÀO VECTOR DB
# ============================================================
def ingest_law_to_vector_db(
    file_path: Union[str, Path],
    force_reindex: bool = False,
) -> Dict[str, Any]:
    """
    Cắt nhỏ -> embed -> lưu vĩnh viễn một văn bản luật vào ChromaDB.

    Args:
        file_path: đường dẫn file luật (thường nằm trong downloaded_laws/).
        force_reindex: True = xóa index cũ của file này rồi index lại.

    Returns:
        dict thống kê: {"source", "chunks", "skipped", "status"}

    Raises:
        VectorDBUnavailableError, EmbeddingError, IngestionError
    """
    path = Path(file_path)
    source = path.name
    collection = get_collection()

    existing = count_chunks_for_source(source)
    if existing and not force_reindex:
        logger.info("'%s' đã có %d chunk trong Vector DB — bỏ qua.", source, existing)
        return {"source": source, "chunks": existing, "skipped": True, "status": "already_indexed"}

    if existing and force_reindex:
        delete_source(source)

    try:
        text = extract_text(path, strict=True)
    except IngestionError:
        raise

    chunks = chunk_legal_text(text, source=source)
    if not chunks:
        raise IngestionError(f"Không tạo được chunk nào từ '{source}'.")

    logger.info("Đang embed %d chunk của '%s'...", len(chunks), source)
    vectors = embed_texts([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT")

    try:
        # upsert (không phải add) để index lại cùng một file không bị lỗi trùng id.
        collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
        )
    except Exception as exc:  # noqa: BLE001
        raise RAGError(f"Không ghi được vào ChromaDB: {exc}") from exc

    logger.info("Đã index '%s': %d chunk.", source, len(chunks))
    return {"source": source, "chunks": len(chunks), "skipped": False, "status": "indexed"}


def ingest_all_persisted_laws(force_reindex: bool = False) -> List[Dict[str, Any]]:
    """Index (hoặc index lại) toàn bộ file đang có trong kho luật cố định."""
    results: List[Dict[str, Any]] = []
    for info in data_ingestion.list_persisted_laws():
        try:
            results.append(ingest_law_to_vector_db(info.path, force_reindex=force_reindex))
        except (RAGError, IngestionError) as exc:
            logger.error("Index thất bại '%s': %s", info.filename, exc)
            results.append(
                {"source": info.filename, "chunks": 0, "skipped": True,
                 "status": "error", "error": str(exc)}
            )
    return results


def rebuild_index() -> List[Dict[str, Any]]:
    """Xóa sạch collection rồi index lại toàn bộ kho luật."""
    try:
        client = get_client()
        client.delete_collection(name=config.CHROMA_COLLECTION_NAME)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không xóa được collection cũ (có thể chưa tồn tại): %s", exc)

    global _collection
    _collection = None
    get_collection()
    return ingest_all_persisted_laws(force_reindex=True)


def count_chunks_for_source(source: str) -> int:
    """Đếm số chunk đang lưu của một file luật."""
    try:
        result = get_collection().get(where={"source": source}, include=[])
        return len(result.get("ids", []))
    except Exception:  # noqa: BLE001
        return 0


def delete_source(source: str) -> int:
    """Xóa toàn bộ chunk của một văn bản khỏi Vector DB."""
    collection = get_collection()
    try:
        existing = collection.get(where={"source": source}, include=[])
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        logger.info("Đã xóa %d chunk của '%s' khỏi Vector DB.", len(ids), source)
        return len(ids)
    except Exception as exc:  # noqa: BLE001
        raise RAGError(f"Không xóa được chunk của '{source}': {exc}") from exc


def collection_stats() -> Dict[str, Any]:
    """Thống kê kho vector: tổng chunk và số chunk theo từng văn bản."""
    try:
        collection = get_collection()
        total = collection.count()
        by_source: Dict[str, int] = {}
        if total:
            data = collection.get(include=["metadatas"])
            for metadata in data.get("metadatas", []) or []:
                name = (metadata or {}).get("source", "unknown")
                by_source[name] = by_source.get(name, 0) + 1
        return {"available": True, "total_chunks": total, "by_source": by_source}
    except RAGError as exc:
        return {"available": False, "total_chunks": 0, "by_source": {}, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "total_chunks": 0, "by_source": {}, "error": str(exc)}


# ============================================================
# 5) TRUY HỒI ĐOẠN LUẬT LIÊN QUAN
# ============================================================
# Từ điển nghiệp vụ kế toán/thuế Việt Nam dùng để dựng truy vấn con.
# Cách này cố ý KHÔNG gọi thêm một lượt LLM để sinh query: giữ tính xác định,
# không tốn thêm chi phí, và không tạo thêm một điểm ảo giác nữa trong luồng.
_DOMAIN_TERMS: Dict[str, Tuple[str, ...]] = {
    "hóa đơn chứng từ hợp lệ": ("hóa đơn", "hoa don", "invoice", "chứng từ", "hd0", "số hóa đơn"),
    "thuế suất thuế giá trị gia tăng": ("vat", "gtgt", "thuế suất", "thue suat", "10%", "8%", "5%"),
    "mã số thuế người mua người bán": ("mã số thuế", "ma so thue", "mst", "tax_code", "tax code"),
    "chi phí được trừ khi tính thuế thu nhập doanh nghiệp": (
        "chi phí", "chi phi", "tiếp khách", "tiep khach", "expense", "khấu trừ", "được trừ",
    ),
    "thời điểm lập hóa đơn và kỳ kê khai": (
        "ngày", "date", "kỳ", "thời điểm", "transaction_date", "quý", "tháng",
    ),
    "thanh toán không dùng tiền mặt": ("tiền mặt", "tien mat", "chuyển khoản", "thanh toán"),
    "xử phạt vi phạm hành chính về thuế": ("xử phạt", "vi phạm", "phạt", "truy thu"),
}


def extract_query_keywords(document_text: str, max_terms: int = 6) -> List[str]:
    """
    Suy ra các chủ đề pháp lý cần tra cứu từ nội dung chứng từ.

    Trả về danh sách cụm truy vấn (tiếng Việt, dạng ngữ nghĩa) thay vì keyword
    rời rạc — vector search hoạt động tốt hơn nhiều với cụm có ngữ nghĩa đầy đủ.
    """
    lowered = (document_text or "").lower()
    scored: List[Tuple[int, str]] = []

    for topic, triggers in _DOMAIN_TERMS.items():
        hits = sum(lowered.count(trigger) for trigger in triggers)
        if hits:
            scored.append((hits, topic))

    scored.sort(reverse=True)
    topics = [topic for _, topic in scored[:max_terms]]

    if not topics:
        topics = ["quy định về hóa đơn chứng từ và thuế giá trị gia tăng"]

    return topics


def _build_subqueries(document_text: str) -> List[str]:
    """
    Dựng danh sách truy vấn con dùng cho multi-query retrieval.

    FIX (nhiễu vector truy vấn RAG): bản cũ ghép thêm 1.200 ký tự trích thô
    của chứng từ (đặc biệt với Excel: ký tự phân cách, số liệu, tên cột...)
    làm một truy vấn riêng. Đưa dữ liệu bảng thô vào embedding như vậy làm
    lệch vector ngữ nghĩa, kéo giảm độ tương đồng cosine với văn bản luật.
    Nay CHỈ dùng các cụm từ khóa nghiệp vụ đã được suy luận qua
    extract_query_keywords() — không còn query nào chứa raw text chứng từ.
    """
    queries = extract_query_keywords(document_text)
    # Khử trùng lặp nhưng giữ nguyên thứ tự ưu tiên.
    seen, unique = set(), []
    for query in queries:
        key = query.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(query)
    return unique[: config.RAG_MAX_SUBQUERIES]


def retrieve_relevant_laws(
    query_document_text: str,
    top_k: Optional[int] = None,
) -> Tuple[str, List[RetrievedChunk]]:
    """
    Truy hồi top-K đoạn luật liên quan nhất tới chứng từ đang kiểm toán.

    Args:
        query_document_text: nội dung chứng từ tài chính.
        top_k: số đoạn luật trả về (mặc định config.RAG_TOP_K).

    Returns:
        (retrieved_law_context, danh sách RetrievedChunk)

    Raises:
        VectorDBUnavailableError nếu ChromaDB không dùng được.
    """
    top_k = top_k or config.RAG_TOP_K
    collection = get_collection()

    total = collection.count()
    if total == 0:
        logger.warning("Vector DB rỗng — chưa có văn bản luật nào được index.")
        return "", []

    subqueries = _build_subqueries(query_document_text)
    logger.info("Multi-query retrieval với %d truy vấn con.", len(subqueries))

    query_vectors = embed_texts(subqueries, task_type="RETRIEVAL_QUERY")

    # Mỗi truy vấn con lấy dư một ít rồi hợp nhất, để tổng thể vẫn đủ đa dạng.
    per_query = max(2, math.ceil(top_k / max(1, len(subqueries))) + 1)

    try:
        response = collection.query(
            query_embeddings=query_vectors,
            n_results=min(per_query, total),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:  # noqa: BLE001
        raise RAGError(f"Truy vấn ChromaDB thất bại: {exc}") from exc

    # --- Hợp nhất kết quả nhiều truy vấn: giữ khoảng cách TỐT NHẤT cho mỗi chunk ---
    best: Dict[str, RetrievedChunk] = {}
    ids_matrix = response.get("ids") or []

    for query_index in range(len(ids_matrix)):
        ids = ids_matrix[query_index] or []
        documents = (response.get("documents") or [[]])[query_index] or []
        metadatas = (response.get("metadatas") or [[]])[query_index] or []
        distances = (response.get("distances") or [[]])[query_index] or []

        for position, chunk_id in enumerate(ids):
            metadata = metadatas[position] if position < len(metadatas) else {}
            distance = float(distances[position]) if position < len(distances) else 1.0
            document = documents[position] if position < len(documents) else ""

            current = best.get(chunk_id)
            if current is None or distance < current.distance:
                best[chunk_id] = RetrievedChunk(
                    chunk_id=chunk_id,
                    text=document,
                    source=(metadata or {}).get("source", "unknown"),
                    article=(metadata or {}).get("article", ""),
                    distance=distance,
                )

    ranked = sorted(best.values(), key=lambda c: c.distance)[:top_k]

    if not ranked:
        return "", []

    context = "\n\n".join(chunk.to_context_block() for chunk in ranked)
    logger.info(
        "Truy hồi %d đoạn luật (tương đồng cao nhất %.2f).",
        len(ranked), ranked[0].similarity,
    )
    return context, ranked


def get_law_context(document_text: str, top_k: Optional[int] = None) -> Tuple[str, List[RetrievedChunk], str]:
    """
    Lấy ngữ cảnh pháp lý với cơ chế dự phòng nhiều tầng:

        1. RAG (ChromaDB)  ->  retrieval_mode = "rag"
        2. Nếu RAG tắt/lỗi/kho vector rỗng: gom toàn bộ downloaded_laws/
                           ->  retrieval_mode = "full_context"
        3. Nếu kho luật cũng rỗng: trả về chuỗi rỗng để tầng trên raise lỗi rõ ràng.

    Returns:
        (law_context, danh sách chunk đã truy hồi, retrieval_mode)
    """
    if config.RAG_ENABLED:
        try:
            context, chunks = retrieve_relevant_laws(document_text, top_k=top_k)
            if context.strip():
                return context, chunks, "rag"
            logger.warning("RAG không trả về kết quả — chuyển sang chế độ full-context.")
        except RAGError as exc:
            logger.warning("RAG không khả dụng (%s) — chuyển sang chế độ full-context.", exc)

    context = data_ingestion.load_all_persisted_laws()
    return context, [], "full_context"
