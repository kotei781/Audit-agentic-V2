"""
rag_engine_3.py
--------------
Phiên bản RAG tối ưu hóa cho kiểm toán tài chính và pháp lý Việt Nam.

Nâng cấp trọng tâm:
1) Hybrid Search: Dense (Chroma vector) + Sparse (BM25).
2) Query Expansion nhẹ bằng Gemini Flash với fallback heuristic.
3) Relaxed legal grounding: ưu tiên cả quy định chung và điều/khoản cụ thể.
4) Giữ nguyên kiểu trả về và cơ chế cache SHA-256 / retry 429 của rag_engine.py.
"""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import config
import data_ingestion
from data_ingestion import IngestionError, compute_sha256, extract_text

logger = config.get_logger("rag_engine_3")
if logger.hasHandlers():
    logger.handlers.clear()


class RAGError(Exception):
    """Lỗi chung của tầng RAG."""


class VectorDBUnavailableError(RAGError):
    """ChromaDB chưa được cài hoặc không khởi tạo được."""


class EmbeddingError(RAGError):
    """Lỗi khi gọi API embedding."""


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    article: str
    distance: float
    sparse_score: float = 0.0
    dense_score: float = 0.0
    combined_score: float = 0.0

    @property
    def similarity(self) -> float:
        return max(0.0, min(1.0, 1.0 - self.distance))

    def to_context_block(self) -> str:
        header = f"[Nguồn: {self.source}"
        if self.article:
            header += f" | {self.article}"
        header += f" | tương đồng: {self.similarity:.2f}]"
        return f"{header}\n{self.text}"


_client = None
_collection = None
_genai_client = None


def _import_chromadb():
    try:
        import chromadb
        from chromadb.config import Settings

        return chromadb, Settings
    except ImportError as exc:  # pragma: no cover
        raise VectorDBUnavailableError(
            "Chưa cài ChromaDB. Chạy: pip install chromadb\n"
            "Hoặc đặt RAG_ENABLED=false trong .env để dùng chế độ full-context."
        ) from exc


def get_client():
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
    global _collection
    if _collection is not None:
        return _collection

    client = get_client()
    try:
        _collection = client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as exc:  # noqa: BLE001
        raise VectorDBUnavailableError(f"Không mở được collection: {exc}") from exc
    return _collection


def is_vector_db_ready() -> bool:
    if not config.RAG_ENABLED:
        return False
    try:
        return get_collection().count() > 0
    except RAGError:
        return False
    except Exception:  # noqa: BLE001
        return False


def _get_genai_client():
    global _genai_client
    if _genai_client is not None:
        return _genai_client

    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover
        raise EmbeddingError("Chưa cài google-genai. Chạy: pip install google-genai") from exc

    config.validate_config()
    api_key = config.GEMINI_API_KEY or os.getenv("AUDIT_GEMINI_KEY_1")
    _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def _l2_normalize(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return list(vector)
    return [v / norm for v in vector]


_RETRY_DELAY_PATTERN = re.compile(r"retrydelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _extract_retry_delay(exc: Exception, default: float = 20.0) -> float:
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
    if not texts:
        return []

    client = _get_genai_client()
    vectors: List[List[float]] = []
    total_batches = max(1, math.ceil(len(texts) / config.EMBEDDING_BATCH_SIZE))

    for batch_index in range(total_batches):
        start = batch_index * config.EMBEDDING_BATCH_SIZE
        end = min(start + config.EMBEDDING_BATCH_SIZE, len(texts))
        batch = list(texts[start:end])
        response = None
        last_error: Optional[Exception] = None

        for attempt in range(1, config.API_MAX_RETRIES + 1):
            try:
                from google.genai import types

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
                    "Embedding batch %d/%d bị chặn hạn ngạch 429 (lần thử %d/%d) — nghỉ %.1fs theo retryDelay rồi thử lại.",
                    batch_index + 1,
                    total_batches,
                    attempt,
                    config.API_MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)

        if response is None:
            raise EmbeddingError(
                f"Lỗi khi gọi API embedding ({config.EMBEDDING_MODEL}): {last_error}"
            ) from last_error

        for item in response.embeddings:
            values = list(item.values or [])
            if not values:
                raise EmbeddingError("API embedding trả về vector rỗng.")
            vectors.append(_l2_normalize(values))

        if end < len(texts):
            time.sleep(1)

    if len(vectors) != len(texts):
        raise EmbeddingError(
            f"Số vector trả về ({len(vectors)}) không khớp số đoạn text ({len(texts)})."
        )

    return vectors


_ARTICLE_PATTERN = re.compile(r"(?im)^\s*(Điều\s+\d+[a-zA-Z]?\s*[:.\-–]?.*)$")
_CHAPTER_PATTERN = re.compile(r"(?im)^\s*(Chương\s+[IVXLCDM\d]+.*)$")
_PAGE_MARKER = re.compile(r"(?m)^--- Trang \d+ ---$")


@dataclass
class LawChunk:
    chunk_id: str
    text: str
    source: str
    article: str
    chapter: str
    chunk_index: int

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "article": self.article or "",
            "chapter": self.chapter or "",
            "chunk_index": self.chunk_index,
            "char_count": len(self.text),
        }


def _split_by_size(text: str, chunk_size: int, overlap: int) -> List[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    pieces: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
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
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    text = _PAGE_MARKER.sub("", text or "").strip()
    if not text:
        return []

    source_hash = compute_sha256(text.encode("utf-8"))[:10]
    markers = [(m.start(), m.group(1).strip()) for m in _ARTICLE_PATTERN.finditer(text)]

    segments: List[Tuple[str, str]] = []
    if not markers:
        segments.append(("", text))
    else:
        if markers[0][0] > 0:
            segments.append(("", text[: markers[0][0]]))
        for index, (position, title) in enumerate(markers):
            end = markers[index + 1][0] if index + 1 < len(markers) else len(text)
            segments.append((title, text[position:end]))

    chapter_marks = [(m.start(), m.group(1).strip()) for m in _CHAPTER_PATTERN.finditer(text)]

    def _chapter_at(offset: int) -> str:
        current = ""
        for position, title in chapter_marks:
            if position <= offset:
                current = title
            else:
                break
        return current

    chunks: List[LawChunk] = []
    cursor = 0
    for article_title, body in segments:
        body = body.strip()
        if not body:
            continue
        chapter = _chapter_at(cursor)
        cursor += len(body)
        for piece in _split_by_size(body, chunk_size, overlap):
            if article_title and not piece.lstrip().lower().startswith("điều"):
                piece = f"[{article_title}]\n{piece}"
            chunks.append(
                LawChunk(
                    chunk_id=f"{source_hash}-{len(chunks):04d}",
                    text=piece,
                    source=source,
                    article=article_title,
                    chapter=chapter,
                    chunk_index=len(chunks),
                )
            )
    return chunks


def ingest_law_to_vector_db(
    file_path: Union[str, Path],
    force_reindex: bool = False,
) -> Dict[str, Any]:
    path = Path(file_path)
    source = path.name
    collection = get_collection()

    try:
        text = extract_text(path, strict=True)
    except IngestionError:
        raise

    content_hash = compute_sha256(text.encode("utf-8"))
    existing = collection.get(where={"source": source}, limit=1, include=["metadatas"])
    if not force_reindex and existing.get("ids"):
        metadata = (existing.get("metadatas") or [{}])[0] or {}
        if metadata.get("content_hash") == content_hash:
            logger.info("'%s' đã được index với hash tương ứng; bỏ qua.", source)
            return {"source": source, "chunks": 0, "skipped": True, "status": "cached"}

    chunks = chunk_legal_text(text, source)
    if not chunks:
        raise IngestionError(f"Không tạo được chunk nào từ {source}")

    ids = [c.chunk_id for c in chunks]
    metadatas: List[Dict[str, Any]] = []
    for chunk in chunks:
        metadata = chunk.to_metadata()
        metadata["content_hash"] = content_hash
        metadata["source_name"] = source
        metadatas.append(metadata)

    try:
        collection.upsert(
            ids=ids,
            documents=[c.text for c in chunks],
            metadatas=metadatas,
            embeddings=embed_texts([c.text for c in chunks], task_type="RETRIEVAL_DOCUMENT"),
        )
    except Exception as exc:  # noqa: BLE001
        raise RAGError(f"Không ghi được vào ChromaDB: {exc}") from exc

    return {"source": source, "chunks": len(chunks), "skipped": False, "status": "indexed"}


def ingest_all_persisted_laws(force_reindex: bool = False) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for info in data_ingestion.list_persisted_laws():
        try:
            results.append(ingest_law_to_vector_db(info.path, force_reindex=force_reindex))
        except (RAGError, IngestionError) as exc:
            logger.error("Index thất bại '%s': %s", info.filename, exc)
            results.append(
                {"source": info.filename, "chunks": 0, "skipped": True, "status": "error", "error": str(exc)}
            )
    return results


def rebuild_index() -> List[Dict[str, Any]]:
    try:
        client = get_client()
        client.delete_collection(name=config.CHROMA_COLLECTION_NAME)
    except Exception:  # noqa: BLE001
        logger.warning("Không xóa được collection cũ (có thể chưa tồn tại).")
    global _collection
    _collection = None
    get_collection()
    return ingest_all_persisted_laws(force_reindex=True)


def count_chunks_for_source(source: str) -> int:
    try:
        return len(get_collection().get(where={"source": source}, include=[]).get("ids", []))
    except Exception:  # noqa: BLE001
        return 0


def delete_source(source: str) -> int:
    collection = get_collection()
    try:
        existing = collection.get(where={"source": source}, include=[])
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception as exc:  # noqa: BLE001
        raise RAGError(f"Không xóa được chunk của '{source}': {exc}") from exc


def collection_stats() -> Dict[str, Any]:
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


_DOMAIN_TERMS: Dict[str, Tuple[str, ...]] = {
    "hóa đơn chứng từ hợp lệ": ("hóa đơn", "hoa don", "invoice", "chứng từ", "hd0", "số hóa đơn"),
    "thuế suất thuế giá trị gia tăng": ("vat", "gtgt", "thuế suất", "thue suat", "10%", "8%", "5%"),
    "mã số thuế người mua người bán": ("mã số thuế", "ma so thue", "mst", "tax_code", "tax code"),
    "chi phí được trừ khi tính thuế": ("chi phí", "chi phi", "expense", "khấu trừ", "được trừ", "tiep khach"),
    "thời điểm lập hóa đơn": ("ngày", "date", "kỳ", "thời điểm", "transaction_date", "quý", "tháng"),
    "thanh toán không dùng tiền mặt": ("tiền mặt", "tien mat", "chuyển khoản", "thanh toán"),
    "xử phạt vi phạm hành chính": ("xử phạt", "phạt", "vi phạm", "truy thu"),
    "nguyên tắc chung về chứng từ": ("quy định chung", "nguyên tắc", "điều kiện hợp lệ", "hóa đơn hợp lệ"),
}


def _tokenize_for_bm25(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-ZÀ-ỹ0-9%]+", (text or "").lower())
    return [token for token in tokens if len(token) > 1]


def extract_query_keywords(document_text: str, max_terms: int = 6) -> List[str]:
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


def _expand_query_terms(document_text: str, max_terms: int = 4) -> List[str]:
    seed_keywords = extract_query_keywords(document_text, max_terms=max_terms)
    expanded: List[str] = []
    for keyword in seed_keywords:
        expanded.append(keyword)
        if "hóa đơn" in keyword.lower() or "chứng từ" in keyword.lower():
            expanded.extend(["điều kiện hóa đơn hợp lệ", "quy định chung về hóa đơn", "hoá đơn điện tử"])
        if "thuế suất" in keyword.lower() or "vat" in keyword.lower():
            expanded.extend(["thuế suất gtgt", "thuế giá trị gia tăng", "căn cứ tính thuế"])
        if "mã số thuế" in keyword.lower():
            expanded.extend(["mã số thuế người mua", "mã số thuế người bán", "định danh doanh nghiệp"])
        if "chi phí" in keyword.lower():
            expanded.extend(["chi phí được khấu trừ", "nguyên tắc chi phí", "căn cứ ghi nhận chi phí"])
    deduped: List[str] = []
    seen = set()
    for term in expanded:
        key = term.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(term)
    return deduped[: max_terms * 2]


def _build_subqueries(document_text: str) -> List[str]:
    queries = _expand_query_terms(document_text, max_terms=4)
    seen, unique = set(), []
    for query in queries:
        key = query.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(query)
    return unique[: config.RAG_MAX_SUBQUERIES]


def _dense_hybrid_candidates(collection, subqueries: Sequence[str], top_k: int) -> Dict[str, Dict[str, Any]]:
    candidates: Dict[str, Dict[str, Any]] = {}
    if not subqueries:
        return candidates
    vector_queries = embed_texts(subqueries, task_type="RETRIEVAL_QUERY")
    per_query = max(2, math.ceil(top_k / max(1, len(subqueries))) + 1)
    try:
        response = collection.query(
            query_embeddings=vector_queries,
            n_results=min(per_query, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:  # noqa: BLE001
        raise RAGError(f"Truy vấn ChromaDB thất bại: {exc}") from exc

    ids_matrix = response.get("ids") or []
    documents = response.get("documents") or []
    metadatas = response.get("metadatas") or []
    distances = response.get("distances") or []
    for q_index, ids in enumerate(ids_matrix):
        docs = (documents[q_index] if q_index < len(documents) else []) or []
        meta_items = (metadatas[q_index] if q_index < len(metadatas) else []) or []
        dist_items = (distances[q_index] if q_index < len(distances) else []) or []
        for rank, chunk_id in enumerate(ids):
            doc = docs[rank] if rank < len(docs) else ""
            metadata = meta_items[rank] if rank < len(meta_items) else {}
            distance = float(dist_items[rank]) if rank < len(dist_items) else 1.0
            vector_score = max(0.0, 1.0 - distance)
            current = candidates.get(chunk_id)
            if current is None or vector_score > current["dense_score"]:
                candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": doc,
                    "source": (metadata or {}).get("source", "unknown"),
                    "article": (metadata or {}).get("article", ""),
                    "rank": rank,
                    "distance": distance,
                    "dense_score": vector_score,
                }
    return candidates


def _sparse_candidates(collection, subqueries: Sequence[str], top_k: int) -> Dict[str, Dict[str, Any]]:
    candidates: Dict[str, Dict[str, Any]] = {}
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return candidates

    result = collection.get(include=["documents", "metadatas"])
    ids = result.get("ids") or []
    docs = result.get("documents") or []
    if not ids or not docs:
        return candidates

    tokenized = [_tokenize_for_bm25(doc) for doc in docs]
    if not any(tokenized):
        return candidates

    bm25 = BM25Okapi(tokenized)
    max_score = 1.0
    for q_index, query in enumerate(subqueries):
        tokens = _tokenize_for_bm25(query)
        if not tokens:
            continue
        scores = bm25.get_scores(tokens)
        ranked_positions = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[: max(3, top_k * 2)]
        for rank, pos in enumerate(ranked_positions):
            chunk_id = ids[pos]
            score = float(scores[pos])
            max_score = max(max_score, score, 1.0)
            current = candidates.get(chunk_id)
            if current is None or score > current["sparse_score"]:
                candidates[chunk_id] = {
                    "chunk_id": chunk_id,
                    "sparse_rank": rank,
                    "sparse_score": score,
                    "text": docs[pos],
                    "source": "unknown",
                    "article": "",
                }

    for chunk_id, item in candidates.items():
        item["sparse_score"] = float(item["sparse_score"]) / max_score if max_score else 0.0
    return candidates


def _fuse_results(dense_map: Dict[str, Dict[str, Any]], sparse_map: Dict[str, Dict[str, Any]], top_k: int) -> List[RetrievedChunk]:
    merged: Dict[str, Dict[str, Any]] = {}
    for chunk_id, item in dense_map.items():
        merged[chunk_id] = {**item, "sparse_score": 0.0, "combined_score": item["dense_score"]}
    for chunk_id, item in sparse_map.items():
        entry = merged.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "text": "",
                "source": "unknown",
                "article": "",
                "rank": 999,
                "distance": 1.0,
                "dense_score": 0.0,
                "sparse_score": 0.0,
                "combined_score": 0.0,
            },
        )
        entry["sparse_score"] = max(entry.get("sparse_score", 0.0), float(item.get("sparse_score", 0.0)))
        entry["sparse_rank"] = min(entry.get("sparse_rank", 999), int(item.get("sparse_rank", 999)))
        entry["text"] = entry.get("text") or item.get("text", "")
        entry["source"] = entry.get("source") or item.get("source", "unknown")
        entry["article"] = entry.get("article") or item.get("article", "")

    combined: List[RetrievedChunk] = []
    for chunk_id, item in merged.items():
        dense_rank = item.get("rank", 999)
        sparse_rank = item.get("sparse_rank", 999)
        dense_term = 1.0 / (60 + dense_rank + 1.0) if dense_rank < 999 else 0.0
        sparse_term = 1.0 / (60 + sparse_rank + 1.0) if sparse_rank < 999 else 0.0
        rrf_score = dense_term + sparse_term
        dense_norm = float(item.get("dense_score", 0.0))
        sparse_norm = float(item.get("sparse_score", 0.0))
        weighted = min(1.0, 0.7 * dense_norm + 0.3 * sparse_norm)
        combined_score = max(rrf_score, weighted)
        item["combined_score"] = combined_score
        if item.get("text"):
            combined.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=item["text"],
                    source=item.get("source", "unknown"),
                    article=item.get("article", ""),
                    distance=float(item.get("distance", 1.0)),
                    sparse_score=sparse_norm,
                    dense_score=dense_norm,
                    combined_score=combined_score,
                )
            )
    combined.sort(key=lambda c: (-c.combined_score, c.distance))
    return combined[:top_k]


def retrieve_relevant_laws(
    query_document_text: str,
    top_k: Optional[int] = None,
) -> Tuple[str, List[RetrievedChunk]]:
    top_k = top_k or config.RAG_TOP_K
    collection = get_collection()

    total = collection.count()
    if total == 0:
        logger.warning("Vector DB rỗng — chưa có văn bản luật nào được index.")
        return "", []

    subqueries = _build_subqueries(query_document_text)
    if not subqueries:
        subqueries = ["quy định chung về hóa đơn và thuế giá trị gia tăng"]

    dense_map = _dense_hybrid_candidates(collection, subqueries, top_k)
    sparse_map = _sparse_candidates(collection, subqueries, top_k)
    merged = _fuse_results(dense_map, sparse_map, top_k)
    if not merged:
        return "", []

    general_chunks = [c for c in merged if "nguyên tắc" in (c.text or "").lower() or "quy định chung" in (c.text or "").lower()]
    ranked = general_chunks + [c for c in merged if c not in general_chunks]
    context = "\n\n".join(chunk.to_context_block() for chunk in ranked[:top_k])
    logger.info("Hybrid retrieval: %d chunk được giữ lại, dense=%d, sparse=%d.", len(ranked), len(dense_map), len(sparse_map))
    return context, ranked[:top_k]


def get_law_context(document_text: str, top_k: Optional[int] = None) -> Tuple[str, List[RetrievedChunk], str]:
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


__all__ = [
    "RAGError",
    "VectorDBUnavailableError",
    "EmbeddingError",
    "RetrievedChunk",
    "LawChunk",
    "get_client",
    "get_collection",
    "embed_texts",
    "chunk_legal_text",
    "ingest_law_to_vector_db",
    "ingest_all_persisted_laws",
    "rebuild_index",
    "delete_source",
    "collection_stats",
    "extract_query_keywords",
    "retrieve_relevant_laws",
    "get_law_context",
]
