import os
import re
import time
import logging
from typing import List, Dict, Any, Optional
from google.genai import types

logger = logging.getLogger("rag_engine")

def embed_with_retry(
    client,
    texts: List[str],
    max_retries: int = 4,
    base_delay: float = 2.0,
    output_dim: int = 768
) -> List[List[float]]:
    """
    Tạo embedding cho một danh sách văn bản với cơ chế Exponential Backoff
    và ép cố định số chiều về 768 (tối ưu bộ nhớ ChromaDB).
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=texts,
                config=types.EmbedContentConfig(output_dimensionality=output_dim)
            )
            return [e.values for e in response.embeddings]

        except Exception as e:
            error_msg = str(e)
            if ("503" in error_msg or "429" in error_msg or "UNAVAILABLE" in error_msg) and attempt < max_retries:
                sleep_time = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Embedding API bị nghẽn/quá tải (lần {attempt}/{max_retries}). "
                    f"Thử lại sau {sleep_time:.1f}s..."
                )
                time.sleep(sleep_time)
            else:
                logger.error(f"Lỗi không thể phục hồi khi tạo Embedding: {e}")
                raise e

def chunk_legal_document(text: str, filename: str) -> List[Dict[str, Any]]:
    """
    Chia nhỏ văn bản luật theo ranh giới Chương / Điều bằng RegEx.
    """
    pattern = r"(Chương\s+[IVXLCDM\d]+\.?|Điều\s+\d+\.)"
    splits = re.split(pattern, text)

    chunks = []
    if len(splits) <= 1:
        chunk_size = 1000
        for i in range(0, len(text), chunk_size):
            chunks.append({
                "text": text[i:i + chunk_size],
                "metadata": {"source": filename, "section": "General"}
            })
        return chunks

    i = 1
    while i < len(splits):
        header_candidate = splits[i].strip()
        content = splits[i + 1].strip() if i + 1 < len(splits) else ""

        chunk_text = f"{header_candidate}\n{content}"
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": filename,
                "section": header_candidate
            }
        })
        i += 2

    return chunks

def index_documents_safely(
    documents: List[Dict[str, Any]],
    chroma_collection,
    gemini_client,
    batch_size: int = 20,
    cooldown_seconds: float = 3.0
):
    """
    Tiến hành Chunking và Indexing dữ liệu vào ChromaDB.
    Hỗ trợ Resume bằng .upsert() và bỏ qua batch hỏng cục bộ.
    """
    total_docs = len(documents)
    logger.info(f"Bắt đầu tiến trình Indexing tối ưu cho {total_docs} văn bản.")

    for doc_idx, doc in enumerate(documents, 1):
        file_name = doc.get("filename", f"doc_{doc_idx}")
        chunks = doc.get("chunks", [])

        if not chunks and "text" in doc:
            chunks = chunk_legal_document(doc["text"], file_name)

        if not chunks:
            logger.warning(f"File '{file_name}' không chứa dữ liệu chữ để index.")
            continue

        logger.info(f"[{doc_idx}/{total_docs}] Đang xử lý: {file_name} ({len(chunks)} chunks)")

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_texts = [c["text"] for c in batch_chunks]
            batch_ids = [f"{file_name}_chunk_{i + idx}" for idx in range(len(batch_chunks))]
            batch_metadatas = [c.get("metadata", {"source": file_name}) for c in batch_chunks]

            try:
                embeddings = embed_with_retry(gemini_client, batch_texts, output_dim=768)

                chroma_collection.upsert(
                    ids=batch_ids,
                    embeddings=embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )

                time.sleep(cooldown_seconds)

            except Exception as e:
                logger.error(f"Lỗi tại batch {i // batch_size + 1} của file {file_name}: {e}")
                continue

        logger.info(f"Hoàn tất xử lý văn bản: {file_name}")

    logger.info("Hoàn tất toàn bộ tiến trình Indexing dữ liệu vào Vector DB.")

def query_rag_engine(
    query: str,
    chroma_collection,
    gemini_client,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Truy vấn Vector DB bằng Vector Similarity để phục vụ kiểm toán.
    """
    try:
        if chroma_collection.count() == 0:
            logger.warning("Vector DB rỗng — chưa có văn bản luật nào được index.")
            return []

        query_embedding = embed_with_retry(gemini_client, [query], output_dim=768)[0]
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            for doc, meta in zip(docs, metas):
                retrieved.append({
                    "text": doc,
                    "metadata": meta
                })
        return retrieved

    except Exception as e:
        logger.error(f"Lỗi xảy ra trong quá trình truy vấn RAG: {e}")
        return []

def collection_stats(chroma_collection=None) -> Dict[str, Any]:
    """
    Lấy thống kê số lượng chunk hiện có trong Vector DB.
    Hỗ trợ gọi hàm linh hoạt kể cả khi chưa truyền đối tượng collection.
    """
    if chroma_collection is None:
        return {"available": False, "total_chunks": 0}

    try:
        count = chroma_collection.count()
        return {"available": True, "total_chunks": count}
    except Exception as e:
        logger.error(f"Lỗi khi lấy thống kê ChromaDB: {e}")
        return {"available": False, "total_chunks": 0}

