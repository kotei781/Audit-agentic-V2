"""
data_ingestion.py
------------------
TẦNG LƯU TRỮ CỐ ĐỊNH (Persistence Storage Layer).

Chịu trách nhiệm:
  1) Nhận file luật do Admin tải lên và lưu VĨNH VIỄN vào `downloaded_laws/`.
  2) Trích xuất text từ PDF / DOCX / TXT / CSV / XLSX.
  3) Gom toàn bộ kho luật thành một `law_context` duy nhất (chế độ dự phòng
     khi không dùng RAG).
  4) Liệt kê / xóa file trong kho luật phục vụ màn hình Admin.
  5) Tiện ích demo: tạo & đọc CSDL SQLite mẫu.

NGUYÊN TẮC THIẾT KẾ:
  * Mọi hàm ghi file đều chống path traversal (chỉ lấy phần tên file).
  * Mọi hàm đọc đều chặn "văn bản rỗng": PDF scan không có lớp text sẽ trả
    về chuỗi rỗng, nếu không chặn thì Agent sẽ chạy trên ngữ cảnh rỗng và
    sinh kết luận vô căn cứ — lỗi thầm lặng nguy hiểm nhất của hệ thống.
  * Mọi file lưu xuống đều được tính SHA-256 phục vụ audit trail.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import config

logger = config.get_logger("data_ingestion")

# ---- Thư viện đọc file: import mềm để lỗi thiếu package hiện rõ ràng ----
try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover
    DocxDocument = None

try:
    import openpyxl
except ImportError:  # pragma: no cover
    openpyxl = None


# ============================================================
# NGOẠI LỆ CHUYÊN BIỆT
# ============================================================
class IngestionError(Exception):
    """Lỗi chung của tầng nạp dữ liệu."""


class UnsupportedFileTypeError(IngestionError):
    """Định dạng file không được hỗ trợ."""


class EmptyDocumentError(IngestionError):
    """File đọc được nhưng không có nội dung text hữu ích (thường do PDF scan)."""


class FileTooLargeError(IngestionError):
    """File vượt quá giới hạn dung lượng cho phép."""


# ============================================================
# MODEL MÔ TẢ FILE TRONG KHO LUẬT
# ============================================================
@dataclass
class LawFileInfo:
    """Metadata một file luật trong kho cố định — dùng cho bảng quản trị."""

    filename: str
    path: Path
    size_bytes: int
    modified_at: datetime
    sha256: str
    extension: str
    char_count: Optional[int] = None
    indexed_chunks: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_human(self) -> str:
        size = float(self.size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"

    def to_row(self) -> Dict[str, Any]:
        return {
            "Tên file": self.filename,
            "Định dạng": self.extension,
            "Dung lượng": self.size_human,
            "Cập nhật": self.modified_at.strftime("%Y-%m-%d %H:%M"),
            "Số chunk đã index": self.indexed_chunks,
            "SHA-256": self.sha256[:12],
        }


# ============================================================
# 1) TIỆN ÍCH AN TOÀN
# ============================================================
_UNSAFE_CHARS = re.compile(r"[^\w\-. ()\u00C0-\u1EF9]", re.UNICODE)


def sanitize_filename(filename: str) -> str:
    """
    Làm sạch tên file trước khi ghi xuống đĩa.

    Chặn path traversal ("../../etc/passwd") bằng cách chỉ giữ phần basename,
    đồng thời loại bỏ ký tự đặc biệt nhưng VẪN GIỮ dấu tiếng Việt cho dễ đọc.
    """
    base = os.path.basename(str(filename or "")).strip()
    base = base.replace("\x00", "")
    cleaned = _UNSAFE_CHARS.sub("_", base).strip(" ._")
    if not cleaned:
        cleaned = f"law_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"
    return cleaned[:150]


def compute_sha256(data: Union[bytes, Path, str]) -> str:
    """Tính SHA-256 của bytes hoặc của một file trên đĩa."""
    hasher = hashlib.sha256()
    if isinstance(data, bytes):
        hasher.update(data)
        return hasher.hexdigest()

    path = Path(data)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _resolve_unique_path(directory: Path, filename: str) -> Path:
    """Sinh đường dẫn không trùng: 'luat.pdf' -> 'luat (1).pdf' nếu đã tồn tại."""
    candidate = directory / filename
    if not candidate.exists():
        return candidate

    stem, suffix = candidate.stem, candidate.suffix
    for index in range(1, 1000):
        candidate = directory / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate
    raise IngestionError("Không tìm được tên file khả dụng trong kho luật.")


def _extract_bytes_and_name(file_obj: Any, filename: Optional[str]) -> tuple[bytes, str]:
    """
    Chuẩn hóa đầu vào đa dạng về (bytes, filename).

    Hỗ trợ đồng thời 2 kiểu gọi được nêu trong tài liệu yêu cầu:
      * save_admin_law_file(file_bytes, filename)     # backend thuần
      * save_admin_law_file(uploaded_file)            # Streamlit UploadedFile
    """
    if isinstance(file_obj, (bytes, bytearray)):
        if not filename:
            raise IngestionError("Truyền vào bytes thì bắt buộc phải có tham số filename.")
        return bytes(file_obj), filename

    if isinstance(file_obj, (str, Path)):
        path = Path(file_obj)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")
        return path.read_bytes(), filename or path.name

    # Streamlit UploadedFile / BytesIO / file handle
    name = filename or getattr(file_obj, "name", None)
    if hasattr(file_obj, "getvalue"):
        return file_obj.getvalue(), name or "uploaded_file"
    if hasattr(file_obj, "read"):
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
        return file_obj.read(), name or "uploaded_file"

    raise IngestionError(f"Không hiểu kiểu dữ liệu đầu vào: {type(file_obj)!r}")


# ============================================================
# 2) LƯU FILE LUẬT CỦA ADMIN VÀO KHO CỐ ĐỊNH
# ============================================================
def save_admin_law_file(
    file_obj: Any,
    filename: Optional[str] = None,
    overwrite: bool = False,
) -> LawFileInfo:
    """
    Lưu VĨNH VIỄN một file luật do Admin tải lên vào `downloaded_laws/`.

    Args:
        file_obj: bytes, đường dẫn, hoặc đối tượng file (Streamlit UploadedFile).
        filename: tên file (bắt buộc khi truyền bytes).
        overwrite: True = ghi đè file trùng tên; False = tự đánh số " (1)".

    Returns:
        LawFileInfo mô tả file đã lưu (kèm SHA-256 và số ký tự trích xuất được).

    Raises:
        UnsupportedFileTypeError, FileTooLargeError, EmptyDocumentError, IngestionError
    """
    try:
        raw_bytes, original_name = _extract_bytes_and_name(file_obj, filename)
    except (IngestionError, FileNotFoundError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Không đọc được dữ liệu file tải lên: {exc}") from exc

    if not raw_bytes:
        raise EmptyDocumentError("File tải lên rỗng (0 byte).")

    if len(raw_bytes) > config.MAX_UPLOAD_BYTES:
        raise FileTooLargeError(
            f"File {original_name} nặng {len(raw_bytes) / 1024 / 1024:.1f} MB, "
            f"vượt giới hạn {config.MAX_UPLOAD_MB} MB."
        )

    safe_name = sanitize_filename(original_name)
    extension = Path(safe_name).suffix.lower()

    if extension not in config.ALLOWED_LAW_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Định dạng '{extension or 'không rõ'}' không được hỗ trợ cho văn bản luật. "
            f"Chỉ chấp nhận: {', '.join(sorted(config.ALLOWED_LAW_EXTENSIONS))}"
        )

    target = (
        config.LAW_STORAGE_DIR / safe_name
        if overwrite
        else _resolve_unique_path(config.LAW_STORAGE_DIR, safe_name)
    )

    # Ghi ra file tạm rồi đổi tên: tránh để lại file hỏng nếu tiến trình chết giữa chừng.
    tmp_path = target.with_suffix(target.suffix + ".part")
    try:
        tmp_path.write_bytes(raw_bytes)
        tmp_path.replace(target)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise IngestionError(f"Không ghi được file vào kho luật: {exc}") from exc

    # Xác thực ngay: file lưu vào kho phải đọc được text, nếu không thì vô dụng.
    try:
        text = extract_text(target)
        char_count = len(text)
    except EmptyDocumentError:
        target.unlink(missing_ok=True)
        raise
    except IngestionError:
        target.unlink(missing_ok=True)
        raise

    info = LawFileInfo(
        filename=target.name,
        path=target,
        size_bytes=len(raw_bytes),
        modified_at=datetime.fromtimestamp(target.stat().st_mtime, tz=timezone.utc),
        sha256=compute_sha256(raw_bytes),
        extension=extension,
        char_count=char_count,
    )
    logger.info("Đã lưu văn bản luật: %s (%s, %d ký tự)", info.filename, info.size_human, char_count)
    return info


def list_persisted_laws() -> List[LawFileInfo]:
    """Liệt kê toàn bộ file luật trong kho cố định (dùng cho màn hình Admin)."""
    results: List[LawFileInfo] = []

    for path in sorted(config.LAW_STORAGE_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in config.ALLOWED_LAW_EXTENSIONS:
            continue
        try:
            stat = path.stat()
            results.append(
                LawFileInfo(
                    filename=path.name,
                    path=path,
                    size_bytes=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                    sha256=compute_sha256(path),
                    extension=path.suffix.lower(),
                )
            )
        except OSError as exc:  # noqa: PERF203
            logger.warning("Bỏ qua file lỗi trong kho luật %s: %s", path.name, exc)

    return results


def delete_law_file(filename: str) -> bool:
    """Xóa một file khỏi kho luật. Trả về True nếu đã xóa được."""
    safe_name = sanitize_filename(filename)
    target = config.LAW_STORAGE_DIR / safe_name

    # Chốt chặn cuối: đảm bảo đường dẫn nằm trong kho luật.
    try:
        target.resolve().relative_to(config.LAW_STORAGE_DIR.resolve())
    except ValueError:
        raise IngestionError("Đường dẫn không hợp lệ — từ chối thao tác xóa.")

    if not target.exists():
        return False

    target.unlink()
    logger.info("Đã xóa văn bản luật khỏi kho: %s", safe_name)
    return True


# ============================================================
# 3) TRÍCH XUẤT TEXT ĐA ĐỊNH DẠNG
# ============================================================
def extract_text_from_pdf(file_path: Union[str, Path]) -> str:
    """
    Trích xuất text từ PDF, có đánh số trang để phục vụ trích dẫn vị trí.

    Ưu tiên pdfplumber (bóc bảng biểu tốt hơn), dự phòng bằng pypdf.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file PDF: {path}")

    pages: List[str] = []

    if pdfplumber is not None:
        try:
            with pdfplumber.open(path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    tables_text = _pdf_tables_to_text(page)
                    body = "\n".join(part for part in (page_text, tables_text) if part)
                    pages.append(f"--- Trang {page_number} ---\n{body}")
            return "\n\n".join(pages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pdfplumber lỗi với %s (%s) — chuyển sang pypdf.", path.name, exc)
            pages = []

    if PdfReader is None:
        raise IngestionError("Chưa cài pdfplumber lẫn pypdf — không đọc được PDF.")

    try:
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            pages.append(f"--- Trang {page_number} ---\n{page.extract_text() or ''}")
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Không đọc được PDF {path.name}: {exc}") from exc

    return "\n\n".join(pages)


def _pdf_tables_to_text(page: Any) -> str:
    """Chuyển bảng trong một trang PDF thành text dạng pipe-separated."""
    try:
        tables = page.extract_tables() or []
    except Exception:  # noqa: BLE001
        return ""

    lines: List[str] = []
    for table_index, table in enumerate(tables, start=1):
        lines.append(f"[Bảng {table_index}]")
        for row in table:
            cells = [(cell or "").replace("\n", " ").strip() for cell in row]
            if any(cells):
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def extract_text_from_docx(file_path: Union[str, Path]) -> str:
    """Trích xuất text từ .docx, bao gồm cả đoạn văn và nội dung bảng biểu."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file Word: {path}")
    if DocxDocument is None:
        raise IngestionError("Chưa cài python-docx — không đọc được file .docx.")

    try:
        document = DocxDocument(str(path))
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Không mở được file Word {path.name}: {exc}") from exc

    parts: List[str] = [p.text for p in document.paragraphs if p.text.strip()]

    for table_index, table in enumerate(document.tables, start=1):
        parts.append(f"--- Bảng {table_index} ---")
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))

    return "\n".join(parts)


def extract_text_from_txt(file_path: Union[str, Path]) -> str:
    """Đọc file text, tự dò encoding (utf-8 -> utf-8-sig -> cp1258 -> latin-1)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    for encoding in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise IngestionError(f"Không giải mã được file text: {path.name}")


def extract_text_from_csv(file_path: Union[str, Path], max_rows: int = 5000) -> str:
    """Chuyển CSV thành bảng text để đưa vào prompt (giới hạn số dòng)."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {path}")

    raw = extract_text_from_txt(path)
    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel

    reader = csv.reader(io.StringIO(raw), dialect)
    lines: List[str] = []
    for row_index, row in enumerate(reader):
        if row_index >= max_rows:
            lines.append(f"... (đã cắt bớt, file có nhiều hơn {max_rows} dòng)")
            break
        if any(cell.strip() for cell in row):
            lines.append(" | ".join(_escape_pipe(cell.strip()) for cell in row))

    return "\n".join(lines)


def extract_text_from_xlsx(file_path: Union[str, Path], max_rows_per_sheet: int = 3000) -> str:
    """Chuyển toàn bộ sheet của file Excel thành text dạng bảng."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file Excel: {path}")
    if openpyxl is None:
        raise IngestionError("Chưa cài openpyxl — không đọc được file .xlsx.")

    try:
        workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"Không mở được file Excel {path.name}: {exc}") from exc

    parts: List[str] = []
    try:
        for sheet in workbook.worksheets:
            parts.append(f"--- Sheet: {sheet.title} ---")
            for row_index, row in enumerate(sheet.iter_rows(values_only=True)):
                if row_index >= max_rows_per_sheet:
                    parts.append(f"... (đã cắt bớt sau {max_rows_per_sheet} dòng)")
                    break
                cells = ["" if v is None else str(v).strip() for v in row]
                if any(cells):
                    parts.append(" | ".join(_escape_pipe(c) for c in cells))
    finally:
        workbook.close()

    return "\n".join(parts)


def _escape_pipe(value: str) -> str:
    """Escape ký tự '|' để không làm vỡ cấu trúc bảng khi đưa vào prompt."""
    return value.replace("|", "\\|")


_EXTRACTORS = {
    ".pdf": extract_text_from_pdf,
    ".docx": extract_text_from_docx,
    ".txt": extract_text_from_txt,
    ".md": extract_text_from_txt,
    ".csv": extract_text_from_csv,
    ".xlsx": extract_text_from_xlsx,
    ".xls": extract_text_from_xlsx,
}


def extract_text(file_path: Union[str, Path], strict: bool = True) -> str:
    """
    Điều phối trích xuất text theo phần mở rộng của file.

    Args:
        file_path: đường dẫn file.
        strict: True thì raise EmptyDocumentError khi nội dung quá ngắn
                (dấu hiệu PDF scan không có lớp text).

    Raises:
        UnsupportedFileTypeError, EmptyDocumentError, IngestionError
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    extractor = _EXTRACTORS.get(extension)
    if extractor is None:
        raise UnsupportedFileTypeError(
            f"Định dạng '{extension or 'không rõ'}' chưa được hỗ trợ. "
            f"Hỗ trợ: {', '.join(sorted(_EXTRACTORS))}"
        )

    text = extractor(path) or ""
    cleaned = normalize_whitespace(text)

    if strict and len(cleaned.strip()) < config.MIN_EXTRACTED_TEXT_LENGTH:
        raise EmptyDocumentError(
            f"Trích xuất được quá ít nội dung từ '{path.name}' "
            f"({len(cleaned.strip())} ký tự, ngưỡng tối thiểu "
            f"{config.MIN_EXTRACTED_TEXT_LENGTH}).\n"
            "Nguyên nhân phổ biến: PDF là bản SCAN ảnh, không có lớp text. "
            "Hãy chạy OCR trước (ví dụ ocrmypdf) rồi tải lại."
        )

    return cleaned


# Alias giữ đúng tên hàm nêu trong tài liệu yêu cầu ROLENAME.docx.
extract_text_from_file = extract_text


def normalize_whitespace(text: str) -> str:
    """Chuẩn hóa khoảng trắng: bỏ dòng trống thừa, gộp space, giữ xuống dòng."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================
# 4) GOM TOÀN BỘ KHO LUẬT THÀNH MỘT NGỮ CẢNH (chế độ FULL-CONTEXT)
# ============================================================
def load_all_persisted_laws(max_chars: Optional[int] = None) -> str:
    """
    Quét toàn bộ `downloaded_laws/`, đọc nội dung và gom thành MỘT chuỗi
    `law_context` duy nhất, có phân tách rõ nguồn từng văn bản.

    Đây là chế độ dự phòng theo đúng yêu cầu ban đầu. Ở chế độ chính
    (RAG — xem rag_engine.py), hệ thống chỉ nạp các đoạn luật liên quan.

    CẢNH BÁO VỀ KHẢ NĂNG MỞ RỘNG: nhồi toàn bộ kho luật vào prompt sẽ đụng
    trần context và gây hiện tượng "lost in the middle" khi kho lớn dần.

    Returns:
        Chuỗi law_context (rỗng nếu kho chưa có file nào).
    """
    blocks: List[str] = []
    total_chars = 0
    limit = max_chars or (config.MAX_DOCUMENT_CHARS * 4)

    for info in list_persisted_laws():
        try:
            text = extract_text(info.path, strict=False)
        except IngestionError as exc:
            logger.warning("Bỏ qua %s khi gom kho luật: %s", info.filename, exc)
            continue

        if not text.strip():
            logger.warning("File %s không có nội dung text — bỏ qua.", info.filename)
            continue

        block = f"\n===== VĂN BẢN: {info.filename} =====\n{text}"
        if total_chars + len(block) > limit:
            blocks.append(
                f"\n[... đã cắt bớt: kho luật vượt {limit} ký tự. "
                "Hãy bật chế độ RAG để xử lý kho lớn.]"
            )
            break

        blocks.append(block)
        total_chars += len(block)

    if not blocks:
        logger.warning("Kho luật rỗng: chưa có văn bản nào trong %s", config.LAW_STORAGE_DIR)
        return ""

    logger.info("Đã gom %d văn bản luật (%d ký tự) ở chế độ full-context.", len(blocks), total_chars)
    return "\n".join(blocks)


def law_corpus_is_empty() -> bool:
    """True nếu kho luật cố định chưa có file nào."""
    return not list_persisted_laws()


# ============================================================
# 5) NẠP CHỨNG TỪ CẦN KIỂM TOÁN (phía User)
# ============================================================
def save_temp_upload(file_obj: Any, filename: Optional[str] = None) -> Path:
    """
    Lưu chứng từ User tải lên vào thư mục tạm để trích xuất.

    Chứng từ KHÔNG được lưu vào kho luật và không lưu vĩnh viễn — đây là dữ
    liệu nghiệp vụ của khách hàng, vòng đời càng ngắn càng tốt.
    """
    raw_bytes, original_name = _extract_bytes_and_name(file_obj, filename)

    if len(raw_bytes) > config.MAX_UPLOAD_BYTES:
        raise FileTooLargeError(
            f"Chứng từ nặng {len(raw_bytes) / 1024 / 1024:.1f} MB, "
            f"vượt giới hạn {config.MAX_UPLOAD_MB} MB."
        )

    safe_name = sanitize_filename(original_name)
    extension = Path(safe_name).suffix.lower()
    if extension not in config.ALLOWED_DOCUMENT_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Định dạng chứng từ '{extension or 'không rõ'}' không được hỗ trợ. "
            f"Chấp nhận: {', '.join(sorted(config.ALLOWED_DOCUMENT_EXTENSIONS))}"
        )

    target = _resolve_unique_path(config.TMP_DIR, safe_name)
    target.write_bytes(raw_bytes)
    return target


def cleanup_temp_uploads(keep_last: int = 0) -> int:
    """Dọn thư mục tạm; trả về số file đã xóa."""
    files = sorted(config.TMP_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    removed = 0
    for path in files[keep_last:]:
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


# ============================================================
# 6) TIỆN ÍCH DEMO — CSDL SQLite MẪU
# ============================================================
def create_sample_financial_database(db_path: Union[str, Path, None] = None) -> Path:
    """
    Tạo CSDL SQLite MẪU chứa giao dịch tài chính để demo toàn luồng.

    Triển khai thật sẽ thay bằng kết nối tới CSDL sản xuất qua SQLAlchemy:
        from sqlalchemy import create_engine
        engine = create_engine("postgresql+psycopg://user:pass@host:5432/db")
        rows = pd.read_sql("SELECT ... WHERE period = :p", engine, params={"p": ...})
    """
    path = Path(db_path or (config.BASE_DIR / "sample_financial_data.db"))

    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_no TEXT,
                transaction_date TEXT,
                description TEXT,
                amount REAL,
                vat_rate REAL,
                vat_amount REAL,
                counterparty_tax_code TEXT
            )
            """
        )
        cursor.execute("SELECT COUNT(*) FROM transactions")
        if cursor.fetchone()[0] == 0:
            cursor.executemany(
                """
                INSERT INTO transactions
                    (invoice_no, transaction_date, description, amount,
                     vat_rate, vat_amount, counterparty_tax_code)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("HD0001", "2026-01-05", "Ban hang hoa A cho cong ty X",
                     100_000_000, 0.10, 10_000_000, "0101234567"),
                    ("HD0002", "2026-01-12", "Chi phi tiep khach khong co hoa don",
                     25_000_000, 0.0, 0.0, ""),
                    ("HD0003", "2026-02-01", "Ban dich vu tu van cho cong ty Y",
                     50_000_000, 0.08, 4_000_000, "0109876543"),
                    ("HD0004", "2026-02-15", "Ban hang hoa B cho cong ty Z",
                     80_000_000, 0.10, 6_400_000, "0105555555"),
                ],
            )
            conn.commit()

    return path.resolve()


def fetch_data_from_sql(
    db_path: Union[str, Path],
    query: str = "SELECT * FROM transactions",
    params: Optional[Iterable[Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Truy vấn CSDL SQLite, trả về danh sách dict.

    LƯU Ý BẢO MẬT: `query` phải là hằng do lập trình viên kiểm soát. Tuyệt đối
    không nối chuỗi từ input người dùng — hãy dùng tham số `params` (?) thay thế.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy CSDL: {path}")

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, tuple(params or ()))
        return [dict(row) for row in cursor.fetchall()]


def sql_rows_to_text(rows: List[Dict[str, Any]]) -> str:
    """
    Chuyển kết quả truy vấn thành bảng text cho prompt.

    Số tiền được format có dấu phân cách để mô hình đọc đúng bậc đơn vị —
    chuỗi "100000000.0" rất dễ bị đọc nhầm bậc so với "100.000.000".
    """
    if not rows:
        return "(Không có dữ liệu)"

    headers = list(rows[0].keys())
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]

    for row in rows:
        cells = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float) and abs(value) >= 1000:
                cells.append(f"{value:,.0f}".replace(",", "."))
            else:
                cells.append(_escape_pipe(str(value)))
        lines.append(" | ".join(cells))

    return "\n".join(lines)


def load_sample_document_text() -> str:
    """Tạo nhanh chứng từ mẫu từ CSDL SQLite để demo."""
    db_path = create_sample_financial_database()
    return sql_rows_to_text(fetch_data_from_sql(db_path))
