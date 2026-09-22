"""
main.py
-------
Điểm khởi chạy CLI của Agentic AI Audit System.

CÁC LỆNH:

    # 1) Nạp văn bản luật vào kho cố định + index vào Vector DB
    python main.py ingest-law --file nghi_dinh_123.pdf
    python main.py ingest-law --dir ./van_ban_luat --reindex

    # 2) Xem kho luật hiện có
    python main.py list-laws

    # 3) Index lại toàn bộ (dùng khi đổi model embedding)
    python main.py reindex

    # 4) Kiểm toán một chứng từ (tự lấy luật từ kho, không cần truyền file luật)
    python main.py audit --file bang_ke_hoa_don.xlsx
    python main.py audit --sample                 # dùng dữ liệu SQL mẫu
    python main.py audit --sample --auto-approve  # CHỈ để test kỹ thuật

Lưu ý: từ phiên bản này, KHÔNG còn tham số --law-pdf / --law-url. Văn bản luật
được quản lý tập trung trong kho cố định thay vì truyền tay mỗi lần chạy —
đó chính là lỗ hổng kiến trúc mà bản refactor này khắc phục.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import config
import data_ingestion
import rag_engine
import report_exporter
from audit_agent import AuditAgentError, NoLawCorpusError, run_agentic_audit
from data_ingestion import IngestionError
from human_review import auto_approve_all, run_cli_review

logger = config.get_logger("main")


# ============================================================
# LỆNH: ingest-law
# ============================================================
def cmd_ingest_law(args: argparse.Namespace) -> int:
    targets: List[Path] = []

    if args.file:
        targets.append(Path(args.file))
    if args.dir:
        directory = Path(args.dir)
        if not directory.is_dir():
            print(f"[LỖI] Không tìm thấy thư mục: {directory}")
            return 1
        targets.extend(
            path for path in sorted(directory.iterdir())
            if path.suffix.lower() in config.ALLOWED_LAW_EXTENSIONS
        )

    if not targets:
        print("[LỖI] Cần chỉ định --file hoặc --dir.")
        return 1

    success = 0
    for path in targets:
        print(f"\n>> Đang nạp: {path.name}")
        try:
            info = data_ingestion.save_admin_law_file(path)
            print(f"   ✅ Đã lưu vào kho: {info.filename} ({info.size_human}, {info.char_count:,} ký tự)")

            if not args.no_index:
                result = rag_engine.ingest_law_to_vector_db(info.path, force_reindex=args.reindex)
                print(f"   ✅ Đã index {result['chunks']} chunk (trạng thái: {result['status']})")
            success += 1

        except IngestionError as exc:
            print(f"   ❌ {exc}")
        except rag_engine.RAGError as exc:
            print(f"   ⚠️  Đã lưu vào kho nhưng chưa index được: {exc}")
            print("      Hệ thống vẫn chạy được ở chế độ full-context.")
            success += 1

    print(f"\n>> Hoàn tất: {success}/{len(targets)} văn bản.")
    return 0 if success else 1


# ============================================================
# LỆNH: list-laws
# ============================================================
def cmd_list_laws(_: argparse.Namespace) -> int:
    laws = data_ingestion.list_persisted_laws()
    stats = rag_engine.collection_stats()
    by_source = stats.get("by_source", {})

    if not laws:
        print("Kho luật đang trống. Dùng: python main.py ingest-law --file <file>")
        return 0

    print(f"\nKHO LUẬT CỐ ĐỊNH — {config.LAW_STORAGE_DIR}")
    print("-" * 92)
    print(f"{'TÊN FILE':<46} {'DUNG LƯỢNG':>12} {'CHUNK':>8} {'CẬP NHẬT':>20}")
    print("-" * 92)

    for info in laws:
        chunks = by_source.get(info.filename, 0)
        print(
            f"{info.filename[:45]:<46} {info.size_human:>12} {chunks:>8} "
            f"{info.modified_at.strftime('%Y-%m-%d %H:%M'):>20}"
        )

    print("-" * 92)
    if stats.get("available"):
        print(f"Vector DB: {stats['total_chunks']} chunk tại {config.CHROMA_DB_DIR}")
    else:
        print(f"Vector DB chưa sẵn sàng: {stats.get('error', 'không rõ nguyên nhân')}")
    return 0


# ============================================================
# LỆNH: reindex
# ============================================================
def cmd_reindex(_: argparse.Namespace) -> int:
    print(">> Đang xây lại toàn bộ chỉ mục Vector DB...")
    try:
        results = rag_engine.rebuild_index()
    except rag_engine.RAGError as exc:
        print(f"[LỖI] {exc}")
        return 1

    total = sum(item.get("chunks", 0) for item in results)
    for item in results:
        icon = "✅" if item.get("status") == "indexed" else "❌"
        print(f"   {icon} {item['source']}: {item.get('chunks', 0)} chunk")

    print(f"\n>> Xong: {len(results)} văn bản, tổng {total} chunk.")
    return 0


# ============================================================
# LỆNH: audit
# ============================================================
def cmd_audit(args: argparse.Namespace) -> int:
    # --- Nạp chứng từ ---
    try:
        if args.sample or not args.file:
            if not args.sample:
                print(">> Không chỉ định --file, dùng dữ liệu SQL mẫu.")
            document_text = data_ingestion.load_sample_document_text()
            document_source = "sample_sql_data"
            document_hash = data_ingestion.compute_sha256(document_text.encode("utf-8"))
        else:
            path = Path(args.file)
            document_text = data_ingestion.extract_text(path, strict=False)
            document_source = path.name
            document_hash = data_ingestion.compute_sha256(path)
    except (IngestionError, FileNotFoundError) as exc:
        print(f"[LỖI] {exc}")
        return 1

    if not document_text.strip():
        print("[LỖI] Không trích xuất được nội dung nào từ chứng từ (PDF scan?).")
        return 1

    # --- Chạy Agent ---
    print(f"\n>> Đang kiểm toán '{document_source}' bằng {config.GEMINI_MODEL}...")
    try:
        report = run_agentic_audit(
            document_text=document_text,
            document_source=document_source,
            document_sha256=document_hash,
            top_k=args.top_k,
        )
    except NoLawCorpusError as exc:
        print(f"[LỖI] {exc}")
        return 1
    except AuditAgentError as exc:
        print(f"[LỖI] {exc}")
        return 1

    summary = report.summary()
    print(
        f">> Tìm thấy {summary['total']} phát hiện "
        f"(vi phạm: {summary['violation']} | chưa đủ căn cứ: {summary['uncertain']} | "
        f"trích dẫn lỗi: {summary['grounding_failed']})"
    )
    print(f">> Chế độ truy hồi luật: {report.retrieval_mode}")

    # --- Human-in-the-Loop ---
    if args.auto_approve:
        print(
            "\n⚠ CẢNH BÁO: đang chạy --auto-approve. Báo cáo sinh ra bị đánh dấu "
            "hitl_bypassed và KHÔNG có giá trị sử dụng chính thức."
        )
        report = auto_approve_all(report)
    else:
        report = run_cli_review(report, reviewer_id=args.reviewer)

    # --- Xuất báo cáo ---
    try:
        paths = report_exporter.export_final_report(report, reviewer_id=args.reviewer)
    except report_exporter.ExportError as exc:
        print(f"[LỖI] {exc}")
        return 1

    print("\n" + "#" * 78)
    print("# ĐÃ XUẤT BÁO CÁO")
    print("#" * 78)
    print(f"Word : {paths['docx']}")
    print(f"JSON : {paths['json']}")
    return 0


# ============================================================
# ARG PARSER
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agentic AI Audit System — kiểm toán chứng từ tài chính bằng Gemini + RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest-law
    ingest = subparsers.add_parser("ingest-law", help="Nạp văn bản luật vào kho cố định")
    ingest.add_argument("--file", type=str, help="Đường dẫn một file luật (.pdf/.docx/.txt)")
    ingest.add_argument("--dir", type=str, help="Thư mục chứa nhiều file luật")
    ingest.add_argument("--reindex", action="store_true", help="Index lại nếu đã tồn tại")
    ingest.add_argument("--no-index", action="store_true", help="Chỉ lưu vào kho, không index Vector DB")
    ingest.set_defaults(func=cmd_ingest_law)

    # list-laws
    listing = subparsers.add_parser("list-laws", help="Liệt kê kho luật hiện có")
    listing.set_defaults(func=cmd_list_laws)

    # reindex
    reindex = subparsers.add_parser("reindex", help="Xây lại toàn bộ Vector DB")
    reindex.set_defaults(func=cmd_reindex)

    # audit
    audit = subparsers.add_parser("audit", help="Kiểm toán một chứng từ tài chính")
    audit.add_argument("--file", type=str, help="Chứng từ cần kiểm toán (.xlsx/.csv/.pdf/.docx)")
    audit.add_argument("--sample", action="store_true", help="Dùng dữ liệu SQL mẫu")
    audit.add_argument("--reviewer", type=str, default=None, help="Mã/tên người kiểm duyệt")
    audit.add_argument("--top-k", type=int, default=None, help="Số đoạn luật truy hồi")
    audit.add_argument(
        "--auto-approve",
        action="store_true",
        help=(
            "Bỏ qua kiểm duyệt con người — CHỈ dùng để test kỹ thuật. Báo cáo "
            "sẽ bị đánh dấu hitl_bypassed."
        ),
    )
    audit.set_defaults(func=cmd_audit)

    return parser


def main() -> int:
    config.setup_logging()
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n>> Đã hủy theo yêu cầu người dùng.")
        return 130
    except config.ConfigError as exc:
        print(f"[LỖI CẤU HÌNH] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
