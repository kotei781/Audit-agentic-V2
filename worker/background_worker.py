"""
worker/background_worker.py
-----------------------------
YÊU CẦU 2: Autonomous Agent chạy ngầm — thay vì người dùng phải upload file
thủ công, tiến trình này TỰ ĐỘNG:

  1. Định kỳ (mặc định mỗi 10 phút, dùng APScheduler) kết nối vào CSDL SQL
     cục bộ, kiểm tra xem bảng `transactions` có dòng mới phát sinh không
     (so với lần quét trước, theo dõi qua `state_store.py`).
  2. Nếu có dòng mới, đưa vào `RequestBatcher` (worker/rate_limited_client.py)
     để gộp nhiều dòng thành 1 lần gọi Gemini thay vì gọi riêng từng dòng.
  3. Khi lô được xử lý xong: nếu có vi phạm/nghi vấn, TỰ ĐỘNG xuất báo cáo
     (.docx + .json qua report_exporter.py có sẵn) và in cảnh báo ra terminal.

RANH GIỚI QUAN TRỌNG VỀ HUMAN-IN-THE-LOOP:
    Toàn bộ kiến trúc gốc (README v2, human_review.py, report_exporter.py)
    đặt nguyên tắc: KHÔNG báo cáo nào có giá trị chính thức nếu chưa qua
    con người duyệt. Worker này KHÔNG được phép âm thầm phá vỡ nguyên tắc đó.

    Vì vậy báo cáo do worker tự sinh ra được đánh dấu rõ ràng là CẢNH BÁO SƠ
    BỘ bằng cách gọi `human_review.auto_approve_all()` — đúng cơ chế
    `hitl_bypassed=True` đã có sẵn trong hệ thống gốc (report_exporter.py in
    cảnh báo đỏ ngay trang đầu file Word). Báo cáo này CẦN một người kiểm
    duyệt thật rà soát lại bằng `python main.py audit --file ...` hoặc qua
    Streamlit UI trước khi coi là kết luận chính thức.

CHẠY WORKER:
    python -m worker.background_worker
    python -m worker.background_worker --interval-minutes 5 --db path/to.db
    python -m worker.background_worker --run-once      # chạy 1 lần rồi thoát, để test

KHÔNG SỬA FILE GỐC NÀO — chỉ import và dùng lại các hàm/class công khai đã
có sẵn trong config.py, data_ingestion.py, audit_agent.py, human_review.py,
report_exporter.py.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import data_ingestion
import human_review
import report_exporter
from audit_agent import AuditAgent, AuditAgentError, NoLawCorpusError
from worker import state_store
from worker.rate_limited_client import RateLimitedGeminiClient, RequestBatcher

logger = config.get_logger("worker.background_worker")


# ============================================================
# CẤU HÌNH RIÊNG CỦA WORKER (không đụng tới config.py gốc)
# ============================================================
DEFAULT_SCAN_INTERVAL_MINUTES = 10
DEFAULT_BATCH_MAX_SIZE = 10
DEFAULT_BATCH_MAX_WAIT_SECONDS = 30.0
DEFAULT_RATE_LIMIT_RPM = 5  # khớp gói Gemini free tier trong yêu cầu của bạn


# ============================================================
# AGENT DÙNG CHUNG — GẮN SẴN RATE LIMITER (không sửa audit_agent.py)
# ============================================================
_shared_agent: Optional[AuditAgent] = None


def get_rate_limited_agent(rpm: float = DEFAULT_RATE_LIMIT_RPM) -> AuditAgent:
    """
    Tạo (hoặc tái sử dụng) một AuditAgent duy nhất cho cả vòng đời worker,
    với client Gemini đã được thay bằng RateLimitedGeminiClient — xem giải
    thích kỹ thuật "drop-in" trong worker/rate_limited_client.py.
    """
    global _shared_agent
    if _shared_agent is None:
        _shared_agent = AuditAgent()
        _shared_agent.client = RateLimitedGeminiClient(
            api_key=config.GEMINI_API_KEY, rpm=rpm
        )
        logger.info("Đã khởi tạo AuditAgent với rate limit %s RPM.", rpm)
    return _shared_agent


# ============================================================
# XỬ LÝ MỘT LÔ (được RequestBatcher gọi khi đủ điều kiện flush)
# ============================================================
def process_batch(rows: List[Dict[str, Any]]) -> None:
    """
    Gộp nhiều dòng chứng từ thành 1 chứng từ ảo duy nhất, gửi 1 lần cho
    Agent, rồi tự động phê duyệt (đánh dấu hitl_bypassed) + xuất báo cáo nếu
    có phát hiện đáng chú ý.
    """
    document_text = data_ingestion.sql_rows_to_text(rows)
    row_ids = [row.get("id") for row in rows if "id" in row]
    document_source = (
        f"autonomous_scan_{datetime.now():%Y%m%d_%H%M%S}_"
        f"ids_{min(row_ids)}-{max(row_ids)}" if row_ids else "autonomous_scan"
    )
    document_hash = data_ingestion.compute_sha256(document_text.encode("utf-8"))

    logger.info("Đang kiểm toán lô %d dòng (%s)...", len(rows), document_source)

    try:
        agent = get_rate_limited_agent()
        report = agent.run_audit(
            document_text=document_text,
            document_source=document_source,
            document_sha256=document_hash,
        )
    except NoLawCorpusError as exc:
        logger.error("Kho luật rỗng — không thể kiểm toán: %s", exc)
        return
    except AuditAgentError as exc:
        logger.error("Lỗi Agent khi xử lý lô %s: %s", document_source, exc)
        return

    # Đánh dấu rõ đây là báo cáo tự động, CHƯA qua người duyệt thật.
    report = human_review.auto_approve_all(
        report,
        reason=(
            "Tự động sinh bởi Background Worker (APScheduler) — đây là CẢNH BÁO "
            "SƠ BỘ, cần người kiểm duyệt rà soát lại trước khi coi là chính thức."
        ),
    )

    summary = report.summary()
    if summary["violation"] == 0 and summary["uncertain"] == 0:
        logger.info("Lô %s: không phát hiện vấn đề (%d dòng).", document_source, len(rows))
        return

    try:
        paths = report_exporter.export_final_report(report, reviewer_id="AUTONOMOUS_WORKER")
    except report_exporter.ExportError as exc:
        logger.error("Không xuất được báo cáo cho lô %s: %s", document_source, exc)
        return

    _print_terminal_alert(document_source, summary, paths)


def _print_terminal_alert(document_source: str, summary: Dict[str, int], paths: Dict[str, Path]) -> None:
    """In cảnh báo nổi bật ra terminal — theo đúng yêu cầu 'in cảnh báo ra terminal'."""
    print("\n" + "!" * 78)
    print("!! CẢNH BÁO TỰ ĐỘNG — WORKER PHÁT HIỆN DẤU HIỆU CẦN CHÚ Ý")
    print("!" * 78)
    print(f"   Lô dữ liệu       : {document_source}")
    print(f"   Vi phạm (YES)    : {summary['violation']}")
    print(f"   Chưa đủ căn cứ   : {summary['uncertain']}")
    print(f"   Trích dẫn lỗi    : {summary['grounding_failed']}")
    print(f"   Báo cáo Word     : {paths['docx']}")
    print(f"   Báo cáo JSON     : {paths['json']}")
    print("   ⚠ Đây là cảnh báo SƠ BỘ (hitl_bypassed=True) — cần người kiểm")
    print("     duyệt rà soát lại trước khi coi là kết luận chính thức.")
    print("!" * 78 + "\n")


# ============================================================
# QUÉT CSDL TÌM DÒNG MỚI
# ============================================================
def scan_for_new_transactions(db_path: Path, batcher: RequestBatcher) -> None:
    """
    Job chạy định kỳ: tìm các dòng có `id` lớn hơn lần quét trước, đưa từng
    dòng vào RequestBatcher để gộp lô trước khi gọi Gemini.
    """
    if not db_path.exists():
        logger.warning("Chưa có CSDL tại %s — tạo CSDL mẫu để demo.", db_path)
        data_ingestion.create_sample_financial_database(db_path)

    last_id = state_store.load_last_id()
    try:
        new_rows = data_ingestion.fetch_data_from_sql(
            db_path,
            query="SELECT * FROM transactions WHERE id > ? ORDER BY id ASC",
            params=(last_id,),
        )
    except FileNotFoundError as exc:
        logger.error("Không đọc được CSDL: %s", exc)
        return

    if not new_rows:
        logger.info("Không có chứng từ mới (đã quét tới id=%d).", last_id)
        return

    logger.info("Phát hiện %d dòng chứng từ mới -> đưa vào hàng đợi xử lý.", len(new_rows))
    for row in new_rows:
        batcher.add(row)

    max_id = max(row["id"] for row in new_rows)
    state_store.save_last_id(max_id)


# ============================================================
# ĐIỂM VÀO CLI
# ============================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Background Worker — tự động quét CSDL và kiểm toán định kỳ."
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="Đường dẫn CSDL SQLite (mặc định: sample_financial_data.db cạnh config.py).",
    )
    parser.add_argument(
        "--interval-minutes", type=float, default=DEFAULT_SCAN_INTERVAL_MINUTES,
        help=f"Chu kỳ quét, phút (mặc định {DEFAULT_SCAN_INTERVAL_MINUTES}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_MAX_SIZE,
        help=f"Số dòng tối đa mỗi lô trước khi tự flush (mặc định {DEFAULT_BATCH_MAX_SIZE}).",
    )
    parser.add_argument(
        "--batch-wait-seconds", type=float, default=DEFAULT_BATCH_MAX_WAIT_SECONDS,
        help=f"Thời gian chờ tối đa trước khi flush lô chưa đầy (mặc định {DEFAULT_BATCH_MAX_WAIT_SECONDS}s).",
    )
    parser.add_argument(
        "--rpm", type=float, default=DEFAULT_RATE_LIMIT_RPM,
        help=f"Giới hạn request/phút gửi tới Gemini (mặc định {DEFAULT_RATE_LIMIT_RPM}).",
    )
    parser.add_argument(
        "--run-once", action="store_true",
        help="Chạy đúng 1 lần quét rồi thoát (dùng để test, không khởi động scheduler).",
    )
    return parser


def main() -> int:
    config.setup_logging()
    args = build_arg_parser().parse_args()

    db_path = Path(args.db) if args.db else (config.BASE_DIR / "sample_financial_data.db")

    batcher = RequestBatcher(
        process_batch=process_batch,
        max_batch_size=args.batch_size,
        max_wait_seconds=args.batch_wait_seconds,
    )

    # Gắn rpm tùy chỉnh cho agent dùng chung (phải tạo trước lần dùng đầu tiên).
    get_rate_limited_agent(rpm=args.rpm)

    if args.run_once:
        print(">> Chạy 1 lần (--run-once)...")
        scan_for_new_transactions(db_path, batcher)
        batcher.stop()
        print(">> Hoàn tất.")
        return 0

    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler()
    scheduler.add_job(
        scan_for_new_transactions,
        trigger="interval",
        minutes=args.interval_minutes,
        args=[db_path, batcher],
        next_run_time=datetime.now(),  # quét ngay 1 lần khi vừa khởi động
        id="scan_for_new_transactions",
        max_instances=1,  # không cho 2 lần quét chạy chồng nếu 1 lần quét chạy lâu
    )

    print(f">> Background Worker đã khởi động — quét CSDL mỗi {args.interval_minutes} phút.")
    print(f">> CSDL: {db_path}")
    print(f">> Rate limit: {args.rpm} RPM | Batch: tối đa {args.batch_size} dòng "
          f"hoặc chờ {args.batch_wait_seconds}s")
    print(">> Nhấn Ctrl+C để dừng.\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n>> Đang dừng worker...")
        batcher.stop()
        print(">> Đã dừng.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
