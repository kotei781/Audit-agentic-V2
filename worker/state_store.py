"""
worker/state_store.py
----------------------
Lưu vết "đã quét tới transaction ID nào" để Background Worker không xử lý
trùng các dòng cũ mỗi lần khởi động lại tiến trình.

Dùng 1 file JSON đơn giản — đủ cho quy mô demo/1 worker duy nhất. Nếu chạy
nhiều worker song song hoặc cần độ tin cậy cao hơn, nên thay bằng 1 bảng
riêng trong CSDL (có transaction) hoặc Redis — KHÔNG thuộc phạm vi yêu cầu
hiện tại nên chưa làm ở đây.
"""

from __future__ import annotations

import json
from pathlib import Path

_STATE_FILE = Path(__file__).resolve().parent / "worker_state.json"


def load_last_id() -> int:
    """Trả về ID lớn nhất đã xử lý lần gần nhất (0 nếu chưa từng chạy)."""
    if not _STATE_FILE.exists():
        return 0
    try:
        data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        return int(data.get("last_processed_id", 0))
    except (json.JSONDecodeError, ValueError, OSError):
        return 0


def save_last_id(value: int) -> None:
    """Ghi lại ID lớn nhất vừa xử lý xong."""
    _STATE_FILE.write_text(
        json.dumps({"last_processed_id": value}, ensure_ascii=False),
        encoding="utf-8",
    )
