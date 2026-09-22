"""
human_review.py
----------------
LUỒNG KIỂM DUYỆT CON NGƯỜI (Human-in-the-Loop) qua giao diện dòng lệnh.

NGUYÊN TẮC: mọi finding đều phải có quyết định của con người trước khi báo cáo
được xuất. Chốt chặn thật nằm ở `report_exporter._ensure_reviewed()` — module
này chỉ là giao diện thu thập quyết định.

KHÁC BIỆT SO VỚI BẢN CŨ:
  * Bản cũ TỰ ĐỘNG approve mọi finding có has_violation == NO, trong khi
    docstring lại khẳng định "mọi finding đều PHẢI được con người xác nhận" —
    mâu thuẫn nội tại. Nay hành vi này được tách thành tùy chọn tường minh
    `auto_approve_compliant` (mặc định False) và luôn ghi rõ vào human_note.
  * Ghi nhận reviewer_id và thời điểm duyệt phục vụ audit trail.
  * Vòng xử lý các mục bị hoãn có kiểm tra đầu vào đầy đủ.
"""

from __future__ import annotations

from typing import List, Optional

import config
from schemas import AuditReport, HumanDecision, ViolationCheckResult, ViolationStatus

logger = config.get_logger("human_review")

_STATUS_LABEL = {
    ViolationStatus.YES: "CÓ VI PHẠM",
    ViolationStatus.NO: "KHÔNG VI PHẠM",
    ViolationStatus.UNCERTAIN: "CHƯA ĐỦ CĂN CỨ",
}


def _print_finding(index: int, finding: ViolationCheckResult) -> None:
    print("\n" + "=" * 78)
    print(f"PHÁT HIỆN #{index}  [{finding.finding_id}]  —  {_STATUS_LABEL[finding.has_violation]}")
    print("=" * 78)
    print(f"Mức rủi ro          : {finding.risk_level.value}")
    print(f"Trích dẫn chứng từ  : {finding.input_citation}")
    print(f"Căn cứ pháp lý      : {finding.rule_citation}")
    print(f"Suy luận của AI     : {finding.reasoning_steps}")
    print(f"Giải thích          : {finding.explanation}")
    print(f"Độ tin cậy của AI   : {finding.confidence_score:.2f}")
    print(f"Hành động đề xuất   : {finding.recommended_action.value}")

    if finding.grounding_verified is False:
        print(
            f"⚠ CẢNH BÁO         : Trích dẫn KHÔNG tìm thấy trong chứng từ gốc "
            f"(điểm khớp {finding.grounding_score:.2f}). {finding.downgraded_reason or ''}"
        )
    elif finding.grounding_verified:
        print(f"Kiểm chứng trích dẫn: ĐẠT (điểm {finding.grounding_score:.2f})")


def _prompt_decision(finding: ViolationCheckResult, reviewer_id: Optional[str]) -> bool:
    """
    Hỏi quyết định cho một finding.

    Returns:
        False nếu người dùng chọn hoãn (s), True nếu đã có quyết định.
    """
    while True:
        choice = (
            input(
                "\n>> Quyết định - [a] Phê duyệt | [r] Từ chối | [s] Hoãn lại: "
            )
            .strip()
            .lower()
        )

        if choice == "a":
            note = input(">> Ghi chú (Enter để bỏ qua): ").strip()
            finding.apply_decision(
                HumanDecision.APPROVED, note or "Người kiểm duyệt đã xác nhận.", reviewer_id
            )
            return True

        if choice == "r":
            note = input(">> Lý do từ chối (Enter để bỏ qua): ").strip()
            finding.apply_decision(
                HumanDecision.REJECTED, note or "Người kiểm duyệt đã từ chối cảnh báo này.", reviewer_id
            )
            return True

        if choice == "s":
            print(">> Đã hoãn, sẽ hỏi lại ở vòng cuối.")
            return False

        print(">> Lựa chọn không hợp lệ. Nhập a, r hoặc s.")


def run_cli_review(
    report: AuditReport,
    reviewer_id: Optional[str] = None,
    auto_approve_compliant: bool = False,
) -> AuditReport:
    """
    Duyệt lần lượt từng finding qua terminal.

    Args:
        report: báo cáo cần kiểm duyệt.
        reviewer_id: mã/tên người kiểm duyệt (ghi vào audit trail).
        auto_approve_compliant: True thì tự ghi nhận "approved" cho các finding
            có has_violation == NO. Mặc định False — người duyệt vẫn phải nhìn
            và xác nhận, vì kết luận "không vi phạm" của AI cũng có thể sai.
    """
    print("\n" + "#" * 78)
    print("# LUỒNG KIỂM DUYỆT CON NGƯỜI (HUMAN-IN-THE-LOOP)")
    print("#" * 78)

    if not reviewer_id:
        reviewer_id = input(">> Mã/tên người kiểm duyệt: ").strip() or "unknown_reviewer"

    if not report.findings:
        print("\nKhông có phát hiện nào cần rà soát.")
        return report

    for index, finding in enumerate(report.findings, start=1):
        _print_finding(index, finding)

        if auto_approve_compliant and finding.has_violation == ViolationStatus.NO:
            finding.apply_decision(
                HumanDecision.APPROVED,
                "Tự động ghi nhận (chế độ auto_approve_compliant): AI kết luận không vi phạm.",
                reviewer_id,
            )
            print(">> Tự động ghi nhận: KHÔNG VI PHẠM.")
            continue

        _prompt_decision(finding, reviewer_id)

    # ---- Vòng xử lý các mục đã hoãn ----
    pending: List[ViolationCheckResult] = report.pending_findings()
    while pending:
        print(f"\n>> Còn {len(pending)} phát hiện chưa quyết định. Xử lý nốt:")
        for index, finding in enumerate(pending, start=1):
            _print_finding(index, finding)
            while finding.human_decision is None:
                choice = input(">> [a] Phê duyệt | [r] Từ chối: ").strip().lower()
                if choice == "a":
                    note = input(">> Ghi chú (Enter để bỏ qua): ").strip()
                    finding.apply_decision(HumanDecision.APPROVED, note or None, reviewer_id)
                elif choice == "r":
                    note = input(">> Lý do từ chối (Enter để bỏ qua): ").strip()
                    finding.apply_decision(HumanDecision.REJECTED, note or None, reviewer_id)
                else:
                    print(">> Lựa chọn không hợp lệ. Nhập a hoặc r.")
        pending = report.pending_findings()

    summary = report.summary()
    print(
        f"\n>> Kiểm duyệt xong: {summary['approved']} phê duyệt / "
        f"{summary['rejected']} từ chối."
    )
    return report


def auto_approve_all(report: AuditReport, reason: str = "Chế độ test nhanh --auto-approve") -> AuditReport:
    """
    Tự động phê duyệt toàn bộ finding — CHỈ dùng để kiểm thử kỹ thuật.

    Báo cáo sinh ra sẽ được đánh dấu `hitl_bypassed = True` và in cảnh báo đỏ
    ngay trên trang đầu file Word, để không ai nhầm nó với báo cáo thật.
    """
    report.hitl_bypassed = True
    for finding in report.findings:
        finding.apply_decision(HumanDecision.APPROVED, reason, "AUTO_APPROVE_BOT")

    logger.warning("Đã bỏ qua Human-in-the-Loop bằng chế độ auto-approve.")
    return report
