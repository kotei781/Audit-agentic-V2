"""
app_streamlit_reports.py
--------------------------
MÀN HÌNH RIÊNG: liệt kê & rà soát lại các báo cáo đã có sẵn trên đĩa
(`audit_reports/`), đặc biệt là báo cáo do `background_worker.py` TỰ ĐỘNG
sinh ra (`hitl_bypassed=True`) — hiện KHÔNG hiển thị ở đâu trong
`app_streamlit.py` gốc, người dùng phải tự mở thư mục ra xem.

TẠI SAO ĐỂ Ở FILE RIÊNG (không sửa app_streamlit.py):
  * Không đụng vào, không cần audit lại UI kiểm toán chính đang hoạt động.
  * Có thể bật/tắt độc lập.
  * Muốn gộp vào cùng 1 app dạng multipage: copy nguyên file này vào thư mục
    `pages/` cạnh `app_streamlit.py` (ví dụ đặt tên
    `pages/2_📂_Bao_cao_da_luu.py`) — Streamlit tự thêm nó vào sidebar điều
    hướng, KHÔNG cần sửa gì thêm ở cả 2 file.

CHẠY ĐỘC LẬP:
    streamlit run app_streamlit_reports.py

NGUYÊN TẮC AN TOÀN DỮ LIỆU — CHỈ ĐỌC + GHI FILE MỚI:
    File này KHÔNG BAO GIỜ ghi đè lên báo cáo gốc do Worker sinh ra. Khi rà
    soát xong và bấm lưu, kết quả được ghi ra file MỚI với hậu tố
    "_reviewed" (.docx + .json), giữ nguyên bản gốc làm bằng chứng đối chiếu
    (đúng tinh thần audit trail đã có trong README của hệ thống gốc).

    Chốt chặn HITL thật sự vẫn nằm ở report_exporter._ensure_reviewed() —
    file này không mở lại chốt chặn đó, chỉ cung cấp giao diện để hoàn tất nó.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Optional

import streamlit as st

import config
import report_exporter
from schemas import AuditReport, HumanDecision, RiskLevel, ViolationStatus

st.set_page_config(
    page_title="Báo cáo đã lưu — AI Audit System",
    page_icon="📂",
    layout="wide",
)

# Phải khớp đúng chuỗi mà worker/human_review.auto_approve_all() dùng khi
# tự động phê duyệt, để nhận diện finding nào CHƯA từng qua mắt người thật.
_BOT_REVIEWER_ID = "AUTO_APPROVE_BOT"

RISK_BADGE = {
    RiskLevel.HIGH: "🔴",
    RiskLevel.MEDIUM: "🟡",
    RiskLevel.LOW: "🟢",
}

STATUS_TEXT = {
    ViolationStatus.YES: "CÓ VI PHẠM",
    ViolationStatus.NO: "KHÔNG VI PHẠM",
    ViolationStatus.UNCERTAIN: "CHƯA ĐỦ CĂN CỨ",
}


# ============================================================
# STATE
# ============================================================
def _init_state() -> None:
    defaults = {
        "selected_report_path": None,
        "loaded_report": None,
        "reviewer_id": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()


# ============================================================
# QUÉT DANH SÁCH BÁO CÁO TRÊN ĐĨA
# ============================================================
def _peek_report_summary(path: Path) -> Optional[dict]:
    """
    Đọc nhanh các trường cần cho bảng danh sách bằng json.loads() thô, KHÔNG
    validate toàn bộ qua Pydantic — nhanh hơn khi kho có nhiều báo cáo. Model
    đầy đủ chỉ được nạp khi người dùng thực sự mở một báo cáo cụ thể (xem
    `_load_full_report`).
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    findings = raw.get("findings", [])
    return {
        "path": path,
        "report_id": raw.get("report_id", "?"),
        "generated_at": raw.get("generated_at", "?"),
        "document_source": raw.get("document_source", "?"),
        "hitl_bypassed": bool(raw.get("hitl_bypassed", False)),
        "total_findings": len(findings),
        "violations": sum(1 for f in findings if f.get("has_violation") == "YES"),
        "bot_pending": sum(1 for f in findings if f.get("reviewer_id") == _BOT_REVIEWER_ID),
    }


def _list_reports() -> list[dict]:
    if not config.OUTPUT_DIR.exists():
        return []
    summaries = []
    for path in sorted(config.OUTPUT_DIR.glob("*.json"), reverse=True):
        summary = _peek_report_summary(path)
        if summary:
            summaries.append(summary)
    return summaries


def _load_full_report(path: Path) -> AuditReport:
    return AuditReport.model_validate_json(path.read_text(encoding="utf-8"))


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("📂 Báo cáo đã lưu")
    st.caption(
        "Xem lại mọi báo cáo trong `audit_reports/`, kể cả báo cáo do "
        "Background Worker tự sinh ra (đánh dấu ⚠ cần rà soát)."
    )
    only_bypassed = st.checkbox("Chỉ hiện báo cáo cần rà soát (worker tự động)", value=True)
    search_text = st.text_input("Tìm theo tên chứng từ", placeholder="VD: autonomous_scan...")

    st.divider()
    st.session_state.reviewer_id = st.text_input(
        "Mã/tên người kiểm duyệt",
        value=st.session_state.reviewer_id,
        placeholder="VD: nguyenvana",
        help="Bắt buộc trước khi lưu lại bản đã rà soát.",
    )

    if st.button("🔄 Làm mới danh sách"):
        st.session_state.selected_report_path = None
        st.session_state.loaded_report = None
        st.rerun()


# ============================================================
# MÀN HÌNH DANH SÁCH
# ============================================================
def render_list() -> None:
    st.title("📂 Danh sách báo cáo đã sinh ra")
    st.caption(f"Đang quét thư mục: `{config.OUTPUT_DIR}`")

    summaries = _list_reports()
    if only_bypassed:
        summaries = [s for s in summaries if s["hitl_bypassed"]]
    if search_text.strip():
        needle = search_text.strip().lower()
        summaries = [s for s in summaries if needle in s["document_source"].lower()]

    if not summaries:
        st.info("Không có báo cáo nào khớp bộ lọc hiện tại.")
        return

    st.caption(f"Tìm thấy {len(summaries)} báo cáo.")

    for summary in summaries:
        badge = "⚠️ CẦN RÀ SOÁT (worker tự động)" if summary["hitl_bypassed"] else "✅ Đã rà soát"
        with st.container(border=True):
            cols = st.columns([3, 2, 1, 1, 2, 1])
            cols[0].markdown(f"**{summary['document_source']}**\n\n`{summary['report_id']}`")
            cols[1].write(summary["generated_at"])
            cols[2].metric("Phát hiện", summary["total_findings"])
            cols[3].metric("Vi phạm", summary["violations"])
            cols[4].write(badge)
            if cols[5].button("Mở", key=f"open_{summary['path'].name}"):
                st.session_state.selected_report_path = str(summary["path"])
                st.session_state.loaded_report = None
                st.rerun()


# ============================================================
# MÀN HÌNH CHI TIẾT + RÀ SOÁT LẠI
# ============================================================
def render_detail(path: Path) -> None:
    if st.button("← Quay lại danh sách"):
        st.session_state.selected_report_path = None
        st.session_state.loaded_report = None
        st.rerun()

    if st.session_state.loaded_report is None:
        try:
            st.session_state.loaded_report = _load_full_report(path)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Không đọc được báo cáo: {exc}")
            with st.expander("Chi tiết kỹ thuật"):
                st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            return

    report: AuditReport = st.session_state.loaded_report

    st.title("📋 Rà soát báo cáo")
    st.caption(f"File nguồn: `{path.name}`")

    if report.hitl_bypassed:
        st.warning(
            "Báo cáo này được Background Worker TỰ ĐỘNG phê duyệt "
            "(`hitl_bypassed=True`) — CHƯA có giá trị sử dụng chính thức. "
            "Rà soát từng phát hiện bên dưới rồi lưu lại bản đã xác nhận.",
            icon="⚠️",
        )
    else:
        st.success("Báo cáo này đã được đánh dấu là đã rà soát (`hitl_bypassed=False`).")

    summary = report.summary()
    cols = st.columns(5)
    cols[0].metric("Tổng phát hiện", summary["total"])
    cols[1].metric("Nghi vi phạm", summary["violation"])
    cols[2].metric("Chưa đủ căn cứ", summary["uncertain"])
    cols[3].metric("Rủi ro cao", summary["high_risk"])
    cols[4].metric("Trích dẫn lỗi", summary["grounding_failed"])

    if not report.findings:
        st.info("Báo cáo này không có phát hiện nào.")
        return

    st.subheader("Từng phát hiện")
    for index, finding in enumerate(report.findings):
        icon = RISK_BADGE[finding.risk_level]
        is_bot_decision = finding.reviewer_id == _BOT_REVIEWER_ID

        title = f"{icon} #{index + 1} — {STATUS_TEXT[finding.has_violation]} (tin cậy {finding.confidence_score:.2f})"
        if is_bot_decision:
            title += "  🤖 Bot tự duyệt — CẦN BẠN XÁC NHẬN LẠI"

        with st.expander(title, expanded=is_bot_decision):
            st.markdown(f"**Trích dẫn từ chứng từ:** {finding.input_citation}")
            st.markdown(f"**Căn cứ pháp lý:** {finding.rule_citation}")
            st.markdown(f"**Giải thích:** {finding.explanation}")

            with st.expander("Xem suy luận từng bước của AI"):
                st.write(finding.reasoning_steps)

            if finding.grounding_verified is False:
                st.warning(
                    f"Trích dẫn không khớp chứng từ gốc (điểm khớp {finding.grounding_score:.2f}). "
                    f"{finding.downgraded_reason or ''}",
                    icon="⚠️",
                )

            decision_label = finding.human_decision.value if finding.human_decision else "chưa có"
            st.caption(
                f"Quyết định hiện tại: **{decision_label}** — bởi `{finding.reviewer_id or 'ẩn danh'}`"
                + (f" ({finding.human_note})" if finding.human_note else "")
            )

            note = st.text_input(
                "Ghi chú rà soát của bạn",
                key=f"note_{report.report_id}_{finding.finding_id}",
                value="" if is_bot_decision else (finding.human_note or ""),
                placeholder="Lý do phê duyệt hoặc từ chối...",
            )

            col1, col2 = st.columns(2)
            reviewer = st.session_state.reviewer_id.strip() or None
            with col1:
                if st.button("✅ Xác nhận / Phê duyệt", key=f"approve_{report.report_id}_{finding.finding_id}"):
                    finding.apply_decision(HumanDecision.APPROVED, note, reviewer)
                    st.rerun()
            with col2:
                if st.button("❌ Từ chối", key=f"reject_{report.report_id}_{finding.finding_id}"):
                    finding.apply_decision(HumanDecision.REJECTED, note, reviewer)
                    st.rerun()

    st.divider()
    _render_export_section(report, path)


def _render_export_section(report: AuditReport, source_path: Path) -> None:
    st.subheader("Lưu bản đã rà soát")

    remaining_bot = sum(1 for f in report.findings if f.reviewer_id == _BOT_REVIEWER_ID)
    reviewer = st.session_state.reviewer_id.strip()

    if remaining_bot:
        st.warning(f"Còn **{remaining_bot}** phát hiện vẫn do bot tự duyệt — hãy xác nhận lại từng cái ở trên.")
    if not reviewer:
        st.warning("Nhập mã/tên người kiểm duyệt ở thanh bên trước khi lưu.")

    can_save = remaining_bot == 0 and bool(reviewer)

    if st.button("💾 Lưu báo cáo đã rà soát (giữ nguyên bản gốc)", type="primary", disabled=not can_save):
        try:
            report.hitl_bypassed = False
            docx_bytes = report_exporter.build_docx_bytes(report, reviewer_id=reviewer)
            json_bytes = report_exporter.build_json_bytes(report)
        except report_exporter.ExportError as exc:
            st.error(str(exc))
            return

        stem = source_path.stem
        docx_path = config.OUTPUT_DIR / f"{stem}_reviewed.docx"
        json_path = config.OUTPUT_DIR / f"{stem}_reviewed.json"
        docx_path.write_bytes(docx_bytes)
        json_path.write_bytes(json_bytes)

        st.success(
            f"Đã lưu `{docx_path.name}` và `{json_path.name}` — bản gốc do worker "
            "sinh ra vẫn được giữ nguyên trong audit_reports/."
        )
        st.session_state.loaded_report = None


# ============================================================
# ĐIỀU HƯỚNG
# ============================================================
selected = st.session_state.selected_report_path
if selected:
    render_detail(Path(selected))
else:
    render_list()
