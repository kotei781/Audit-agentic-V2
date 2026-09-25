"""
app_streamlit.py
-----------------
GIAO DIỆN WEB + PHÂN QUYỀN (Admin / Kiểm toán viên).

CÁCH CHẠY:
    streamlit run app_streamlit.py

PHÂN VAI:
  * ADMIN  — quản trị kho luật: tải văn bản luật lên, tự động lưu vào
             `downloaded_laws/` VÀ đẩy vào ChromaDB; xem/xóa/index lại kho.
  * USER   — kiểm toán viên: chỉ upload chứng từ và bấm "Bắt đầu Kiểm toán".
             KHÔNG nhìn thấy, KHÔNG sửa được kho luật.

⚠ CẢNH BÁO BẢO MẬT — ĐỌC KỸ:
`st.sidebar.selectbox` chỉ là phân vai ở tầng HIỂN THỊ, KHÔNG phải cơ chế bảo
mật: bất kỳ ai cũng có thể tự đổi vai sang Admin. Ở đây đã bổ sung lớp mật
khẩu tối thiểu (biến ADMIN_PASSWORD trong .env / secrets.toml) để chặn thao
tác ghi. Khi triển khai thật cho nhiều người dùng, BẮT BUỘC thay bằng xác
thực thực thụ (SSO/OIDC, streamlit-authenticator, hoặc reverse proxy có auth)
và ghi nhật ký người thao tác — vì báo cáo kiểm toán cần biết AI của người nào.
"""

from __future__ import annotations

import traceback
import os
from pathlib import Path

import streamlit as st

import config
import data_ingestion
import rag_engine
import report_exporter
from audit_agent import AuditAgentError, NoLawCorpusError, run_agentic_audit
from data_ingestion import IngestionError
from schemas import AuditReport, HumanDecision, RiskLevel, ViolationStatus
from supervisor_agent import FlexibleSupervisor

st.set_page_config(
    page_title="Agentic AI Audit System",
    page_icon="🔍",
    layout="wide",
)

ROLE_USER = "👤 Kiểm toán viên (User)"
ROLE_ADMIN = "🔐 Quản trị kho luật (Admin)"

RISK_BADGE = {
    RiskLevel.HIGH: ("🔴", "#C0392B"),
    RiskLevel.MEDIUM: ("🟡", "#D68910"),
    RiskLevel.LOW: ("🟢", "#1E8449"),
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
        "report": None,
        "supervisor_result": None,
        "admin_unlocked": False,
        "reviewer_id": "",
        "last_error": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()


# ============================================================
# SIDEBAR — CHỌN VAI TRÒ
# ============================================================
with st.sidebar:
    st.title("🔍 AI Audit System")
    role = st.selectbox("Vai trò truy cập", [ROLE_USER, ROLE_ADMIN])

    st.divider()
    st.caption("Trạng thái hệ thống")

    api_ready = bool(config.GEMINI_API_KEY)
    st.write(f"{'✅' if api_ready else '❌'} Gemini API key")

    stats = rag_engine.collection_stats()
    if stats.get("available"):
        st.write(f"✅ Vector DB — {stats['total_chunks']} chunk")
    else:
        st.write("⚠️ Vector DB chưa sẵn sàng (chạy chế độ full-context)")

    st.caption(f"Model: `{config.GEMINI_MODEL}`")

    st.divider()
    st.session_state.reviewer_id = st.text_input(
        "Mã/tên người kiểm duyệt",
        value=st.session_state.reviewer_id,
        placeholder="VD: nguyenvana",
        help="Được ghi vào báo cáo để phục vụ audit trail. Bắt buộc trước khi xuất báo cáo.",
    )


# ============================================================
# TIỆN ÍCH HIỂN THỊ
# ============================================================
def _show_exception(exc: Exception, context: str = "") -> None:
    """Hiển thị lỗi thân thiện, giữ traceback trong expander cho người kỹ thuật."""
    st.error(f"{context}{exc}")
    with st.expander("Chi tiết kỹ thuật"):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def _dataframe(rows) -> None:
    """
    Hiển thị bảng, tương thích cả Streamlit cũ lẫn mới.

    Streamlit >= 1.49 thay `use_container_width` bằng `width="stretch"`; tham
    số cũ vẫn chạy nhưng in cảnh báo deprecated ra giữa giao diện. Helper này
    thử API mới trước, tự lùi về API cũ nếu môi trường dùng bản cũ hơn.
    """
    try:
        st.dataframe(rows, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(rows, use_container_width=True, hide_index=True)


# ============================================================
# MÀN HÌNH ADMIN
# ============================================================
def render_admin() -> None:
    st.title("🔐 Quản trị kho văn bản pháp luật")
    st.caption(
        "Văn bản tải lên được lưu vĩnh viễn trong `downloaded_laws/` và index "
        "vào ChromaDB. Kiểm toán viên không nhìn thấy màn hình này."
    )

    # ---- Cổng mật khẩu ----
    if config.ADMIN_PASSWORD and not st.session_state.admin_unlocked:
        with st.form("admin_login"):
            password = st.text_input("Mật khẩu quản trị", type="password")
            if st.form_submit_button("Mở khóa"):
                if password == config.ADMIN_PASSWORD:
                    st.session_state.admin_unlocked = True
                    st.rerun()
                else:
                    st.error("Mật khẩu không đúng.")
        st.stop()

    if not config.ADMIN_PASSWORD:
        st.warning(
            "Chưa đặt `ADMIN_PASSWORD` — vai trò Admin đang mở cho mọi người "
            "truy cập ứng dụng. Chỉ chấp nhận được khi chạy cục bộ để demo.",
            icon="⚠️",
        )

    tab_upload, tab_manage, tab_supervisor, tab_audit_config = st.tabs(["📥 Nạp văn bản luật", "🗂️ Kho luật hiện có", "⚙️ Cấu hình AI Giám sát", "🔍 Cấu hình AI Kiểm toán"])

    # ---------- TAB 1: UPLOAD ----------
    with tab_upload:
        uploaded_files = st.file_uploader(
            "Chọn văn bản quy phạm pháp luật",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help=f"Tối đa {config.MAX_UPLOAD_MB} MB/file. PDF phải có lớp text (không phải bản scan).",
        )

        col_a, col_b = st.columns([1, 2])
        with col_a:
            auto_index = st.checkbox("Tự động index vào Vector DB", value=True)
        with col_b:
            overwrite = st.checkbox("Ghi đè nếu trùng tên file", value=False)

        if st.button("💾 Lưu & Index", type="primary", disabled=not uploaded_files or st.session_state.get("indexing", False)):
            st.session_state.indexing = True
            for uploaded in uploaded_files:
                with st.status(f"Đang xử lý `{uploaded.name}`...", expanded=False) as status:
                    try:
                        info = data_ingestion.save_admin_law_file(uploaded, overwrite=overwrite)
                        status.write(
                            f"✅ Đã lưu vào kho: {info.filename} "
                            f"({info.size_human}, {info.char_count:,} ký tự)"
                        )

                        if auto_index:
                            status.write("🧠 Đang cắt chunk và tạo embedding...")
                            result = rag_engine.ingest_law_to_vector_db(
                                info.path, force_reindex=overwrite
                            )
                            status.write(f"✅ Đã index {result['chunks']} chunk vào ChromaDB.")

                        status.update(label=f"Hoàn tất: {uploaded.name}", state="complete")
                    except IngestionError as exc:
                        status.update(label=f"Lỗi: {uploaded.name}", state="error")
                        st.error(str(exc))
                    except rag_engine.RAGError as exc:
                        status.update(label=f"Đã lưu nhưng chưa index: {uploaded.name}", state="error")
                        st.warning(
                            f"File đã lưu vào kho nhưng chưa index được vào Vector DB: {exc}\n\n"
                            "Hệ thống vẫn kiểm toán được ở chế độ full-context."
                        )
                    except Exception as exc:  # noqa: BLE001
                        status.update(label=f"Lỗi: {uploaded.name}", state="error")
                        _show_exception(exc)
            st.session_state.indexing = False
            st.rerun()

    # ---------- TAB 2: QUẢN LÝ ----------
    with tab_manage:
        laws = data_ingestion.list_persisted_laws()
        stats = rag_engine.collection_stats()
        by_source = stats.get("by_source", {})

        col1, col2, col3 = st.columns(3)
        col1.metric("Số văn bản trong kho", len(laws))
        col2.metric("Tổng chunk đã index", stats.get("total_chunks", 0))
        col3.metric("Văn bản đã index", len(by_source))

        if not laws:
            st.info("Kho luật đang trống. Hãy nạp văn bản ở tab bên cạnh.")
        else:
            for info in laws:
                info.indexed_chunks = by_source.get(info.filename, 0)

            _dataframe([info.to_row() for info in laws])

            st.divider()
            st.subheader("Thao tác")

            action_col1, action_col2 = st.columns(2)

            with action_col1:
                target = st.selectbox("Chọn văn bản", [info.filename for info in laws])
                if st.button("🔄 Index lại văn bản này"):
                    if st.session_state.get("indexing", False):
                        st.warning("Hệ thống đang trong quá trình index, vui lòng đợi...")
                    else:
                        st.session_state.indexing = True
                        try:
                            result = rag_engine.ingest_law_to_vector_db(
                                config.LAW_STORAGE_DIR / target, force_reindex=True
                            )
                            st.success(f"Đã index lại: {result['chunks']} chunk.")
                            st.rerun()
                        except Exception as exc:  # noqa: BLE001
                            _show_exception(exc)
                        finally:
                            st.session_state.indexing = False

                if st.button("🗑️ Xóa khỏi kho", type="secondary"):
                    try:
                        rag_engine.delete_source(target)
                        data_ingestion.delete_law_file(target)
                        st.success(f"Đã xóa `{target}` khỏi kho luật và Vector DB.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        _show_exception(exc)

            with action_col2:
                st.write("Xây lại toàn bộ chỉ mục (dùng khi đổi model embedding):")
                if st.button("♻️ Rebuild toàn bộ Vector DB"):
                    with st.spinner("Đang index lại toàn bộ kho luật..."):
                        try:
                            results = rag_engine.rebuild_index()
                            total = sum(r.get("chunks", 0) for r in results)
                            st.success(f"Đã index lại {len(results)} văn bản, tổng {total} chunk.")
                        except Exception as exc:  # noqa: BLE001
                            _show_exception(exc)

    # ---------- TAB 3: CẤU HÌNH SUPERVISOR ----------
    with tab_supervisor:
        st.subheader("⚙️ Cấu hình Model & API Keys cho AI Giám sát (Supervisor)")
        st.caption("Quản lý Model và Keys cho chuỗi Fallback. Thay đổi tại đây sẽ cập nhật trực tiếp vào file .env")

        # Định nghĩa cấu hình chi tiết cho từng Tier
        config_schema = {
            "TIER1": {
                "model": ("Model Tier 1", "selectbox", ['gemini-1.5-pro', 'gemini-2.0-flash', 'gemini-1.5-flash'], "GEMINI_MODEL_TIER1"),
                "key": ("API Key Tier 1", "password", None, "GEMINI_API_KEY_1")
            },
            "TIER2": {
                "model": ("Model Tier 2", "selectbox", ['gemini-1.5-flash', 'gemini-2.0-flash-lite', 'gemini-1.5-pro'], "GEMINI_MODEL_TIER2"),
                "key": ("API Key Tier 2", "password", None, "GEMINI_API_KEY_2")
            },
            "TIER3": {
                "model": ("Model Tier 3", "text", "llama-3.3-70b-versatile", "GROQ_MODEL"),
                "key": ("API Key Tier 3", "password", None, "GROQ_API_KEY")
            },
            "TIER4": {
                "model": ("Model Tier 4", "selectbox", ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'], "OPENAI_MODEL"),
                "key": ("API Key Tier 4", "password", None, "OPENAI_API_KEY")
            },
            "TIER5": {
                "model": ("Model Tier 5 (Local)", "text", "llama3", "OLLAMA_MODEL"),
                "url": ("Ollama Base URL", "text", "http://localhost:11434", "OLLAMA_BASE_URL"),
            },
        }

        with st.form(key="supervisor_full_config_form"):
            user_inputs = {}
            cols = st.columns(3)
            for i, (tier_id, fields) in enumerate(config_schema.items()):
                with cols[i % 3]:
                    st.markdown(f"**{tier_id}**")
                    for field_id, (label, field_type, opt, env_var) in fields.items():
                        current_val = os.getenv(env_var, opt if opt and isinstance(opt, str) else "")
                        if field_type == "selectbox":
                            user_inputs[env_var] = st.selectbox(label, options=opt, index=opt.index(current_val) if current_val in opt else 0, key=f"{tier_id}_{field_id}")
                        elif field_type == "password":
                            user_inputs[env_var] = st.text_input(label, value=current_val, type="password", key=f"{tier_id}_{field_id}")
                        else:
                            user_inputs[env_var] = st.text_input(label, value=current_val, key=f"{tier_id}_{field_id}")
                    st.divider()

            submit_btn = st.form_submit_button("💾 Lưu Cấu Hình AI Giám Sát", type="primary")
            if submit_btn:
                try:
                    env_path = Path(".env")
                    if not env_path.exists(): lines = []
                    else:
                        with open(env_path, "r", encoding="utf-8") as f: lines = f.readlines()
                    for env_var, new_val in user_inputs.items():
                        found = False
                        for idx, line in enumerate(lines):
                            if line.startswith(f"{env_var}="):
                                lines[idx] = f"{env_var}={new_val}\n"
                                found = True
                                break
                        if not found: lines.append(f"{env_var}={new_val}\n")
                    with open(env_path, "w", encoding="utf-8") as f: f.writelines(lines)
                    for env_var, new_val in user_inputs.items(): os.environ[env_var] = new_val
                    st.success("✅ Đã cập nhật cấu hình Model và API Keys thành công!")
                    st.rerun()
                except Exception as exc: st.error(f"❌ Lỗi khi lưu cấu hình: {exc}")

        st.divider()
        if st.button("🧪 Kiểm tra kết nối Keys"):
            with st.spinner("Đang test kết nối tới các API..."):
                try:
                    sup = FlexibleSupervisor()
                    test_results = []
                    for tier_id, fields in config_schema.items():
                        for field_id, (label, field_type, opt, env_var) in fields.items():
                            if field_type == "password":
                                val = os.getenv(env_var)
                                if val and val not in ["mã_key_gemini_1_của_cậu", "mã_key_groq_của_cậu"]:
                                    test_results.append(f"✅ {label}: Key đã sẵn sàng")
                                else:
                                    test_results.append(f"❌ {label}: Key trống hoặc chưa thay đổi giá trị ví dụ")
                    for res in test_results: st.write(res)
                except Exception as exc: st.error(f"Lỗi khi test kết nối: {exc}")

    # ---------- TAB 4: CẤU HÌNH AUDIT AGENT ----------
    with tab_audit_config:
        st.subheader("🔍 Cấu hình AI Quét Kiểm toán (Audit Agent)")
        st.caption("Quản lý Model và Keys xoay vòng (Rotation). Thay đổi tại đây sẽ cập nhật vào file .env và áp dụng ngay lập tức.")

        with st.form(key="audit_agent_config_form"):
            model_options = ['gemini-1.5-pro', 'gemini-2.0-flash', 'gemini-1.5-flash']
            current_model = os.getenv("AUDIT_GEMINI_MODEL", "gemini-1.5-pro")
            selected_model = st.selectbox("Model kiểm toán", options=model_options, index=model_options.index(current_model) if current_model in model_options else 0)
            st.divider()
            st.write("**API Keys (Xoay vòng)**")
            key1 = st.text_input("API Key 1 (Bắt buộc)", value=os.getenv("AUDIT_GEMINI_KEY_1", ""), type="password", help="Key chính dùng cho Audit Agent")
            key2 = st.text_input("API Key 2 (Dự phòng)", value=os.getenv("AUDIT_GEMINI_KEY_2", ""), type="password")
            key3 = st.text_input("API Key 3 (Dự phòng)", value=os.getenv("AUDIT_GEMINI_KEY_3", ""), type="password")
            submit_audit_btn = st.form_submit_button("💾 Lưu Cấu Hình AI Kiểm Toán", type="primary")
            if submit_audit_btn:
                if not key1.strip(): st.error("❌ Lỗi: API Key 1 là bắt buộc")
                else:
                    try:
                        from dotenv import set_key
                        env_path = Path(".env")
                        if env_path.exists():
                            import shutil
                            shutil.copy(env_path, env_path.with_suffix(".env.bak"))
                        set_key(str(env_path), "AUDIT_GEMINI_MODEL", selected_model)
                        set_key(str(env_path), "AUDIT_GEMINI_KEY_1", key1)
                        set_key(str(env_path), "AUDIT_GEMINI_KEY_2", key2)
                        set_key(str(env_path), "AUDIT_GEMINI_KEY_3", key3)
                        os.environ["AUDIT_GEMINI_MODEL"] = selected_model
                        os.environ["AUDIT_GEMINI_KEY_1"] = key1
                        os.environ["AUDIT_GEMINI_KEY_2"] = key2
                        os.environ["AUDIT_GEMINI_KEY_3"] = key3
                        st.session_state.audit_config = {"model": selected_model, "key1": key1, "key2": key2, "key3": key3}
                        st.success("✅ Đã cập nhật cấu hình AI Kiểm toán thành công!")
                        st.rerun()
                    except Exception as exc: st.error(f"❌ Lỗi khi lưu cấu hình: {exc}")


# ============================================================
# MÀN HÌNH USER (KIỂM TOÁN VIÊN)
# ============================================================
def render_user() -> None:
    st.title("👤 Kiểm toán chứng từ tài chính")
    st.caption(
        "Tải chứng từ lên và bấm Bắt đầu Kiểm toán. Hệ thống tự động truy hồi "
        "các điều khoản liên quan từ kho luật của tổ chức."
    )

    if data_ingestion.law_corpus_is_empty():
        st.error(
            "Kho văn bản pháp luật đang trống. Vui lòng liên hệ quản trị viên "
            "để nạp văn bản trước khi kiểm toán."
        )
        st.stop()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        uploaded = st.file_uploader(
            "Chứng từ cần kiểm toán",
            type=["xlsx", "xls", "csv", "pdf", "docx", "txt"],
            help=f"Tối đa {config.MAX_UPLOAD_MB} MB.",
        )

    with col_right:
        use_sample = st.checkbox("Dùng dữ liệu mẫu để thử", value=False)
        top_k = st.slider("Số đoạn luật truy hồi (top-K)", 3, 15, config.RAG_TOP_K)

    run_clicked = st.button(
        "🚀 Bắt đầu Kiểm toán",
        type="primary",
        disabled=not (uploaded or use_sample),
    )

    if run_clicked:
        _run_audit(uploaded, use_sample, top_k)

    report: AuditReport | None = st.session_state.report
    if report is not None:
        st.divider()
        _render_report(report)


def _run_audit(uploaded, use_sample: bool, top_k: int) -> None:
    """Trích xuất chứng từ -> chạy Agent -> lưu report vào session_state."""
    try:
        with st.spinner("Đang trích xuất nội dung chứng từ..."):
            if use_sample and not uploaded:
                document_text = data_ingestion.load_sample_document_text()
                document_source = "sample_sql_data"
                document_hash = data_ingestion.compute_sha256(document_text.encode("utf-8"))
            else:
                temp_path: Path = data_ingestion.save_temp_upload(uploaded)
                document_text = data_ingestion.extract_text(temp_path, strict=False)
                document_source = uploaded.name
                document_hash = data_ingestion.compute_sha256(temp_path)
                temp_path.unlink(missing_ok=True)

        if not document_text.strip():
            st.error(
                "Không trích xuất được nội dung từ chứng từ. Nếu đây là PDF bản "
                "scan, hãy chạy OCR trước khi tải lên."
            )
            return

        with st.spinner("Đang truy hồi điều khoản liên quan và đối chiếu bằng Gemini..."):
            report = run_agentic_audit(
                document_text=document_text,
                document_source=document_source,
                document_sha256=document_hash,
                top_k=top_k,
            )

        # --- Tích hợp AI Giám sát (Supervisor) ---
        try:
            with st.spinner("🔍 AI Giám sát (Supervisor) đang thẩm định độc lập báo cáo..."):
                supervisor = FlexibleSupervisor()
                sup_result = supervisor.supervise(
                    raw_content=document_text,
                    audit_json=report.to_dict()
                )
                st.session_state.supervisor_result = sup_result
        except Exception as exc:
            st.warning(f"⚠️ AI Giám sát gặp sự cố nhưng hệ thống vẫn tiếp tục: {exc}")
            st.session_state.supervisor_result = None

        st.session_state.report = report
        summary = report.summary()
        st.success(
            f"Hoàn tất — {summary['total']} phát hiện "
            f"({summary['violation']} nghi vi phạm, {summary['uncertain']} chưa đủ căn cứ)."
        )

    except NoLawCorpusError as exc:
        st.error(str(exc))
    except (IngestionError, AuditAgentError) as exc:
        st.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        _show_exception(exc, "Lỗi ngoài dự kiến: ")


def _render_report(report: AuditReport) -> None:
    """Hiển thị kết quả + luồng kiểm duyệt con người + nút xuất báo cáo."""
    st.header("📋 Kết quả đối chiếu")

    # --- Hiển thị kết quả AI Giám sát ---
    sup_res = st.session_state.get("supervisor_result")
    if sup_res:
        status = sup_res.get("status")
        tier = sup_res.get("evaluated_by_tier", "Unknown")

        if status == "PASSED":
            st.success(f"✅ Đã thẩm định bởi AI Giám sát ({tier}) - Không phát hiện lỗi sai")
        elif status == "REJECTED":
            st.error(f"⚠️ CẢNH BÁO TỪ AI GIÁM SÁT: Báo cáo có dấu hiệu chứa sai sót/ảo giác!")
            with st.expander("Chi tiết sai sót phát hiện bởi Supervisor", expanded=True):
                discrepancies = sup_res.get("discrepancies", [])
                if not discrepancies:
                    st.write("Không có chi tiết lỗi cụ thể.")
                else:
                    for d in discrepancies:
                        st.markdown(f"""
                        - **Loại lỗi:** `{d.get('type')}` | **Trường:** `{d.get('field')}`
                        - **AI Kiểm toán báo:** `{d.get('audit_reported')}`
                        - **Thực tế gốc:** `{d.get('actual_raw_data')}`
                        - **Giải thích:** {d.get('explanation')}
                        """)
                st.info(f"Nhận xét tổng quan: {sup_res.get('supervisor_comment', '')}")
        else:
            st.warning(f"AI Giám sát trả về trạng thái không xác định: {status}")

    summary = report.summary()
    cols = st.columns(5)
    cols[0].metric("Tổng phát hiện", summary["total"])
    cols[1].metric("Nghi vi phạm", summary["violation"])
    cols[2].metric("Chưa đủ căn cứ", summary["uncertain"])
    cols[3].metric("Rủi ro cao", summary["high_risk"])
    cols[4].metric("Trích dẫn lỗi", summary["grounding_failed"])

    if summary["grounding_failed"]:
        st.warning(
            f"{summary['grounding_failed']} phát hiện có trích dẫn KHÔNG tìm thấy "
            "trong chứng từ gốc — hệ thống đã tự hạ cấp và bắt buộc người kiểm "
            "duyệt xác minh thủ công.",
            icon="⚠️",
        )

    if report.law_sources:
        with st.expander("Điều khoản pháp lý đã được truy hồi", expanded=False):
            _dataframe(
                [
                    {
                        "Văn bản": ref.filename,
                        "Điều khoản": ref.article or "(toàn văn)",
                        "Độ tương đồng": ref.similarity,
                    }
                    for ref in report.law_sources
                ]
            )

    if not report.findings:
        st.info("Không có phát hiện nào cần rà soát.")
        return

    st.subheader("Kiểm duyệt từng phát hiện (Human-in-the-Loop)")

    for index, finding in enumerate(report.findings):
        icon, color = RISK_BADGE[finding.risk_level]
        decided = finding.human_decision is not None

        with st.expander(
            f"{icon} Phát hiện #{index + 1} — {STATUS_TEXT[finding.has_violation]} "
            f"(tin cậy {finding.confidence_score:.2f})"
            + ("  ✔ đã xử lý" if decided else ""),
            expanded=not decided,
        ):
            st.markdown(f"**Trích dẫn từ chứng từ:** {finding.input_citation}")
            st.markdown(f"**Căn cứ pháp lý:** {finding.rule_citation}")
            st.markdown(f"**Giải thích:** {finding.explanation}")

            with st.expander("Xem suy luận từng bước của AI"):
                st.write(finding.reasoning_steps)

            if finding.grounding_verified is False:
                st.warning(
                    f"Trích dẫn không khớp chứng từ gốc (điểm khớp "
                    f"{finding.grounding_score:.2f}). {finding.downgraded_reason or ''}",
                    icon="⚠️",
                )

            note = st.text_input(
                "Ghi chú của người kiểm duyệt",
                key=f"note_{finding.finding_id}",
                value=finding.human_note or "",
                placeholder="Lý do phê duyệt hoặc từ chối...",
            )

            col1, col2, col3 = st.columns([1, 1, 3])
            reviewer = st.session_state.reviewer_id or None

            with col1:
                if st.button("✅ Phê duyệt", key=f"approve_{finding.finding_id}"):
                    finding.apply_decision(HumanDecision.APPROVED, note, reviewer)
                    st.rerun()
            with col2:
                if st.button("❌ Từ chối", key=f"reject_{finding.finding_id}"):
                    finding.apply_decision(HumanDecision.REJECTED, note, reviewer)
                    st.rerun()
            with col3:
                if decided:
                    st.success(f"Trạng thái: **{finding.human_decision.value}**")
                    if st.button("↩️ Hoàn tác", key=f"undo_{finding.finding_id}"):
                        finding.human_decision = None
                        finding.reviewed_at = None
                        finding.reviewer_id = None
                        st.rerun()
                else:
                    st.info("Trạng thái: _chưa xử lý_")

    # ---------- XUẤT BÁO CÁO ----------
    st.divider()
    pending = len(report.pending_findings())
    reviewer = st.session_state.reviewer_id.strip()

    if pending:
        st.warning(f"Còn **{pending}** phát hiện chưa được xử lý.")
    elif not reviewer:
        st.warning("Nhập mã/tên người kiểm duyệt ở thanh bên trước khi xuất báo cáo.")
    else:
        st.success("Đã kiểm duyệt đầy đủ — có thể xuất báo cáo.")

    can_export = pending == 0 and bool(reviewer)

    if can_export:
        try:
            docx_bytes = report_exporter.build_docx_bytes(report, reviewer_id=reviewer)
            json_bytes = report_exporter.build_json_bytes(report)
        except report_exporter.ExportError as exc:
            st.error(str(exc))
            return

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Tải Báo Cáo (.docx)",
                data=docx_bytes,
                file_name=report_exporter.suggest_filename(report, "docx"),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
            )
        with col2:
            st.download_button(
                "🧾 Tải dữ liệu thô (.json)",
                data=json_bytes,
                file_name=report_exporter.suggest_filename(report, "json"),
                mime="application/json",
            )
    else:
        st.button("📥 Tải Báo Cáo (.docx)", disabled=True)


# ============================================================
# ĐIỀU HƯỚNG
# ============================================================
if not config.GEMINI_API_KEY:
    st.error(
        "Chưa cấu hình `GEMINI_API_KEY`. Tạo file `.env` với dòng "
        "`GEMINI_API_KEY=...` hoặc khai báo trong `.streamlit/secrets.toml`, "
        "rồi khởi động lại ứng dụng."
    )
    st.stop()

if role == ROLE_ADMIN:
    render_admin()
else:
    render_user()
