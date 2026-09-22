# Agentic AI Audit System — v2 (RAG + Persistence + RBAC)

Hệ thống rà soát chứng từ tài chính tự động: đối chiếu chứng từ với kho văn bản
pháp luật của tổ chức bằng Google Gemini, có kiểm chứng trích dẫn và bắt buộc
kiểm duyệt của con người trước khi xuất báo cáo.

---

## 1. Lỗ hổng kiến trúc đã khắc phục

| Vấn đề ở bản v1 | Cách xử lý ở v2 |
|---|---|
| Mỗi lần kiểm toán phải truyền tay file luật (`--law-pdf`) | Kho luật cố định `downloaded_laws/`, Admin nạp một lần, Agent tự lấy |
| Nhồi toàn bộ văn bản luật vào prompt → không scale, "lost in the middle" | RAG với ChromaDB lưu vĩnh viễn tại `chroma_db/`, chỉ truy hồi top-K đoạn liên quan |
| Prompt yêu cầu "suy luận từng bước" nhưng schema bắt kết luận ngay token đầu | Thêm trường `reasoning_steps` **đặt trước** `has_violation` |
| Schema định nghĩa 2 nơi (dict thô + Pydantic) → chắc chắn lệch | Một nguồn duy nhất: `schemas.AgentResponse` dùng trực tiếp làm `response_schema` |
| `input_citation` không được kiểm chứng → AI có thể bịa trích dẫn | `verify_grounding()` đối chiếu ngược với chứng từ gốc; không khớp → hạ cấp xuống `UNCERTAIN` |
| Chứng từ nhét thẳng vào prompt → prompt injection | Delimiter `<<<DOCUMENT>>>` + quy tắc "dữ liệu không phải chỉ thị" trong system prompt |
| Không kiểm tra `finish_reason` → JSON vỡ khi chạm trần token | Kiểm tra `MAX_TOKENS`/`SAFETY` và báo lỗi có ngữ nghĩa |
| Không retry khi API lỗi tạm thời | Backoff lũy thừa cho 429/500/503 |
| `config.py` import `streamlit` → CLI phụ thuộc UI | Lazy import trong `_read_secret()` |
| Báo cáo không tái lập được (không ghi model, prompt version, người duyệt) | Audit trail đầy đủ trong `AuditReport` |
| `datetime.now()` naive theo giờ máy | `datetime.now(timezone.utc)` |
| `--auto-approve` bỏ qua HITL mà báo cáo không đánh dấu | Cờ `hitl_bypassed` + cảnh báo đỏ trên trang đầu file Word |
| PDF scan → text rỗng → Agent chạy trên ngữ cảnh rỗng | Chặn bằng `MIN_EXTRACTED_TEXT_LENGTH`, gợi ý chạy OCR |
| Tải luật từ URL tùy ý (SSRF) | Gỡ bỏ; luật do Admin chủ động nạp |
| Ai cũng sửa được kho luật | Phân vai Admin/User + `ADMIN_PASSWORD` |
| Chỉ xuất JSON | Xuất cả `.docx` (báo cáo trình bày) và `.json` (dữ liệu thô) |

---

## 2. Kiến trúc

```
                    ┌─────────────────────────────────────┐
   ADMIN ──upload──►│ data_ingestion.save_admin_law_file()│
                    │        downloaded_laws/  (vĩnh viễn)│
                    └──────────────┬──────────────────────┘
                                   │ chunk theo "Điều N" + embed
                                   ▼
                    ┌─────────────────────────────────────┐
                    │ rag_engine  →  ChromaDB (chroma_db/)│
                    └──────────────┬──────────────────────┘
                                   │ retrieve_relevant_laws(top_k)
                                   ▼
   USER ──chứng từ──► audit_agent.run_agentic_audit()
                                   │
                    ┌──────────────┴──────────────┐
                    │ Gemini + RESPONSE_SCHEMA    │
                    │ verify_grounding()          │
                    └──────────────┬──────────────┘
                                   ▼
                    Human-in-the-Loop (UI / CLI)
                                   ▼
                    report_exporter → .docx + .json
```

| File | Vai trò |
|---|---|
| `config.py` | Cấu hình tập trung, đọc secret, logging |
| `schemas.py` | Pydantic model — nguồn schema duy nhất |
| `data_ingestion.py` | Lưu trữ cố định + trích xuất PDF/DOCX/TXT/CSV/XLSX |
| `rag_engine.py` | Chunking theo Điều, embedding Gemini, ChromaDB, multi-query retrieval |
| `audit_agent.py` | Điều phối: RAG → prompt → Gemini → parse → kiểm chứng |
| `report_exporter.py` | Xuất `.docx` / `.json`, chốt chặn HITL |
| `human_review.py` | Kiểm duyệt qua CLI |
| `app_streamlit.py` | Giao diện web, phân vai Admin/User |
| `main.py` | CLI: `ingest-law`, `list-laws`, `reindex`, `audit` |

---

## 3. Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env      # rồi mở .env và điền GEMINI_API_KEY
```

Lấy API key tại <https://aistudio.google.com/apikey>.

---

## 4. Sử dụng

### 4.1. Giao diện web

```bash
streamlit run app_streamlit.py
```

- **Admin** — nạp văn bản luật, xem/xóa/index lại kho.
- **User** — upload chứng từ (`.xlsx`, `.csv`, `.pdf`, `.docx`), bấm *Bắt đầu Kiểm toán*, duyệt từng phát hiện, tải báo cáo `.docx`.

### 4.2. Dòng lệnh

```bash
python main.py ingest-law --file nghi_dinh_123.pdf   # nạp 1 văn bản
python main.py ingest-law --dir ./van_ban_luat       # nạp cả thư mục
python main.py list-laws                             # xem kho
python main.py reindex                               # xây lại Vector DB
python main.py audit --file bang_ke.xlsx --reviewer nguyenvana
python main.py audit --sample                        # chạy thử với dữ liệu mẫu
```

---

## 5. Giới hạn cần biết trước khi dùng thật

**Phân quyền hiện tại chưa phải bảo mật thật.** `st.sidebar.selectbox` chỉ là
phân vai ở tầng hiển thị; `ADMIN_PASSWORD` là lớp chặn tối thiểu. Triển khai
đa người dùng cần SSO/OIDC hoặc reverse proxy có xác thực, kèm nhật ký thao tác.

**Dữ liệu chứng từ được gửi ra API bên thứ ba.** Mã số thuế, số tiền, tên đối
tác đều đi tới Google. Trước khi dùng với dữ liệu khách hàng thật, cần rà soát
nghĩa vụ theo Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân, cân nhắc
masking/tokenization hoặc chuyển sang Vertex AI có chọn region kèm DPA.

**`confidence_score` không phải xác suất đã hiệu chuẩn.** Đây là tự đánh giá
của mô hình. Không dùng làm ngưỡng tự động hóa khi chưa hiệu chuẩn trên tập có nhãn.

**`temperature=0.0` giảm phương sai, không loại bỏ ảo giác.** Cơ chế chống ảo
giác thật sự của hệ thống là `verify_grounding()` + phạm vi căn cứ đóng + HITL.

**Chưa có tầng rule engine.** Các kiểm tra số học thuần (`vat_amount ==
amount × vat_rate`, mã số thuế đủ 10/13 chữ số, thuế suất thuộc tập hợp lệ)
hiện vẫn do LLM làm — đắt hơn, chậm hơn và kém chính xác hơn vài dòng Python.
Đây là hạng mục ưu tiên tiếp theo: xử lý deterministic ở tầng L1, chỉ đẩy phần
cần diễn giải ngôn ngữ pháp lý sang LLM.

**Chưa có OCR.** PDF bản scan sẽ bị từ chối ngay khi nạp. Chạy `ocrmypdf` trước.
