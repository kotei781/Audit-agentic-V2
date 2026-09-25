"""
schemas.py
----------
Định nghĩa cấu trúc dữ liệu (Pydantic v2) cho toàn hệ thống.

BA THAY ĐỔI QUAN TRỌNG SO VỚI BẢN CŨ:

1. TÁCH MODEL AI-FACING VÀ MODEL NỘI BỘ
   `AIFinding` là schema DUY NHẤT gửi cho Gemini (dùng trực tiếp làm
   response_schema). `ViolationCheckResult` kế thừa nó và bổ sung các trường
   phục vụ Human-in-the-Loop + kiểm chứng. Nhờ vậy AI không bao giờ tự sinh
   ra `human_decision`, và ta chỉ còn MỘT nguồn định nghĩa schema (bản cũ
   định nghĩa 2 lần: dict thô trong audit_agent.py và Pydantic ở đây).

2. THÊM TRƯỜNG `reasoning_steps` ĐẶT TRƯỚC `has_violation`
   Với Structured Output, thứ tự trường = thứ tự sinh token. Bản cũ yêu cầu
   model "phân tích step-by-step" nhưng lại bắt nó xuất `has_violation` ngay
   ở token đầu tiên — chỉ thị đó hoàn toàn vô tác dụng. Đặt reasoning lên
   trước mới thực sự kích hoạt được chain-of-thought.

3. AUDIT TRAIL ĐẦY ĐỦ
   Báo cáo kiểm toán chỉ có giá trị khi tái lập được: ai duyệt, lúc nào,
   model nào, phiên bản prompt nào, lấy từ đoạn luật nào.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    """Thời điểm hiện tại theo UTC (timezone-aware).

    Bản cũ dùng datetime.now() naive theo giờ máy — không chấp nhận được với
    một hệ thống kiểm toán chạy trên nhiều máy/nhiều múi giờ.
    """
    return datetime.now(timezone.utc)


# ============================================================
# ENUM
# ============================================================
class ViolationStatus(str, Enum):
    YES = "YES"
    NO = "NO"
    UNCERTAIN = "UNCERTAIN"


class RecommendedAction(str, Enum):
    FLAG_FOR_HUMAN_REVIEW = "FLAG_FOR_HUMAN_REVIEW"
    APPROVE = "APPROVE"


class RiskLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class HumanDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


# ============================================================
# MODEL AI-FACING — dùng trực tiếp làm response_schema của Gemini
# ============================================================
class AIFinding(BaseModel):
    """
    Một phát hiện do AI sinh ra. Thứ tự khai báo trường ở đây chính là thứ tự
    Gemini sinh JSON, nên `reasoning_steps` BẮT BUỘC đứng đầu.

    Lưu ý kỹ thuật: KHÔNG đặt ràng buộc ge/le trên confidence_score ở model
    này. response_schema của Gemini là tập con của OpenAPI 3.0 và có thể từ
    chối một số keyword ràng buộc số học. Giá trị được kẹp lại (clamp) bằng
    validator phía Python — validator không ảnh hưởng tới JSON Schema sinh ra.
    """

    reasoning_steps: str = Field(
        description=(
            "Suy luận từng bước TRƯỚC khi kết luận, theo đúng trình tự: "
            "(1) Mục dữ liệu nào trong chứng từ đang được xét; "
            "(2) Điều/Khoản nào trong văn bản luật được cung cấp có liên quan; "
            "(3) Đối chiếu cụ thể giữa (1) và (2); "
            "(4) Kết luận rút ra. Viết ngắn gọn, mỗi bước 1-2 câu."
        )
    )
    has_violation: ViolationStatus = Field(
        description=(
            "YES nếu chắc chắn vi phạm; NO nếu chắc chắn không vi phạm; "
            "UNCERTAIN nếu văn bản luật được cung cấp KHÔNG đủ căn cứ để kết "
            "luận. Khi phân vân, BẮT BUỘC chọn UNCERTAIN thay vì đoán."
        )
    )
    input_citation: str = Field(
        description=(
            "Trích dẫn NGUYÊN VĂN (copy chính xác từng ký tự) đoạn text trong "
            "chứng từ đầu vào. Không diễn giải, không tóm tắt, không sửa lại "
            "chính tả. Trích dẫn này sẽ được đối chiếu tự động với tài liệu "
            "gốc; nếu không khớp, phát hiện sẽ bị hạ cấp thành UNCERTAIN."
        )
    )
    rule_citation: str = Field(
        description=(
            "Trích dẫn CHÍNH XÁC Điều/Khoản của văn bản quy định được cung cấp "
            "trong prompt. Nếu không có điều khoản nào trong phần luật được "
            "cung cấp áp dụng được, ghi 'KHÔNG CÓ CĂN CỨ TRONG NGỮ CẢNH ĐƯỢC "
            "CUNG CẤP' và đặt has_violation = UNCERTAIN."
        )
    )
    explanation: str = Field(
        description="Giải thích ngắn gọn, khách quan lý do đi tới kết luận trên."
    )
    confidence_score: float = Field(
        description=(
            "Độ tin cậy vào kết luận, từ 0.0 đến 1.0. Đây là tự đánh giá của "
            "mô hình, KHÔNG phải xác suất đã hiệu chuẩn."
        )
    )
    recommended_action: RecommendedAction = Field(
        description="FLAG_FOR_HUMAN_REVIEW nếu cần người xác minh, APPROVE nếu không."
    )

    @field_validator("confidence_score")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        """Kẹp confidence_score về [0.0, 1.0] thay vì để ValidationError làm hỏng cả batch."""
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0


class AgentResponse(BaseModel):
    """Bao ngoài cho toàn bộ phản hồi của Agent — đây là RESPONSE_SCHEMA."""

    findings: List[AIFinding] = Field(
        default_factory=list,
        description=(
            "Danh sách phát hiện. Nếu rà soát xong không có vấn đề nào, trả về "
            "mảng rỗng []. Mỗi mục dữ liệu đáng ngờ là MỘT phần tử riêng."
        ),
    )


# Alias giữ đúng tên gọi trong tài liệu yêu cầu.
RESPONSE_SCHEMA = AgentResponse


# ============================================================
# MODEL NỘI BỘ — AI output + verification + Human-in-the-Loop
# ============================================================
class ViolationCheckResult(AIFinding):
    """Phát hiện của AI sau khi đã bổ sung kiểm chứng và quyết định của con người."""

    finding_id: str = Field(default_factory=lambda: uuid4().hex[:12])

    # ----- Kết quả kiểm chứng tự động (anti-hallucination) -----
    grounding_verified: Optional[bool] = Field(
        default=None,
        description=(
            "True nếu input_citation thực sự tồn tại trong tài liệu gốc. "
            "None nghĩa là chưa chạy kiểm chứng."
        ),
    )
    grounding_score: float = Field(
        default=0.0, description="Tỷ lệ khớp của input_citation với tài liệu gốc (0-1)."
    )
    downgraded_reason: Optional[str] = Field(
        default=None,
        description="Lý do hệ thống tự hạ cấp kết luận của AI (nếu có).",
    )
    retrieved_chunk_ids: List[str] = Field(
        default_factory=list,
        description="ID các chunk luật mà RAG đã cung cấp cho lần phân tích này.",
    )

    # ----- Human-in-the-Loop -----
    human_decision: Optional[HumanDecision] = None
    human_note: Optional[str] = None
    reviewer_id: Optional[str] = None
    reviewed_at: Optional[datetime] = None

    @property
    def risk_level(self) -> RiskLevel:
        """Mức rủi ro suy ra từ kết luận + độ tin cậy, dùng để tô màu trên UI."""
        if self.has_violation == ViolationStatus.YES:
            return RiskLevel.HIGH if self.confidence_score >= 0.7 else RiskLevel.MEDIUM
        if self.has_violation == ViolationStatus.UNCERTAIN:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def apply_decision(
        self,
        decision: HumanDecision,
        note: Optional[str] = None,
        reviewer_id: Optional[str] = None,
    ) -> None:
        """Ghi nhận quyết định của người kiểm duyệt kèm dấu vết thời gian."""
        self.human_decision = decision
        self.human_note = note or None
        self.reviewer_id = reviewer_id
        self.reviewed_at = utc_now()

    @classmethod
    def from_ai_finding(cls, finding: AIFinding, **extra: Any) -> "ViolationCheckResult":
        return cls(**finding.model_dump(), **extra)

    @classmethod
    def build_parse_failure(cls, error: str, raw_payload: Any = None) -> "ViolationCheckResult":
        """
        Tạo một finding UNCERTAIN khi dữ liệu AI trả về sai định dạng.

        Nguyên tắc: KHÔNG tự "vá" hay bịa dữ liệu để hợp lệ hóa. Giữ nguyên
        payload thô để con người kiểm tra được chuyện gì đã xảy ra.
        """
        return cls(
            reasoning_steps="(Không có — phản hồi của mô hình không parse được.)",
            has_violation=ViolationStatus.UNCERTAIN,
            input_citation="(Không xác định — dữ liệu AI trả về sai định dạng)",
            rule_citation="N/A",
            explanation=(
                "Phản hồi của Gemini không khớp schema bắt buộc, cần con người "
                f"kiểm tra thủ công. Chi tiết lỗi: {error}\n"
                f"Payload thô: {str(raw_payload)[:500]}"
            ),
            confidence_score=0.0,
            recommended_action=RecommendedAction.FLAG_FOR_HUMAN_REVIEW,
            grounding_verified=False,
            downgraded_reason="parse_failure",
        )


# ============================================================
# BÁO CÁO
# ============================================================
class LawSourceRef(BaseModel):
    """Tham chiếu tới một nguồn luật đã được dùng trong lần kiểm toán."""

    filename: str
    chunk_id: Optional[str] = None
    article: Optional[str] = None
    similarity: Optional[float] = None


class AuditReport(BaseModel):
    """Toàn bộ báo cáo kiểm toán sinh ra từ một phiên chạy hệ thống."""

    report_id: str = Field(default_factory=lambda: uuid4().hex[:16])
    generated_at: datetime = Field(default_factory=utc_now)

    document_source: str = Field(description="Tên/đường dẫn chứng từ đã kiểm toán.")
    document_sha256: Optional[str] = Field(
        default=None, description="Hash chứng từ đầu vào — chứng minh tính toàn vẹn."
    )

    law_sources: List[LawSourceRef] = Field(default_factory=list)
    retrieval_mode: str = Field(
        default="rag",
        description="'rag' = truy hồi theo ngữ nghĩa; 'full_context' = nạp toàn bộ kho luật.",
    )

    # ----- Audit trail: metadata tái lập -----
    provider_used: str = Field(default="gemini")
    model_used: str = Field(default="")
    embedding_model: str = Field(default="")
    prompt_version: str = Field(default="")
    temperature: float = Field(default=0.0)
    hitl_bypassed: bool = Field(
        default=False,
        description=(
            "True nếu báo cáo được tạo ở chế độ --auto-approve (bỏ qua kiểm "
            "duyệt người thật). Báo cáo dạng này KHÔNG có giá trị sử dụng chính thức."
        ),
    )

    findings: List[ViolationCheckResult] = Field(default_factory=list)
    runtime_metadata: Dict[str, Any] = Field(default_factory=dict)

    # [NEW FEATURE] Thêm thông tin về việc cắt bớt tài liệu đầu vào
    document_truncated: bool = Field(
        default=False,
        description="True nếu chứng từ gốc bị cắt bớt do vượt quá MAX_DOCUMENT_CHARS."
    )
    truncation_details: Optional[str] = Field(
        default=None,
        description="Chi tiết số ký tự bị cắt bớt."
    )

    # ----- Truy vấn tiện ích -----
    def approved_findings(self) -> List[ViolationCheckResult]:
        return [f for f in self.findings if f.human_decision == HumanDecision.APPROVED]

    def rejected_findings(self) -> List[ViolationCheckResult]:
        return [f for f in self.findings if f.human_decision == HumanDecision.REJECTED]

    def pending_findings(self) -> List[ViolationCheckResult]:
        return [f for f in self.findings if f.human_decision is None]

    def is_fully_reviewed(self) -> bool:
        return not self.pending_findings()

    def violations(self) -> List[ViolationCheckResult]:
        return [f for f in self.findings if f.has_violation == ViolationStatus.YES]

    def summary(self) -> Dict[str, int]:
        """Bảng thống kê nhanh, dùng cho dashboard và trang đầu báo cáo Word."""
        return {
            "total": len(self.findings),
            "violation": len(self.violations()),
            "uncertain": len(
                [f for f in self.findings if f.has_violation == ViolationStatus.UNCERTAIN]
            ),
            "compliant": len(
                [f for f in self.findings if f.has_violation == ViolationStatus.NO]
            ),
            "high_risk": len([f for f in self.findings if f.risk_level == RiskLevel.HIGH]),
            "grounding_failed": len(
                [f for f in self.findings if f.grounding_verified is False]
            ),
            "approved": len(self.approved_findings()),
            "rejected": len(self.rejected_findings()),
            "pending": len(self.pending_findings()),
        }
