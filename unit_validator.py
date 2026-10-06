"""
unit_validator.py
-----------------
Module xác thực đơn vị đo lường thông minh.
Xử lý nhiễu OCR và quản lý xung đột đa tiền tệ cho chứng từ quốc tế.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set


@dataclass
class UnitViolation:
    """Đối tượng lỗi đơn vị tương thích với Audit Agent."""

    type: str
    detail: str
    severity: str = "ERROR"
    action: Optional[str] = None
    explanation: str = ""
    text_segment: str = ""
    row: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnitValidator:
    def __init__(self):
        # Từ điển phân loại đơn vị chi tiết
        self.UNIT_CATALOG = {
            "CURRENCY": {"VND", "VNĐ", "USD", "EUR", "JPY", "GBP", "CNY", "Đ", "đ"},
            "MASS": {"KG", "G", "TẤN", "LBS", "POUND"},
            "VOLUME": {"LÍT", "L", "M3", "ML", "GALLON"},
            "AREA": {"M2", "SQFT", "HECTA"},
            "ENERGY": {"KWH", "W", "BTU"},
            "COUNT": {"CHIẾC", "CÁI", "BỘ", "KIỆN", "LÔ", "UNIT"},
        }

        # Bản đồ sửa lỗi OCR phổ biến (Common OCR Misinterpretations)
        self.OCR_FIX_MAP = {
            "K9": "KG",
            "K0": "KG",
            "1ÍT": "LÍT",
            "L1T": "LÍT",
            "VNDS": "VND",
            "VND_": "VND",
            "0USD": "USD",
            "VNDS": "VND",
            "USD$": "USD",
            "US$": "USD",
        }

    def _normalize_unit(self, unit_str: str) -> str:
        """Chuẩn hóa đơn vị: Xóa khoảng trắng, viết hoa, sửa lỗi OCR."""
        if not unit_str:
            return ""

        normalized = str(unit_str).strip().upper()
        normalized = re.sub(r"[^A-ZÀ-Ỹ0-9_\-/]", "", normalized)
        return self.OCR_FIX_MAP.get(normalized, normalized)

    def identify_unit_type(self, unit_str: str) -> Optional[str]:
        """Xác định loại đơn vị (ví dụ: CURRENCY, MASS)."""
        norm = self._normalize_unit(unit_str)
        for category, units in self.UNIT_CATALOG.items():
            if norm in units:
                return category
        return None

    def _coerce_rows(self, data: Any) -> List[Dict[str, Any]]:
        """Chuyển dữ liệu văn bản / DataFrame / dict / list thành list[dict]."""
        if data is None:
            return []
        if isinstance(data, str):
            rows: List[Dict[str, Any]] = []
            for line in data.splitlines():
                if not line.strip():
                    continue
                matches = re.findall(r"[-+]?\d[\d,\s.]*", line)
                if len(matches) >= 3:
                    unit_match = re.findall(r"(VND|VNĐ|USD|EUR|JPY|GBP|CNY|KG|G|L|ML|M3|TẤN|CHIẾC|CÁI|BỘ|LÔ|UNIT)", line, flags=re.IGNORECASE)
                    rows.append(
                        {
                            "unit": unit_match[0] if unit_match else "",
                            "quantity": matches[0],
                            "unit_price": matches[1],
                            "total_amount": matches[2],
                            "source_text": line.strip(),
                        }
                    )
            return rows

        if isinstance(data, dict):
            return [data]

        if hasattr(data, "to_dict"):
            try:
                return list(data.to_dict("records"))
            except Exception:
                pass

        if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
            result: List[Dict[str, Any]] = []
            for item in data:
                if isinstance(item, dict):
                    result.append(item)
            return result

        return []

    def validate_unit_consistency(self, data: Any) -> List[UnitViolation]:
        """
        Xác thực tính nhất quán của đơn vị trong cùng một bảng.
        Xử lý ngoại lệ cho chứng từ đa tiền tệ (USD/VND).
        """
        rows = self._coerce_rows(data)
        violations: List[UnitViolation] = []
        found_currencies: Set[str] = set()

        if isinstance(data, str):
            for token in re.findall(r"\b(?:VND|VNĐ|USD|EUR|JPY|GBP|CNY|Đ|đ)\b", data, flags=re.IGNORECASE):
                norm = self._normalize_unit(token)
                if self.identify_unit_type(norm) == "CURRENCY":
                    found_currencies.add(norm)

        for idx, row in enumerate(rows):
            unit = row.get("unit") or row.get("currency") or row.get("currency_code") or ""
            norm_unit = self._normalize_unit(unit)
            if self.identify_unit_type(norm_unit) == "CURRENCY":
                found_currencies.add(norm_unit)

        if len(found_currencies) > 1:
            normalized_set = {u for u in found_currencies if u in {"USD", "VND", "VNĐ", "Đ", "đ"}}
            if len(normalized_set) == len(found_currencies):
                violations.append(
                    UnitViolation(
                        type="MULTICURRENCY_DETECTED",
                        detail=f"Chứng từ chứa nhiều loại tiền tệ ({', '.join(sorted(found_currencies))}). Hệ thống chuyển sang chế độ xác thực tỷ giá quy đổi.",
                        severity="INFO",
                        action="CHECK_EXCHANGE_RATE",
                        explanation="Chứng từ ghi đồng thời nhiều loại tiền tệ, cần đối chiếu tỷ giá quy đổi trước khi kết luận.",
                        text_segment="multi_currency_detected",
                    )
                )
            else:
                violations.append(
                    UnitViolation(
                        type="UNIT_INCONSISTENCY",
                        detail=f"Phát hiện xung đột tiền tệ không chuẩn: {', '.join(sorted(found_currencies))}",
                        severity="ERROR",
                        action="MANUAL_REVIEW",
                        explanation="Có sự mâu thuẫn về đơn vị tiền tệ giữa các dòng/chứng từ, không nên tự kết luận hợp lệ.",
                        text_segment="currency_conflict",
                    )
                )

        return violations

    def check_unit_presence(self, data: Any, required_category: str) -> List[UnitViolation]:
        """Kiểm tra xem một danh mục đơn vị bắt buộc có xuất hiện hay không."""
        rows = self._coerce_rows(data)
        violations: List[UnitViolation] = []
        has_category = False

        for row in rows:
            unit = row.get("unit") or row.get("currency") or row.get("currency_code") or ""
            if self.identify_unit_type(unit) == required_category:
                has_category = True
                break

        if not has_category:
            violations.append(
                UnitViolation(
                    type="MISSING_UNIT_CATEGORY",
                    detail=f"Chứng từ thiếu đơn vị đo lường thuộc nhóm {required_category}.",
                    severity="WARNING",
                    explanation=f"Không tìm thấy đơn vị thuộc nhóm {required_category} trong dữ liệu đã phân tích.",
                    text_segment="missing_unit_category",
                )
            )

        return violations


def validate_unit_consistency(data: Any) -> List[UnitViolation]:
    """Wrapper mức module, giữ tương thích với code cũ."""
    return UnitValidator().validate_unit_consistency(data)


def check_unit_presence(data: Any, required_category: str) -> List[UnitViolation]:
    """Wrapper mức module, giữ tương thích với code cũ."""
    return UnitValidator().check_unit_presence(data, required_category)
