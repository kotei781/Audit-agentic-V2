"""
unit_validator.py
-----------------
Module xác thực đơn vị đo lường thông minh.
Xử lý nhiễu OCR và quản lý xung đột đa tiền tệ cho chứng từ quốc tế.
"""

from __future__ import annotations
import re
from typing import List, Dict, Optional, Set, Tuple, Any

class UnitValidator:
    def __init__(self):
        # Từ điển phân loại đơn vị chi tiết
        self.UNIT_CATALOG = {
            "CURRENCY": {"VND", "VNĐ", "USD", "EUR", "JPY", "GBP", "CNY", "đ"},
            "MASS": {"KG", "G", "TẤN", "LBS", "POUND"},
            "VOLUME": {"LÍT", "L", "M3", "ML", "GALLON"},
            "AREA": {"M2", "SQFT", "HECTA"},
            "ENERGY": {"KWH", "W", "BTU"},
            "COUNT": {"CHIẾC", "CÁI", "BỘ", "KIỆN", "LÔ", "UNIT"}
        }

        # Bản đồ sửa lỗi OCR phổ biến (Common OCR Misinterpretations)
        self.OCR_FIX_MAP = {
            "K9": "KG", "K0": "KG", "1ÍT": "LÍT", "L1T": "LÍT",
            "VNDS": "VND", "VND_": "VND", "0USD": "USD"
        }

    def _normalize_unit(self, unit_str: str) -> str:
        """Chuẩn hóa đơn vị: Xóa khoảng trắng, viết hoa, sửa lỗi OCR."""
        if not unit_str: return ""

        normalized = str(unit_str).strip().upper()
        # Áp dụng bản đồ sửa lỗi OCR
        return self.OCR_FIX_MAP.get(normalized, normalized)

    def identify_unit_type(self, unit_str: str) -> Optional[str]:
        """Xác định loại đơn vị (ví dụ: CURRENCY, MASS)."""
        norm = self._normalize_unit(unit_str)
        for category, units in self.UNIT_CATALOG.items():
            if norm in units:
                return category
        return None

    def validate_unit_consistency(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Xác thực tính nhất quán của đơn vị trong cùng một bảng.
        Xử lý ngoại lệ cho chứng từ đa tiền tệ (USD/VND).
        """
        violations = []
        found_currencies: Set[str] = set()

        for idx, row in enumerate(data):
            unit = row.get('unit', '')
            norm_unit = self._normalize_unit(unit)

            if self.identify_unit_type(norm_unit) == "CURRENCY":
                found_currencies.add(norm_unit)

        # LOGIC XỬ LÝ XUNG ĐỘT TIỀN TỆ (SỬA LỖI FALSE POSITIVE)
        if len(found_currencies) > 1:
            # Ngoại lệ: Cho phép song song USD và VND (Tiêu chuẩn XNK)
            # Chuẩn hóa lại tập hợp để so khớp
            normalized_set = {u for u in found_currencies if u in {"USD", "VND", "VNĐ", "đ"}}

            if len(normalized_set) == len(found_currencies):
                # Tất cả đều nằm trong whitelist USD/VND -> Chuyển sang chế độ kiểm tra tỷ giá
                violations.append({
                    "type": "MULTICURRENCY_DETECTED",
                    "detail": f"Chứng từ chứa nhiều loại tiền tệ ({', '.join(found_currencies)}). "
                              "Hệ thống chuyển sang chế độ xác thực tỷ giá quy đổi.",
                    "severity": "INFO",
                    "action": "CHECK_EXCHANGE_RATE"
                })
            else:
                # Nếu có loại tiền tệ lạ khác, vẫn báo lỗi không nhất quán
                violations.append({
                    "type": "UNIT_INCONSISTENCY",
                    "detail": f"Phát hiện xung đột tiền tệ không chuẩn: {', '.join(found_currencies)}",
                    "severity": "ERROR",
                    "action": "MANUAL_REVIEW"
                })

        return violations

    def check_unit_presence(self, data: List[Dict[str, Any]], required_category: str) -> List[Dict[str, Any]]:
        """Kiểm tra xem một danh mục đơn vị bắt buộc có xuất hiện hay không."""
        violations = []
        has_category = False

        for row in data:
            unit = row.get('unit', '')
            if self.identify_unit_type(unit) == required_category:
                has_category = True
                break

        if not has_category:
            violations.append({
                "type": "MISSING_UNIT_CATEGORY",
                "detail": f"Chứng từ thiếu đơn vị đo lường thuộc nhóm {required_category}.",
                "severity": "WARNING"
            })

        return violations
