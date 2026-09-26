"""
calc_engine.py
---------------
Module tính toán độc lập cho kiểm toán tài chính.
Đảm bảo độ chính xác tuyệt đối thông qua decimal.Decimal và logic kiểm tra chéo.
"""

from __future__ import annotations
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import config

logger = logging.getLogger("calc_engine")

class CalcEngine:
    def __init__(self, tolerance: str = '1'):
        """
        Args:
            tolerance: Sai số cho phép (mặc định là 1 đơn vị tiền tệ).
        """
        self.tolerance = Decimal(tolerance)

    def _to_decimal(self, value: Any) -> Decimal:
        """
        Chuyển đổi mọi loại dữ liệu sang Decimal, xử lý sạch chuỗi số.
        """
        if value is None:
            return Decimal('0')
        if isinstance(value, (int, float, Decimal)):
            return Decimal(str(value))

        s = str(value).strip().replace(',', '')
        try:
            return Decimal(s)
        except InvalidOperation:
            return Decimal('0')

    def validate_document_math(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        HÀM CHÍNH: Kiểm tra logic số liệu của toàn bộ chứng từ.
        Đầu vào: Dữ liệu đã cấu trúc (List of dicts hoặc DataFrame).
        """
        if isinstance(data, pd.DataFrame):
            records = data.to_dict('records')
        else:
            records = data

        errors = []
        if not records:
            return errors

        # 1. Kiểm tra từng dòng: Số lượng * Đơn giá = Thành tiền
        line_totals = []
        for idx, row in enumerate(records):
            qty = self._to_decimal(row.get('quantity', 0))
            price = self._to_decimal(row.get('unit_price', 0))
            reported_total = self._to_decimal(row.get('total_amount', 0))

            calculated_total = (qty * price).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

            if abs(calculated_total - reported_total) > self.tolerance:
                errors.append({
                    "row": idx + 1,
                    "type": "LINE_MATH_ERROR",
                    "detail": f"Dòng {idx+1}: {qty} * {price} = {calculated_total}, nhưng trong file ghi {reported_total}",
                    "expected": str(calculated_total),
                    "actual": str(reported_total),
                    "citation": f"Row {idx+1}: Total Amount"
                })
            line_totals.append(reported_total)

        # 2. Kiểm tra Tổng cộng tiền hàng (SUM of totals = Grand Total)
        grand_total_reported = self._to_decimal(records[0].get('grand_total', 0)) if records else Decimal('0')
        actual_sum_of_lines = sum(line_totals, Decimal('0'))

        if abs(actual_sum_of_lines - grand_total_reported) > self.tolerance:
            errors.append({
                "row": "Footer",
                "type": "COLUMN_SUM_ERROR",
                "detail": f"Tổng các dòng ({actual_sum_of_lines}) không khớp với Tổng cộng ghi trên chứng từ ({grand_total_reported})",
                "expected": str(actual_sum_of_lines),
                "actual": str(grand_total_reported),
                "citation": "Grand Total Field"
            })

        # 3. Kiểm tra Thuế: Tiền hàng * Thuế suất = Tiền thuế
        tax_rate = self._to_decimal(records[0].get('tax_rate', 0))
        tax_amount_reported = self._to_decimal(records[0].get('tax_amount', 0))

        calculated_tax = (actual_sum_of_lines * tax_rate).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        if abs(calculated_tax - tax_amount_reported) > self.tolerance:
            errors.append({
                "row": "Taxation",
                "type": "TAX_CALC_ERROR",
                "detail": f"Tiền hàng {actual_sum_of_lines} * Thuế suất {tax_rate} = {calculated_tax}, nhưng ghi là {tax_amount_reported}",
                "expected": str(calculated_tax),
                "actual": str(tax_amount_reported),
                "citation": "Tax Amount Field"
            })

        return errors

    def check_outliers(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Phát hiện các con số phi lý (quá thấp hoặc quá cao) dựa trên ngưỡng cấu hình.
        """
        outliers = []
        # Lấy ngưỡng từ config nếu có, nếu không mặc định một giá trị tối thiểu cực thấp
        min_threshold = getattr(config, 'MIN_REASONABLE_PRICE', Decimal('100'))

        for idx, row in enumerate(data):
            price = self._to_decimal(row.get('unit_price', 0))
            if Decimal('0') < price < min_threshold:
                outliers.append({
                    "row": idx + 1,
                    "type": "UNREASONABLE_VALUE",
                    "detail": f"Đơn giá {price} thấp hơn ngưỡng hợp lý ({min_threshold}). Nghi vấn nhập liệu sai hoặc thiếu sót."
                })
        return outliers