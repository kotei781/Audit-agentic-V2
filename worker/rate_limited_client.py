"""
worker/rate_limited_client.py
------------------------------
YÊU CẦU 1: Quản lý gọi API Gemini an toàn với gói FREE TIER giới hạn ở mức
thấp (ví dụ 5 RPM), gồm 3 cơ chế độc lập nhưng phối hợp với nhau:

  1. TokenBucket           — Rate Limiting: CHẶN NHỊP chủ động, không để một
                             request nào được gửi đi nếu sẽ vượt quá RPM cho
                             phép. Đây là phòng bệnh hơn chữa bệnh — khác với
                             retry (chỉ xử lý SAU KHI đã bị 429).
  2. retry_with_backoff    — Decorator retry với Exponential Backoff + jitter,
                             dùng lại được cho BẤT KỲ hàm nào, không riêng Gemini.
  3. RequestBatcher        — Hàng đợi gom nhóm nhiều "work item" (vd: từng dòng
                             chứng từ) thành MỘT request duy nhất, thay vì bắn
                             N request cho N dòng — giảm số request thực tế cần
                             tới TokenBucket cấp phép.

THIẾT KẾ "DROP-IN" — KHÔNG CẦN SỬA audit_agent.py:
    `RateLimitedGeminiClient` implement đúng interface `.models.generate_content()`
    giống hệt `google.genai.Client`, nên có thể GÁN THAY THẾ attribute `client`
    của một `AuditAgent` đã khởi tạo, từ BÊN NGOÀI file gốc:

        from audit_agent import AuditAgent
        from worker.rate_limited_client import RateLimitedGeminiClient
        import config

        agent = AuditAgent()
        agent.client = RateLimitedGeminiClient(api_key=config.GEMINI_API_KEY, rpm=5)
        # Từ giờ mọi lệnh gọi bên trong agent._call_gemini() đều tự động được
        # chặn nhịp + retry, mà audit_agent.py không hề bị đụng tới.

LƯU Ý: audit_agent.py đã có sẵn MỘT lớp retry cơ bản trong `_call_gemini()`
(exponential backoff, không có jitter, không có rate limiting chủ động). Khi
gắn RateLimitedGeminiClient theo cách trên, hệ thống sẽ có 2 lớp retry lồng
nhau (không sai, chỉ hơi dư — lớp ngoài của audit_agent hiếm khi phải kích
hoạt vì lớp trong đã xử lý gần hết). Nếu muốn gọn hơn, cần sửa `_call_gemini()`
để bỏ lớp cũ — đây là thay đổi trên file gốc nên KHÔNG tự ý làm, cần bạn
xác nhận trước.
"""

from __future__ import annotations

import functools
import logging
import random
import threading
import time
from typing import Any, Callable, List, Optional, TypeVar

logger = logging.getLogger("audit_system.worker.rate_limited_client")

T = TypeVar("T")


# ============================================================
# 1) TOKEN BUCKET — RATE LIMITING CHỦ ĐỘNG
# ============================================================
class TokenBucket:
    """
    Thuật toán Token Bucket kinh điển: bucket có sức chứa `capacity` token,
    được nạp lại liên tục với tốc độ `rate_per_minute` token/phút. Mỗi lần
    gọi `acquire()` sẽ tiêu 1 token; nếu bucket rỗng, luồng gọi sẽ BỊ CHẶN
    (ngủ) cho tới khi có đủ token — tức là tự động giãn cách các request,
    không bao giờ để vượt quá RPM cấu hình.

    Thread-safe: nhiều luồng có thể cùng gọi acquire() an toàn.
    """

    def __init__(self, rate_per_minute: float, capacity: Optional[int] = None):
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute phải > 0")

        self.rate_per_second = rate_per_minute / 60.0
        # Capacity mặc định = rate_per_minute -> cho phép "burst" tối đa bằng
        # đúng 1 phút hạn ngạch, không cho tích lũy vượt mức đó.
        self.capacity = float(capacity if capacity is not None else max(1, int(rate_per_minute)))
        self._tokens = self.capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill_locked(self) -> None:
        """Nạp lại token theo thời gian đã trôi qua. PHẢI được gọi khi đã giữ lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)
            self._last_refill = now

    def acquire(self, tokens: float = 1.0) -> float:
        """
        Chặn tới khi có đủ `tokens`. Trả về số giây đã phải chờ (0.0 nếu
        không phải chờ) — hữu ích để log/giám sát.
        """
        waited = 0.0
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited

                deficit = tokens - self._tokens
                sleep_for = deficit / self.rate_per_second

            # Ngủ NGOÀI lock để không khóa các luồng khác trong lúc chờ.
            sleep_for = min(sleep_for, 1.0)  # chờ từng đoạn ngắn, dễ dừng/giám sát
            time.sleep(sleep_for)
            waited += sleep_for


# ============================================================
# 2) RETRY VỚI EXPONENTIAL BACKOFF (DECORATOR DÙNG LẠI ĐƯỢC)
# ============================================================
_RETRIABLE_MARKERS = (
    "429", "resource_exhausted", "rate limit", "rate_limit",
    "quota", "500", "503", "unavailable", "timeout", "deadline",
)


def is_retriable_error(exc: BaseException) -> bool:
    """Nhận diện lỗi tạm thời (nên thử lại) so với lỗi vĩnh viễn (nên dừng ngay)."""
    message = str(exc).lower()
    return any(marker in message for marker in _RETRIABLE_MARKERS)


def retry_with_backoff(
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    jitter: float = 0.3,
    retriable: Callable[[BaseException], bool] = is_retriable_error,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator retry với Exponential Backoff + jitter ngẫu nhiên (chống hiệu
    ứng "thundering herd" khi nhiều tiến trình cùng retry đồng loạt).

    Công thức delay: base_delay * 2^(lần_thử - 1), cộng thêm nhiễu ngẫu nhiên
    trong khoảng [0, jitter * delay], và không bao giờ vượt `max_delay`.

    Dùng được cho BẤT KỲ hàm nào, không riêng Gemini:

        @retry_with_backoff(max_retries=5, base_delay=2.0)
        def goi_api_nao_do(...):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Optional[BaseException] = None

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - cố ý bắt rộng để phân loại lại
                    last_exc = exc
                    if not retriable(exc) or attempt == max_retries:
                        raise

                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    delay += random.uniform(0, jitter * delay)
                    logger.warning(
                        "Gọi %s thất bại (lần %d/%d): %s — thử lại sau %.1fs",
                        func.__name__, attempt, max_retries, exc, delay,
                    )
                    time.sleep(delay)

            # Không bao giờ tới được dòng này (vòng lặp luôn raise hoặc return),
            # nhưng giữ lại để công cụ kiểm tra kiểu (mypy) không báo thiếu return.
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ============================================================
# 3) HÀNG ĐỢI GOM NHÓM (BATCHING QUEUE)
# ============================================================
class RequestBatcher:
    """
    Gom nhiều "work item" (ví dụ: từng dòng chứng từ mới phát sinh) vào một
    bộ đệm, rồi xả (flush) TẤT CẢ cùng lúc thành MỘT lệnh gọi xử lý duy nhất
    — thay vì gọi API riêng cho từng item. Đây là đòn bẩy giảm số request
    MẠNH nhất: 20 dòng chứng từ gộp thành 1 request thay vì 20 request.

    Điều kiện xả tự động (cái nào tới trước):
      * Đủ `max_batch_size` item, HOẶC
      * Đã chờ đủ `max_wait_seconds` kể từ item đầu tiên trong lô hiện tại
        (tránh việc item lẻ tẻ bị "kẹt" vô thời hạn nếu không đủ số lượng).

    Việc "vẫn đảm bảo tính chính xác" khi gộp nhiều dòng vào 1 prompt đã
    được xử lý ở tầng schema của hệ thống (`schemas.AIFinding` + system prompt
    trong audit_agent.py yêu cầu "mỗi dòng dữ liệu đáng ngờ là MỘT phần tử
    riêng trong findings") — batcher này KHÔNG thay đổi gì ở tầng đó, chỉ
    kiểm soát khi nào một lô được gửi đi.
    """

    def __init__(
        self,
        process_batch: Callable[[List[Any]], None],
        max_batch_size: int = 10,
        max_wait_seconds: float = 20.0,
    ):
        self._process_batch = process_batch
        self._max_batch_size = max_batch_size
        self._max_wait_seconds = max_wait_seconds

        self._buffer: List[Any] = []
        self._first_item_at: Optional[float] = None
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._flusher_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flusher_thread.start()

    def add(self, item: Any) -> None:
        """Thêm 1 item vào hàng đợi. Tự flush ngay nếu vừa đủ max_batch_size."""
        should_flush_now = False
        with self._lock:
            self._buffer.append(item)
            if self._first_item_at is None:
                self._first_item_at = time.monotonic()
            should_flush_now = len(self._buffer) >= self._max_batch_size

        if should_flush_now:
            self.flush()

    def flush(self) -> None:
        """Xả toàn bộ item đang chờ (nếu có) thành một lệnh xử lý duy nhất."""
        with self._lock:
            if not self._buffer:
                return
            batch, self._buffer = self._buffer, []
            self._first_item_at = None

        logger.info("RequestBatcher: xả 1 lô gồm %d item.", len(batch))
        try:
            self._process_batch(batch)
        except Exception:  # noqa: BLE001
            logger.exception("Lỗi khi xử lý lô %d item — dữ liệu lô này đã bị loại khỏi hàng đợi.", len(batch))

    def _flush_loop(self) -> None:
        """Luồng nền: kiểm tra định kỳ điều kiện timeout để flush."""
        while not self._stop_event.wait(timeout=1.0):
            with self._lock:
                timed_out = (
                    self._first_item_at is not None
                    and (time.monotonic() - self._first_item_at) >= self._max_wait_seconds
                )
            if timed_out:
                self.flush()

    def stop(self) -> None:
        """Dừng luồng nền và xả nốt phần dữ liệu còn lại trước khi thoát."""
        self._stop_event.set()
        self._flusher_thread.join(timeout=5.0)
        self.flush()


# ============================================================
# CLIENT TỔNG HỢP — DROP-IN CHO google.genai.Client
# ============================================================
class _RateLimitedModels:
    """Bọc `client.models` gốc — chặn nhịp + retry trước khi gọi generate_content thật."""

    def __init__(self, real_models: Any, call_with_protection: Callable[..., Any]):
        self._real_models = real_models
        self._call_with_protection = call_with_protection

    def generate_content(self, **kwargs: Any) -> Any:
        return self._call_with_protection(self._real_models.generate_content, **kwargs)


class RateLimitedGeminiClient:
    """
    Bọc `google.genai.Client` với đầy đủ Rate Limiting (Token Bucket) + Retry
    (Exponential Backoff). Giữ NGUYÊN interface `.models.generate_content(...)`
    nên dùng thay thế trực tiếp cho `genai.Client` ở bất kỳ đâu, kể cả gán đè
    vào một AuditAgent đã tồn tại (xem docstring đầu file).

    Args:
        api_key: Gemini API key.
        rpm: giới hạn request/phút (mặc định 5 — khớp gói free tier).
        max_retries, base_delay, max_delay: cấu hình backoff, xem retry_with_backoff.
    """

    def __init__(
        self,
        api_key: str,
        rpm: float = 5,
        max_retries: int = 5,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
    ):
        from google import genai  # import trễ — không bắt buộc cài SDK để dùng TokenBucket/RequestBatcher riêng lẻ

        self._real_client = genai.Client(api_key=api_key)
        self._bucket = TokenBucket(rate_per_minute=rpm)
        self.rpm = rpm

        @retry_with_backoff(max_retries=max_retries, base_delay=base_delay, max_delay=max_delay)
        def _call_with_protection(fn: Callable[..., Any], **kwargs: Any) -> Any:
            waited = self._bucket.acquire()
            if waited > 0.01:
                logger.info("Rate limiter: đã chờ %.1fs để không vượt %d RPM.", waited, rpm)
            return fn(**kwargs)

        self.models = _RateLimitedModels(self._real_client.models, _call_with_protection)
