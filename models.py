"""Audit schema và cơ chế lưu trữ audit trail cho hệ thống HITL.

Định nghĩa schema dữ liệu AgentProposal, AuditEntry bằng Pydantic
và các hàm tiện ích đọc/ghi log append-only vào audit_log.json.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

AUDIT_LOG_PATH = Path(__file__).resolve().parent / "audit_log.json"


class AgentProposal(BaseModel):
    """Cấu trúc đầu ra chuẩn hóa (structured output) mà agent bắt buộc phải trả về."""

    proposed_action: Literal["send_email", "increase_credit_limit", "no_action"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    action_amount: Optional[float] = Field(
        default=None,
        description="Số tiền điều chỉnh hạn mức (VND), áp dụng riêng cho increase_credit_limit",
    )


class AuditEntry(BaseModel):
    """Bản ghi audit bất biến (immutable) lưu vết từng quyết định trong hệ thống."""

    # 6 trường bắt buộc theo chuẩn nghiệp vụ
    timestamp: str
    agent_id: str
    action: str
    confidence: float
    reviewer_id: str
    decision: str

    # Các trường ngữ cảnh bổ trợ giúp truy vết chi tiết
    customer_id: Optional[str] = None
    agent_source: Optional[str] = None  # "llm:<model>" hoặc "mock"
    reasoning: Optional[str] = None
    original_action: Optional[str] = None
    action_amount: Optional[float] = None
    original_amount: Optional[float] = None
    route: Optional[str] = None
    executed: Optional[bool] = None

    @staticmethod
    def now() -> str:
        """Tạo chuỗi timestamp định dạng chuẩn ISO-8601 (độ phân giải giây)."""
        return datetime.now().isoformat(timespec="seconds")


def load_audit_log(path: Path | str | None = None) -> List[dict]:
    """Đọc toàn bộ danh sách bản ghi audit trail từ file lưu trữ."""
    target_file = Path(path or AUDIT_LOG_PATH)
    if not target_file.exists():
        return []
    try:
        raw_content = target_file.read_text(encoding="utf-8")
        parsed_entries = json.loads(raw_content)
    except (json.JSONDecodeError, OSError):
        return []
    return parsed_entries if isinstance(parsed_entries, list) else []


def append_audit_entry(entry: AuditEntry, path: Path | str | None = None) -> AuditEntry:
    """Ghi thêm một bản ghi audit mới theo cơ chế append-only (không làm mất lịch sử cũ)."""
    target_file = Path(path or AUDIT_LOG_PATH)
    history_records = load_audit_log(target_file)
    history_records.append(entry.model_dump())
    target_file.write_text(
        json.dumps(history_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return entry

