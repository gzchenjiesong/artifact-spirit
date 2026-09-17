"""AL5 可观测性：`status` / `layers` / `reflect` / `audit_view` / `doctor` / `render`。

**全部是 AL3 之上的只读投影**（INV-1）——它们能告诉你发生了什么，
但不能改变任何东西。这条纪律让"排障"永远是安全的。
"""

from __future__ import annotations

from .audit_view import AuditView, format_audit_text, format_restorable_text
from .render import format_review_text, format_trace_text
from .status import (
    audit_view,
    doctor,
    doctor_text,
    layers,
    reflect,
    status,
    status_text,
)

__all__ = [
    "AuditView",
    "audit_view",
    "doctor",
    "doctor_text",
    "format_audit_text",
    "format_restorable_text",
    "format_review_text",
    "format_trace_text",
    "layers",
    "reflect",
    "status",
    "status_text",
]
