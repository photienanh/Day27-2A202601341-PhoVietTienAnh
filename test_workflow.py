"""Bộ kiểm thử toàn diện luồng HITL theo danh mục tiêu chuẩn của bài Lab.

Thực thi kiểm tra:
    python test_workflow.py

Lưu ý: Dữ liệu audit phát sinh trong quá trình test được ghi tách biệt vào audit_log_test.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import models

# Định tuyến đường dẫn audit log sang file test độc lập trước khi load graph
models.AUDIT_LOG_PATH = Path(__file__).resolve().parent / "audit_log_test.json"
if models.AUDIT_LOG_PATH.exists():
    models.AUDIT_LOG_PATH.unlink()

from graph import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    GraphState,
    build_graph,
    calibrate_confidence,
    evaluate_customer,
    make_config,
    route_action,
    start_evaluation,
    submit_human_decision,
)

TOTAL_PASSED, TOTAL_FAILED = 0, 0


def assert_condition(label: str, is_valid: bool, extra_info: str = "") -> None:
    global TOTAL_PASSED, TOTAL_FAILED
    if is_valid:
        TOTAL_PASSED += 1
        print(f"  [PASS] {label}")
    else:
        TOTAL_FAILED += 1
        print(f"  [FAIL] {label} {extra_info}")


def print_section_header(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
print_section_header("1. State + Agent reasoning")
eval_output = evaluate_customer({"customer_id": "CUST001"})
assert_condition("agent trả về proposed_action", "proposed_action" in eval_output)
assert_condition("agent trả về confidence_score", "confidence_score" in eval_output)
assert_condition("agent trả về reasoning", bool(eval_output.get("reasoning")))
assert_condition(
    "0.0 <= confidence_score <= 1.0",
    0.0 <= eval_output["confidence_score"] <= 1.0,
    f"got {eval_output['confidence_score']}",
)
assert_condition(
    "GraphState khai báo đủ 5 field bắt buộc",
    {"customer_id", "proposed_action", "confidence_score", "reasoning", "human_decision"}
    <= set(GraphState.__annotations__),
)

# ---------------------------------------------------------------------------
print_section_header("2. Hard rule: increase_credit_limit @ 0.99 -> Human Review")
test_state = {"proposed_action": "increase_credit_limit", "confidence_score": 0.99}
assert_condition(
    "confidence 0.99 KHÔNG override được hard policy",
    route_action(test_state) == "execute_high_risk_action",
    f"got {route_action(test_state)}",
)

print_section_header("3. Auto-execute: send_email @ 0.90")
test_state = {"proposed_action": "send_email", "confidence_score": 0.90}
assert_condition(
    "route -> execute_low_risk_action",
    route_action(test_state) == "execute_low_risk_action",
)

print_section_header("4. Escalation: send_email @ 0.82")
test_state = {"proposed_action": "send_email", "confidence_score": 0.82}
assert_condition(
    "route -> execute_high_risk_action",
    route_action(test_state) == "execute_high_risk_action",
)
assert_condition("threshold đang dùng là 0.85", CONFIDENCE_THRESHOLD == 0.85)

print_section_header("5. Confidence calibration")
base_confidence = 0.92
calibrated_score = calibrate_confidence(
    base_confidence,
    {"income_verified": False, "data_age_days": 15, "tenure_months": 24},
)
assert_condition(
    "thu nhập chưa xác minh bị trừ điểm",
    calibrated_score < base_confidence,
    f"{base_confidence} -> {calibrated_score}",
)
assert_condition(
    "sau calibration rơi xuống dưới threshold",
    calibrated_score < CONFIDENCE_THRESHOLD,
    f"got {calibrated_score}",
)

# ---------------------------------------------------------------------------
print_section_header("6. Interrupt: graph dừng TRƯỚC execute_high_risk_action")
workflow_engine = build_graph()
flow_snapshot = start_evaluation(workflow_engine, "CUST001", "t-approve")
assert_condition("graph ở pending state", flow_snapshot["is_pending"])
assert_condition(
    "next node là execute_high_risk_action",
    flow_snapshot["next"] == ["execute_high_risk_action"],
)
assert_condition("action high-risk CHƯA chạy", not flow_snapshot["values"].get("executed", False))
assert_condition("state vẫn giữ customer data", flow_snapshot["values"]["customer_id"] == "CUST001")
assert_condition(
    "state giữ proposed_action",
    flow_snapshot["values"]["proposed_action"] == "increase_credit_limit",
)

print_section_header("7. Approve -> resume -> execute")
flow_snapshot = submit_human_decision(
    workflow_engine, make_config("t-approve"), "approve", "operator_01"
)
assert_condition("graph đã chạy xong", not flow_snapshot["is_pending"])
assert_condition("action được thực thi", flow_snapshot["values"]["executed"] is True)
assert_condition(
    "human_decision = approve", flow_snapshot["values"]["human_decision"] == "approve"
)
print(f"       -> {flow_snapshot['values']['execution_result']}")

print_section_header("8. Reject -> resume -> abort")
start_evaluation(workflow_engine, "CUST001", "t-reject")
flow_snapshot = submit_human_decision(
    workflow_engine, make_config("t-reject"), "reject", "operator_02"
)
assert_condition("action KHÔNG được thực thi", flow_snapshot["values"]["executed"] is False)
print(f"       -> {flow_snapshot['values']['execution_result']}")

print_section_header("9. Edit -> sửa số tiền -> execute")
before_edit_snap = start_evaluation(workflow_engine, "CUST001", "t-edit")
original_amount_val = before_edit_snap["values"]["action_amount"]
flow_snapshot = submit_human_decision(
    workflow_engine,
    make_config("t-edit"),
    "edit",
    "operator_03",
    edited_amount=20_000_000,
)
assert_condition("số tiền đã được human sửa", flow_snapshot["values"]["action_amount"] == 20_000_000)
assert_condition(
    "giữ lại đề xuất gốc của agent",
    flow_snapshot["values"]["original_amount"] == original_amount_val,
)
assert_condition("action được thực thi sau khi sửa", flow_snapshot["values"]["executed"] is True)
print(f"       -> {flow_snapshot['values']['execution_result']}")

print_section_header("10. Auto-execute path (CUST002) không bị interrupt")
flow_snapshot = start_evaluation(workflow_engine, "CUST002", "t-auto")
assert_condition("graph chạy thẳng tới END", not flow_snapshot["is_pending"])
assert_condition("route = low_risk", flow_snapshot["values"].get("route") == "low_risk")
assert_condition(
    "confidence >= threshold",
    flow_snapshot["values"]["confidence_score"] >= CONFIDENCE_THRESHOLD,
)

print_section_header("11. Escalation thật (CUST003 - thu nhập chưa xác minh)")
flow_snapshot = start_evaluation(workflow_engine, "CUST003", "t-escalate")
assert_condition("low-risk action nhưng vẫn bị chặn để review", flow_snapshot["is_pending"])
assert_condition(
    "proposed_action = send_email",
    flow_snapshot["values"]["proposed_action"] == "send_email",
)
assert_condition(
    "confidence < threshold",
    flow_snapshot["values"]["confidence_score"] < CONFIDENCE_THRESHOLD,
    f"got {flow_snapshot['values']['confidence_score']}",
)

# ---------------------------------------------------------------------------
print_section_header("12. Audit trail")
logged_audit_entries = models.load_audit_log(models.AUDIT_LOG_PATH)
recorded_decisions = [entry["decision"] for entry in logged_audit_entries]
assert_condition("có entry cho approve", "approve" in recorded_decisions)
assert_condition("có entry cho reject", "reject" in recorded_decisions)
assert_condition("có entry cho edit", "edit" in recorded_decisions)
assert_condition("có entry cho auto_execute", "auto_execute" in recorded_decisions)
assert_condition(
    "lịch sử cũ không bị ghi đè (append-only)",
    len(logged_audit_entries) >= 4,
    f"got {len(logged_audit_entries)}",
)
mandatory_fields = {"timestamp", "agent_id", "action", "confidence", "reviewer_id", "decision"}
assert_condition(
    "mỗi entry có đủ 6 field bắt buộc",
    all(mandatory_fields <= set(item) for item in logged_audit_entries),
)

print(f"\n{'=' * 52}\nPASSED: {TOTAL_PASSED}   FAILED: {TOTAL_FAILED}\n{'=' * 52}")
sys.exit(1 if TOTAL_FAILED else 0)

