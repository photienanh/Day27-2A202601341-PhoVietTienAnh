"""Workflow LangGraph đánh giá rủi ro rời bỏ khách hàng (Churn Risk) với Human-in-the-Loop.

Sơ đồ vận hành:
    customer data -> evaluate_customer -> route_action (ưu tiên hard rule trước confidence)
        -> execute_low_risk_action  (tự động thực thi an toàn)
        -> execute_high_risk_action (TẠM DỪNG / INTERRUPT -> chờ con người duyệt -> tiếp tục)

Hệ thống sử dụng MemorySaver() làm checkpointer để duy trì state xuyên suốt quá trình review.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from models import AgentProposal, AuditEntry, append_audit_entry

# Tải cấu hình biến môi trường từ .env
load_dotenv()

# ---------------------------------------------------------------------------
# Cấu hình chính sách và tham số hệ thống
# ---------------------------------------------------------------------------

AGENT_ID = "churn-risk-agent"

#: Ngưỡng độ tin cậy tối thiểu cho phép hệ thống tự động thực thi
CONFIDENCE_THRESHOLD = 0.85

#: Danh mục hành động rủi ro cao: BẮT BUỘC duyệt thủ công, bỏ qua điểm confidence
HARD_RULE_ACTIONS = {"increase_credit_limit", "waive_fee", "close_account"}

#: Danh mục hành động rủi ro thấp: có thể tự động thực thi nếu confidence đạt chuẩn
LOW_RISK_ACTIONS = {"send_email", "no_action"}

#: Mô hình LLM mặc định (tự động fallback sang mock nếu thiếu API key)
LLM_MODEL = os.getenv("HITL_LLM_MODEL", "gpt-4o-mini")


# ---------------------------------------------------------------------------
# Bước 1 - GraphState: Trạng thái xuyên suốt toàn bộ workflow
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    # 5 trường cốt lõi theo đặc tả hệ thống
    customer_id: str
    proposed_action: str
    confidence_score: float
    reasoning: str
    human_decision: Optional[str]

    # Các trường mở rộng phục vụ thực thi, kiểm toán và hỗ trợ chỉnh sửa
    customer_data: Dict[str, Any]
    use_llm: bool  # True: gọi OpenAI; False: dùng rule-based mock agent
    agent_source: str  # "llm:<model>" hoặc "mock"
    action_amount: Optional[float]
    original_action: Optional[str]
    original_amount: Optional[float]
    raw_confidence: float
    reviewer_id: Optional[str]
    route: str
    executed: bool
    execution_result: str


# ---------------------------------------------------------------------------
# Dữ liệu khách hàng mẫu (Mock Database)
# ---------------------------------------------------------------------------

CUSTOMER_DB: Dict[str, Dict[str, Any]] = {
    "CUST001": {
        "name": "Nguyen Van A",
        "total_operating_income": 480_000_000,  # Tổng thu nhập 12 tháng (VND)
        "churn_probability": 0.81,
        "tenure_months": 54,
        "current_credit_limit": 200_000_000,
        "income_verified": True,
        "data_age_days": 12,
    },
    "CUST002": {
        "name": "Tran Thi B",
        "total_operating_income": 90_000_000,
        "churn_probability": 0.52,
        "tenure_months": 18,
        "current_credit_limit": 50_000_000,
        "income_verified": True,
        "data_age_days": 20,
    },
    "CUST003": {
        "name": "Le Van C",
        "total_operating_income": 75_000_000,
        "churn_probability": 0.46,
        "tenure_months": 9,
        "current_credit_limit": 40_000_000,
        "income_verified": False,  # Chưa xác minh -> bị phạt điểm confidence
        "data_age_days": 15,
    },
    "CUST004": {
        "name": "Pham Thi D",
        "total_operating_income": 40_000_000,
        "churn_probability": 0.88,
        "tenure_months": 30,
        "current_credit_limit": 30_000_000,
        "income_verified": False,  # Chưa xác minh dù xác suất rời bỏ cao
        "data_age_days": 120,
    },
}

HIGH_VALUE_TOI = 100_000_000


# ---------------------------------------------------------------------------
# Bước 2 - Đánh giá và phân tích đề xuất của Agent
# ---------------------------------------------------------------------------


def calibrate_confidence(raw_confidence: float, customer: Dict[str, Any]) -> float:
    """Hiệu chỉnh điểm tin cậy dựa trên các tiêu chí chất lượng dữ liệu thực tế."""
    adjusted_score = raw_confidence
    if not customer.get("income_verified", False):
        adjusted_score -= 0.10
    if customer.get("data_age_days", 0) > 90:
        adjusted_score -= 0.05
    if customer.get("tenure_months", 0) < 6:
        adjusted_score -= 0.05
    return round(max(0.0, min(1.0, adjusted_score)), 2)


SYSTEM_PROMPT = """Bạn là churn-risk agent của một ngân hàng bán lẻ.
Đọc hồ sơ khách hàng và đề xuất ĐÚNG MỘT hành động giữ chân khách:
- "send_email": gửi email chăm sóc / ưu đãi phí (low-risk, có thể hoàn tác)
- "increase_credit_limit": tăng hạn mức tín dụng (high-risk, ảnh hưởng tài chính)
- "no_action": không cần làm gì

Quy tắc:
- confidence_score trong [0.0, 1.0], phản ánh mức chắc chắn THỰC SỰ. Nếu thu nhập chưa
  được xác minh hoặc dữ liệu đã cũ thì phải hạ điểm xuống, đừng tự tin quá mức.
- Chỉ điền action_amount khi đề xuất increase_credit_limit (đơn vị VND).
- reasoning viết bằng tiếng Việt, ngắn gọn, nêu rõ căn cứ từ số liệu.
- Bạn KHÔNG có quyền phê duyệt. Đề xuất của bạn sẽ đi qua policy engine và có thể bị
  bắt buộc chuyển cho con người duyệt, bất kể bạn tự tin đến đâu."""


def _llm_propose(customer_id: str, customer: Dict[str, Any]) -> Optional[AgentProposal]:
    """Tạo đề xuất thông qua mô hình LLM với structured output."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from langchain_openai import ChatOpenAI

        model_client = ChatOpenAI(model=LLM_MODEL, temperature=0).with_structured_output(
            AgentProposal
        )
        profile_text = (
            f"customer_id: {customer_id}\n"
            f"Tổng thu nhập hoạt động 12 tháng (TOI): {customer['total_operating_income']:,.0f} VND\n"
            f"Xác suất rời bỏ (churn probability): {customer['churn_probability']:.2f}\n"
            f"Thời gian gắn bó: {customer['tenure_months']} tháng\n"
            f"Hạn mức tín dụng hiện tại: {customer['current_credit_limit']:,.0f} VND\n"
            f"Thu nhập đã được xác minh: {'có' if customer['income_verified'] else 'CHƯA'}\n"
            f"Dữ liệu cập nhật cách đây: {customer['data_age_days']} ngày"
        )
        return model_client.invoke(
            [("system", SYSTEM_PROMPT), ("human", f"Hồ sơ khách hàng:\n{profile_text}")]
        )
    except Exception as err:  # noqa: BLE001
        print(f"[agent] LLM lỗi ({err.__class__.__name__}: {err}) -> fallback mock agent")
        return None


def _mock_propose(customer: Dict[str, Any]) -> AgentProposal:
    """Tạo đề xuất theo luật xác định (deterministic mock) để chạy offline."""
    churn_rate = customer["churn_probability"]
    income_toi = customer["total_operating_income"]

    if churn_rate >= 0.70 and income_toi >= HIGH_VALUE_TOI:
        suggested_action = "increase_credit_limit"
        target_amount = float(round(income_toi * 0.10, -6))
        base_confidence = 0.96
        explanation = (
            f"Churn probability {churn_rate:.2f} rất cao trên khách hàng high-value "
            f"(TOI {income_toi:,.0f} VND, tenure {customer['tenure_months']} tháng). "
            f"Đề xuất tăng hạn mức thêm {target_amount:,.0f} VND để giữ chân khách."
        )
    elif churn_rate >= 0.70:
        suggested_action = "increase_credit_limit"
        target_amount = float(round(income_toi * 0.10, -6))
        base_confidence = 0.95
        explanation = (
            f"Churn probability {churn_rate:.2f} cao nhưng TOI chỉ {income_toi:,.0f} VND. "
            "Agent vẫn đề xuất tăng hạn mức - action high-risk, bắt buộc người duyệt."
        )
    elif churn_rate >= 0.40:
        suggested_action = "send_email"
        target_amount = None
        base_confidence = 0.92
        explanation = (
            f"Churn probability {churn_rate:.2f} ở mức trung bình, chưa cần hành động "
            "tài chính rủi ro. Đề xuất gửi email retention kèm ưu đãi phí."
        )
    else:
        suggested_action = "send_email"
        target_amount = None
        base_confidence = 0.90
        explanation = f"Churn probability {churn_rate:.2f} thấp. Chỉ cần email chăm sóc định kỳ."

    return AgentProposal(
        proposed_action=suggested_action,
        confidence_score=base_confidence,
        reasoning=explanation,
        action_amount=target_amount,
    )


def evaluate_customer(state: GraphState) -> Dict[str, Any]:
    """Node thẩm định: phân tích thông tin khách hàng và đưa ra đề xuất ban đầu."""
    cid = state["customer_id"]
    customer_info = CUSTOMER_DB.get(cid, CUSTOMER_DB["CUST002"])

    is_llm_mode = state.get("use_llm", os.getenv("HITL_USE_LLM") == "1")
    proposal = _llm_propose(cid, customer_info) if is_llm_mode else None
    agent_origin = f"llm:{LLM_MODEL}" if proposal else "mock"
    if proposal is None:
        proposal = _mock_propose(customer_info)

    initial_confidence = proposal.confidence_score
    reasoning_text = proposal.reasoning

    final_confidence = calibrate_confidence(initial_confidence, customer_info)
    if final_confidence != initial_confidence:
        reasoning_text += (
            f" [Calibration: raw {initial_confidence:.2f} -> {final_confidence:.2f} "
            "do thu nhập chưa xác minh / dữ liệu cũ / khách hàng mới.]"
        )

    return {
        "customer_data": customer_info,
        "agent_source": agent_origin,
        "proposed_action": proposal.proposed_action,
        "action_amount": proposal.action_amount,
        "original_action": proposal.proposed_action,
        "original_amount": proposal.action_amount,
        "raw_confidence": initial_confidence,
        "confidence_score": final_confidence,
        "reasoning": reasoning_text,
        "human_decision": None,
        "executed": False,
    }


# ---------------------------------------------------------------------------
# Bước 3 - Định tuyến quyết định (Conditional Routing)
# ---------------------------------------------------------------------------


def route_action(state: GraphState) -> str:
    """Điều hướng luồng xử lý: tự động thực thi hay chuyển cho con người xem xét."""
    action_type = state["proposed_action"]
    score = state["confidence_score"]

    # Ưu tiên 1: Hard policy rule luôn bắt buộc human review
    if action_type in HARD_RULE_ACTIONS:
        return "execute_high_risk_action"

    # Ưu tiên 2: Tự động thực thi cho hành động rủi ro thấp có độ tin cậy cao
    if score >= CONFIDENCE_THRESHOLD and action_type in LOW_RISK_ACTIONS:
        return "execute_low_risk_action"

    # Ưu tiên 3: Leo thang (escalate) khi độ tin cậy chưa đạt chuẩn
    return "execute_high_risk_action"


# ---------------------------------------------------------------------------
# Bước 4 - Các Node thực thi và ghi nhận Audit Trail
# ---------------------------------------------------------------------------


def _log(
    state: GraphState, *, reviewer_id: str, decision: str, executed: bool, route: str
) -> AuditEntry:
    record = AuditEntry(
        timestamp=AuditEntry.now(),
        agent_id=AGENT_ID,
        action=state.get("proposed_action", ""),
        confidence=state.get("confidence_score", 0.0),
        reviewer_id=reviewer_id,
        decision=decision,
        customer_id=state.get("customer_id"),
        agent_source=state.get("agent_source"),
        reasoning=state.get("reasoning"),
        original_action=state.get("original_action"),
        action_amount=state.get("action_amount"),
        original_amount=state.get("original_amount"),
        route=route,
        executed=executed,
    )
    return append_audit_entry(record)


def execute_low_risk_action(state: GraphState) -> Dict[str, Any]:
    """Tự động thực thi hành động rủi ro thấp và lưu audit trail."""
    action_name = state["proposed_action"]
    message = f"AUTO-EXECUTED: {action_name} cho khách hàng {state['customer_id']}."
    _log(state, reviewer_id="system-auto", decision="auto_execute", executed=True, route="low_risk")
    return {"executed": True, "execution_result": message, "route": "low_risk"}


def execute_high_risk_action(state: GraphState) -> Dict[str, Any]:
    """Thực thi sau khi workflow được tiếp tục (resume) với quyết định từ con người."""
    human_choice = (state.get("human_decision") or "").strip().lower()
    reviewer_name = state.get("reviewer_id") or "unknown_operator"
    action_name = state.get("proposed_action", "")
    amt = state.get("action_amount")
    amt_text = f" ({amt:,.0f} VND)" if amt else ""
    orig_amt = state.get("original_amount")
    orig_amt_text = f" {orig_amt:,.0f} VND" if orig_amt else ""

    if human_choice == "approve":
        was_executed = True
        summary = f"APPROVED & EXECUTED: {action_name}{amt_text} cho {state['customer_id']}."
    elif human_choice == "edit":
        was_executed = True
        summary = (
            f"EDITED & EXECUTED: {action_name}{amt_text} cho {state['customer_id']} "
            f"(agent đề xuất ban đầu: {state.get('original_action')}{orig_amt_text})."
        )
    elif human_choice == "reject":
        was_executed = False
        summary = f"REJECTED: không thực hiện {action_name} cho {state['customer_id']}."
    else:
        human_choice = "no_decision"
        was_executed = False
        summary = "ABORTED: không có human decision, hành động high-risk bị hủy."

    _log(
        state,
        reviewer_id=reviewer_name,
        decision=human_choice,
        executed=was_executed,
        route="high_risk",
    )
    return {"executed": was_executed, "execution_result": summary, "route": "high_risk"}


# ---------------------------------------------------------------------------
# Bước 5 - Xây dựng và biên dịch StateGraph
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None):
    """Khởi tạo và biên dịch StateGraph với cơ chế ngắt quãng (interrupt_before)."""
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("evaluate_customer", evaluate_customer)
    graph_builder.add_node("execute_low_risk_action", execute_low_risk_action)
    graph_builder.add_node("execute_high_risk_action", execute_high_risk_action)

    graph_builder.add_edge(START, "evaluate_customer")
    graph_builder.add_conditional_edges(
        "evaluate_customer",
        route_action,
        {
            "execute_low_risk_action": "execute_low_risk_action",
            "execute_high_risk_action": "execute_high_risk_action",
        },
    )
    graph_builder.add_edge("execute_low_risk_action", END)
    graph_builder.add_edge("execute_high_risk_action", END)

    saver = checkpointer or MemorySaver()
    return graph_builder.compile(
        checkpointer=saver,
        interrupt_before=["execute_high_risk_action"],
    )


# ---------------------------------------------------------------------------
# Các hàm điều khiển Workflow dùng chung (Helper API)
# ---------------------------------------------------------------------------


def make_config(thread_id: str) -> Dict[str, Any]:
    """Tạo cấu hình định danh phiên làm việc (thread_id)."""
    return {"configurable": {"thread_id": thread_id}}


def get_snapshot(graph, config: Dict[str, Any]) -> Dict[str, Any]:
    """Trích xuất trạng thái hiện thời của graph và cờ chờ phê duyệt."""
    state_snap = graph.get_state(config)
    return {
        "values": dict(state_snap.values),
        "next": list(state_snap.next),
        "is_pending": bool(state_snap.next),
    }


def start_evaluation(
    graph, customer_id: str, thread_id: str, use_llm: bool = False
) -> Dict[str, Any]:
    """Kích hoạt tiến trình đánh giá khách hàng."""
    session_config = make_config(thread_id)
    graph.invoke({"customer_id": customer_id, "use_llm": use_llm}, session_config)
    return get_snapshot(graph, session_config)


def submit_human_decision(
    graph,
    config: Dict[str, Any],
    decision: str,
    reviewer_id: str,
    edited_action: Optional[str] = None,
    edited_amount: Optional[float] = None,
) -> Dict[str, Any]:
    """Ghi nhận quyết định của người duyệt vào state và tiếp tục luồng thực thi."""
    state_patch: Dict[str, Any] = {"human_decision": decision, "reviewer_id": reviewer_id}
    if decision == "edit":
        if edited_action:
            state_patch["proposed_action"] = edited_action
        if edited_amount is not None:
            state_patch["action_amount"] = float(edited_amount)

    graph.update_state(config, state_patch)
    graph.invoke(None, config)
    return get_snapshot(graph, config)
