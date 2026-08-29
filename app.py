"""Giao diện Phê duyệt Người dùng (Human Approval Interface) qua Streamlit.

Khởi chạy ứng dụng:
    streamlit run app.py
"""

from __future__ import annotations

import os
import uuid

import pandas as pd
import streamlit as st

from graph import (
    CONFIDENCE_THRESHOLD,
    CUSTOMER_DB,
    HARD_RULE_ACTIONS,
    LOW_RISK_ACTIONS,
    build_graph,
    get_snapshot,
    make_config,
    start_evaluation,
    submit_human_decision,
)
from models import AUDIT_LOG_PATH, load_audit_log

st.set_page_config(page_title="HITL Churn Risk Console", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_graph():
    """Biên dịch StateGraph một lần duy nhất và duy trì bộ nhớ qua các lần rerun."""
    return build_graph()


graph = get_graph()

# ---------------------------------------------------------------------------
# Sidebar: Cấu hình phiên đánh giá và định danh Reviewer
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Cấu hình")
    active_reviewer_id = st.text_input("Reviewer ID", value="operator_01")

    selected_customer_id = st.selectbox(
        "Khách hàng",
        list(CUSTOMER_DB),
        format_func=lambda cid: f"{cid} - {CUSTOMER_DB[cid]['name']}",
    )

    customer_profile = CUSTOMER_DB[selected_customer_id]
    st.caption(
        f"TOI: {customer_profile['total_operating_income']:,.0f} VND\n\n"
        f"Churn probability: {customer_profile['churn_probability']:.2f}\n\n"
        f"Hạn mức hiện tại: {customer_profile['current_credit_limit']:,.0f} VND\n\n"
        f"Thu nhập đã xác minh: {'có' if customer_profile['income_verified'] else 'CHƯA'}"
    )

    has_openai_env = bool(os.getenv("OPENAI_API_KEY"))
    is_llm_active = st.toggle(
        "Dùng LLM thật (OpenAI)",
        value=False,
        disabled=not has_openai_env,
        help="Bật: gọi OpenAI với structured output. Tắt: mock agent rule-based, chạy offline."
        if has_openai_env
        else "Chưa cấu hình OPENAI_API_KEY trong file .env — đang dùng mock agent.",
    )

    if st.button("▶️ Chạy agent đánh giá", type="primary", use_container_width=True):
        st.session_state.thread_id = f"{selected_customer_id}-{uuid.uuid4().hex[:8]}"
        start_evaluation(
            graph,
            selected_customer_id,
            st.session_state.thread_id,
            use_llm=is_llm_active,
        )
        st.rerun()

    st.divider()
    st.subheader("📋 Chính sách áp dụng")
    st.markdown(
        f"""
- **Confidence threshold:** `{CONFIDENCE_THRESHOLD}`
- **Hard rule (luôn cần người duyệt):** `{', '.join(sorted(HARD_RULE_ACTIONS))}`
- **Low-risk actions:** `{', '.join(sorted(LOW_RISK_ACTIONS))}`
- **Interrupt:** `interrupt_before=["execute_high_risk_action"]`
"""
    )

st.title("🛡️ Human-in-the-Loop · Churn Risk Console")

# ---------------------------------------------------------------------------
# Trích xuất trạng thái phiên làm việc từ Checkpointer
# ---------------------------------------------------------------------------
active_thread_id = st.session_state.get("thread_id")
if not active_thread_id:
    st.info("Vui lòng chọn khách hàng ở thanh bên và nhấn **▶️ Chạy agent đánh giá** để bắt đầu.")
    st.stop()

session_config = make_config(active_thread_id)
state_snapshot = get_snapshot(graph, session_config)
current_state = state_snapshot["values"]

st.caption(
    f"Thread ID: `{active_thread_id}`  ·  Next node: `{state_snapshot['next'] or 'END'}`"
    f"  ·  Agent: `{current_state.get('agent_source', '-')}`"
)

# ---------------------------------------------------------------------------
# Hiển thị Thẻ đề xuất của Agent (Action Card)
# ---------------------------------------------------------------------------
current_confidence = current_state.get("confidence_score", 0.0)
suggested_action = current_state.get("proposed_action", "-")
suggested_amount = current_state.get("action_amount")

st.subheader("🤖 Đề xuất của agent")
col_cid, col_act, col_conf, col_amt = st.columns(4)
col_cid.metric("Customer ID", current_state.get("customer_id", "-"))
col_act.metric("Proposed Action", suggested_action)
col_conf.metric(
    "Confidence",
    f"{current_confidence:.2f}",
    delta=f"{current_confidence - CONFIDENCE_THRESHOLD:+.2f} vs threshold",
)
col_amt.metric("Số tiền", f"{suggested_amount:,.0f} VND" if suggested_amount else "—")

st.progress(
    min(current_confidence, 1.0),
    text=f"confidence {current_confidence:.2f} / threshold {CONFIDENCE_THRESHOLD}",
)
st.markdown("**Reasoning**")
st.info(current_state.get("reasoning", "-"))

if suggested_action in HARD_RULE_ACTIONS:
    st.warning(
        f"⛔ **Policy override:** `{suggested_action}` nằm trong danh mục rủi ro cao (hard rule) "
        "nên LUÔN bắt buộc duyệt thủ công, kể cả khi điểm tin cậy đạt mức cao."
    )
elif state_snapshot["is_pending"]:
    st.warning(
        f"⚠️ **Escalation:** Điểm tin cậy {current_confidence:.2f} < ngưỡng {CONFIDENCE_THRESHOLD}. "
        "Hành động này bị chặn lại để chuyển cho con người xem xét."
    )

st.divider()

# ---------------------------------------------------------------------------
# Bảng điều khiển Phê duyệt / Từ chối / Chỉnh sửa (Approve / Reject / Edit)
# ---------------------------------------------------------------------------
if state_snapshot["is_pending"]:
    st.subheader("🧑‍⚖️ Human review — graph đang tạm dừng")
    st.caption(
        "Workflow đã tạm dừng TRƯỚC node `execute_high_risk_action`. Chưa có tác vụ nào được thực thi; "
        "toàn bộ trạng thái được bảo lưu an toàn trong bộ nhớ checkpoint."
    )

    btn_approve_col, btn_reject_col = st.columns(2)
    if btn_approve_col.button("✅ Approve", type="primary", use_container_width=True):
        submit_human_decision(graph, session_config, "approve", active_reviewer_id)
        st.rerun()
    if btn_reject_col.button("❌ Reject", use_container_width=True):
        submit_human_decision(graph, session_config, "reject", active_reviewer_id)
        st.rerun()

    with st.expander("✏️ Edit — sửa đề xuất trước khi thực thi"):
        chosen_action = st.selectbox(
            "Action sau khi sửa",
            sorted(HARD_RULE_ACTIONS | LOW_RISK_ACTIONS),
            index=sorted(HARD_RULE_ACTIONS | LOW_RISK_ACTIONS).index(suggested_action)
            if suggested_action in (HARD_RULE_ACTIONS | LOW_RISK_ACTIONS)
            else 0,
        )
        chosen_amount = st.number_input(
            "Số tiền sau khi sửa (VND)",
            min_value=0.0,
            value=float(suggested_amount or 0.0),
            step=1_000_000.0,
            format="%.0f",
        )
        if st.button("💾 Lưu chỉnh sửa & Thực thi", use_container_width=True):
            submit_human_decision(
                graph,
                session_config,
                "edit",
                active_reviewer_id,
                edited_action=chosen_action,
                edited_amount=chosen_amount,
            )
            st.rerun()
else:
    operator_decision = current_state.get("human_decision")
    execution_outcome = current_state.get("execution_result", "-")
    if current_state.get("executed"):
        st.success(f"**Kết quả:** {execution_outcome}")
    else:
        st.error(f"**Kết quả:** {execution_outcome}")
    st.caption(
        f"Route: `{current_state.get('route', '-')}`  ·  Human decision: "
        f"`{operator_decision or 'không cần (auto-execute)'}`"
    )

# ---------------------------------------------------------------------------
# Hiển thị Lịch sử kiểm toán (Audit Trail)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🧾 Audit trail")
st.caption(f"File: `{AUDIT_LOG_PATH.name}` (append-only)")

audit_records = load_audit_log()
if not audit_records:
    st.caption("Chưa có bản ghi audit nào.")
else:
    audit_dataframe = pd.DataFrame(audit_records)[
        [
            "timestamp",
            "customer_id",
            "agent_source",
            "action",
            "confidence",
            "reviewer_id",
            "decision",
            "executed",
        ]
    ]
    st.dataframe(audit_dataframe.iloc[::-1], use_container_width=True, hide_index=True)
    with st.expander("Xem JSON đầy đủ"):
        st.json(audit_records[-5:])

