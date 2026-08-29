# Day 27 — Hệ thống Agent Human-in-the-Loop (HITL)

LangGraph workflow đánh giá **churn risk** của khách hàng ngân hàng: agent đề xuất hành động
kèm confidence score, policy quyết định route, graph **tạm dừng trước mọi hành động high-risk**
để con người Approve / Reject / Edit qua Streamlit, và mọi quyết định được ghi vào audit trail.

```
Customer Data
      |
      v
evaluate_customer  --> proposed_action + confidence_score + reasoning
      |
      v
route_action  (HARD RULE xét trước, rồi mới tới confidence)
      |
      +----------------------------+
      | Low-risk & conf >= 0.85    | High-risk / conf < 0.85
      v                            v
execute_low_risk_action      execute_high_risk_action
   (auto execute)                  X  <-- interrupt_before: graph DỪNG tại đây
                                   |
                                   v
                            Streamlit Review
                          Approve / Reject / Edit
                                   |
                    update_state() + invoke(None, config)
                                   |
                                   v
                        Execute hoặc Abort --> Audit Log
```

## 1. Cài đặt

```bash
git clone https://github.com/<YOUR_USERNAME>/Day27-HITL.git
cd Day27-HITL

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

Yêu cầu Python 3.10+ (đã test trên Python 3.12, langgraph 1.2.11, streamlit 1.40).

### Chế độ agent: mock (mặc định) hoặc LLM thật

| | Mock agent | LLM thật (OpenAI) |
|---|---|---|
| Cần API key | Không | Có |
| Tính chất | Rule-based, deterministic, chạy offline | `gpt-4o-mini` + `with_structured_output(AgentProposal)` |
| Dùng khi | Chấm bài / chạy test / demo đủ cả 3 nhánh routing | Muốn thấy agent reasoning thật |

**Mặc định là mock** vì nó deterministic và demo được đủ cả ba nhánh (auto-execute,
escalation, hard rule). Khi bật LLM, `gpt-4o-mini` thường trả confidence ~0.80 cho mọi
khách hàng nên **mọi case đều rơi vào human review** — đúng về mặt an toàn nhưng nhánh
auto-execute sẽ không xuất hiện để quan sát.

Bật LLM thật:

```bash
cp .env.example .env        # rồi điền OPENAI_API_KEY thật vào .env
```

`.env` đã nằm trong `.gitignore` nên **không bao giờ được commit**. Sau đó bật toggle
**"Dùng LLM thật (OpenAI)"** ở sidebar Streamlit, hoặc đặt `HITL_USE_LLM=1` khi chạy CLI.

Nếu không có key, gọi LLM lỗi mạng, hoặc hết quota → workflow **tự động fallback về mock
agent**, không bao giờ sập. Nguồn agent của từng lần chạy được ghi vào state và audit log
ở field `agent_source` (`llm:gpt-4o-mini` hoặc `mock`).

## 2. Chạy

### Chạy Streamlit UI (luồng chính)

```bash
streamlit run app.py
```

Thao tác trong UI:

1. Sidebar: nhập **Reviewer ID**, chọn khách hàng → bấm **▶️ Chạy agent đánh giá**.
2. Xem **Action Card**: `proposed_action`, `confidence_score`, `reasoning`, số tiền đề xuất.
3. Nếu graph bị interrupt → chọn **✅ Approve**, **❌ Reject** hoặc mở **✏️ Edit** để sửa
   action/số tiền rồi **💾 Lưu chỉnh sửa & Thực thi**.
4. Kết quả thực thi + bảng **Audit trail** hiển thị ngay bên dưới.

### Chạy workflow bằng CLI / kiểm tra toàn bộ checklist

```bash
python test_workflow.py
```

Script chạy 35 assertion phủ đủ checklist "Kiểm tra kết quả" của Lab: state, agent reasoning,
hard rule, auto-execute, escalation, interrupt, resume Approve/Reject/Edit, và audit trail.
Kết quả hiện tại: **PASSED: 35 · FAILED: 0** (audit của test ghi riêng vào `audit_log_test.json`).

## 3. Cấu trúc project

| File | Nội dung |
|---|---|
| `graph.py` | `GraphState`, `evaluate_customer` (LLM + mock), `route_action`, `execute_low_risk_action`, `execute_high_risk_action`, `MemorySaver`, `interrupt_before`, mock customer DB |
| `models.py` | `AuditEntry` + `AgentProposal` (Pydantic) + đọc/ghi audit trail append-only |
| `.env.example` | Mẫu biến môi trường (không chứa key thật) |
| `app.py` | Streamlit approval interface (Approve / Reject / Edit) |
| `test_workflow.py` | Test toàn bộ luồng HITL theo checklist của Lab |
| `audit_log.json` | Audit trail (append-only) |

### GraphState

```python
class GraphState(TypedDict, total=False):
    customer_id: str
    proposed_action: str
    confidence_score: float
    reasoning: str
    human_decision: Optional[str]
    # + field hỗ trợ: customer_data, action_amount, original_action,
    #   original_amount, raw_confidence, reviewer_id, route, executed, execution_result
```

### AuditEntry

```python
class AuditEntry(BaseModel):
    timestamp: str
    agent_id: str
    action: str
    confidence: float
    reviewer_id: str
    decision: str
    # + optional: customer_id, reasoning, original_action, action_amount,
    #   original_amount, route, executed
```

## 4. Confidence threshold & Hard policy rule

```python
CONFIDENCE_THRESHOLD = 0.85
HARD_RULE_ACTIONS    = {"increase_credit_limit", "waive_fee", "close_account"}
LOW_RISK_ACTIONS     = {"send_email", "no_action"}
```

`route_action(state)` xét theo đúng thứ tự sau — **hard rule luôn được kiểm tra TRƯỚC
confidence**, nếu không confidence 0.99 sẽ override được policy:

| # | Rule | Điều kiện | Route |
|---|---|---|---|
| 1 | **Policy Override** | `action in HARD_RULE_ACTIONS` | `execute_high_risk_action` (kể cả confidence = 0.99) |
| 2 | **Auto-Execute** | `confidence >= 0.85` và action low-risk | `execute_low_risk_action` |
| 3 | **Escalate** | `confidence < 0.85` | `execute_high_risk_action` |

Graph được compile:

```python
memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["execute_high_risk_action"],
)
```

`MemorySaver` giữ state trong lúc graph tạm dừng, nên customer data không bị mất khi chờ người
duyệt. Trong `app.py`, graph được tạo qua `@st.cache_resource` để **không bị build lại mỗi lần
Streamlit rerun** (build lại là mất luôn pending state).

### Confidence calibration (trước bước routing)

Confidence do agent tự báo được hiệu chỉnh bằng chất lượng dữ liệu đầu vào trước khi routing:

| Tín hiệu | Phạt |
|---|---|
| `income_verified = False` | −0.10 |
| `data_age_days > 90` | −0.05 |
| `tenure_months < 6` | −0.05 |

Ví dụ CUST003: agent tự báo `0.92` nhưng thu nhập chưa xác minh → calibrated `0.82` → rơi
xuống dưới threshold → bị escalate cho người duyệt thay vì auto-execute.

## 5. Dữ liệu demo

| Customer | TOI (VND) | Churn | Agent đề xuất | Confidence | Kết quả routing |
|---|---|---|---|---|---|
| CUST001 | 480,000,000 | 0.81 | `increase_credit_limit` (+48tr) | 0.96 | **Human review** (hard rule) |
| CUST002 | 90,000,000 | 0.52 | `send_email` | 0.92 | **Auto-execute** |
| CUST003 | 75,000,000 | 0.46 | `send_email` | 0.92 → **0.82** | **Human review** (escalate) |
| CUST004 | 40,000,000 | 0.88 | `increase_credit_limit` | 0.95 → 0.80 | **Human review** (hard rule) |

CUST004 là ca đáng chú ý: agent rất "tự tin" đề xuất tăng hạn mức cho khách hàng có thu nhập
**chưa xác minh** — đúng kịch bản mà hard rule sinh ra để chặn.

## 6. Approve / Reject / Edit hoạt động thế nào

Cả ba đều đi qua cùng một cơ chế: ghi quyết định vào state rồi resume graph.

```python
graph.update_state(config, {"human_decision": decision, "reviewer_id": reviewer_id})
graph.invoke(None, config)          # resume từ đúng chỗ đã interrupt
```

| Quyết định | `execute_high_risk_action` làm gì |
|---|---|
| **Approve** | Thực thi action như agent đề xuất, `executed = True` |
| **Reject** | Hủy hành động, `executed = False` |
| **Edit** | `update_state` ghi đè `proposed_action` / `action_amount` do người duyệt nhập, sau đó thực thi bản đã sửa; đề xuất gốc của agent vẫn được giữ trong `original_action` / `original_amount` để audit |
| *(thiếu quyết định)* | Fail-safe: coi như `no_decision` → **không thực thi** |

Ví dụ Edit: agent đề xuất tăng `48,000,000 VND`, reviewer sửa còn `20,000,000 VND` → audit ghi
lại cả con số gốc lẫn con số sau khi sửa.

## 7. Audit log

Lưu tại `audit_log.json` ở thư mục gốc của repo, ghi theo kiểu **append-only**
(đọc danh sách hiện có → append entry mới → ghi lại cả danh sách, không bao giờ ghi đè lịch sử).
Mọi nhánh đều ghi audit: auto-execute, approve, reject, edit và cả trường hợp abort.

```json
[
  {
    "timestamp": "2026-08-29T09:00:00",
    "agent_id": "churn-risk-agent",
    "action": "increase_credit_limit",
    "confidence": 0.96,
    "reviewer_id": "operator_01",
    "decision": "approve",
    "customer_id": "CUST001",
    "original_action": "increase_credit_limit",
    "action_amount": 20000000.0,
    "original_amount": 48000000.0,
    "route": "high_risk",
    "executed": true
  }
]
```

Trong production nên thay file JSON bằng bảng **append-only trên PostgreSQL** (chỉ cấp quyền
`INSERT`, không `UPDATE`/`DELETE`) để audit trail thực sự bất biến và kiểm toán được.

## 8. Reflection

**Câu 1 — `interrupt_before` hay `interrupt_after` khi muốn người viết lại email trước khi nó
đi tới routing node?**
Dùng `interrupt_after` trên chính node sinh email. `interrupt_before` dừng *trước khi* node
chạy, lúc đó email chưa tồn tại nên không có gì để sửa. `interrupt_after` để node generate xong,
đưa bản nháp vào state rồi mới dừng — người duyệt sửa nội dung bằng `update_state`, và bản đã
sửa mới chảy tiếp sang routing node. Nguyên tắc chung: `interrupt_before` cho hành động *phá
hủy/không hoàn tác được* (cần chặn trước khi xảy ra), `interrupt_after` cho *nội dung* cần review.

**Câu 2 — Chống Alert Fatigue khi 500 email/ngày bị kẹt ở confidence 0.82:**
- **Batch approval:** gom các action cùng loại vào một bảng, cho phép chọn nhiều dòng và
  Approve hàng loạt, thay vì 500 lần bấm từng cái.
- **Risk-tiering threshold:** mỗi loại action một ngưỡng riêng thay vì một ngưỡng 0.85 dùng
  chung — `send_email` là hành động có thể hoàn tác, threshold 0.70 là đủ; `increase_credit_limit`
  giữ hard rule.
- **Calibrate confidence** (đã làm trong repo) để 0.82 phản ánh rủi ro thật, thay vì để cả cụm
  bị kẹt sát dưới ngưỡng vì một hằng số cứng.
- **Sampling / audit ngược:** auto-execute toàn bộ nhưng bốc ngẫu nhiên 5% cho người review sau,
  và đưa ngay các case sai vào vòng hiệu chỉnh threshold.
- **Sắp xếp theo tác động:** ưu tiên hiển thị case có TOI cao / khách hàng VIP lên đầu, phần
  đuôi rủi ro thấp thì auto-approve sau timeout.
- **Undo window:** cho phép hoàn tác trong 15 phút thay vì bắt duyệt trước với hành động nhẹ.

**Câu 3 — Vì sao không nên tin confidence LLM tự chấm, và calibrate thế nào?**
Confidence của LLM là xác suất token của một chuỗi tự mô tả, không phải xác suất *đúng*. Nó
được tối ưu để nghe thuyết phục, nên thường cao đều bất kể chất lượng dữ liệu đầu vào — agent
có thể nói `0.95` khi đọc sai thu nhập khách hàng, vì nó tự tin vào *câu trả lời của chính nó*
chứ không biết dữ liệu nguồn đã sai. Nó cũng có thể bị đẩy lên bởi cách viết prompt (self-report
bias) và không hề hiệu chỉnh theo tần suất sai thực tế. Cách calibrate trước routing:
1. **Trừ điểm theo chất lượng dữ liệu** — `calibrate_confidence()` trong `graph.py`: thu nhập
   chưa xác minh, dữ liệu quá cũ, khách hàng mới đều làm giảm điểm.
2. **Đối chiếu với nguồn sự thật** — verify TOI/thu nhập từ core banking; lệch quá ngưỡng thì
   ép confidence về 0.
3. **Hiệu chỉnh theo lịch sử** — dùng chính `audit_log.json` để đo tỉ lệ đúng thật của agent ở
   từng khoảng confidence (Platt scaling / isotonic regression), rồi ánh xạ điểm tự báo sang
   xác suất đúng thực nghiệm.
4. **Tách nguồn tín hiệu** — lấy điểm từ mô hình churn/rule engine bên ngoài thay vì để LLM tự
   chấm chính mình.
5. **Và quan trọng nhất: hard rule không phụ thuộc confidence.** Với hành động tài chính, không
   con số confidence nào được phép bỏ qua người duyệt.
