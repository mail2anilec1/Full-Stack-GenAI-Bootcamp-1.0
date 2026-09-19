"""Streamlit interface for the LangGraph human-in-the-loop workflow."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime
from uuid import uuid4

import streamlit as st

from HITL import HITLWorkflow

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


st.set_page_config(
    page_title="HITL Control Desk",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    :root {
        --ink: #132238;
        --muted: #607086;
        --line: #dce5ee;
        --navy: #0b1f33;
        --blue: #246bfd;
        --mint: #18a77b;
        --paper: #f5f8fb;
    }
    .stApp { background: linear-gradient(145deg, #f4f8fc 0%, #eef3f8 100%); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1f33 0%, #102c49 100%);
    }
    [data-testid="stSidebar"] * { color: #eef6ff; }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: #132238 !important; }
    .hero {
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(36, 107, 253, .15);
        border-radius: 22px;
        background: linear-gradient(120deg, #ffffff 0%, #f0f6ff 100%);
        box-shadow: 0 14px 35px rgba(20, 49, 80, .08);
        margin-bottom: 1.2rem;
    }
    .eyebrow {
        color: #246bfd;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .hero h1 { color: #132238; margin: .35rem 0 .3rem; font-size: 2.25rem; }
    .hero p { color: #607086; margin: 0; max-width: 780px; }
    .step-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: .65rem;
        margin: .5rem 0 1.25rem;
    }
    .step {
        background: rgba(255,255,255,.82);
        border: 1px solid #dce5ee;
        border-radius: 13px;
        padding: .72rem .85rem;
        color: #607086;
        font-size: .84rem;
    }
    .step strong { display: block; color: #132238; font-size: .93rem; }
    .step.active { border-color: #246bfd; box-shadow: inset 0 0 0 1px #246bfd; }
    .review-card {
        border-radius: 18px;
        border: 1px solid #d8e3ef;
        border-left: 5px solid #246bfd;
        background: white;
        padding: 1.2rem 1.3rem;
        box-shadow: 0 10px 28px rgba(20, 49, 80, .07);
        margin: .35rem 0 1rem;
    }
    .review-title { color: #132238; font-size: 1.1rem; font-weight: 800; }
    .review-copy { color: #607086; font-size: .9rem; margin-top: .2rem; }
    .status-pill {
        display: inline-block;
        padding: .25rem .6rem;
        border-radius: 999px;
        font-size: .76rem;
        font-weight: 800;
        color: #765400;
        background: #fff1bd;
        border: 1px solid #f3d66c;
    }
    div[data-testid="stTextArea"] textarea {
        background: #ffffff;
        color: #132238;
        border-radius: 14px;
        min-height: 160px;
    }
    div[data-testid="stMetric"] {
        background: #f7faff;
        border: 1px solid #dce5ee;
        padding: .8rem;
        border-radius: 13px;
    }
    .small-note { color: #718096; font-size: .78rem; }
    @media (max-width: 700px) {
        .step-row { grid-template-columns: 1fr; }
        .hero h1 { font-size: 1.7rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)


EXAMPLES = {
    "Damaged item": (
        "Customer CUST-101 received a damaged product. Please refund ₹5000."
    ),
    "Late delivery": (
        "Customer CUST-204 says an order arrived 12 days late and requests a "
        "refund of ₹2750. Review the request."
    ),
    "Policy exception": (
        "Customer CUST-309 is asking for a ₹15000 refund outside the normal "
        "return window because of a medical emergency."
    ),
}


def set_example(text: str) -> None:
    st.session_state.request_text = text


def credential_fingerprint(model_name: str, api_key: str) -> str:
    raw = f"{model_name}:{api_key or 'environment'}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_workflow(model_name: str, api_key: str) -> HITLWorkflow:
    fingerprint = credential_fingerprint(model_name, api_key)
    if st.session_state.get("workflow_fingerprint") != fingerprint:
        st.session_state.workflow = HITLWorkflow(
            model_name=model_name,
            api_key=api_key or None,
        )
        st.session_state.workflow_fingerprint = fingerprint
    return st.session_state.workflow


def initialize_state() -> None:
    defaults = {
        "request_text": EXAMPLES["Damaged item"],
        "proposal": None,
        "thread_id": None,
        "final_message": None,
        "history": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


initialize_state()

with st.sidebar:
    st.markdown("## ◈ Control Desk")
    st.caption("LangGraph · Human approval checkpoint")
    st.divider()

    model_name = st.selectbox(
        "OpenAI model",
        options=["gpt-4.1-mini", "gpt-4o-mini"],
        index=0,
        disabled=st.session_state.proposal is not None,
    )
    api_key = st.text_input(
        "OpenAI API key (optional)",
        type="password",
        help="Leave empty to use OPENAI_API_KEY from your environment or .env file.",
        disabled=st.session_state.proposal is not None,
    )
    env_ready = bool(os.getenv("OPENAI_API_KEY"))
    if api_key or env_ready:
        st.success("API credentials available")
    else:
        st.warning("Add an API key to analyze a request.")

    st.divider()
    st.markdown("**Try an example**")
    for label, example in EXAMPLES.items():
        st.button(
            label,
            key=f"example_{label}",
            on_click=set_example,
            args=(example,),
            use_container_width=True,
            disabled=st.session_state.proposal is not None,
        )

    st.divider()
    st.markdown("**Session activity**")
    st.caption(f"Completed reviews: {len(st.session_state.history)}")
    if st.button("Start a fresh request", use_container_width=True):
        st.session_state.proposal = None
        st.session_state.thread_id = None
        st.session_state.final_message = None
        st.rerun()

st.markdown(
    """
<section class="hero">
    <div class="eyebrow">Human-in-the-loop operations</div>
    <h1>Review before the agent acts.</h1>
    <p>Submit a customer request, inspect the model's structured proposal, then
    approve or reject the action from one controlled workspace.</p>
</section>
""",
    unsafe_allow_html=True,
)

current_step = 2 if st.session_state.proposal else (3 if st.session_state.final_message else 1)
st.markdown(
    f"""
<div class="step-row">
    <div class="step {'active' if current_step == 1 else ''}"><strong>01 · Analyze</strong>Submit the customer request</div>
    <div class="step {'active' if current_step == 2 else ''}"><strong>02 · Review</strong>Inspect the proposed action</div>
    <div class="step {'active' if current_step == 3 else ''}"><strong>03 · Resolve</strong>Approve or reject safely</div>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([1.12, 0.88], gap="large")

with left:
    st.subheader("Customer request")
    request_text = st.text_area(
        "Describe the support case",
        key="request_text",
        label_visibility="collapsed",
        placeholder="Example: Customer CUST-101 received a damaged item...",
        disabled=st.session_state.proposal is not None,
    )

    analyze_clicked = st.button(
        "Analyze and create proposal  →",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.proposal is not None,
    )
    st.markdown(
        '<div class="small-note">No action executes until a human reviewer approves it.</div>',
        unsafe_allow_html=True,
    )

    if analyze_clicked:
        if not request_text.strip():
            st.warning("Enter a customer request before starting the workflow.")
        elif not (api_key or env_ready):
            st.error("Provide an OpenAI API key in the sidebar or set OPENAI_API_KEY.")
        else:
            try:
                with st.spinner("The agent is preparing a structured proposal..."):
                    workflow = get_workflow(model_name, api_key)
                    thread_id = f"refund-{uuid4().hex}"
                    proposal = workflow.start_review(request_text, thread_id)
                st.session_state.thread_id = thread_id
                st.session_state.proposal = proposal
                st.session_state.final_message = None
                st.rerun()
            except Exception as exc:
                st.error(f"The proposal could not be created: {exc}")

    if st.session_state.final_message:
        st.success(st.session_state.final_message)

with right:
    st.subheader("Approval checkpoint")
    proposal = st.session_state.proposal

    if proposal is None:
        st.markdown(
            """
<div class="review-card">
    <span class="status-pill">WAITING FOR REQUEST</span>
    <div class="review-title" style="margin-top:.8rem">No proposal yet</div>
    <div class="review-copy">The structured action, customer, amount, and
    reasoning will appear here when analysis reaches the interrupt.</div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<div class="review-card">
    <span class="status-pill">PAUSED · HUMAN DECISION REQUIRED</span>
    <div class="review-title" style="margin-top:.8rem">AI proposal ready</div>
    <div class="review-copy">The LangGraph run is checkpointed and will remain
    paused until you choose an outcome.</div>
</div>
""",
            unsafe_allow_html=True,
        )

        action_col, amount_col = st.columns(2)
        action_col.metric("Proposed action", str(proposal["action"]).replace("_", " ").title())
        amount_col.metric("Amount", f"₹{float(proposal['amount']):,.2f}")
        st.markdown("**Customer**")
        st.write(proposal["customer"])
        st.markdown("**Agent reasoning**")
        st.info(proposal["reason"])

        approve_col, reject_col = st.columns(2)
        approved = approve_col.button(
            "✓ Approve action",
            type="primary",
            use_container_width=True,
        )
        rejected = reject_col.button(
            "✕ Reject action",
            use_container_width=True,
        )

        if approved or rejected:
            decision = bool(approved)
            try:
                with st.spinner("Resuming the checkpointed workflow..."):
                    workflow = get_workflow(model_name, api_key)
                    result = workflow.resume_review(
                        st.session_state.thread_id,
                        approved=decision,
                    )
                st.session_state.final_message = result["final_message"]
                st.session_state.history.insert(
                    0,
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "customer": proposal["customer"],
                        "action": proposal["action"],
                        "decision": "Approved" if decision else "Rejected",
                        "message": result["final_message"],
                    },
                )
                st.session_state.proposal = None
                st.session_state.thread_id = None
                st.rerun()
            except Exception as exc:
                st.error(f"The workflow could not be resumed: {exc}")

if st.session_state.history:
    st.divider()
    with st.expander("Session review history", expanded=False):
        for item in st.session_state.history:
            st.markdown(
                f"**{item['time']} · {item['customer']} · {item['decision']}**"
            )
            st.caption(
                f"Proposed action: {item['action'].replace('_', ' ').title()} — "
                f"{item['message']}"
            )
