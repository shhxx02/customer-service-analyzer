"""
app.py

Enhanced Customer Service Call Analyzer with Modern UI.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4
import html

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sentiment_utils import (
    score_message,
    conversation_overall,
    detect_escalation,
    detect_intent,
    urgency_score,
    sentence_level_scores,
    worst_sentence,
    moving_average,
    top_negative_messages,
    adaptive_reply,
    generate_text_report,
)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Service Analyzer",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .sentiment-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .sentiment-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .sentiment-neutral {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session State
# -----------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True


def make_message(role: str, text: str, score: float, label: str) -> dict:
    """Create a standardized message dictionary."""
    msg = {
        "id": str(uuid4())[:8],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "role": role,
        "text": text,
        "score": float(score),
        "label": label,
    }
    
    if role == "user":
        msg["intent"] = detect_intent(text)
        msg["urgency"] = round(urgency_score(text), 2)
        msg["worst_sentence"] = worst_sentence(text)["sentence"]
    else:
        msg["intent"] = "agent"
        msg["urgency"] = 0.0
        msg["worst_sentence"] = ""
    
    return msg


def get_sentiment_badge(label: str) -> str:
    """Return HTML for sentiment badge."""
    if label == "Positive":
        return '<span class="sentiment-positive">😊 Positive</span>'
    elif label == "Negative":
        return '<span class="sentiment-negative">😟 Negative</span>'
    else:
        return '<span class="sentiment-neutral">😐 Neutral</span>'


# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="header-container">
    <h1 style="margin: 0; font-size: 2.5rem;">📞 Customer Service Call Analyzer</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
        Real-time sentiment analysis and conversation insights
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    
    escalation_window = st.slider(
        "Escalation Threshold",
        min_value=2,
        max_value=6,
        value=3,
        help="Number of consecutive negative messages to trigger escalation"
    )
    
    st.markdown("---")
    
    st.header("📊 Quick Stats")
    total_msgs = len(st.session_state.conversation)
    user_msgs = len([m for m in st.session_state.conversation if m["role"] == "user"])
    agent_msgs = total_msgs - user_msgs
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("User Messages", user_msgs)
    with col2:
        st.metric("Agent Responses", agent_msgs)
    
    if user_msgs > 0:
        user_scores = [m["score"] for m in st.session_state.conversation if m["role"] == "user"]
        avg_score, overall_label = conversation_overall(user_scores)
        
        st.metric(
            "Overall Sentiment",
            overall_label,
            delta=f"{avg_score:.3f}",
            delta_color="normal" if avg_score >= 0 else "inverse"
        )
    
    st.markdown("---")
    
    # Escalation Alert
    user_labels = [m["label"] for m in st.session_state.conversation if m["role"] == "user"]
    if detect_escalation(user_labels, window=escalation_window):
        st.error(f"⚠️ **Escalation Detected!**\n\nCustomer has been negative for {escalation_window}+ consecutive messages.")
    else:
        st.success("✅ No escalation detected")
    
    st.markdown("---")
    
    if st.button("🔄 Reset Conversation", use_container_width=True):
        st.session_state.conversation = []
        st.session_state.show_welcome = True
        st.rerun()

# -----------------------------
# Main Layout
# -----------------------------
chat_col, analysis_col = st.columns([3, 2])

# =============================
# CHAT SECTION
# =============================
with chat_col:
    st.subheader("💬 Conversation")
    
    if st.session_state.show_welcome and len(st.session_state.conversation) == 0:
        st.info("👋 Welcome! Start chatting to see real-time sentiment analysis.")
    
    if st.session_state.conversation:
        for msg in st.session_state.conversation:
            with st.chat_message(msg["role"]):
                st.write(msg["text"])
                
                if msg["role"] == "user":
                    col1, col2, col3 = st.columns([2, 2, 4])
                    with col1:
                        st.markdown(get_sentiment_badge(msg["label"]), unsafe_allow_html=True)
                    with col2:
                        if msg["urgency"] > 0.5:
                            st.markdown(f"🔥 Urgency: {msg['urgency']:.2f}")
                    with col3:
                        if msg["intent"] != "other":
                            st.markdown(f"🏷️ Intent: {msg['intent'].title()}")

# Chat input
user_text = st.chat_input("Type your message here...")

if user_text:
    st.session_state.show_welcome = False
    
    user_scores = score_message(user_text)
    user_msg = make_message(
        role="user",
        text=user_text,
        score=user_scores["compound"],
        label=user_scores["label"],
    )
    st.session_state.conversation.append(user_msg)
    
    agent_text = adaptive_reply(user_scores["label"])
    agent_scores = score_message(agent_text)
    agent_msg = make_message(
        role="agent",
        text=agent_text,
        score=agent_scores["compound"],
        label=agent_scores["label"],
    )
    st.session_state.conversation.append(agent_msg)
    
    st.rerun()

# =============================
# ANALYSIS SECTION
# =============================
with analysis_col:
    st.subheader("🔍 Detailed Analysis")
    
    with st.expander("📝 Sentence-Level Breakdown", expanded=False):
        user_messages = [m for m in st.session_state.conversation if m["role"] == "user"]
        
        if not user_messages:
            st.info("No user messages to analyze yet.")
        else:
            for idx, m in enumerate(user_messages, start=1):
                st.markdown(f"**Message {idx}:** {m['text']}")
                
                sentences = sentence_level_scores(m["text"])
                if sentences:
                    for s in sentences:
                        sentiment_color = {
                            "Positive": "green",
                            "Negative": "red",
                            "Neutral": "gray"
                        }[s['label']]
                        
                        st.markdown(
                            f"- {s['sentence']} "
                            f":{sentiment_color}[{s['label']} ({s['compound']:.3f})]"
                        )
                else:
                    st.caption("_No sentence breakdown available_")
                
                st.markdown("---")
    
    with st.expander("⚠️ Critical Messages", expanded=True):
        worst_msgs = top_negative_messages(st.session_state.conversation, top_k=3)
        
        if worst_msgs:
            for i, m in enumerate(worst_msgs, start=1):
                st.markdown(
                    f"**{i}.** {m['text']}\n\n"
                    f"- Score: {m['score']:.3f}\n"
                    f"- Intent: {m.get('intent', 'N/A').title()}\n"
                    f"- Urgency: {m.get('urgency', 0):.2f}"
                )
                st.markdown("---")
        else:
            st.info("No negative messages detected.")

# =============================
# ANALYTICS DASHBOARD
# =============================
st.markdown("---")
st.header("📈 Conversation Analytics")

user_scores = [m["score"] for m in st.session_state.conversation if m["role"] == "user"]

if user_scores:
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        avg_score, overall_label = conversation_overall(user_scores)
        st.metric("Overall Sentiment", overall_label, f"{avg_score:.3f}")
    
    with metric_col2:
        positive_count = sum(1 for m in st.session_state.conversation 
                           if m["role"] == "user" and m["label"] == "Positive")
        st.metric("Positive Messages", positive_count)
    
    with metric_col3:
        negative_count = sum(1 for m in st.session_state.conversation 
                           if m["role"] == "user" and m["label"] == "Negative")
        st.metric("Negative Messages", negative_count)
    
    with metric_col4:
        avg_urgency = sum(m.get("urgency", 0) for m in st.session_state.conversation 
                         if m["role"] == "user") / len(user_scores)
        st.metric("Avg Urgency", f"{avg_urgency:.2f}")
    
    st.subheader("📊 Sentiment Trend")
    smoothed = moving_average(user_scores, window=3)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(smoothed) + 1), smoothed, 
            marker="o", linewidth=2, markersize=8, 
            color="#667eea", label="Smoothed Sentiment")
    ax.fill_between(range(1, len(smoothed) + 1), smoothed, 0, alpha=0.3, color="#667eea")
    
    ax.set_xlabel("User Message Number", fontsize=11, fontweight="bold")
    ax.set_ylabel("Compound Score", fontsize=11, fontweight="bold")
    ax.set_ylim(-1.0, 1.0)
    ax.axhline(0.05, linestyle="--", alpha=0.4, color="green", label="Positive Threshold")
    ax.axhline(-0.05, linestyle="--", alpha=0.4, color="red", label="Negative Threshold")
    ax.axhline(0, linestyle="-", alpha=0.2, color="gray")
    ax.set_title("Conversation Sentiment Over Time", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.2)
    
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("💾 Export Options")
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if st.button("📄 Export as CSV", use_container_width=True):
            df = pd.DataFrame(st.session_state.conversation)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV File",
                data=csv_bytes,
                file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with export_col2:
        if st.button("📋 Generate Report", use_container_width=True):
            report = generate_text_report(st.session_state.conversation)
            st.download_button(
                "⬇️ Download Report",
                data=report,
                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            
            with st.expander("Preview Report"):
                st.text(report)
else:
    st.info("💡 Start a conversation to see analytics and insights appear here.")

st.markdown("---")
st.caption("Built with Streamlit • Powered by VADER Sentiment Analysis")