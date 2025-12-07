"""
sentiment_utils.py

Logic for analysing customer messages:
- Message-level sentiment (VADER)
- Conversation-level sentiment
- Escalation detection
- Simple keyword-based intent detection
- Simple urgency scoring
- Sentence-level breakdown and "worst sentence"
- Moving average for mood trend
- Top negative messages
- Text summary report
"""

from __future__ import annotations

import random
import re
from statistics import mean
from typing import Dict, List, Tuple

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# -------------------------------------------------
# Initialise VADER sentiment analyser
# -------------------------------------------------

# Ensure VADER data is available
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

_sia = SentimentIntensityAnalyzer()


# -------------------------------------------------
# 1. Message-level sentiment (Tier 2)
# -------------------------------------------------

def score_message(text: str) -> Dict[str, float | str]:
    """
    Analyse sentiment of a single message using VADER.

    Returns a dict with:
        neg, neu, pos, compound (floats)
        label: 'Positive' | 'Neutral' | 'Negative'
    """
    if text is None:
        text = ""

    scores = _sia.polarity_scores(text)
    compound = scores.get("compound", 0.0)
    lower_text = text.lower()

    # ---- Special handling for "borderline neutral" phrases ----
    # For customer service, phrases like "not bad" or "ok" are closer to Neutral.
    for phrase in _NEUTRAL_PHRASES:
        if phrase in lower_text:
            scores["compound"] = 0.0
            scores["label"] = "Neutral"
            return scores

    # ---- Normal VADER threshold-based classification ----
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    scores["label"] = label
    return scores



# -------------------------------------------------
# 2. Conversation-level sentiment (Tier 1)
# -------------------------------------------------

def conversation_overall(compound_scores: List[float]) -> Tuple[float, str]:
    """
    Given a list of compound scores, return (average, label).
    """
    if not compound_scores:
        return 0.0, "Neutral"

    avg = sum(compound_scores) / len(compound_scores)

    if avg >= 0.05:
        label = "Positive"
    elif avg <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return avg, label


# -------------------------------------------------
# 3. Escalation detection
# -------------------------------------------------

def detect_escalation(labels: List[str], window: int = 3) -> bool:
    """
    Return True if there are at least `window` consecutive 'Negative' labels.
    """
    consecutive = 0
    for lab in labels:
        if lab == "Negative":
            consecutive += 1
            if consecutive >= window:
                return True
        else:
            consecutive = 0
    return False


# -------------------------------------------------
# 4. Intent detection (simple keyword-based)
# -------------------------------------------------

_INTENT_KEYWORDS = {
    "billing": ["bill", "billing", "charge", "charged", "invoice", "payment"],
    "refund": ["refund", "refunds", "return", "money back", "replace", "exchange"],
    "delivery": ["delivery", "delivered", "shipping", "courier", "track", "tracking", "late"],
    "technical": ["broken", "not working", "error", "bug", "crash", "slow", "issue"],
    "account": ["login", "password", "account", "profile", "signup", "sign up"],
}


def detect_intent(text: str) -> str:
    """
    Very simple keyword-based intent detection.
    Returns one of: 'billing', 'refund', 'delivery', 'technical', 'account', 'other'.
    """
    if not text:
        return "other"

    lower_text = text.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in lower_text:
                return intent
    return "other"


# -------------------------------------------------
# 5. Urgency score (rule-based 0–1)
# -------------------------------------------------

_URGENT_WORDS = [
    "urgent", "asap", "immediately", "right now", "soon", "please help", "help me",
]

_NEUTRAL_PHRASES = [
    "not bad",
    "ok",
    "okay",
    "its ok",
    "it's ok",
    "its fine",
    "it's fine",
    "fine",
    "average",
    "not negative",
]



def urgency_score(text: str) -> float:
    """
    Simple urgency estimate based on:
      - urgent words
      - exclamation marks
      - ALL CAPS words
    Returns a float in [0.0, 1.0].
    """
    if not text:
        return 0.0

    score = 0.0
    lower_text = text.lower()

    # urgent words
    for word in _URGENT_WORDS:
        if word in lower_text:
            score += 0.4

    # exclamation marks
    exclaims = text.count("!")
    score += min(0.2, 0.05 * exclaims)

    # ALL CAPS words
    words = re.findall(r"\w+", text)
    if words:
        caps_count = sum(1 for w in words if w.isupper())
        frac_caps = caps_count / len(words)
        score += min(0.4, frac_caps)

    return max(0.0, min(1.0, score))


# -------------------------------------------------
# 6. Sentence-level breakdown and worst sentence
# -------------------------------------------------

def sentence_level_scores(text: str) -> List[Dict[str, str | float]]:
    """
    Split text into sentences and return sentiment for each.
    Uses score_message() so logic is consistent.
    """
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    result: List[Dict[str, str | float]] = []

    for s in sentences:
        if not s:
            continue
        sc = score_message(s)
        result.append({
            "sentence": s,
            "compound": sc["compound"],
            "label": sc["label"],
        })

    return result



def worst_sentence(text: str) -> Dict[str, str | float]:
    """
    Return the most negative sentence in the text.
    """
    sents = sentence_level_scores(text)
    if not sents:
        return {"sentence": "", "compound": 0.0, "label": "Neutral"}
    return min(sents, key=lambda s: s["compound"])


# -------------------------------------------------
# 7. Moving average for mood trend
# -------------------------------------------------

def moving_average(values: List[float], window: int = 3) -> List[float]:
    """
    Compute a simple moving average over the list (for graph smoothing).
    """
    if not values:
        return []

    result: List[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start:i + 1]
        result.append(mean(window_vals))
    return result


# -------------------------------------------------
# 8. Top negative messages
# -------------------------------------------------

def top_negative_messages(conversation: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Return up to top_k most negative user messages from the conversation.
    """
    user_msgs = [m for m in conversation if m.get("role") == "user"]
    sorted_msgs = sorted(user_msgs, key=lambda m: m.get("score", 0.0))
    return sorted_msgs[:top_k]


# -------------------------------------------------
# 9. Adaptive replies (template-based)
# -------------------------------------------------

_NEGATIVE_REPLIES = [
    "I’m really sorry you're facing this issue. I’ll help you right away.",
    "I understand this must be frustrating. Let me look into it for you.",
    "I apologise for the inconvenience. I’ll try to resolve it quickly.",
]

_NEUTRAL_REPLIES = [
    "Thanks for the update. Could you share a few more details?",
    "Okay, I understand. Can you provide the order number?",
    "Got it. I’ll check this and get back to you.",
]

_POSITIVE_REPLIES = [
    "Happy to hear that! Let me know if you need anything else.",
    "Great! Glad it worked for you.",
    "Awesome! I’m here if you need anything further.",
]


def adaptive_reply(label: str) -> str:
    """
    Choose a reply based on the user's sentiment label.
    """
    if label == "Negative":
        return random.choice(_NEGATIVE_REPLIES)
    if label == "Positive":
        return random.choice(_POSITIVE_REPLIES)
    return random.choice(_NEUTRAL_REPLIES)


# -------------------------------------------------
# 10. Text report
# -------------------------------------------------

def generate_text_report(conversation: List[Dict]) -> str:
    """
    Generate a multi-line text summary of the conversation.
    """
    user_scores = [m["score"] for m in conversation if m.get("role") == "user"]
    avg, overall_label = conversation_overall(user_scores)

    lines: List[str] = []
    lines.append("Conversation Summary")
    lines.append("====================")
    lines.append(f"Total messages: {len(conversation)}")
    lines.append(f"User messages: {len([m for m in conversation if m.get('role') == 'user'])}")
    lines.append(f"Overall sentiment: {overall_label} (average compound = {avg:.3f})")
    lines.append("")

    worst_msgs = top_negative_messages(conversation, top_k=3)
    if worst_msgs:
        lines.append("Top negative user messages:")
        for i, m in enumerate(worst_msgs, start=1):
            lines.append(f"{i}. \"{m['text']}\" (score = {m['score']:.3f})")
    else:
        lines.append("No user messages to highlight.")

    return "\n".join(lines)
