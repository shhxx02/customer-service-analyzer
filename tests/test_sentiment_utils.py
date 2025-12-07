# tests/test_sentiment_utils.py

import os
import sys

# Ensure project root (folder containing sentiment_utils.py) is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sentiment_utils import (
    score_message,
    conversation_overall,
    detect_escalation,
    detect_intent,
    urgency_score,
)


def test_score_message_positive():
    result = score_message("I absolutely love this service!")
    assert result["label"] == "Positive"
    assert result["compound"] > 0.1


def test_score_message_negative():
    result = score_message("Your service disappoints me. I am very unhappy.")
    assert result["label"] == "Negative"
    assert result["compound"] < -0.1


def test_conversation_overall_and_escalation():
    scores = [0.4, -0.6, -0.8]
    avg, label = conversation_overall(scores)
    assert label == "Negative"

    labels = ["Positive", "Negative", "Negative", "Negative"]
    assert detect_escalation(labels, window=3) is True


def test_detect_intent_and_urgency():
    intent = detect_intent("I want a refund for my broken product")
    assert intent in ("refund", "billing", "technical")

    urgent = urgency_score("This is urgent, please help me ASAP!!!")
    assert urgent > 0.4
