# Customer Service Call Analyzer

A clean, interactive chatbot that performs real-time sentiment analysis, conversation insights, escalation detection, and intent/urgency recognition for customer support scenarios.  
Developed for the LiaPlus Assignment (Tier 1 + Tier 2 + enhancements).

---

## 🔗 Live Demo
Run the app here:  
https://your-app-name.streamlit.app


## Features

### Core Features (Tier 1 and Tier 2)
- Per-message sentiment analysis using VADER
- Sentence-level sentiment breakdown
- Overall conversation sentiment
- Escalation detection (multiple negative messages)
- Intent classification (billing, refund, delivery, technical, account)
- Urgency scoring
- Automatic agent responses based on sentiment

### Additional Enhancements
- Chat-style UI built in Streamlit
- Right-panel analytics dashboard
- Sentiment trend graph
- Highlighting of most negative messages
- CSV export and text report generation

---

## Quick Start

### Requirements
- Python 3.8+
- pip

### Installation

1. Clone the repository
```
git clone <your-repo-url>
cd customer-service-analyzer
```


2. Create and activate a virtual environment
```
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. Install dependencies
```
pip install -r requirements.txt
```


4. Launch the application
```
streamlit run app.py
```


Open your browser at:
http://localhost:8501

---

## Dependencies

- streamlit  
- nltk  
- pandas  
- matplotlib  
- pytest (optional)

---

## Project Structure

```
customer-service-analyzer/
│
├── app.py                 # Main Streamlit application (Enhanced UI)
├── sentiment_utils.py     # Sentiment analysis logic and utilities
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── tests/                 # Test suite (optional)
    └── test_sentiment.py
```


## Sentiment Logic Overview

### Message-Level Sentiment (Tier 2)
Each customer message is analyzed using VADER compound score:

- compound >= 0.05 → Positive  
- compound <= -0.05 → Negative  
- otherwise → Neutral  

Additional Tier 2 features:
- Sentence-level scoring
- Worst sentence extraction
- Intent detection
- Urgency scoring

### Conversation-Level Sentiment (Tier 1)
Calculated as the average sentiment of all user messages:

```
overall = sum(scores) / len(scores)
```




### Escalation Logic
Triggered when the user sends several consecutive negative messages.

---

## Example Outputs

### Example 1
User: "Your service disappoints me"  
Sentiment: Negative  

Agent: Apologizes and offers help  

User: "Last time was better"  
Sentiment: Positive  

Overall: Neutral

### Example 2
Three negative messages in a row  
Escalation detected

---

## Running Tests

```
pytest -q
```


Tests validate:
- Sentiment classification
- Escalation detection
- Intent detection
- Urgency scoring

---






