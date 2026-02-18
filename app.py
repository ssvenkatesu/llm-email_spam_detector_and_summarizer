"""
AI-Based Email Spam & Phishing Detector
HYBRID VERSION (LLM + RULE ENGINE)
"""

import os
import json
import re
import subprocess
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
from dotenv import load_dotenv


# =============================================================================
# SPM SECTION
# =============================================================================
SPM_RISKS = ["Evolving phishing techniques require ongoing model updates."]
SPM_TASK_PIPELINE = ["dataset_update", "retraining", "deployment"]


# =============================================================================
# RULE BASED SPAM ENGINE (NEW)
# =============================================================================

SPAM_KEYWORDS = [
    "winner", "lottery", "claim now", "urgent", "verify account",
    "bank", "crypto", "investment", "limited time", "payment required",
    "click here", "free money", "work from home", "earn $",
    "suspended", "account security", "invoice pending"
]

PHISHING_PATTERNS = [
    r"http[s]?://",
    r"verify.*account",
    r"update.*details",
    r"bank.*security",
    r"send.*personal details",
    r"payment.*required"
]


def rule_based_spam_score(email_text: str):
    text = email_text.lower()
    score = 0.0
    detected = []

    for kw in SPAM_KEYWORDS:
        if kw in text:
            score += 0.08
            detected.append(kw)

    for pattern in PHISHING_PATTERNS:
        if re.search(pattern, text):
            score += 0.15
            detected.append(pattern)

    score = min(score, 1.0)
    return score, list(set(detected))


# =============================================================================
# OOAD CLASSES
# =============================================================================

class Email:

    def __init__(self, content: str):
        self.content = content
        self.spam_label = None
        self.spam_score = None
        self.keywords_detected = None
        self.summary = None
        self.reasoning = None

    def set_classification_result(self, result: dict):
        self.spam_label = result["spam_label"]
        self.spam_score = result["spam_score"]
        self.keywords_detected = result["keywords_detected"]
        self.summary = result["summary"]
        self.reasoning = result["reasoning"]

    def to_result_dict(self):
        return {
            "spam_label": self.spam_label,
            "spam_score": self.spam_score,
            "keywords_detected": self.keywords_detected or [],
            "summary": self.summary or "No summary",
            "reasoning": self.reasoning or "No reasoning",
        }


# =============================================================================
# NORMALIZER (HYBRID LOGIC)
# =============================================================================

def _normalize_result(result: dict, email_text: str = ""):

    result.setdefault("spam_label", "not_spam")
    result.setdefault("spam_score", 0.0)
    result.setdefault("keywords_detected", [])
    result.setdefault("summary", "No summary")
    result.setdefault("reasoning", "No reasoning")

    ai_score = float(result["spam_score"])

    rule_score, rule_keywords = rule_based_spam_score(email_text)

    # HYBRID FINAL SCORE
    final_score = (0.6 * ai_score) + (0.4 * rule_score)

    result["spam_score"] = max(0.0, min(1.0, final_score))
    result["keywords_detected"] = list(
        set(result["keywords_detected"] + rule_keywords)
    )

    # FINAL DECISION
    result["spam_label"] = "spam" if result["spam_score"] >= 0.45 else "not_spam"

    return result


# =============================================================================
# CLASSIFIER
# =============================================================================

class Classifier:

    def __init__(self):
        self._last_used = "Groq Cloud ✓"

    def _analyze_with_groq(self, email_text, model="llama-3.3-70b-versatile"):

        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ KEY missing")

        client = Groq(api_key=api_key)

        prompt = f"""
You are a STRICT spam detection AI.

Return JSON ONLY:

{{
"spam_label":"spam or not_spam",
"spam_score":0-1,
"keywords_detected":["risk words"],
"summary":"3 sentence summary",
"reasoning":"detailed reasoning"
}}

EMAIL:
{email_text}
"""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        output = response.choices[0].message.content.strip()

        cleaned = re.sub(r"```(json)?", "", output).replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1

        result = json.loads(cleaned[start:end])

        return _normalize_result(result, email_text)

    def _analyze_with_ollama(self, email_text, model="llama3"):

        prompt = f"""
Spam detector JSON only.

{{
"spam_label":"spam or not_spam",
"spam_score":0-1,
"keywords_detected":["risk"],
"summary":"short summary",
"reasoning":"reason"
}}

EMAIL:
{email_text}
"""

        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=60
        )

        output = result.stdout.strip()
        cleaned = re.sub(r"```(json)?", "", output).replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1

        parsed = json.loads(cleaned[start:end])

        return _normalize_result(parsed, email_text)

    def classify_email(self, email_text, local_model="llama3"):
        try:
            out = self._analyze_with_groq(email_text)
            self._last_used = "Groq Cloud ✓"
            return out
        except:
            out = self._analyze_with_ollama(email_text, local_model)
            self._last_used = f"Ollama Local ({local_model}) ✓"
            return out

    @property
    def last_used(self):
        return self._last_used


# =============================================================================
# USER
# =============================================================================

class User:

    def __init__(self, history):
        self.history = history

    def add_classification(self, result):
        self.history.append(result)

    def view_report(self):
        if not self.history:
            return None, 0, 0
        df = pd.DataFrame(self.history)
        spam_count = (df["spam_label"] == "spam").sum()
        safe_count = (df["spam_label"] == "not_spam").sum()
        return df, spam_count, safe_count


# =============================================================================
# STREAMLIT APP
# =============================================================================

if "history" not in st.session_state:
    st.session_state.history = []


def main():

    st.set_page_config(
        page_title="Hybrid Spam Detector",
        page_icon="📧",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ===============================
    # CUSTOM CSS (MODERN UI)
    # ===============================
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg,#0f172a,#111827);
        color:white;
    }

    .card {
        background: rgba(255,255,255,0.06);
        padding:20px;
        border-radius:15px;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin-bottom:15px;
    }

    .title {
        text-align:center;
        font-size:38px;
        font-weight:700;
        color:#00d87a;
    }

    .subtitle {
        text-align:center;
        color:#cbd5e1;
        margin-bottom:20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'>📧 Hybrid AI Spam & Phishing Detector</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>LLM + Rule Engine Powered Detection System</div>", unsafe_allow_html=True)

    classifier = Classifier()
    user = User(st.session_state.history)

    # ===============================
    # SIDEBAR
    # ===============================
    with st.sidebar:
        st.header("⚙️ Settings")
        local_model = st.selectbox("Offline Model", ["llama3", "mistral"])
        st.markdown("---")
        st.write("### System Info")
        st.success(classifier.last_used)

    tab1, tab2 = st.tabs(["📩 Analyze Email", "📊 Dashboard"])

    # ===============================
    # TAB 1 — ANALYSIS
    # ===============================
    with tab1:

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        email_content = st.text_area(
            "Paste Email Content",
            height=250,
            placeholder="Paste suspicious email here..."
        )

        if st.button("🚀 Analyze Email", use_container_width=True):

            if not email_content.strip():
                st.warning("Paste email first")
                st.stop()

            with st.spinner("Analyzing email..."):
                result = classifier.classify_email(email_content, local_model)

            user.add_classification(result)

            label = result["spam_label"]
            score = result["spam_score"]

            col1, col2 = st.columns(2)

            with col1:
                if label == "spam":
                    st.error(f"🚫 SPAM DETECTED — Score: {score:.2f}")
                else:
                    st.success(f"🟢 SAFE EMAIL — Score: {score:.2f}")

            with col2:
                st.metric("Spam Probability", f"{score*100:.1f}%")

            st.progress(score)

            st.subheader("📝 Summary")
            st.info(result["summary"])

            st.subheader("🧠 Reasoning")
            st.write(result["reasoning"])

            st.subheader("⚠️ Keywords Detected")
            st.warning(", ".join(result["keywords_detected"]) or "None")

        st.markdown("</div>", unsafe_allow_html=True)

    # ===============================
    # TAB 2 — DASHBOARD
    # ===============================
    with tab2:

        df, spam_count, safe_count = user.view_report()

        if df is None:
            st.info("No history yet")
            return

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Emails", len(df))
        col2.metric("Spam Emails", spam_count)
        col3.metric("Safe Emails", safe_count)

        st.subheader("📈 Spam Score Trend")
        st.line_chart(df["spam_score"])

        fig = go.Figure(go.Pie(
            labels=["Spam", "Safe"],
            values=[spam_count, safe_count],
            hole=0.55
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color="white"
        )

        st.subheader("📊 Classification Distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 History Table")
        st.dataframe(df[["spam_label", "spam_score", "summary"]])

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()