import os
import json
import re
import subprocess
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from groq import Groq
from dotenv import load_dotenv

# =============== GLOBAL STORAGE ================
if "history" not in st.session_state:
    st.session_state.history = []


# =========================================================
#  🟢 PRIMARY ENGINE — GROQ API (FAST + FREE + POWERFUL)
# =========================================================
def analyze_with_groq(email_text, model="llama-3.3-70b-versatile"):
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("❌ GROQ_API_KEY missing in .env")

    client = Groq(api_key=api_key)

    prompt = f"""
    You are a STRICT email spam + phishing detection AI. Be AGGRESSIVE.

    If not 100% clean → classify as spam.

    Return only JSON:

    {{
      "spam_label": "spam" or "not_spam",
      "spam_score": float (0-1),
      "keywords_detected": ["risk terms"],
      "summary": "3 sentence summary",
      "reasoning": "detailed justification"
    }}

    Email to analyze:
    {email_text}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":prompt}]
    )

    output = response.choices[0].message.content.strip()

    # ---------------------------------------
    # 🛠 FIXED JSON EXTRACTOR (Rock Solid)
    # ---------------------------------------
    cleaned = re.sub(r"```(json)?", "", output).replace("```", "").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1

    result = None

    # Try extracting {...}
    if start != -1 and end != -1:
        try:
            result = json.loads(cleaned[start:end])
        except:
            pass

    # Full parse fallback
    if result is None:
        try:
            result = json.loads(cleaned)
        except:
            raise RuntimeError("❌ GPT returned invalid JSON:\n" + output[:400])

    # Normalize output
    result.setdefault("spam_label","not_spam")
    result.setdefault("spam_score",0.0)
    result.setdefault("keywords_detected",[])
    result.setdefault("summary","No summary")
    result.setdefault("reasoning","No reasoning")

    result["spam_score"] = float(result["spam_score"])

    if result["spam_score"] >= 0.50:
        result["spam_label"]="spam"
    else:
        result["spam_label"]="not_spam"

    return result


# =========================================================
#  🔥 OFFLINE FALLBACK — OLLAMA
# =========================================================
def analyze_with_ollama(email_text, model="llama3"):

    prompt = f"""
    Spam detector JSON only.

    {{
      "spam_label":"spam" or "not_spam",
      "spam_score":0-1,
      "keywords_detected":["risk"],
      "summary":"short summary",
      "reasoning":"explain classification"
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

    if result.returncode != 0:
        raise RuntimeError(f"Ollama error: {result.stderr.strip()}")

    output = result.stdout.strip()
    cleaned = re.sub(r"```(json)?", "", output).replace("```", "").strip()

    start = cleaned.find("{")
    end   = cleaned.rfind("}")+1

    if start == -1 or end == 0:
        raise RuntimeError(f"❌ Ollama returned invalid JSON:\n{output[:400]}")

    try:
        parsed = json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        raise RuntimeError(f"❌ Ollama returned invalid JSON:\n{output[:400]}")

    # Normalize output
    parsed.setdefault("spam_label","not_spam")
    parsed.setdefault("spam_score",0.0)
    parsed.setdefault("keywords_detected",[])
    parsed.setdefault("summary","No summary")
    parsed.setdefault("reasoning","No reasoning")

    parsed["spam_score"]=float(parsed.get("spam_score",0))
    parsed["spam_score"] = max(0.0, min(1.0, parsed["spam_score"]))

    if parsed["spam_score"]>=0.5:
        parsed["spam_label"]="spam"
    else:
        parsed["spam_label"]="not_spam"

    return parsed


# =========================================================
#  🎨 STREAMLIT UI
# =========================================================
def main():

    st.set_page_config(page_title="Hybrid AI Email Spam and Phishing Detector", page_icon="⚡", layout="wide")

    st.markdown("""
    <h1 style="
        text-align:center;font-size:42px;font-weight:800;
        background:linear-gradient(90deg,#00eaff,#00ff87);
        -webkit-background-clip:text;color:transparent;">
        📧 Hybrid AI Email Spam and Phishing Detector (Groq + Ollama)
    </h1>
    <p style="text-align:center;color:#bbb;">
        Cloud AI ⚡ + Local AI 🔥 + Live Spam Analytics 📊
    </p>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔍 Analyze Email", "📊 Analytics Dashboard"])


    # ================= TAB 1 =================
    with tab1:

        email = st.text_area("Paste Email Content 👇", height=220)
        local_model = st.selectbox("Offline Model", ["llama3","mistral","llama3:70b"])

        if st.button("Analyze Email 🚀", use_container_width=True):
            if email.strip()=="":
                st.warning("Paste an email first ⚠")
                return

            with st.spinner("Analyzing email with Groq..."):
                try:
                    result = analyze_with_groq(email)
                    used="Groq Cloud ✓"
                except Exception as e:
                    st.warning(f"Groq failed → switching to local model\n{e}")
                    result = analyze_with_ollama(email,local_model)
                    used=f"Ollama Local ({local_model}) ✓"

            st.session_state.history.append(result)

            color = "#ff4d4d" if result["spam_label"]=="spam" else "#00d87a"

            st.markdown(f"""
            <div style="background:{color}25;border:2px solid {color};
                        padding:14px;border-radius:12px;text-align:center;
                        font-size:23px;font-weight:bold;">
               { "🚫 SPAM" if result['spam_label']=="spam" else "🟢 SAFE" }
               — Score {result['spam_score']:.2f}
               <br><span style="font-size:13px;opacity:.6;">Model Used: {used}</span>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("📝 Summary")
            st.write(result["summary"])

            st.subheader("🔍 Reasoning")
            st.write(result["reasoning"])

            st.subheader("⚠ Trigger Keywords")
            st.write(", ".join(result["keywords_detected"]) or "No keywords detected")


    # ================= TAB 2 =================
    with tab2:

        st.header("📊 Spam Analysis Dashboard")

        if len(st.session_state.history)==0:
            st.info("No emails scanned yet.")
            return

        df=pd.DataFrame(st.session_state.history)

        st.subheader("Trend — Spam Score Over Time")
        st.line_chart(df["spam_score"])

        st.subheader("Spam vs Safe Ratio")
        fig = go.Figure(go.Pie(labels=["Spam","Safe"], values=[
            sum(df.spam_label=="spam"),
            sum(df.spam_label=="not_spam"),
        ], hole=0.45))
        st.plotly_chart(fig)

        st.subheader("History Log")
        st.table(df[["spam_label","spam_score","summary"]])


if __name__ == "__main__":
    main()
