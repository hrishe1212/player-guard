# ⚽ PlayerGuard AI — Sports Club Player Welfare & Injury Risk Escalation System

An AI Agent and Human-in-the-Loop Clinical Decision Support application for **Northbridge Meridian FC** (Policy Ref: `NMFC-PG-POL-001`).

---

## 🌟 Key Capabilities
1. **Decision Tree Injury Risk Prediction**:
   - Model: Decision Tree Classifier with `max_depth=3` and `max_leaf_nodes=None` (unlimited leaves).
   - Trained on 5,430 multimodal player workload and biomechanical telemetry sessions (`sports_multimodal_data.csv`).
   - Achieves **97.6% accuracy** and **95.6% recall** on elevated risk detection.
2. **Policy RAG (Retrieval-Augmented Generation)**:
   - Indexes all 73 formalized governance rules from `PlayerGuard_AI_Policy.docx`.
   - Lexical and semantic retrieval prioritizing mandatory governance rules (`PW-001`, `MOD-003`, `ESC-001`, `ESC-002`, `AUD-001`).
3. **Groq LLM Clinical Briefing Agent**:
   - Uses Groq's ultra-low latency API (`qwen/qwen3.8-27b` / `openai/gpt-oss-120b`).
   - Formats briefings strictly according to the **Section 15 Standard LLM Briefing Template**.
   - Enforces zero-hallucination and non-diagnostic constraints (Rule `PW-001`).
4. **Human-in-the-Loop Dashboard (Streamlit)**:
   - Final classification options:
     - 🟢 **Eligible to Play**
     - 🟡 **Put Under Review**
     - 🔴 **Medical Assistance**
   - Traceable medical audit trail logging compliant with Rule `AUD-001`.
5. **Automated Medical Escalation Email Dispatch**:
   - Automated SSL SMTP dispatch to team doctors and clinical staff via Gmail.

---

## 🚀 How to Run Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Credentials:**
   Ensure `Credentials.json` contains your valid API keys:
   ```json
   {
       "groq_api_key": "YOUR_GROQ_KEY",
       "gmail_user": "YOUR_GMAIL@gmail.com",
       "gmail_app_password": "YOUR_APP_PASSWORD"
   }
   ```

3. **Train the Model (if needed):**
   ```bash
   python train_tree.py
   ```

4. **Launch Streamlit App:**
   ```bash
   streamlit run app.py
   ```

---

## 🌐 How to Publish to Streamlit Cloud

1. Create a GitHub repository and push this directory:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for PlayerGuard AI"
   git remote add origin https://github.com/YOUR_USERNAME/playerguard-ai.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **"New app"**.
3. Select your repository, branch (`main`), and main file path (`app.py`).
4. Under **Advanced settings > Secrets**, paste your credentials:
   ```toml
   GROQ_API_KEY = "gsk_..."
   GMAIL_USER = "..."
   GMAIL_APP_PASSWORD = "..."
   ```
5. Click **Deploy!** 🚀
