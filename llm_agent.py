import os
import json
from groq import Groq

def generate_agent_briefing(
    player_id,
    timestamp,
    risk_label,
    risk_prob,
    key_metrics,
    symptoms,
    retrieved_rules,
    api_key=None
):
    if not api_key:
        creds_path = os.path.join(os.path.dirname(__file__), 'Credentials.json')
        if os.path.exists(creds_path):
            with open(creds_path, 'r') as f:
                api_key = json.load(f).get('groq_api_key', '')

    if not api_key:
        return "Error: Groq API Key is not configured."

    client = Groq(api_key=api_key)

    rules_context = ""
    for r in retrieved_rules:
        rules_context += f"\n- **Rule [{r['rule_id']}] {r['title']}**:\n  {r['content']}\n"

    system_prompt = """You are the PlayerGuard AI Briefing System for Northbridge Meridian FC.
You provide clinical, medical, and sports performance decision support to team doctors, physiotherapists, and sports scientists.

CRITICAL POLICY CONSTRAINTS (NMFC-PG-POL-001):
1. PW-001 (Human Authority): AI provides decision support ONLY. You must NEVER clear, bench, prescribe medication, or declare return-to-play for a player. All decisions are strictly reserved to human authorised professionals.
2. RET-002: You MUST cite retrieved rule IDs (e.g., PW-001, ESC-001, MOD-003, SYM-001) for every policy statement.
3. RET-003: Do NOT invent thresholds, medical diagnoses, or treatment procedures.
4. ORDER & READABILITY RULES:
   - "Required Next Step" MUST come BEFORE "Applicable Policy".
   - Every single bullet point MUST be on its own line.
   - Never put multiple bullet items on the same line.
   - List each symptom and each contributing metric on its own individual line.

EXACT FORMATTING STRUCTURE (Use exact markdown headers and double line breaks):

### 1. Player Record
- **Player Identifier**: <Player Name, Kit Number & ID>
- **Record Timestamp**: <Timestamp>

### 2. Model Output
- **Predicted Risk Category**: <Elevated-risk (Class 1) OR Lower-risk (Class 0)>
- **Estimated Risk Probability**: <Value>%
- **Contributing Biomechanical & Sensor Telemetry**:
  - Heart Rate: <Value> bpm
  - EMG Muscle Load: <Value> mV
  - Ground Reaction Force (GRF): <Value> N
  - Fatigue Index: <Value> pts
  - Training Session Duration: <Value> min

### 3. Relevant Context & Physical Observations
- **Reported Symptoms**:
  (Break each reported symptom into its own distinct bullet point):
  - <Symptom 1>
  - <Symptom 2 (if present)>
- **Tactical / Operational Context**: <Position, session notes>

### 4. Required Next Step (Immediate Action)
- **Responsible Human Role**: <Team Doctor / Physiotherapist / Sports Scientist per ESC-001/ESC-002>
- **Clinical Action**: <Exact clinical evaluation, examination, or monitoring required>
- **Escalation Path**: <Who to contact if unresolved per ESC rules>

### 5. Applicable Policy & Governing Rules
- **Rule <Rule-ID> (<Rule Title>)**: <Clear explanation of what the rule mandates>
- **Rule PW-001 (Human Decision-Making Authority)**: <Explain that diagnosis, clearance, and benching belong strictly to authorised human staff>

### 6. Limitations
- Algorithmic estimate for decision support only; does not provide clinical diagnosis or medical clearance (PW-001, MOD-008).
"""

    user_prompt = f"""Draft the policy-compliant PlayerGuard AI briefing for this player case:

PLAYER DATA:
- Player Identifier: {player_id}
- Timestamp: {timestamp}
- Model Risk Output: {"Elevated Risk (Class 1)" if risk_label == 1 else "Lower Risk (Class 0)"}
- Model Risk Probability: {risk_prob:.1%}
- Key Sensor & Workload Metrics: {json.dumps(key_metrics, indent=2)}
- Recorded Symptoms / Notes: {symptoms if symptoms else "None recorded"}

RETRIEVED CLUB POLICY RULES (RAG):
{rules_context}
"""

    models_to_try = ['qwen/qwen3.8-27b', 'openai/gpt-oss-120b', 'openai/gpt-oss-20b']
    last_err = ""
    for m in models_to_try:
        try:
            response = client.chat.completions.create(
                model=m,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content
        except Exception as e:
            last_err = str(e)
            continue

    return f"LLM Generation Error: {last_err}"

if __name__ == '__main__':
    from rag_engine import PolicyRAG
    rag = PolicyRAG()
    rules = rag.retrieve("elevated risk player", risk_label=1, symptoms="Tightness in right calf; mild stiffness during acceleration")
    briefing = generate_agent_briefing(
        player_id="Leo Gallagher (#11) - PG-CUSTOM-11",
        timestamp="2026-10-28 17:35:00",
        risk_label=1,
        risk_prob=0.956,
        key_metrics={"heart_rate": 176.0, "emg_amplitude": 1.22, "ground_reaction_force": 664.0, "fatigue_index": 42.7, "training_duration": 56.8},
        symptoms="Sharp fatigue in right hamstring during late sprint repetitions; mild gait stiffness observed.",
        retrieved_rules=rules
    )
    print("Agent Briefing Output:\n", briefing)
