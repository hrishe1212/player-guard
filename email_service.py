import os
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def get_credentials():
    creds_path = os.path.join(os.path.dirname(__file__), 'Credentials.json')
    if os.path.exists(creds_path):
        with open(creds_path, 'r') as f:
            return json.load(f)
    return {
        'groq_api_key': os.getenv('GROQ_API_KEY', ''),
        'gmail_user': os.getenv('GMAIL_USER', ''),
        'gmail_app_password': os.getenv('GMAIL_APP_PASSWORD', '')
    }

def send_player_alert(
    recipient_email,
    player_id,
    risk_level,
    human_decision,
    briefing_text,
    rules_cited=None,
    reviewer="Head Medical Officer"
):
    creds = get_credentials()
    sender = creds.get('gmail_user')
    password = creds.get('gmail_app_password')

    if not sender or not password:
        return False, "Gmail credentials not configured."

    subject = f"[PlayerGuard AI Alert] {player_id} - {human_decision.upper()} ({risk_level.upper()})"
    
    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #1e293b; background: #f8fafc; padding: 20px; }}
            .container {{ max-width: 650px; margin: auto; background: #ffffff; border-radius: 10px; border: 1px solid #e2e8f0; overflow: hidden; }}
            .header {{ background: #0f172a; color: #ffffff; padding: 24px; text-align: center; }}
            .header h1 {{ margin: 0; font-size: 20px; letter-spacing: 0.5px; }}
            .badge {{ display: inline-block; padding: 4px 12px; border-radius: 9999px; font-weight: bold; font-size: 13px; text-transform: uppercase; margin-top: 10px; }}
            .badge-danger {{ background: #fee2e2; color: #dc2626; }}
            .badge-warning {{ background: #fef3c7; color: #d97706; }}
            .badge-success {{ background: #dcfce7; color: #16a34a; }}
            .content {{ padding: 24px; }}
            .field-row {{ margin-bottom: 14px; }}
            .field-label {{ font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: bold; margin-bottom: 4px; }}
            .field-value {{ font-size: 15px; color: #0f172a; }}
            .briefing-box {{ background: #f1f5f9; padding: 16px; border-left: 4px solid #3b82f6; border-radius: 4px; font-family: monospace; white-space: pre-wrap; font-size: 13px; margin: 16px 0; }}
            .footer {{ background: #f8fafc; padding: 16px; font-size: 12px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>NORTHBRIDGE MERIDIAN FC — PlayerGuard AI</h1>
                <div class="badge {'badge-danger' if 'Medical' in human_decision or 'Elevated' in risk_level else ('badge-warning' if 'Review' in human_decision else 'badge-success')}">
                    Decision: {human_decision}
                </div>
            </div>
            <div class="content">
                <div class="field-row">
                    <div class="field-label">Player Identifier</div>
                    <div class="field-value"><strong>{player_id}</strong></div>
                </div>
                <div class="field-row">
                    <div class="field-label">Model Risk Estimate</div>
                    <div class="field-value">{risk_level} (Decision Tree Depth=3)</div>
                </div>
                <div class="field-row">
                    <div class="field-label">Human Reviewer (Authorised Professional per PW-001)</div>
                    <div class="field-value">{reviewer}</div>
                </div>
                <div class="field-row">
                    <div class="field-label">Clinical / Administrative Action</div>
                    <div class="field-value">{human_decision}</div>
                </div>
                <div class="field-row">
                    <div class="field-label">AI Agent Policy Briefing (Section 15 Template)</div>
                    <div class="briefing-box">{briefing_text}</div>
                </div>
                {f'<div class="field-row"><div class="field-label">Governing Rule IDs</div><div class="field-value">{", ".join(rules_cited)}</div></div>' if rules_cited else ''}
            </div>
            <div class="footer">
                PlayerGuard AI Decision Support System • Document Ref: NMFC-PG-POL-001<br>
                Notice: AI outputs provide decision support only. Clinical decisions rest with authorized medical staff (PW-001).
            </div>
        </div>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"PlayerGuard AI <{sender}>"
    msg["To"] = recipient_email

    msg.attach(MIMEText(briefing_text, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, [recipient_email], msg.as_string())
        return True, "Notification email successfully dispatched!"
    except Exception as e:
        return False, f"Email delivery error: {str(e)}"

if __name__ == '__main__':
    creds = get_credentials()
    print("Email Service configured with account:", creds.get('gmail_user'))
