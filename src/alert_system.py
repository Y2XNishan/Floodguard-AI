"""Alert delivery and subscription helpers for FloodGuard AI."""

from __future__ import annotations

import csv
import json
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SUBSCRIPTIONS_FILE = DATA_DIR / "subscriptions.csv"
ALERT_LOG_FILE = DATA_DIR / "alert_logs.csv"

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

SUBSCRIPTION_FIELDS = [
    "id",
    "name",
    "district",
    "state",
    "alert_type",
    "contact",
    "risk_threshold",
    "send_daily",
    "subscribed_date",
    "is_active",
]

ALERT_LOG_FIELDS = [
    "timestamp",
    "subscription_id",
    "district",
    "alert_type",
    "risk_level",
    "status",
]


def _ensure_csv(path: Path, headers: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(headers)


def _risk_color(risk_level: str) -> str:
    return {
        "low": "#00C853",
        "moderate": "#FFD600",
        "high": "#FF6D00",
        "severe": "#D50000",
    }.get(risk_level.lower(), "#FFD600")


def _risk_emoji(risk_level: str) -> str:
    return {
        "low": "🟢",
        "moderate": "🟡",
        "high": "🟠",
        "severe": "🔴",
    }.get(risk_level.lower(), "🟡")


def _actions_for_risk(risk_level: str) -> list[str]:
    return {
        "low": ["Monitor weather updates", "Keep an emergency kit ready"],
        "moderate": ["Move valuables to higher ground", "Avoid low-lying areas"],
        "high": ["Prepare for evacuation", "Contact local authorities"],
        "severe": ["EVACUATE NOW", "Call 112 for emergency help"],
    }.get(risk_level.lower(), ["Monitor official flood updates", "Keep your phone charged"])


def _row_value(row: Any, names: list[str], default: Any = "") -> Any:
    if isinstance(row, dict):
        for name in names:
            if name in row:
                return row[name]
        return default
    for name in names:
        if hasattr(row, name):
            return getattr(row, name)
    return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_configured(value: str | None) -> bool:
    if not value:
        return False
    normalized = value.strip().lower()
    return not (
        normalized.startswith("your_")
        or normalized.endswith("_here")
        or normalized in {"xxxx_xxxx_xxxx_xxxx", "your_token"}
    )


def _iter_forecast_rows(forecast_data: Any) -> list[Any]:
    if forecast_data is None:
        return []
    if hasattr(forecast_data, "to_dict"):
        try:
            return forecast_data.to_dict("records")
        except TypeError:
            return []
    if isinstance(forecast_data, list):
        return forecast_data
    return []


def _forecast_table_html(forecast_data: Any) -> str:
    rows = _iter_forecast_rows(forecast_data)
    if not rows:
        return "<p style='color:#aaa'>Forecast data unavailable.</p>"

    html_rows = []
    for row in rows[:7]:
        date = _row_value(row, ["date_display", "date", "day"], "Day")
        rain = _row_value(row, ["rainfall_mm", "precipitation_mm", "rain"], 0)
        risk = _row_value(row, ["risk_probability", "flood_probability", "flood_probability_pct"], 0)
        risk = _to_float(risk)
        risk = risk * 100 if risk <= 1 else risk
        rain = _to_float(rain)
        html_rows.append(
            "<tr>"
            f"<td style='padding:8px;border-bottom:1px solid #333'>{date}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #333'>{rain:.1f} mm</td>"
            f"<td style='padding:8px;border-bottom:1px solid #333'>{risk:.1f}%</td>"
            "</tr>"
        )

    return (
        "<table style='width:100%;border-collapse:collapse;color:white'>"
        "<thead><tr>"
        "<th style='text-align:left;padding:8px;color:#00b4d8'>Day</th>"
        "<th style='text-align:left;padding:8px;color:#00b4d8'>Rain</th>"
        "<th style='text-align:left;padding:8px;color:#00b4d8'>Risk</th>"
        "</tr></thead><tbody>"
        + "".join(html_rows)
        + "</tbody></table>"
    )


def _forecast_text(forecast_data: Any) -> str:
    rows = _iter_forecast_rows(forecast_data)
    if not rows:
        return "Forecast data unavailable."

    lines = []
    for row in rows[:7]:
        day = _row_value(row, ["date_display", "date", "day"], "Day")
        rain = _row_value(row, ["rainfall_mm", "precipitation_mm", "rain"], 0)
        risk = _row_value(row, ["risk_probability", "flood_probability", "flood_probability_pct"], 0)
        risk = _to_float(risk)
        risk = risk * 100 if risk <= 1 else risk
        rain = _to_float(rain)
        emoji = "🌧️" if rain >= 20 else "☁️"
        lines.append(f"{emoji} {day}: {rain:.1f} mm rain, {risk:.1f}% risk")
    return "\n".join(lines)


def _risk_level_from_pct(risk_pct: float) -> str:
    if risk_pct >= 85:
        return "Severe"
    if risk_pct >= 65:
        return "High"
    if risk_pct >= 35:
        return "Moderate"
    return "Low"


def _normalize_alert_type(alert_type: str) -> str:
    value = (alert_type or "").strip().lower()
    if "telegram" in value:
        return "Telegram"
    if "sms" in value:
        return "SMS"
    return "Email"


def send_email_alert(
    to_email,
    district="Test District",
    state="Test State",
    risk_score=75,
    risk_level="High",
    risk_pct=75,
    forecast_data=None,
):
    """Send a Gmail SMTP alert using an app password."""
    risk_score = round(float(risk_score), 1)
    if forecast_data is None:
        forecast_data = []

    gmail_user = os.getenv("GMAIL_USER")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")

    if not _is_configured(gmail_user) or not _is_configured(gmail_password):
        return False, "Gmail credentials not configured in .env"

    try:
        subject = f"🚨 FLOOD ALERT: {district}, {state} - {risk_level} RISK"

        risk_key = str(risk_level).strip().upper()
        if risk_key in {"VERY HIGH", "EXTREME", "SEVERE"}:
            badge_color = "#ef4444"
        elif risk_key == "HIGH":
            badge_color = "#f97316"
        elif risk_key == "MODERATE":
            badge_color = "#f59e0b"
        else:
            badge_color = "#22c55e"
        actions = _actions_for_risk(str(risk_level))
        actions_html = "".join([f'<p style="color:#cbd5e1;margin:6px 0;font-family:Arial,sans-serif;">✅ {a}</p>' for a in actions])

        html_body = f"""
<html>
<body style="margin:0;padding:0;background-color:#0f172a;">
<table width="100%" cellpadding="0" cellspacing="0" bgcolor="#0f172a">
<tr><td align="center" style="padding:24px;">
<table width="600" cellpadding="0" cellspacing="0">

  <!-- Header -->
  <tr><td bgcolor="#1e293b" style="border-top:4px solid #06b6d4;border-radius:12px;padding:24px;text-align:center;">
    <h1 style="color:#06b6d4;margin:0;font-size:24px;font-family:Arial,sans-serif;">🌊 FloodGuard AI</h1>
    <p style="color:#94a3b8;margin:8px 0 0;font-family:Arial,sans-serif;">India Flood Risk Prediction & Agricultural Intelligence</p>
  </td></tr>

  <tr><td height="12"></td></tr>

  <!-- Alert Badge -->
  <tr><td bgcolor="#1e293b" style="border-radius:12px;padding:20px;text-align:center;">
    <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin: 16px 0;">
      <tr>
        <td align="center">
          <div style="display: inline-block; background-color: #e53e3e; color: #ffffff; font-size: 13px; font-weight: bold; padding: 8px 20px; border-radius: 4px; font-family: Arial, sans-serif; letter-spacing: 1px;">
            🚨 FLOOD ALERT NOTIFICATION
          </div>
        </td>
      </tr>
    </table>
  </td></tr>

  <tr><td height="12"></td></tr>

  <!-- Location -->
  <tr><td bgcolor="#1e293b" style="border-radius:12px;padding:24px;">
    <p style="color:#94a3b8;margin:0 0 12px;font-size:12px;font-family:Arial,sans-serif;text-transform:uppercase;letter-spacing:1px;">Location Details</p>
    <p style="color:#f1f5f9;margin:8px 0;font-family:Arial,sans-serif;">📍 <b>District:</b> <span style="color:#06b6d4;">{district}</span></p>
    <p style="color:#f1f5f9;margin:8px 0;font-family:Arial,sans-serif;">🗺️ <b>State:</b> <span style="color:#06b6d4;">{state}</span></p>
  </td></tr>

  <tr><td height="12"></td></tr>

  <!-- Risk -->
  <tr><td bgcolor="#1e293b" style="border-radius:12px;padding:24px;">
    <p style="color:#94a3b8;margin:0 0 12px;font-size:12px;font-family:Arial,sans-serif;text-transform:uppercase;letter-spacing:1px;">Risk Assessment</p>
    <p style="color:#f1f5f9;margin:8px 0;font-family:Arial,sans-serif;">🎯 <b>Risk Score:</b> <span style="color:#06b6d4;font-size:22px;font-weight:bold;">{risk_score}%</span></p>
    <p style="color:#f1f5f9;margin:8px 0;font-family:Arial,sans-serif;">⚠️ <b>Risk Level:</b> <span style="background-color:{badge_color};color:#ffffff;padding:4px 14px;border-radius:12px;font-weight:bold;">{risk_level}</span></p>
  </td></tr>

  <tr><td height="12"></td></tr>

  <!-- Actions -->
  <tr><td bgcolor="#1e293b" style="border-radius:12px;padding:24px;">
    <p style="color:#94a3b8;margin:0 0 12px;font-size:12px;font-family:Arial,sans-serif;text-transform:uppercase;letter-spacing:1px;">📋 Recommended Actions</p>
    {actions_html}
  </td></tr>

  <tr><td height="12"></td></tr>

  <!-- Footer -->
  <tr><td style="text-align:center;padding:16px;">
    <p style="color:#64748b;font-size:12px;font-family:Arial,sans-serif;margin:0;">Stay safe 🙏 — FloodGuard AI Team</p>
    <p style="color:#475569;font-size:11px;font-family:Arial,sans-serif;margin:4px 0 0;">Emergency: 1070 | 112 | Powered by FloodGuard AI</p>
  </td></tr>

</table>
</td></tr>
</table>
</body>
</html>
"""

        msg = MIMEText(html_body, "html")
        msg["From"] = gmail_user
        msg["To"] = to_email
        msg["Subject"] = subject

        # Debug: log Content-Type to verify it's text/html
        print(f"[send_email_alert] Content-Type: {msg.get_content_type()}")

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(gmail_user, gmail_password)
            server.send_message(msg)

        return True, "Email sent successfully"
    except Exception as exc:
        return False, str(exc)


def send_telegram_alert(
    bot_token: str,
    chat_id: str,
    district: str,
    state: str,
    risk_score: float,
    risk_level: str,
) -> tuple[bool, str]:
    """Send a Telegram alert through the Bot API."""
    if not _is_configured(bot_token) or not _is_configured(chat_id):
        return False, "Telegram credentials not configured"

    message = f"""
🚨 *FLOOD ALERT - FloodGuard AI*

📍 *District:* {district}, {state}
⚠️ *Risk Level:* {risk_level}
📊 *Risk Score:* {risk_score}%

{"🔴 IMMEDIATE ACTION REQUIRED!" if float(risk_score) > 60 else "⚠️ Please take precautions."}

Emergency: 1070, 112
    """

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
            },
            timeout=20,
        )

        if response.status_code == 200:
            return True, "Telegram alert sent"
        return False, response.text
    except Exception as exc:
        return False, str(exc)


def subscribe_user(
    name: str,
    district: str,
    state: str,
    alert_type: str,
    contact: str,
    risk_threshold: int = 60,
    send_daily: bool = True,
) -> str:
    """Save a user subscription and return its generated id."""
    _ensure_csv(SUBSCRIPTIONS_FILE, SUBSCRIPTION_FIELDS)
    contact = str(contact or "").strip()
    if not contact:
        raise ValueError("An alert contact is required")
    try:
        risk_threshold = int(risk_threshold)
    except (TypeError, ValueError):
        risk_threshold = 60
    risk_threshold = min(max(risk_threshold, 0), 100)
    subscription_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    row = {
        "id": subscription_id,
        "name": name,
        "district": district,
        "state": state,
        "alert_type": _normalize_alert_type(alert_type),
        "contact": contact,
        "risk_threshold": str(risk_threshold),
        "send_daily": str(bool(send_daily)),
        "subscribed_date": datetime.now().isoformat(timespec="seconds"),
        "is_active": "True",
    }
    with SUBSCRIPTIONS_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUBSCRIPTION_FIELDS)
        writer.writerow(row)
    return subscription_id


def get_subscriptions() -> list[dict[str, str]]:
    """Load all active alert subscriptions."""
    _ensure_csv(SUBSCRIPTIONS_FILE, SUBSCRIPTION_FIELDS)
    with SUBSCRIPTIONS_FILE.open("r", newline="", encoding="utf-8") as file:
        return [
            row
            for row in csv.DictReader(file)
            if row.get("is_active", "True").lower() == "true"
        ]


def unsubscribe(subscription_id: str) -> bool:
    """Deactivate a subscription by id."""
    _ensure_csv(SUBSCRIPTIONS_FILE, SUBSCRIPTION_FIELDS)
    with SUBSCRIPTIONS_FILE.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    found = False
    for row in rows:
        if row.get("id") == subscription_id:
            row["is_active"] = "False"
            found = True

    with SUBSCRIPTIONS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUBSCRIPTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return found


def _call_with_district_state(func: Callable[..., Any], district: str, state: str) -> Any:
    try:
        return func(district, state)
    except TypeError:
        return func(district)


def _parse_risk_result(result: Any) -> tuple[float, str]:
    if isinstance(result, dict):
        risk_pct = result.get("risk_pct", result.get("risk_probability", result.get("flood_probability", 0)))
        risk_pct = _to_float(risk_pct)
        if risk_pct <= 1:
            risk_pct *= 100
        risk_level = result.get("risk_level") or _risk_level_from_pct(risk_pct)
        return risk_pct, str(risk_level)
    risk_pct = _to_float(result)
    if risk_pct <= 1:
        risk_pct *= 100
    return risk_pct, _risk_level_from_pct(risk_pct)


def send_daily_alerts(forecast_func: Callable[..., Any], risk_func: Callable[..., Any]) -> dict[str, Any]:
    """Send morning alerts to subscribers and log each attempt."""
    subscriptions = get_subscriptions()
    summary = {"checked": len(subscriptions), "sent": 0, "failed": 0, "skipped": 0}
    district_cache: dict[tuple[str, str], dict[str, Any]] = {}

    for subscription in subscriptions:
        district = subscription["district"]
        state = subscription["state"]
        key = (district, state)
        if key not in district_cache:
            try:
                risk_pct, risk_level = _parse_risk_result(_call_with_district_state(risk_func, district, state))
                forecast_data = _call_with_district_state(forecast_func, district, state)
                district_cache[key] = {
                    "risk_pct": risk_pct,
                    "risk_level": risk_level,
                    "forecast_data": forecast_data,
                }
            except Exception as exc:
                district_cache[key] = {"error": str(exc)}

        cached = district_cache[key]
        if "error" in cached:
            log_alert(subscription["id"], district, subscription["alert_type"], "Unknown", "failed")
            summary["failed"] += 1
            continue

        risk_pct = float(cached["risk_pct"])
        risk_level = str(cached["risk_level"])
        send_daily = subscription.get("send_daily", "True").lower() == "true"
        threshold = float(subscription.get("risk_threshold", 60))
        if not send_daily and risk_pct <= threshold:
            summary["skipped"] += 1
            continue

        alert_type = _normalize_alert_type(subscription["alert_type"])
        status = "failed"
        if alert_type == "Email":
            sent, _message = send_email_alert(
                subscription["contact"],
                district,
                state,
                risk_pct,
                risk_level,
            )
            status = "sent" if sent else "failed"
        elif alert_type == "Telegram":
            sent, _message = send_telegram_alert(
                os.getenv("TELEGRAM_BOT_TOKEN", ""),
                subscription["contact"],
                district,
                state,
                risk_pct,
                risk_level,
            )
            status = "sent" if sent else "failed"
        else:
            summary["skipped"] += 1
            log_alert(subscription["id"], district, alert_type, risk_level, "skipped")
            continue

        log_alert(subscription["id"], district, alert_type, risk_level, status)
        if status == "sent":
            summary["sent"] += 1
        else:
            summary["failed"] += 1

    return summary


def log_alert(
    subscription_id: str,
    district: str,
    alert_type: str,
    risk_level: str,
    status: str,
) -> None:
    """Append one alert delivery attempt to the alert log."""
    _ensure_csv(ALERT_LOG_FILE, ALERT_LOG_FIELDS)
    with ALERT_LOG_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ALERT_LOG_FIELDS)
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "subscription_id": subscription_id,
                "district": district,
                "alert_type": alert_type,
                "risk_level": risk_level,
                "status": status,
            }
        )


def get_alert_stats() -> dict[str, int]:
    """Return subscription and delivery statistics."""
    subscriptions = get_subscriptions()
    _ensure_csv(ALERT_LOG_FILE, ALERT_LOG_FIELDS)
    with ALERT_LOG_FILE.open("r", newline="", encoding="utf-8") as file:
        logs = list(csv.DictReader(file))

    today = datetime.now().date().isoformat()
    successful_logs = [log for log in logs if log.get("status") in {"sent", "demo"}]
    return {
        "total_subscribers": len(subscriptions),
        "email_subscribers": sum(1 for sub in subscriptions if _normalize_alert_type(sub.get("alert_type", "")) == "Email"),
        "telegram_subscribers": sum(1 for sub in subscriptions if _normalize_alert_type(sub.get("alert_type", "")) == "Telegram"),
        "sms_subscribers": sum(1 for sub in subscriptions if _normalize_alert_type(sub.get("alert_type", "")) == "SMS"),
        "alerts_sent_today": sum(1 for log in successful_logs if log.get("timestamp", "").startswith(today)),
        "alerts_sent_total": len(successful_logs),
        "active_districts": len({sub.get("district") for sub in subscriptions if sub.get("district")}),
    }
