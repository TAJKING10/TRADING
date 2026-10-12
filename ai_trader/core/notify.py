"""WhatsApp delivery for the daily digest via CallMeBot.

Why CallMeBot over Twilio: Twilio's WhatsApp sandbox session expires if you
don't message it every ~3 days, which breaks an unattended daily cron job.
CallMeBot is a one-time opt-in (no business account, no recurring re-auth) —
the right fit for a personal "just notify me" channel. It's an unofficial
third-party service, not Meta's official Business API, so treat it as
best-effort, not a guaranteed-delivery channel.

One-time setup (only you can do this, from your own phone):
  1. Save +34 644 59 71 30 as a contact.
  2. WhatsApp it: "I allow callmebot to send me messages"
  3. It replies with your personal API key. Put WHATSAPP_PHONE + WHATSAPP_APIKEY in .env.
"""
import logging
import urllib.parse

import requests

logger = logging.getLogger("notify")

CALLMEBOT_URL = "https://api.callmebot.com/whatsapp.php"
MAX_LEN = 1500  # keep digests short enough for a WhatsApp message


def send_whatsapp(phone: str, api_key: str, message: str) -> bool:
    if not phone or not api_key:
        return False

    text = message if len(message) <= MAX_LEN else message[: MAX_LEN - 20] + "\n...(truncated)"
    params = {"phone": phone, "text": text, "apikey": api_key}
    url = f"{CALLMEBOT_URL}?{urllib.parse.urlencode(params)}"

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            logger.info("WhatsApp digest sent to %s", phone)
            return True
        logger.error("WhatsApp send failed (status=%s): %s", resp.status_code, resp.text[:200])
        return False
    except requests.RequestException as e:
        logger.error("WhatsApp send failed: %s", e)
        return False
