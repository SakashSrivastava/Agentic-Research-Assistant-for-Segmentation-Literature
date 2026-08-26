"""Security helpers: logging, rate limiting, HTML sanitization, input validation,
signed expiring tokens (email verification + password reset), and email delivery.

Everything here runs server-side only. Tokens are signed with FLASK_SECRET_KEY and
carry their own expiry; nothing sensitive is exposed to the client.
"""
from __future__ import annotations

import logging
import re
import smtplib
from email.message import EmailMessage
from urllib.parse import urlparse

import bleach
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from src import config

# ---------------- logging (stdout, container-friendly) ----------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("seglit")


def audit(event: str, **fields) -> None:
    """Structured audit line for auth attempts, API errors, and abuse signals."""
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    log.info("AUDIT %s %s", event, extra)


# ---------------- rate limiter (attached to the app in app.py) ----------------
# memory:// is per-process; fine for a single instance. Point at redis:// for
# multi-worker / multi-instance so limits are shared.
limiter = Limiter(key_func=get_remote_address, default_limits=["300 per hour"],
                  storage_uri="memory://", strategy="fixed-window")


def user_or_ip() -> str:
    """Rate-limit key: the logged-in user id when available, else the client IP."""
    from flask_login import current_user
    if getattr(current_user, "is_authenticated", False):
        return f"user:{current_user.id}"
    return get_remote_address()


# ---------------- HTML sanitization (XSS) ----------------
_ALLOWED_TAGS = ["p", "br", "hr", "a", "strong", "em", "b", "i", "u", "code", "pre",
                 "blockquote", "ul", "ol", "li", "h1", "h2", "h3", "h4", "span",
                 "table", "thead", "tbody", "tr", "th", "td"]
_ALLOWED_ATTRS = {"a": ["href", "target", "rel"]}


def sanitize_html(html: str) -> str:
    """Whitelist-sanitize LLM/markdown-generated HTML before it is marked |safe."""
    return bleach.clean(html, tags=_ALLOWED_TAGS, attributes=_ALLOWED_ATTRS,
                        protocols=["http", "https"], strip=True)


# ---------------- input validation ----------------
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def valid_email(email: str) -> bool:
    return bool(email) and len(email) <= 254 and bool(_EMAIL_RE.match(email))


def valid_password(pw: str) -> bool:
    # 8..128: floor for strength, ceiling to stop hashing-DoS via giant inputs.
    return 8 <= len(pw) <= 128


def safe_next(nxt: str | None) -> str | None:
    """Open-redirect guard: allow only same-site relative paths."""
    if not nxt or not nxt.startswith("/") or nxt.startswith("//"):
        return None
    u = urlparse(nxt)
    return nxt if not u.scheme and not u.netloc else None


# ---------------- signed, expiring tokens ----------------
def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(config.FLASK_SECRET_KEY)


def make_token(purpose: str, value: str) -> str:
    return _serializer().dumps(value, salt=purpose)


def read_token(purpose: str, token: str, max_age: int) -> str | None:
    """Return the signed value if the token is valid and unexpired, else None."""
    try:
        return _serializer().loads(token, salt=purpose, max_age=max_age)
    except (BadSignature, SignatureExpired):
        return None


# ---------------- email delivery ----------------
def send_email(to: str, subject: str, body: str) -> bool:
    """Send via SMTP if configured; otherwise log the body and return False so the
    caller can surface the link in-app for local development."""
    if not (config.SMTP_HOST and config.SMTP_USER and config.SMTP_PASSWORD):
        log.warning("SMTP not configured; '%s' to %s NOT emailed. Body:\n%s", subject, to, body)
        return False
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = config.SMTP_FROM, to, subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=15) as s:
            s.starttls()
            s.login(config.SMTP_USER, config.SMTP_PASSWORD)
            s.send_message(msg)
        log.info("Sent '%s' email to %s", subject, to)
        return True
    except Exception as e:  # noqa: BLE001
        log.error("Failed to send email to %s: %s", to, e)
        return False
