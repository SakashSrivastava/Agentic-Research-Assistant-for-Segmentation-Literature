"""Flask web app: accounts, per-user history, admin analytics, and the agent.

  python -m src.app      # then open http://localhost:5000

Security: hashed passwords, expiring sessions, email verification + password reset
(signed expiring tokens), CSRF on every POST, per-route rate limiting, HTML
sanitization, HTTPS/HSTS + hardened headers in prod, and audit logging. User data
lives in a separate writable SQLite DB (src/db.py); the corpus DB stays read-only.
"""
from __future__ import annotations

import re
import secrets
import sqlite3
from datetime import timedelta
from functools import wraps

import markdown as md
from flask import (Flask, abort, flash, redirect, render_template, request,
                   session, url_for)
from flask_login import (LoginManager, UserMixin, current_user, login_required,
                         login_user, logout_user)
from werkzeug.middleware.proxy_fix import ProxyFix

from src import agent, config, db, qcache, security

app = Flask(__name__, template_folder="../templates")
app.secret_key = config.FLASK_SECRET_KEY
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=config.HTTPS_ONLY,      # cookies only over HTTPS in prod
    PERMANENT_SESSION_LIFETIME=timedelta(hours=config.SESSION_HOURS),
    MAX_CONTENT_LENGTH=256 * 1024,                # reject oversized request bodies
)
# Trust the proxy's X-Forwarded-Proto/Host so request.is_secure works behind a LB.
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
db.init_db()
security.limiter.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"

EXAMPLES = [
    "Which architectures report the best Dice on head and neck segmentation, on which datasets, and how many cases each?",
    "Which methods report the highest Dice for pancreas segmentation, and on how many patients?",
    "What Dice scores are reported for brain tumor / glioma segmentation, and by which methods?",
    "Which methods use uncertainty estimation in medical image segmentation?",
]
RETRIEVAL = {"recall5": 0.59, "paper_hit5": 0.85, "table_fidelity": 100}

VERIFY_MAX_AGE = 24 * 3600     # email verification link valid 24h
RESET_MAX_AGE = 3600           # password reset link valid 1h


class User(UserMixin):
    def __init__(self, row: dict):
        self.id = row["id"]
        self.email = row["email"]
        self.is_admin = bool(row["is_admin"])
        self.email_verified = bool(row.get("email_verified", 1))


@login_manager.user_loader
def load_user(user_id):
    row = db.get_user(user_id)
    return User(row) if row else None


def admin_required(f):
    @wraps(f)
    @login_required
    def wrapper(*a, **k):
        if not current_user.is_admin:
            security.audit("admin_denied", user=current_user.id, path=request.path)
            abort(403)
        return f(*a, **k)
    return wrapper


# ---------------- request guards: HTTPS, session, CSRF ----------------

@app.before_request
def _guard():
    session.permanent = True
    if config.HTTPS_ONLY and request.path != "/health" and not request.is_secure:
        return redirect(request.url.replace("http://", "https://", 1), code=301)
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_urlsafe(32)
    if request.method == "POST" and request.form.get("_csrf") != session.get("_csrf"):
        security.audit("csrf_reject", path=request.path, ip=request.remote_addr)
        abort(400)


@app.after_request
def _security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; script-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
    if config.HTTPS_ONLY:
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return resp


@app.context_processor
def _inject_csrf():
    return {"csrf_token": session.get("_csrf", "")}


# ---------------- error handlers (logged) ----------------

@app.errorhandler(400)
def _e400(e):
    return render_template("error.html", code=400, msg="Bad request."), 400


@app.errorhandler(403)
def _e403(e):
    return render_template("error.html", code=403, msg="You do not have access to that page."), 403


@app.errorhandler(404)
def _e404(e):
    return render_template("error.html", code=404, msg="Page not found."), 404


@app.errorhandler(429)
def _e429(e):
    security.audit("rate_limited", path=request.path, ip=request.remote_addr)
    return render_template("error.html", code=429,
                           msg="Too many requests. Please slow down and try again shortly."), 429


@app.errorhandler(500)
def _e500(e):
    security.log.exception("Unhandled error at %s", request.path)
    return render_template("error.html", code=500, msg="Something went wrong on our end."), 500


# ---------------- helpers ----------------

def corpus_stats() -> dict:
    try:
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        one = lambda sql: con.execute(sql).fetchone()[0] or 0
        stats = {
            "papers": one("SELECT COUNT(*) FROM papers"),
            "chunks": one("SELECT COALESCE(SUM(n_chunks),0) FROM papers"),
            "metric_rows": one("SELECT COUNT(*) FROM metrics"),
            "architectures": one("SELECT COUNT(DISTINCT architecture) FROM metrics"),
            "datasets": one("SELECT COUNT(DISTINCT dataset) FROM metrics WHERE dataset IS NOT NULL"),
            "anatomies": one("SELECT COUNT(DISTINCT anatomical_target) FROM papers WHERE anatomical_target IS NOT NULL"),
        }
        con.close()
        return stats
    except Exception:  # noqa: BLE001
        return {}


def _linkify(html: str) -> str:
    return re.sub(r"arxiv_(\d+\.\d+)",
                  r'<a href="https://arxiv.org/abs/\1" target="_blank" rel="noopener">arxiv_\1</a>',
                  html)


def _render_answer(markdown_text: str) -> str:
    # Render markdown, then whitelist-sanitize before it is marked |safe (XSS guard).
    html = md.markdown(markdown_text or "", extensions=["tables", "fenced_code"])
    return security.sanitize_html(_linkify(html))


def _send_link_email(email: str, kind: str) -> None:
    """Send a verification or reset link; in dev (no SMTP) surface it in-app."""
    if kind == "verify":
        token = security.make_token("verify-email", email)
        link = f"{config.BASE_URL}{url_for('verify_email', token=token)}"
        subject, body = "Verify your email", f"Confirm your account (valid 24h):\n{link}"
    else:
        token = security.make_token("reset-password", email)
        link = f"{config.BASE_URL}{url_for('reset_password', token=token)}"
        subject, body = "Reset your password", f"Reset your password (valid 1h):\n{link}"
    if not security.send_email(email, subject, body):
        flash(f"Email delivery is not configured, so here is your {kind} link: {link}", "info")


# ---------------- auth ----------------

@app.route("/signup", methods=["GET", "POST"])
@security.limiter.limit("5 per hour", methods=["POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        pw2 = request.form.get("confirm", "")
        if not security.valid_email(email):
            flash("Enter a valid email address.", "error")
        elif not security.valid_password(pw):
            flash("Password must be 8 to 128 characters.", "error")
        elif pw != pw2:
            flash("Passwords do not match.", "error")
        else:
            user = db.create_user(email, pw)
            if not user:
                flash("That email is already registered. Try logging in.", "error")
            else:
                security.audit("signup", user=user["id"], email=email)
                if config.REQUIRE_EMAIL_VERIFICATION:
                    _send_link_email(email, "verify")
                    return render_template("verify_notice.html", email=email)
                db.set_email_verified(email)          # verification disabled -> log in directly
                login_user(User(db.get_user(user["id"])), remember=False)
                return redirect(url_for("index"))
    return render_template("signup.html")


@app.route("/verify/<token>")
@security.limiter.limit("20 per hour")
def verify_email(token):
    email = security.read_token("verify-email", token, VERIFY_MAX_AGE)
    if not email:
        flash("That verification link is invalid or has expired. Request a new one.", "error")
        return redirect(url_for("login"))
    db.set_email_verified(email)
    security.audit("email_verified", email=email)
    flash("Email verified. You can now log in.", "success")
    return redirect(url_for("login"))


@app.route("/resend-verification", methods=["POST"])
@security.limiter.limit("5 per hour")
def resend_verification():
    email = request.form.get("email", "").strip().lower()
    row = db.get_user_by_email(email)
    if row and not row["email_verified"]:
        _send_link_email(email, "verify")
    # Generic response either way, to avoid revealing which emails are registered.
    flash("If that email needs verification, we've sent a new link.", "success")
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
@security.limiter.limit("10 per minute; 40 per hour", methods=["POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        row = db.get_user_by_email(email)
        if row and db.verify_password(row, request.form.get("password", "")):
            if config.REQUIRE_EMAIL_VERIFICATION and not row["email_verified"]:
                security.audit("login_unverified", email=email)
                return render_template("verify_notice.html", email=email, unverified=True)
            login_user(User(row), remember=False)
            security.audit("login_ok", user=row["id"], ip=request.remote_addr)
            return redirect(security.safe_next(request.args.get("next")) or url_for("index"))
        security.audit("login_fail", email=email, ip=request.remote_addr)
        flash("Incorrect email or password.", "error")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/forgot", methods=["GET", "POST"])
@security.limiter.limit("5 per hour", methods=["POST"])
def forgot():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        if security.valid_email(email) and db.get_user_by_email(email):
            _send_link_email(email, "reset")
            security.audit("reset_requested", email=email)
        # Generic response to avoid user enumeration.
        flash("If that email is registered, a password reset link has been sent.", "success")
        return redirect(url_for("login"))
    return render_template("forgot.html")


@app.route("/reset/<token>", methods=["GET", "POST"])
@security.limiter.limit("10 per hour")
def reset_password(token):
    email = security.read_token("reset-password", token, RESET_MAX_AGE)
    if not email:
        flash("That reset link is invalid or has expired. Request a new one.", "error")
        return redirect(url_for("forgot"))
    if request.method == "POST":
        pw = request.form.get("password", "")
        pw2 = request.form.get("confirm", "")
        if not security.valid_password(pw):
            flash("Password must be 8 to 128 characters.", "error")
        elif pw != pw2:
            flash("Passwords do not match.", "error")
        else:
            row = db.get_user_by_email(email)
            if row:
                db.update_password(row["id"], pw)
                db.set_email_verified(email)   # a successful reset also proves email ownership
                security.audit("password_reset", email=email)
                flash("Password updated. You can now log in.", "success")
            return redirect(url_for("login"))
    return render_template("reset.html", token=token)


# ---------------- app ----------------

@app.route("/", methods=["GET", "POST"])
@login_required
@security.limiter.limit("15 per hour", key_func=security.user_or_ip, methods=["POST"])
def index():
    if config.REQUIRE_EMAIL_VERIFICATION and not current_user.email_verified:
        return render_template("verify_notice.html", email=current_user.email, unverified=True)

    question = request.form.get("question", "").strip() if request.method == "POST" else ""
    question = question[:config.MAX_QUESTION_CHARS]            # bound input length
    user_key = request.form.get("user_api_key", "").strip() or None  # BYOK, never stored/logged
    nocache = request.form.get("nocache") == "1"
    result = None
    if question:
        hit = None if nocache else qcache.find(question)
        if hit:
            row, sim = hit
            db.log_query(current_user.id, question, row["answer"], row["steps"], 0,
                         ok=True, own_key=False, cached=True)
            db.cache_bump_hit(row["id"])
            result = {"answer": _render_answer(row["answer"]), "trace": [], "steps": row["steps"],
                      "tokens": 0, "from_cache": True, "cache_id": row["id"],
                      "matched": row["question"], "similarity": round(sim, 3)}
        elif not user_key and db.queries_today(current_user.id) >= config.FREE_DAILY_QUERIES:
            result = {"error": f"You've used your {config.FREE_DAILY_QUERIES} free searches for "
                      "today. Add your own free Groq API key in Settings to keep going without "
                      "limits. Your key stays in your browser and is never stored on our servers."}
        else:
            try:
                out = agent.answer(question, verbose=False, api_key=user_key)
                steps, tokens, ans = out["steps"], sum(out["tokens"]), out["answer"]
                db.log_query(current_user.id, question, ans, steps, tokens,
                             ok=True, own_key=bool(user_key))
                cache_id = None if ans.strip().startswith("(stopped") \
                    else qcache.store(question, ans, steps, tokens)
                result = {"answer": _render_answer(ans), "trace": out["trace"], "steps": steps,
                          "tokens": tokens, "from_cache": False, "cache_id": cache_id}
            except Exception as ex:  # noqa: BLE001
                security.log.error("agent error for user %s: %s", current_user.id, ex)
                db.log_query(current_user.id, question, str(ex), 0, 0, ok=False, own_key=bool(user_key))
                result = {"error": str(ex)}
    free_left = max(0, config.FREE_DAILY_QUERIES - db.queries_today(current_user.id))
    return render_template("index.html", question=question, result=result,
                           examples=EXAMPLES, stats=corpus_stats(), retrieval=RETRIEVAL,
                           recent=db.user_history(current_user.id, limit=6),
                           free_left=free_left, free_total=config.FREE_DAILY_QUERIES)


@app.route("/feedback", methods=["POST"])
@login_required
@security.limiter.limit("60 per hour", key_func=security.user_or_ip)
def feedback():
    cache_id = request.form.get("cache_id", "")
    vote = request.form.get("vote", "")
    if cache_id.isdigit() and vote in ("up", "down"):
        db.cache_vote(int(cache_id), vote == "up")
        return {"ok": True}
    return {"ok": False}, 400


@app.route("/history")
@login_required
def history():
    items = db.user_history(current_user.id, limit=100)   # scoped to the owner (no IDOR)
    for it in items:
        it["answer_html"] = _render_answer(it["answer"]) if it["ok"] else None
    return render_template("history.html", items=items)


@app.route("/admin")
@admin_required
def admin():
    return render_template("admin.html", stats=db.admin_stats(), cache=db.cache_stats())


@app.route("/health")
@security.limiter.exempt
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
