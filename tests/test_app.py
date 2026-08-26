"""Integration tests for the web app: auth, admin, BYOK + free allowance, the
semantic answer cache, and the security hardening. Runs against an isolated temp
database and makes no LLM calls, so it is fast and free.

  python tests/test_app.py
"""
import os
import sys
import tempfile

# Make the project importable and isolate all state BEFORE importing the app.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USER_DB_PATH"] = os.path.join(tempfile.gettempdir(), "seglit_test_users.db")
os.environ["ADMIN_EMAIL"] = "admin@test.com"
os.environ["FLASK_SECRET_KEY"] = "test-secret"
os.environ["REQUIRE_EMAIL_VERIFICATION"] = "false"   # main flow; verification tested separately
try:
    os.remove(os.environ["USER_DB_PATH"])
except OSError:
    pass

from src import app as appmod  # noqa: E402
from src import db, security   # noqa: E402

app = appmod.app
app.config["TESTING"] = True
security.limiter.enabled = False   # off for the main flow; re-enabled in the rate-limit test


def tok(c):
    """Fetch the CSRF token from the session for a POST."""
    c.get("/login")
    with c.session_transaction() as s:
        return s["_csrf"]


def run():
    c = app.test_client()

    r = c.get("/")
    assert r.status_code in (302, 303) and "/login" in r.headers["Location"]
    print("PASS  / requires login (redirects)")

    r = c.post("/login", data={"email": "a@b.com", "password": "x"})
    assert r.status_code == 400
    print("PASS  POST without CSRF token -> 400")

    t = tok(c)
    r = c.post("/signup", data={"_csrf": t, "email": "user@test.com",
                                "password": "password123", "confirm": "password123"})
    assert r.status_code in (302, 303)
    print("PASS  signup creates user + logs in")

    r = c.get("/")
    assert r.status_code == 200 and b"Search the segmentation" in r.data
    print("PASS  dashboard renders for logged-in user")

    r = c.get("/admin")
    assert r.status_code == 403
    print("PASS  non-admin gets 403 on /admin")

    r = c.get("/history")
    assert r.status_code == 200
    print("PASS  history page renders")

    t = tok(c)
    r = c.post("/signup", data={"_csrf": t, "email": "user@test.com",
                                "password": "password123", "confirm": "password123"})
    assert r.status_code in (302, 303)
    print("PASS  logged-in user redirected away from /signup")

    c.get("/logout")
    t = tok(c)
    r = c.post("/signup", data={"_csrf": t, "email": "user@test.com",
                                "password": "password123", "confirm": "password123"})
    assert b"already registered" in r.data
    print("PASS  duplicate email rejected when logged out")

    t = tok(c)
    r = c.post("/login", data={"_csrf": t, "email": "user@test.com", "password": "wrong"})
    assert b"Incorrect email or password" in r.data
    print("PASS  wrong password rejected")

    t = tok(c)
    r = c.post("/signup", data={"_csrf": t, "email": "admin@test.com",
                                "password": "password123", "confirm": "password123"})
    assert r.status_code in (302, 303)
    r = c.get("/admin")
    assert r.status_code == 403
    print("PASS  signing up with ADMIN_EMAIL does NOT grant admin (no web backdoor)")

    assert db.set_admin("admin@test.com", True) == 1
    r = c.get("/admin")
    assert r.status_code == 200 and b"Usage dashboard" in r.data
    print("PASS  server-side grant promotes to admin; dashboard opens")

    row = db.get_user_by_email("user@test.com")
    assert row and row["password_hash"] != "password123" and len(row["password_hash"]) > 40
    print("PASS  passwords stored hashed, not plaintext")

    # BYOK free allowance
    uid = row["id"]
    c.get("/logout")
    t = tok(c)
    c.post("/login", data={"_csrf": t, "email": "user@test.com", "password": "password123"})
    for i in range(appmod.config.FREE_DAILY_QUERIES):
        db.log_query(uid, f"seed{i}", "a", 1, 100, ok=True, own_key=False)
    t = tok(c)
    r = c.post("/", data={"_csrf": t, "question": "one more please", "user_api_key": ""})
    assert b"free searches for" in r.data
    print("PASS  free allowance blocks the next shared-key search (no Groq call)")

    before = db.queries_today(uid)
    db.log_query(uid, "byok", "a", 1, 100, ok=True, own_key=True)
    assert db.queries_today(uid) == before
    print("PASS  own-key queries do not count against the free allowance")

    # semantic answer cache + feedback
    from src import qcache
    cid = qcache.store("What Dice does U-Net get on the pancreas?", "**U-Net** reports 0.81 Dice.", 2, 1200)
    hit = qcache.find("What Dice does U-Net achieve on pancreas?")
    assert hit and hit[0]["id"] == cid
    print(f"PASS  semantic cache reuses a near-identical question (similarity {hit[1]:.3f})")

    assert qcache.find("What is the capital of France?") is None
    print("PASS  unrelated question does not hit the cache")

    db.cache_vote(cid, up=False); db.cache_vote(cid, up=False)
    assert qcache.find("What Dice does U-Net achieve on pancreas?") is None
    print("PASS  net-downvoted answer is no longer reused")

    t = tok(c)
    r = c.post("/feedback", data={"_csrf": t, "cache_id": str(cid), "vote": "up"})
    assert r.status_code == 200 and r.get_json().get("ok")
    print("PASS  feedback route records a vote")

    # security hardening
    db.create_user("verify@test.com", "password123")
    assert db.get_user_by_email("verify@test.com")["email_verified"] == 0
    vtok = security.make_token("verify-email", "verify@test.com")
    assert c.get(f"/verify/{vtok}").status_code in (302, 303)
    assert db.get_user_by_email("verify@test.com")["email_verified"] == 1
    print("PASS  email verification token verifies the account")

    assert c.get("/verify/not-a-real-token").status_code in (302, 303)
    print("PASS  invalid verification token is rejected")

    rtok = security.make_token("reset-password", "verify@test.com")
    assert c.get(f"/reset/{rtok}").status_code == 200
    t = tok(c)
    r = c.post(f"/reset/{rtok}", data={"_csrf": t, "password": "newpassword1", "confirm": "newpassword1"})
    assert r.status_code in (302, 303)
    assert db.verify_password(db.get_user_by_email("verify@test.com"), "newpassword1")
    print("PASS  password reset token sets a new password")

    assert security.read_token("reset-password", rtok, -1) is None
    assert security.read_token("verify-email", rtok, 3600) is None
    print("PASS  expired and wrong-purpose tokens are rejected")

    assert security.safe_next("/history") == "/history"
    assert security.safe_next("//evil.com") is None
    assert security.safe_next("https://evil.com") is None
    print("PASS  open-redirect guard blocks external next targets")

    clean = security.sanitize_html('<p>ok</p><script>alert(1)</script><a href="javascript:alert(2)">x</a>')
    assert "<script" not in clean and "javascript:" not in clean and "ok" in clean
    print("PASS  HTML sanitizer strips script and javascript: URLs")

    r = c.get("/login")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
    assert "Content-Security-Policy" in r.headers
    print("PASS  security headers set (nosniff, frame-deny, CSP)")

    security.limiter.enabled = True
    got_429 = False
    for _ in range(15):
        tt = tok(c)
        if c.post("/login", data={"_csrf": tt, "email": "user@test.com", "password": "wrong"}).status_code == 429:
            got_429 = True
            break
    security.limiter.enabled = False
    assert got_429, "expected a 429 after repeated login attempts"
    print("PASS  login rate limiting returns 429 after repeated attempts")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run()
