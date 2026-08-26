"""Grant or revoke admin, from the server shell only.

Admin cannot be self-assigned through the website: signup always creates a
regular user, and this command is the only way to grant admin. It requires shell
access to the machine, so only the operator can run it. The target must already
have signed up.

  python -m src.manage_admin list
  python -m src.manage_admin grant  you@email.com
  python -m src.manage_admin revoke someone@email.com

With no email, grant/revoke fall back to ADMIN_EMAIL from the environment.
"""
from __future__ import annotations

import argparse

from src import config, db


def main() -> None:
    db.init_db()
    ap = argparse.ArgumentParser(description="Grant or revoke admin (server-side only).")
    ap.add_argument("action", choices=["grant", "revoke", "list"])
    ap.add_argument("email", nargs="?", help="target email (defaults to ADMIN_EMAIL)")
    args = ap.parse_args()

    if args.action == "list":
        con = db._conn()
        rows = con.execute("SELECT email, is_admin, created_at FROM users ORDER BY created_at").fetchall()
        con.close()
        if not rows:
            print("No users yet.")
        for r in rows:
            print(f"{'[admin]' if r['is_admin'] else '       '}  {r['email']}  ({r['created_at']})")
        return

    email = (args.email or config.ADMIN_EMAIL or "").strip().lower()
    if not email:
        print("Provide an email, or set ADMIN_EMAIL in the environment.")
        return
    changed = db.set_admin(email, args.action == "grant")
    if changed:
        verb = "granted to" if args.action == "grant" else "revoked from"
        print(f"Admin {verb} {email}.")
    else:
        print(f"No user with email {email}. They must sign up first, then run this.")


if __name__ == "__main__":
    main()
