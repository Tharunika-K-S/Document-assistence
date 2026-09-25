from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import sqlite3
import os
from datetime import datetime

auth_bp = Blueprint("auth", __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def current_user():
    user_id = session.get("user_id")

    if not user_id:
        return None

    conn = get_db()
    user = conn.execute(
        "SELECT id, name, email, created_at FROM users WHERE id = ?",
        (user_id,)
    ).fetchone()
    conn.close()

    return user


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please login to continue.", "info")
            return redirect(url_for("auth.login", next=request.path))

        return view(*args, **kwargs)

    return wrapped


@auth_bp.app_context_processor
def inject_current_user():
    return {"logged_in_user": current_user()}


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("auth.account"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please enter email and password.", "error")
            return render_template("login.html")

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            session["user_email"] = user["email"]

            next_url = request.args.get("next") or url_for("index")
            if not next_url.startswith("/"):
                next_url = url_for("index")

            return redirect(next_url)

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("auth.account"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not name or not email or not password:
            flash("Please fill all required fields.", "error")
            return render_template("register.html")

        if len(name) < 2:
            flash("Please enter a valid name.", "error")
            return render_template("register.html")

        if "@" not in email or "." not in email:
            flash("Please enter a valid email address.", "error")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must contain at least 6 characters.", "error")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        conn = get_db()

        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?",
            (email,)
        ).fetchone()

        if existing:
            conn.close()
            flash("An account with this email already exists.", "error")
            return render_template("register.html")

        password_hash = generate_password_hash(password)

        cursor = conn.execute(
            """
            INSERT INTO users
                (name, email, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                name,
                email,
                password_hash,
                datetime.now().isoformat(timespec="seconds")
            )
        )

        conn.commit()
        user_id = cursor.lastrowid
        conn.close()

        session.clear()
        session["user_id"] = user_id
        session["user_name"] = name
        session["user_email"] = email

        return redirect(url_for("auth.account"))

    return render_template("register.html")


@auth_bp.route("/account")
def account():
    user = current_user()

    if not user:
        return redirect(url_for("auth.login"))

    return render_template("account.html", user=user)


@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))
