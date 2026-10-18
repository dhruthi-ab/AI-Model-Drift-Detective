"""
ChurnGuard AI - Lightweight SQLite Authentication & Telemetry Server
"""

import http.server
import socketserver
import json
import sqlite3
import hashlib
import os
import datetime
from urllib.parse import urlparse

PORT = 3000
DB_FILE = os.path.join(os.path.dirname(__file__), "churnguard_users.db")
STATIC_DIR = os.path.dirname(__file__)


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at TEXT NOT NULL,
        last_login TEXT,
        login_count INTEGER DEFAULT 0
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS auth_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        action TEXT NOT NULL,
        status TEXT NOT NULL,
        ip_address TEXT,
        timestamp TEXT NOT NULL
    )
    """)
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        seed_users = [
            ("Sarah Lin", "sarah.lin@mlops.corp", hashlib.sha256("DemoSecure2026!".encode()).hexdigest(), "Lead MLOps Engineer"),
            ("Alex Rivera", "alex.rivera@datascience.corp", hashlib.sha256("DemoSecure2026!".encode()).hexdigest(), "Senior Data Scientist"),
            ("Admin User", "admin@churnguard.ai", hashlib.sha256("AdminRoot2026!".encode()).hexdigest(), "Security Admin")
        ]
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cursor.executemany("""
            INSERT INTO users (name, email, password_hash, role, created_at, last_login, login_count)
            VALUES (?, ?, ?, ?, ?, ?, 1)
        """, [(u[0], u[1], u[2], u[3], now, now) for u in seed_users])

    conn.commit()
    conn.close()


class ChurnGuardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def _send_json(self, status_code, data):
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            self._send_json(400, {"success": False, "error": "Invalid JSON body"})
            return

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        client_ip = self.client_address[0]

        if parsed.path == "/api/auth/register":
            name = payload.get("name", "").strip()
            email = payload.get("email", "").strip().lower()
            password = payload.get("password", "").strip()
            role = payload.get("role", "MLOps Engineer").strip()

            if not email or not password or not name:
                self._send_json(400, {"success": False, "error": "Name, email and password required"})
                conn.close()
                return

            password_hash = hashlib.sha256(password.encode()).hexdigest()
            try:
                cursor.execute("""
                    INSERT INTO users (name, email, password_hash, role, created_at, last_login, login_count)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                """, (name, email, password_hash, role, now, now))
                user_id = cursor.lastrowid
                conn.commit()
                self._send_json(201, {"success": True, "user": {"id": user_id, "name": name, "email": email, "role": role}})
            except sqlite3.IntegrityError:
                self._send_json(409, {"success": False, "error": "Email already exists"})
            finally:
                conn.close()
            return

        elif parsed.path == "/api/auth/login":
            email = payload.get("email", "").strip().lower()
            password = payload.get("password", "").strip()
            password_hash = hashlib.sha256(password.encode()).hexdigest()

            cursor.execute("SELECT id, name, email, role, password_hash, login_count FROM users WHERE email = ?", (email,))
            row = cursor.fetchone()

            if row and row[4] == password_hash:
                new_count = (row[5] or 0) + 1
                cursor.execute("UPDATE users SET last_login = ?, login_count = ? WHERE id = ?", (now, new_count, row[0]))
                conn.commit()
                self._send_json(200, {"success": True, "user": {"id": row[0], "name": row[1], "email": row[2], "role": row[3]}})
            else:
                self._send_json(401, {"success": False, "error": "Invalid email or password"})
            conn.close()
            return

        conn.close()
        self._send_json(404, {"error": "Not found"})

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/auth/users":
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, email, role, created_at, last_login, login_count FROM users ORDER BY id DESC")
            rows = cursor.fetchall()
            conn.close()
            self._send_json(200, {"success": True, "users": [{ "id": r[0], "name": r[1], "email": r[2], "role": r[3], "createdAt": r[4], "lastLogin": r[5], "loginCount": r[6]} for r in rows]})
            return
        super().do_GET()


if __name__ == "__main__":
    init_db()
    with socketserver.TCPServer(("", PORT), ChurnGuardHandler) as httpd:
        print(f"🚀 Running on http://localhost:{PORT}")
        httpd.serve_forever()
