"""
FinPilot AI — Premium Banking Analytics Platform
Single-file Streamlit application (IDBI Innovate Hackathon)

Run with: streamlit run app.py
"""

####################################
# Imports
####################################
import os
import io
import re
import sqlite3
import hashlib
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Optional / soft-dependency imports — the app must never crash on missing packages
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False

try:
    from mistralai.client import Mistral
    MISTRAL_SDK_AVAILABLE = True
except ImportError:
    try:
        # Older/alternate mistralai versions expose Mistral at the top level
        from mistralai import Mistral
        MISTRAL_SDK_AVAILABLE = True
    except ImportError:
        MISTRAL_SDK_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import plotly.io as pio
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False


####################################
# Configuration
####################################
APP_NAME = "FinPilot AI"
DB_PATH = "finpilot.db"

DEMO_EMAIL = "admin@example.com"
DEMO_PASSWORD = "password123"

COLORS = {
    "primary": "#046A38",
    "secondary": "#1F7A8C",
    "accent": "#00C896",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "background": "#F5F7FA",
    "card": "#FFFFFF",
}

DEFAULT_SETTINGS = {
    "theme": "Light",
    "currency": "INR",
    "forecast_months": 6,
    "groq_model": "llama-3.3-70b-versatile",
    "enable_ai": True,
}

TRANSACTION_CATEGORIES = [
    "Salary", "Rent", "Food", "Shopping", "Travel", "Utilities",
    "Insurance", "EMI", "Investment", "Healthcare", "Entertainment", "Other",
]

CURRENCY_SYMBOLS = {"INR": "\u20b9", "USD": "$", "EUR": "\u20ac", "GBP": "\u00a3"}

# ---- Expanded data model -------------------------------------------------
# Each entry: table_name -> list of "column TYPE" definitions (id/user_id/created_at
# are added automatically). New tables auto-initialize in init_database().
ADDITIONAL_TABLES = {
    "income_sources": ["source TEXT", "amount REAL", "frequency TEXT", "date TEXT"],
    "savings_accounts": ["name TEXT", "amount REAL", "date TEXT"],
    "investments": ["inv_type TEXT", "name TEXT", "amount REAL", "current_value REAL", "date TEXT"],
    "loans": ["loan_type TEXT", "principal REAL", "interest_rate REAL", "tenure_months INTEGER",
              "emi REAL", "start_date TEXT"],
    "goals": ["name TEXT", "category TEXT", "target_amount REAL", "current_amount REAL", "target_date TEXT"],
    "emergency_fund": ["amount REAL", "date TEXT"],
    "assets": ["name TEXT", "category TEXT", "value REAL", "date TEXT"],
    "liabilities": ["name TEXT", "category TEXT", "amount REAL", "date TEXT"],
    "recurring_bills": ["name TEXT", "amount REAL", "due_day INTEGER", "category TEXT"],
    "credit_cards": ["name TEXT", "limit_amount REAL", "outstanding REAL"],
    "insurance": ["policy_type TEXT", "provider TEXT", "premium REAL", "coverage REAL", "renewal_date TEXT"],
    "tax_saving": ["instrument TEXT", "amount REAL", "financial_year TEXT"],
    "networth_history": ["date TEXT", "net_worth REAL", "assets REAL", "liabilities REAL"],
}

INVESTMENT_TYPES = ["Mutual Fund", "Stocks", "Gold", "Fixed Deposit", "PPF", "NPS", "Crypto", "Other"]
GOAL_CATEGORIES = ["Car", "House", "Vacation", "Emergency Fund", "Education", "Marriage", "Retirement", "Other"]
ASSET_CATEGORIES = ["Real Estate", "Vehicle", "Gold/Jewellery", "Cash", "Bank Balance", "Electronics", "Other"]
LIABILITY_CATEGORIES = ["Home Loan", "Car Loan", "Personal Loan", "Credit Card Debt", "Education Loan", "Other"]
INSURANCE_TYPES = ["Life", "Health", "Vehicle", "Home", "Term"]

# Keyword map used for automatic category detection on CSV import / free-text entry.
CATEGORY_KEYWORDS = {
    "Salary": ["salary", "payroll", "wages", "stipend"],
    "Rent": ["rent", "landlord", "lease"],
    "Food": ["food", "restaurant", "swiggy", "zomato", "grocery", "cafe", "dining"],
    "Shopping": ["amazon", "flipkart", "myntra", "mall", "shopping", "store"],
    "Travel": ["uber", "ola", "flight", "irctc", "travel", "fuel", "petrol", "cab"],
    "Utilities": ["electricity", "water bill", "gas bill", "broadband", "wifi", "utility", "recharge"],
    "Insurance": ["insurance", "premium", "lic", "policy"],
    "EMI": ["emi", "loan installment", "loan emi"],
    "Investment": ["mutual fund", "sip", "stocks", "investment", "ppf", "nps", "zerodha", "groww"],
    "Healthcare": ["hospital", "pharmacy", "doctor", "medical", "clinic", "medicine"],
    "Entertainment": ["netflix", "spotify", "movie", "prime video", "hotstar", "entertainment"],
}

st.set_page_config(
    page_title=APP_NAME,
    page_icon="\U0001f4a0",
    layout="wide",
    initial_sidebar_state="expanded",
)


####################################
# Custom CSS
####################################
def inject_custom_css():
    """Inject premium banking-grade CSS (glassmorphism, gradients, cards)."""
    st.markdown(f"""
    <link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}
        h1, h2, h3, h4, .brand-title {{
            font-family: 'Sora', sans-serif !important;
        }}
        .stApp {{
            background: linear-gradient(135deg, {COLORS['background']} 0%, #E9EEF4 100%);
        }}
        #MainMenu, footer, header {{visibility: hidden;}}

        section[data-testid="stSidebar"] {{
            background: rgba(255, 255, 255, 0.65);
            backdrop-filter: blur(18px);
            border-right: 1px solid rgba(255,255,255,0.4);
        }}

        .glass-card {{
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(14px);
            border-radius: 18px;
            padding: 22px 24px;
            box-shadow: 0 8px 32px rgba(4, 106, 56, 0.08);
            border: 1px solid rgba(255,255,255,0.5);
            transition: transform 0.25s ease, box-shadow 0.25s ease;
            margin-bottom: 14px;
            animation: fadeInUp 0.4s ease;
        }}
        .glass-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 14px 40px rgba(4, 106, 56, 0.16);
        }}
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .metric-label {{
            font-size: 13px;
            color: #6B7280;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .metric-value {{
            font-size: 26px;
            font-weight: 700;
            color: #111827;
            margin-top: 4px;
        }}
        .metric-delta-pos {{ color: {COLORS['accent']}; font-weight: 600; font-size: 13px; }}
        .metric-delta-neg {{ color: {COLORS['danger']}; font-weight: 600; font-size: 13px; }}

        .stButton > button {{
            background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.55rem 1.4rem;
            font-weight: 600;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px rgba(4, 106, 56, 0.25);
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 22px rgba(4, 106, 56, 0.35);
        }}

        .hero {{
            text-align: center;
            padding: 70px 20px 40px 20px;
        }}
        .hero-title {{
            font-size: 52px;
            font-weight: 800;
            background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['accent']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .hero-sub {{
            font-size: 18px;
            color: #4B5563;
            max-width: 640px;
            margin: 14px auto 0 auto;
        }}
        .feature-card {{
            background: white;
            border-radius: 16px;
            padding: 26px;
            text-align: center;
            box-shadow: 0 6px 20px rgba(0,0,0,0.06);
            transition: transform 0.2s ease;
            height: 100%;
        }}
        .feature-card:hover {{ transform: translateY(-6px); }}
        .feature-icon {{ font-size: 34px; margin-bottom: 10px; }}

        .badge-good {{ background: rgba(0,200,150,0.15); color: {COLORS['accent']}; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: 12px;}}
        .badge-warn {{ background: rgba(245,158,11,0.15); color: {COLORS['warning']}; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: 12px;}}
        .badge-bad {{ background: rgba(239,68,68,0.15); color: {COLORS['danger']}; padding: 4px 12px; border-radius: 999px; font-weight: 600; font-size: 12px;}}

        .progress-track {{
            background: rgba(0,0,0,0.06);
            border-radius: 999px;
            height: 10px;
            width: 100%;
            overflow: hidden;
            margin-top: 6px;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['accent']});
            transition: width 0.6s ease;
        }}
        .skeleton {{
            background: linear-gradient(90deg, #eee 25%, #f5f5f5 50%, #eee 75%);
            background-size: 200% 100%;
            animation: shimmer 1.4s infinite;
            border-radius: 12px;
            height: 90px;
        }}
        @keyframes shimmer {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 14px;
            overflow: hidden;
        }}
    </style>
    """, unsafe_allow_html=True)


def plotly_theme():
    """Consistent modern Plotly layout template."""
    return dict(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", color="#374151"),
        colorway=[COLORS["primary"], COLORS["secondary"], COLORS["accent"],
                  COLORS["warning"], COLORS["danger"], "#8B5CF6"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


def progress_bar(label, pct, sub=""):
    """Renders a label + percentage + native Streamlit progress bar + caption.
    Uses st.progress() (a real widget) instead of a hand-rolled HTML/CSS bar —
    raw injected <div> width styling does not reliably re-render on every
    Streamlit rerun, which was causing the bar to visually 'stick' even when
    the underlying percentage had changed."""
    pct_clamped = max(0.0, min(100.0, float(pct)))
    color = COLORS["accent"] if pct_clamped >= 66 else COLORS["warning"] if pct_clamped >= 33 else COLORS["danger"]
    st.markdown(
        f'<div style="display:flex; justify-content:space-between; font-size:13px; '
        f'color:#374151; margin-bottom:2px;"><span>{label}</span>'
        f'<span style="font-weight:600; color:{color};">{pct_clamped:.0f}%</span></div>',
        unsafe_allow_html=True,
    )
    st.progress(pct_clamped / 100.0)
    if sub:
        st.markdown(
            f'<div style="font-size:12px; color:#6B7280; margin-top:-8px; margin-bottom:12px;">{sub}</div>',
            unsafe_allow_html=True,
        )


####################################
# Database Functions
####################################
@contextmanager
def get_conn():
    """Context-managed SQLite connection with row factory."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _table_columns(conn, table):
    try:
        return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _migrate_schema(conn):
    """Add columns to existing tables that predate a feature (safe, additive only)."""
    cols = _table_columns(conn, "transactions")
    if cols and "merchant" not in cols:
        conn.execute("ALTER TABLE transactions ADD COLUMN merchant TEXT")
    if cols and "is_duplicate" not in cols:
        conn.execute("ALTER TABLE transactions ADD COLUMN is_duplicate INTEGER DEFAULT 0")


def _init_additional_tables(conn):
    """Create every table declared in ADDITIONAL_TABLES if it does not already exist."""
    c = conn.cursor()
    for table, columns in ADDITIONAL_TABLES.items():
        cols_sql = ", ".join(columns)
        c.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                {cols_sql},
                created_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)


def init_database():
    """Create tables if they do not already exist, and seed demo data once."""
    try:
        with get_conn() as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    name TEXT,
                    created_at TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    date TEXT,
                    category TEXT,
                    description TEXT,
                    amount REAL,
                    type TEXT,
                    merchant TEXT,
                    is_duplicate INTEGER DEFAULT 0,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS budgets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    category TEXT,
                    amount REAL,
                    month TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    title TEXT,
                    content TEXT,
                    created_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    user_id INTEGER PRIMARY KEY,
                    theme TEXT,
                    currency TEXT,
                    forecast_months INTEGER,
                    groq_model TEXT,
                    enable_ai INTEGER,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)

            _init_additional_tables(conn)
            _migrate_schema(conn)

            c.execute("SELECT id FROM users WHERE email = ?", (DEMO_EMAIL,))
            row = c.fetchone()
            if row is None:
                pw_hash = hash_password(DEMO_PASSWORD)
                c.execute(
                    "INSERT INTO users (email, password_hash, name, created_at) VALUES (?, ?, ?, ?)",
                    (DEMO_EMAIL, pw_hash, "Demo User", datetime.now().isoformat()),
                )
                user_id = c.lastrowid
                c.execute(
                    """INSERT INTO settings (user_id, theme, currency, forecast_months, groq_model, enable_ai)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (user_id, DEFAULT_SETTINGS["theme"], DEFAULT_SETTINGS["currency"],
                     DEFAULT_SETTINGS["forecast_months"], DEFAULT_SETTINGS["groq_model"], 1),
                )
                seed_demo_transactions(conn, user_id)
                seed_demo_extended_data(conn, user_id)
        return True
    except sqlite3.Error as e:
        st.error(f"Database initialization failed: {e}")
        return False


def seed_demo_transactions(conn, user_id):
    """Generate ~9 months of realistic synthetic transactions for the demo user."""
    rng = np.random.default_rng(42)
    c = conn.cursor()
    start = datetime.now() - timedelta(days=270)
    rows = []
    merchants = {
        "Salary": "Employer Pvt Ltd", "Rent": "Landlord", "Food": "Swiggy/Zomato",
        "Shopping": "Amazon", "Travel": "Uber", "Utilities": "State Electricity Board",
        "Insurance": "LIC", "EMI": "HDFC Bank", "Investment": "Zerodha",
        "Healthcare": "Apollo Pharmacy", "Entertainment": "Netflix",
    }
    for m in range(9):
        month_date = start + timedelta(days=30 * m)
        rows.append((user_id, month_date.strftime("%Y-%m-%d"), "Salary", "Monthly Salary",
                     float(rng.normal(85000, 2000)), "income", merchants["Salary"]))
        expense_plan = {
            "Rent": rng.normal(18000, 500),
            "Food": rng.normal(8000, 1200),
            "Shopping": rng.normal(5000, 2000),
            "Travel": rng.normal(3000, 1500),
            "Utilities": rng.normal(2500, 400),
            "Insurance": rng.normal(2000, 100),
            "EMI": rng.normal(9000, 200),
            "Investment": rng.normal(7000, 1500),
            "Healthcare": rng.normal(1500, 800),
            "Entertainment": rng.normal(2000, 900),
        }
        for cat, amt in expense_plan.items():
            day = month_date + timedelta(days=int(rng.integers(1, 27)))
            rows.append((user_id, day.strftime("%Y-%m-%d"), cat, f"{cat} expense",
                        float(max(amt, 200)), "expense", merchants.get(cat, "")))
    c.executemany(
        "INSERT INTO transactions (user_id, date, category, description, amount, type, merchant) "
        "VALUES (?,?,?,?,?,?,?)",
        rows,
    )


def seed_demo_extended_data(conn, user_id):
    """Seed the new tables (investments, loans, goals, assets, liabilities, etc.) with realistic demo data."""
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")

    investments = [
        ("Mutual Fund", "Axis Bluechip Fund", 60000, 71000),
        ("Stocks", "Nifty 50 Basket", 40000, 46500),
        ("Gold", "Digital Gold", 15000, 16200),
        ("PPF", "Public Provident Fund", 50000, 54000),
        ("Fixed Deposit", "SBI FD 3yr", 100000, 106000),
    ]
    c.executemany(
        "INSERT INTO investments (user_id, inv_type, name, amount, current_value, date, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        [(user_id, t, n, a, cv, today, today) for t, n, a, cv in investments],
    )

    c.execute(
        "INSERT INTO loans (user_id, loan_type, principal, interest_rate, tenure_months, emi, start_date, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (user_id, "Home Loan", 2500000, 8.5, 180, 9000, today, today),
    )

    goals = [
        ("New Car", "Car", 800000, 220000, (datetime.now() + timedelta(days=365)).strftime("%Y-%m-%d")),
        ("Emergency Buffer", "Emergency Fund", 300000, 140000, (datetime.now() + timedelta(days=200)).strftime("%Y-%m-%d")),
        ("Dream Vacation", "Vacation", 150000, 40000, (datetime.now() + timedelta(days=150)).strftime("%Y-%m-%d")),
    ]
    c.executemany(
        "INSERT INTO goals (user_id, name, category, target_amount, current_amount, target_date, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        [(user_id, n, cat, t, cur, d, today) for n, cat, t, cur, d in goals],
    )

    c.execute("INSERT INTO emergency_fund (user_id, amount, date, created_at) VALUES (?,?,?,?)",
              (user_id, 140000, today, today))

    assets = [("Apartment", "Real Estate", 4500000), ("Car", "Vehicle", 600000),
              ("Gold Jewellery", "Gold/Jewellery", 250000)]
    c.executemany(
        "INSERT INTO assets (user_id, name, category, value, date, created_at) VALUES (?,?,?,?,?,?)",
        [(user_id, n, cat, v, today, today) for n, cat, v in assets],
    )

    liabilities = [("Home Loan Outstanding", "Home Loan", 2200000), ("Credit Card Dues", "Credit Card Debt", 18000)]
    c.executemany(
        "INSERT INTO liabilities (user_id, name, category, amount, date, created_at) VALUES (?,?,?,?,?,?)",
        [(user_id, n, cat, a, today, today) for n, cat, a in liabilities],
    )

    c.execute("INSERT INTO credit_cards (user_id, name, limit_amount, outstanding, created_at) VALUES (?,?,?,?,?)",
              (user_id, "HDFC Regalia", 200000, 18000, today))

    c.execute(
        "INSERT INTO insurance (user_id, policy_type, provider, premium, coverage, renewal_date, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (user_id, "Term", "LIC", 12000, 10000000, (datetime.now() + timedelta(days=300)).strftime("%Y-%m-%d"), today),
    )


def verify_user(email: str, password: str):
    """Return user row dict if credentials match, else None."""
    try:
        with get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = c.fetchone()
            if row and row["password_hash"] == hash_password(password):
                return dict(row)
        return None
    except sqlite3.Error as e:
        st.error(f"Login failed due to a database error: {e}")
        return None


@st.cache_data(ttl=30, show_spinner=False)
def load_transactions(user_id: int) -> pd.DataFrame:
    """Load all transactions for a user as a DataFrame (cached briefly for performance)."""
    try:
        with get_conn() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM transactions WHERE user_id = ? ORDER BY date DESC",
                conn, params=(user_id,),
            )
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            if "merchant" not in df.columns:
                df["merchant"] = ""
            df["merchant"] = df["merchant"].fillna("")
        return df
    except sqlite3.Error as e:
        st.error(f"Could not load transactions: {e}")
        return pd.DataFrame(columns=["id", "user_id", "date", "category", "description", "amount", "type", "merchant"])


def add_transaction(user_id, date, category, description, amount, ttype, merchant=None):
    merchant = merchant if merchant else extract_merchant(description)
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO transactions (user_id, date, category, description, amount, type, merchant) "
                "VALUES (?,?,?,?,?,?,?)",
                (user_id, date, category, description, amount, ttype, merchant),
            )
        load_transactions.clear()
        return True
    except sqlite3.Error as e:
        st.error(f"Could not add transaction: {e}")
        return False


def update_transaction(txn_id, date, category, description, amount, ttype):
    try:
        with get_conn() as conn:
            conn.execute(
                "UPDATE transactions SET date=?, category=?, description=?, amount=?, type=?, merchant=? WHERE id=?",
                (date, category, description, amount, ttype, extract_merchant(description), txn_id),
            )
        load_transactions.clear()
        return True
    except sqlite3.Error as e:
        st.error(f"Could not update transaction: {e}")
        return False


def delete_transaction(txn_id):
    try:
        with get_conn() as conn:
            conn.execute("DELETE FROM transactions WHERE id=?", (txn_id,))
        load_transactions.clear()
        return True
    except sqlite3.Error as e:
        st.error(f"Could not delete transaction: {e}")
        return False


def bulk_delete_transactions(txn_ids):
    if not txn_ids:
        return False
    try:
        with get_conn() as conn:
            qmarks = ",".join("?" * len(txn_ids))
            conn.execute(f"DELETE FROM transactions WHERE id IN ({qmarks})", txn_ids)
        load_transactions.clear()
        return True
    except sqlite3.Error as e:
        st.error(f"Could not bulk delete transactions: {e}")
        return False


def bulk_update_category(txn_ids, new_category):
    if not txn_ids:
        return False
    try:
        with get_conn() as conn:
            qmarks = ",".join("?" * len(txn_ids))
            conn.execute(f"UPDATE transactions SET category=? WHERE id IN ({qmarks})", [new_category] + txn_ids)
        load_transactions.clear()
        return True
    except sqlite3.Error as e:
        st.error(f"Could not bulk update transactions: {e}")
        return False


def get_settings(user_id):
    try:
        with get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM settings WHERE user_id=?", (user_id,))
            row = c.fetchone()
            if row:
                d = dict(row)
                d["enable_ai"] = bool(d["enable_ai"])
                return d
    except sqlite3.Error as e:
        st.warning(f"Could not load settings, using defaults: {e}")
    return {"user_id": user_id, **DEFAULT_SETTINGS}


def save_settings(user_id, theme, currency, forecast_months, groq_model, enable_ai):
    try:
        with get_conn() as conn:
            conn.execute("""
                INSERT INTO settings (user_id, theme, currency, forecast_months, groq_model, enable_ai)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(user_id) DO UPDATE SET
                    theme=excluded.theme, currency=excluded.currency,
                    forecast_months=excluded.forecast_months,
                    groq_model=excluded.groq_model, enable_ai=excluded.enable_ai
            """, (user_id, theme, currency, forecast_months, groq_model, int(enable_ai)))
        return True
    except sqlite3.Error as e:
        st.error(f"Could not save settings: {e}")
        return False


def reset_database():
    """Wipe and re-initialize the entire database."""
    try:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        clear_all_caches()
        init_database()
        return True
    except OSError as e:
        st.error(f"Could not reset database: {e}")
        return False


def clear_all_caches():
    load_transactions.clear()
    get_records.clear()


####################################
# Generic CRUD for the expanded data model (income, investments, loans, goals, ...)
####################################
def add_record(table, user_id, **fields):
    """Insert a row into any table declared in ADDITIONAL_TABLES."""
    try:
        cols = list(fields.keys())
        placeholders = ",".join(["?"] * len(cols))
        with get_conn() as conn:
            conn.execute(
                f"INSERT INTO {table} (user_id, {', '.join(cols)}, created_at) "
                f"VALUES (?, {placeholders}, ?)",
                [user_id, *[fields[c] for c in cols], datetime.now().isoformat()],
            )
        get_records.clear()
        return True
    except sqlite3.Error as e:
        st.error(f"Could not save record to {table}: {e}")
        return False


@st.cache_data(ttl=30, show_spinner=False)
def get_records(table: str, user_id: int) -> pd.DataFrame:
    """Load all rows for a user from any table declared in ADDITIONAL_TABLES."""
    try:
        with get_conn() as conn:
            return pd.read_sql_query(f"SELECT * FROM {table} WHERE user_id=? ORDER BY id DESC", conn, params=(user_id,))
    except sqlite3.Error as e:
        st.error(f"Could not load {table}: {e}")
        return pd.DataFrame()


def update_record(table, record_id, **fields):
    try:
        set_clause = ", ".join(f"{c}=?" for c in fields)
        with get_conn() as conn:
            conn.execute(f"UPDATE {table} SET {set_clause} WHERE id=?", [*fields.values(), record_id])
        get_records.clear()
        return True
    except sqlite3.Error as e:
        st.error(f"Could not update {table}: {e}")
        return False


def delete_record(table, record_id):
    try:
        with get_conn() as conn:
            conn.execute(f"DELETE FROM {table} WHERE id=?", (record_id,))
        get_records.clear()
        return True
    except sqlite3.Error as e:
        st.error(f"Could not delete from {table}: {e}")
        return False


####################################
# Smart transaction detection (auto-category, merchant, duplicates)
####################################
def auto_detect_category(description: str, fallback: str = "Other") -> str:
    if not description:
        return fallback
    text = str(description).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category
    return fallback


def extract_merchant(description: str) -> str:
    """Best-effort merchant name extraction from a free-text description."""
    if not description:
        return ""
    text = re.sub(r"[^A-Za-z0-9&/ ]", " ", str(description)).strip()
    words = [w for w in text.split() if len(w) > 2]
    return " ".join(words[:2]).title() if words else str(description)[:20].title()


def find_duplicate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Flag likely duplicate transactions (same date, amount and category)."""
    if df.empty:
        return df
    dup_mask = df.duplicated(subset=["date", "amount", "category", "type"], keep=False)
    return df[dup_mask].sort_values(["date", "amount"])


def enrich_csv_rows(csv_df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detect category/merchant/type for a freshly uploaded CSV of transactions."""
    df = csv_df.copy()
    if "category" not in df.columns:
        df["category"] = ""
    if "type" not in df.columns:
        df["type"] = ""
    if "merchant" not in df.columns:
        df["merchant"] = ""

    missing_cat = df["category"].isna() | (df["category"].astype(str).str.strip() == "")
    df.loc[missing_cat, "category"] = df.loc[missing_cat, "description"].apply(auto_detect_category)

    missing_type = df["type"].isna() | (df["type"].astype(str).str.strip() == "")
    df.loc[missing_type, "type"] = df.loc[missing_type, "category"].apply(
        lambda c: "income" if c == "Salary" else "expense"
    )

    missing_merchant = df["merchant"].isna() | (df["merchant"].astype(str).str.strip() == "")
    df.loc[missing_merchant, "merchant"] = df.loc[missing_merchant, "description"].apply(extract_merchant)

    return df


####################################
# Authentication
####################################
def login_page():
    inject_custom_css()
    st.markdown(f"""
    <div class="hero">
        <div class="brand-title" style="font-size:40px;">\U0001f4a0 {APP_NAME}</div>
        <div class="hero-sub">AI-powered banking analytics &mdash; sign in to your financial cockpit.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        with st.container(border=True):
            st.markdown("#### Sign in")
            email = st.text_input("Email", value="", placeholder="admin@example.com")
            password = st.text_input("Password", type="password", placeholder="********")
            st.caption(f"Demo credentials \u2014 **{DEMO_EMAIL}** / **{DEMO_PASSWORD}**")
            login_clicked = st.button("Log in", width='stretch')

            if login_clicked:
                if not email or not password:
                    st.warning("Please enter both email and password.")
                else:
                    with st.spinner("Verifying credentials..."):
                        user = verify_user(email.strip(), password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.session_state.page = "Dashboard"
                        st.success("Login successful. Redirecting...")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")

    render_landing_features()


def render_landing_features():
    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
    features = [
        ("\U0001f4ca", "Live Dashboards", "Real-time view of balance, cash flow and spending patterns."),
        ("\U0001f9e0", "AI Financial Advisor", "Ask natural-language questions, powered by Groq."),
        ("\U0001f4c8", "Smart Forecasting", "Prophet-driven projections for income, expenses & savings."),
        ("\U0001f3e6", "Loan Eligibility", "ML-based risk scoring using scikit-learn."),
        ("\U0001f49a", "Health Score", "A single number that captures your financial wellbeing."),
        ("\U0001f4c4", "AI Reports", "Auto-generated, downloadable PDF financial reports."),
        ("\U0001f4b0", "Net Worth Tracker", "Assets, liabilities and investments in one live number."),
        ("\U0001f3af", "Goal Tracker", "Track savings goals with completion forecasts."),
        ("\U0001f4c8", "Investment Analyzer", "Allocation, diversification and growth across asset classes."),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div style="font-weight:700; font-size:16px; margin-bottom:6px;">{title}</div>
                <div style="color:#6B7280; font-size:13.5px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)


def logout():
    for key in ["authenticated", "user", "page"]:
        st.session_state.pop(key, None)
    st.rerun()


####################################
# Financial Analytics
####################################
def compute_summary(df: pd.DataFrame) -> dict:
    """
    Single source of truth for all headline financial numbers (avoids duplicate calc).

    NOTE on income/expenses: these are ALL-TIME totals across every transaction on record
    (used for lifetime balance and savings-rate). For anything compared against a monthly
    figure (EMIs, loan affordability, 'Monthly Income/Expenses' dashboard cards) use
    avg_monthly_income / avg_monthly_expenses / avg_monthly_savings instead — mixing the
    two was the root cause of inflated loan eligibility and mislabeled dashboard KPIs.
    """
    if df.empty:
        return dict(income=0.0, expenses=0.0, savings=0.0, balance=0.0,
                     investments=0.0, emergency_fund=0.0, savings_rate=0.0,
                     months_tracked=0, avg_monthly_income=0.0, avg_monthly_expenses=0.0,
                     avg_monthly_savings=0.0)

    income = df.loc[df["type"] == "income", "amount"].sum()
    expenses = df.loc[df["type"] == "expense", "amount"].sum()
    investments = df.loc[(df["type"] == "expense") & (df["category"] == "Investment"), "amount"].sum()
    savings = income - expenses
    savings_rate = (savings / income * 100) if income > 0 else 0.0
    emergency_fund = max(savings, 0) * 0.6
    balance = savings

    months_tracked = max(df["date"].dt.to_period("M").nunique(), 1)
    avg_monthly_income = float(income) / months_tracked
    avg_monthly_expenses = float(expenses) / months_tracked
    avg_monthly_savings = avg_monthly_income - avg_monthly_expenses

    return dict(
        income=float(income), expenses=float(expenses), savings=float(savings),
        balance=float(balance), investments=float(investments),
        emergency_fund=float(emergency_fund), savings_rate=float(savings_rate),
        months_tracked=int(months_tracked), avg_monthly_income=avg_monthly_income,
        avg_monthly_expenses=avg_monthly_expenses, avg_monthly_savings=avg_monthly_savings,
    )


def monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate income/expense/savings by month."""
    if df.empty:
        return pd.DataFrame(columns=["month", "income", "expenses", "savings"])
    tmp = df.copy()
    tmp["month"] = tmp["date"].dt.to_period("M").dt.to_timestamp()
    grouped = tmp.groupby(["month", "type"])["amount"].sum().unstack(fill_value=0)
    for col in ["income", "expense"]:
        if col not in grouped.columns:
            grouped[col] = 0
    grouped["savings"] = grouped["income"] - grouped["expense"]
    grouped = grouped.rename(columns={"expense": "expenses"}).reset_index()
    return grouped[["month", "income", "expenses", "savings"]]


def format_currency(amount, currency="INR"):
    symbol = CURRENCY_SYMBOLS.get(currency, "\u20b9")
    return f"{symbol}{amount:,.0f}"


def metric_card(label, value, delta=None, delta_positive=True):
    delta_html = ""
    if delta is not None:
        cls = "metric-delta-pos" if delta_positive else "metric-delta-neg"
        arrow = "\u25b2" if delta_positive else "\u25bc"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    st.markdown(f"""
    <div class="glass-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


####################################
# Net Worth Engine
####################################
def compute_net_worth(user_id, summary):
    """Aggregate assets, liabilities, investments and loans into a single live net-worth figure."""
    assets_df = get_records("assets", user_id)
    liabilities_df = get_records("liabilities", user_id)
    investments_df = get_records("investments", user_id)
    loans_df = get_records("loans", user_id)

    total_assets = float(assets_df["value"].sum()) if not assets_df.empty else 0.0
    total_liabilities = float(liabilities_df["amount"].sum()) if not liabilities_df.empty else 0.0
    total_investments = float(investments_df["current_value"].sum()) if not investments_df.empty else 0.0
    total_loans = float(loans_df["principal"].sum()) if not loans_df.empty else 0.0

    net_worth = total_assets + total_investments + summary["balance"] - total_liabilities - total_loans

    return dict(
        assets=total_assets, liabilities=total_liabilities, investments=total_investments,
        loans=total_loans, balance=summary["balance"], net_worth=float(net_worth),
    )


def snapshot_networth(user_id, networth: dict):
    """Log today's net worth to history (one snapshot per day)."""
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        with get_conn() as conn:
            existing = conn.execute(
                "SELECT id FROM networth_history WHERE user_id=? AND date=?", (user_id, today)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE networth_history SET net_worth=?, assets=?, liabilities=? WHERE id=?",
                    (networth["net_worth"], networth["assets"], networth["liabilities"], existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO networth_history (user_id, date, net_worth, assets, liabilities, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (user_id, today, networth["net_worth"], networth["assets"], networth["liabilities"],
                     datetime.now().isoformat()),
                )
        get_records.clear()
    except sqlite3.Error:
        pass


def get_networth_history(user_id) -> pd.DataFrame:
    df = get_records("networth_history", user_id)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    return df


####################################
# Emergency Fund Analysis
####################################
def analyze_emergency_fund(user_id, df, summary):
    ef_df = get_records("emergency_fund", user_id)
    current_fund = float(ef_df["amount"].sum()) if not ef_df.empty else summary["emergency_fund"]

    months = df["date"].dt.to_period("M").nunique() if not df.empty else 1
    avg_monthly_expense = (summary["expenses"] / max(months, 1)) if summary["expenses"] else 0
    recommended = avg_monthly_expense * 6
    months_covered = (current_fund / avg_monthly_expense) if avg_monthly_expense > 0 else 0
    gap = max(recommended - current_fund, 0)
    progress_pct = min((current_fund / recommended * 100) if recommended > 0 else 0, 100)

    if months_covered >= 6:
        risk = "Low"
    elif months_covered >= 3:
        risk = "Medium"
    else:
        risk = "High"

    return dict(current_fund=current_fund, recommended=recommended, months_covered=round(months_covered, 1),
                gap=gap, progress_pct=progress_pct, risk=risk, avg_monthly_expense=avg_monthly_expense)


####################################
# Goal Tracker
####################################
def analyze_goals(user_id, summary):
    goals_df = get_records("goals", user_id)
    if goals_df.empty:
        return []

    monthly_savings = max(summary["savings"], 0) / 1.0  # treated as available monthly surplus proxy
    results = []
    for _, g in goals_df.iterrows():
        target = float(g["target_amount"] or 0)
        current = float(g["current_amount"] or 0)
        remaining = max(target - current, 0)
        progress_pct = min((current / target * 100) if target > 0 else 0, 100)

        try:
            target_date = pd.to_datetime(g["target_date"])
            days_remaining = max((target_date - pd.Timestamp.now()).days, 0)
            months_remaining = max(days_remaining / 30.4, 0.1)
        except Exception:
            months_remaining = 12

        required_monthly = remaining / months_remaining if months_remaining > 0 else remaining
        if monthly_savings <= 0:
            completion_probability = 10.0 if remaining <= 0 else 5.0
        else:
            ratio = monthly_savings / required_monthly if required_monthly > 0 else 2.0
            completion_probability = float(np.clip(ratio * 60, 5, 98))

        results.append(dict(
            id=int(g["id"]), name=g["name"], category=g["category"], target=target, current=current,
            remaining=remaining, progress_pct=progress_pct, target_date=g["target_date"],
            months_remaining=round(months_remaining, 1), required_monthly=round(required_monthly, 0),
            completion_probability=round(completion_probability, 1),
        ))
    return results


####################################
# Investment Analyzer
####################################
def analyze_investments(user_id):
    df = get_records("investments", user_id)
    if df.empty:
        return dict(total_invested=0.0, total_value=0.0, growth_pct=0.0, allocation={}, diversification_score=0,
                     rows=df)

    total_invested = float(df["amount"].sum())
    total_value = float(df["current_value"].sum())
    growth_pct = ((total_value - total_invested) / total_invested * 100) if total_invested > 0 else 0.0

    allocation = df.groupby("inv_type")["current_value"].sum().to_dict()
    distinct_types = df["inv_type"].nunique()
    diversification_score = float(np.clip(distinct_types / len(INVESTMENT_TYPES) * 100, 0, 100))

    return dict(total_invested=total_invested, total_value=total_value, growth_pct=round(growth_pct, 2),
                allocation=allocation, diversification_score=round(diversification_score, 1), rows=df)


####################################
# Loan Analyzer (expands the ML-based eligibility model below)
####################################
def analyze_loans(user_id, summary):
    df = get_records("loans", user_id)
    total_emi = float(df["emi"].sum()) if not df.empty else 0.0
    total_principal = float(df["principal"].sum()) if not df.empty else 0.0

    monthly_income = max(summary.get("avg_monthly_income", summary["income"]), 1)
    debt_ratio = total_emi / monthly_income * 100
    safe_emi = max(monthly_income * 0.40 - total_emi, 0)

    # Rough affordability estimate for a new loan at ~9% p.a. over 5 years (annuity formula)
    annual_rate, years = 0.09, 5
    r = annual_rate / 12
    n = years * 12
    if r > 0 and safe_emi > 0:
        max_loan = safe_emi * (1 - (1 + r) ** (-n)) / r
    else:
        max_loan = 0.0

    if debt_ratio < 20:
        risk = "Low"
    elif debt_ratio < 40:
        risk = "Medium"
    else:
        risk = "High"

    affordability_index = float(np.clip(100 - debt_ratio * 1.5, 0, 100))

    return dict(total_emi=total_emi, total_principal=total_principal, debt_ratio=round(debt_ratio, 1),
                safe_emi=round(safe_emi, 0), max_loan=round(max_loan, 0), risk=risk,
                affordability_index=round(affordability_index, 1), rows=df)


####################################
# Smart Budget Engine
####################################
def save_budget_allocations(user_id, month, allocations: dict):
    try:
        with get_conn() as conn:
            for category, amount in allocations.items():
                existing = conn.execute(
                    "SELECT id FROM budgets WHERE user_id=? AND category=? AND month=?",
                    (user_id, category, month),
                ).fetchone()
                if existing:
                    conn.execute("UPDATE budgets SET amount=? WHERE id=?", (amount, existing["id"]))
                else:
                    conn.execute(
                        "INSERT INTO budgets (user_id, category, amount, month) VALUES (?,?,?,?)",
                        (user_id, category, amount, month),
                    )
        return True
    except sqlite3.Error as e:
        st.error(f"Could not save budget: {e}")
        return False


def get_budget_allocations(user_id, month) -> dict:
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT category, amount FROM budgets WHERE user_id=? AND month=?", (user_id, month)
            ).fetchall()
        return {r["category"]: r["amount"] for r in rows}
    except sqlite3.Error:
        return {}


def compute_budget_utilization(user_id, df):
    """Compare this month's actual spend per category against saved budget allocations."""
    month = datetime.now().strftime("%Y-%m")
    budget = get_budget_allocations(user_id, month)
    if not budget or df.empty:
        return dict(total_budget=0.0, total_spent=0.0, utilization_pct=0.0, remaining=0.0, by_category={})

    this_month = df[(df["type"] == "expense") & (df["date"].dt.strftime("%Y-%m") == month)]
    spent_by_cat = this_month.groupby("category")["amount"].sum().to_dict()

    total_budget = sum(budget.values())
    total_spent = sum(spent_by_cat.get(cat, 0) for cat in budget)
    utilization_pct = (total_spent / total_budget * 100) if total_budget > 0 else 0.0

    by_category = {cat: dict(budget=amt, spent=spent_by_cat.get(cat, 0)) for cat, amt in budget.items()}

    return dict(total_budget=total_budget, total_spent=total_spent, utilization_pct=round(utilization_pct, 1),
                remaining=total_budget - total_spent, by_category=by_category)


####################################
# Risk, Stability & Credit metrics
####################################
def compute_debt_to_income(summary, loan_analysis):
    monthly_income = max(summary.get("avg_monthly_income", summary["income"]), 1)
    return round(loan_analysis["total_emi"] / monthly_income * 100, 1)


def compute_financial_stability_index(health, networth, emergency, budget_util):
    """Composite 0-100 index blending health score, net worth trend proxy, EF coverage, budget discipline."""
    ef_component = np.clip(emergency["months_covered"] / 6 * 100, 0, 100)
    nw_component = 100 if networth["net_worth"] > 0 else 30
    budget_component = 100 - min(budget_util.get("utilization_pct", 0), 100) if budget_util.get("total_budget") else 60
    return round(float(np.clip(
        health["score"] * 0.4 + ef_component * 0.25 + nw_component * 0.15 + budget_component * 0.2, 0, 100
    )), 1)


def compute_risk_score(debt_to_income, emergency, health):
    """Higher = riskier. Inverse-ish of stability."""
    risk = (debt_to_income * 0.4) + (max(6 - emergency["months_covered"], 0) * 8) + (100 - health["score"]) * 0.3
    return round(float(np.clip(risk, 0, 100)), 1)


def compute_credit_utilization(user_id):
    df = get_records("credit_cards", user_id)
    if df.empty:
        return 0.0
    total_limit = df["limit_amount"].sum()
    total_outstanding = df["outstanding"].sum()
    return round((total_outstanding / total_limit * 100) if total_limit > 0 else 0.0, 1)


def compute_goal_completion_pct(goals_list):
    if not goals_list:
        return 0.0
    return round(float(np.mean([g["progress_pct"] for g in goals_list])), 1)


def quick_predict_next_month(df, metric):
    """Lightweight single-step-ahead prediction reused for dashboard 'Predicted X' cards."""
    ms = monthly_series(df)
    if len(ms) < 2:
        return float(ms[metric].iloc[-1]) if len(ms) == 1 else 0.0
    y = ms[metric].values
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, 1)
    return float(np.polyval(coeffs, len(y)))


####################################
# Financial Health Score
####################################
def calculate_health_score(df: pd.DataFrame, summary: dict) -> dict:
    """
    Weighted composite score (0-100) from six pillars:
    savings rate, debt ratio, investment ratio, emergency fund,
    expense stability, income stability.
    """
    if df.empty or summary["income"] == 0:
        return dict(score=0, status="No Data", components={}, suggestions=["Add transactions to compute your score."])

    income = summary["income"]
    expenses = summary["expenses"]

    savings_rate = max(min(summary["savings_rate"], 100), -100)
    savings_score = np.clip((savings_rate + 20) / 60 * 100, 0, 100)

    debt = df.loc[(df["type"] == "expense") & (df["category"] == "EMI"), "amount"].sum()
    debt_ratio = (debt / income * 100) if income > 0 else 0
    debt_score = np.clip(100 - debt_ratio * 2.5, 0, 100)

    invest_ratio = (summary["investments"] / income * 100) if income > 0 else 0
    invest_score = np.clip(invest_ratio * 5, 0, 100)

    months = df["date"].dt.to_period("M").nunique() if not df.empty else 1
    avg_monthly_expense = expenses / max(months, 1)
    emergency_months = (summary["emergency_fund"] / avg_monthly_expense) if avg_monthly_expense > 0 else 0
    emergency_score = np.clip(emergency_months / 6 * 100, 0, 100)

    ms = monthly_series(df)
    if len(ms) > 1 and ms["expenses"].mean() > 0:
        expense_cv = ms["expenses"].std() / ms["expenses"].mean()
        expense_stability_score = np.clip(100 - expense_cv * 100, 0, 100)
    else:
        expense_stability_score = 70

    if len(ms) > 1 and ms["income"].mean() > 0:
        income_cv = ms["income"].std() / ms["income"].mean()
        income_stability_score = np.clip(100 - income_cv * 150, 0, 100)
    else:
        income_stability_score = 70

    weights = dict(savings=0.25, debt=0.2, invest=0.15, emergency=0.2, exp_stability=0.1, inc_stability=0.1)
    final_score = (
        savings_score * weights["savings"] +
        debt_score * weights["debt"] +
        invest_score * weights["invest"] +
        emergency_score * weights["emergency"] +
        expense_stability_score * weights["exp_stability"] +
        income_stability_score * weights["inc_stability"]
    )
    final_score = round(float(np.clip(final_score, 0, 100)), 1)

    if final_score >= 75:
        status = "Excellent"
    elif final_score >= 55:
        status = "Good"
    elif final_score >= 35:
        status = "Fair"
    else:
        status = "Needs Attention"

    suggestions = []
    if savings_score < 50:
        suggestions.append("Try to raise your savings rate \u2014 aim for at least 20% of income.")
    if debt_score < 50:
        suggestions.append("Your EMI/debt burden is high relative to income; consider consolidating loans.")
    if invest_score < 40:
        suggestions.append("Increase monthly investments to build long-term wealth.")
    if emergency_score < 50:
        suggestions.append("Build an emergency fund covering at least 6 months of expenses.")
    if expense_stability_score < 50:
        suggestions.append("Your expenses vary significantly month to month \u2014 try tighter budgeting.")
    if not suggestions:
        suggestions.append("Great job! Keep maintaining your current financial discipline.")

    return dict(
        score=final_score, status=status,
        components=dict(
            savings=round(savings_score, 1), debt=round(debt_score, 1), invest=round(invest_score, 1),
            emergency=round(emergency_score, 1), exp_stability=round(expense_stability_score, 1),
            inc_stability=round(income_stability_score, 1),
        ),
        suggestions=suggestions,
    )


def render_gauge(score: float, title="Financial Health Score"):
    color = COLORS["accent"] if score >= 65 else COLORS["warning"] if score >= 40 else COLORS["danger"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "rgba(239,68,68,0.15)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.15)"},
                {"range": [70, 100], "color": "rgba(0,200,150,0.15)"},
            ],
        },
    ))
    fig.update_layout(**plotly_theme(), height=280)
    st.plotly_chart(fig, width='stretch')


####################################
# Loan Prediction
####################################
@st.cache_resource(show_spinner=False)
def train_loan_model():
    """Train a lightweight logistic regression on synthetic but realistic data."""
    if not SKLEARN_AVAILABLE:
        return None, None

    rng = np.random.default_rng(7)
    n = 2000
    age = rng.integers(21, 60, n)
    salary = rng.normal(60000, 25000, n).clip(10000, None)
    credit_score = rng.integers(300, 900, n)
    emi = rng.normal(8000, 6000, n).clip(0, None)
    existing_loans = rng.integers(0, 4, n)

    debt_to_income = emi / (salary + 1)
    logit = (
        0.006 * (credit_score - 600)
        - 4.0 * debt_to_income
        - 0.35 * existing_loans
        + 0.00002 * (salary - 40000)
        - 0.01 * (age - 40)
    )
    prob = 1 / (1 + np.exp(-logit))
    approved = (prob + rng.normal(0, 0.12, n)) > 0.5

    X = np.column_stack([age, salary, credit_score, emi, existing_loans])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=500)
    model.fit(X_scaled, approved.astype(int))
    return model, scaler


def predict_loan_eligibility(age, salary, credit_score, emi, existing_loans):
    model, scaler = train_loan_model()
    if model is None:
        return dict(eligible=False, probability=0.0, risk_score=100,
                     error="scikit-learn is not installed; loan prediction unavailable.")
    X = np.array([[age, salary, credit_score, emi, existing_loans]])
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0][1])
    eligible = prob >= 0.5
    risk_score = round((1 - prob) * 100, 1)

    suggestions = []
    if credit_score < 650:
        suggestions.append("Improve your credit score above 700 for better approval odds.")
    if emi / max(salary, 1) > 0.4:
        suggestions.append("Your EMI-to-income ratio is high; consider reducing existing debt.")
    if existing_loans >= 2:
        suggestions.append("Multiple existing loans reduce eligibility \u2014 consider consolidation.")
    if not suggestions:
        suggestions.append("Your profile looks strong for loan approval.")

    return dict(eligible=eligible, probability=round(prob * 100, 1), risk_score=risk_score,
                 suggestions=suggestions, error=None)


####################################
# Forecasting
####################################
def run_forecast(df: pd.DataFrame, metric: str, periods_months: int = 6):
    """
    Forecast a monthly metric ('income', 'expenses', or 'savings') using Prophet
    if available, otherwise fall back to a simple linear trend so the app never breaks.
    """
    ms = monthly_series(df)
    if ms.empty or len(ms) < 3:
        return None, "Not enough historical data to forecast (need at least 3 months)."

    hist = ms[["month", metric]].rename(columns={"month": "ds", metric: "y"})

    if PROPHET_AVAILABLE:
        try:
            m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
            m.fit(hist)
            future = m.make_future_dataframe(periods=periods_months, freq="MS")
            forecast = m.predict(future)
            result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            return result, None
        except Exception as e:
            return _linear_fallback_forecast(hist, periods_months), f"Prophet failed ({e}); used linear fallback."
    else:
        return _linear_fallback_forecast(hist, periods_months), "Prophet not installed; used linear fallback."


def _linear_fallback_forecast(hist: pd.DataFrame, periods_months: int) -> pd.DataFrame:
    x = np.arange(len(hist))
    y = hist["y"].values
    coeffs = np.polyfit(x, y, 1) if len(x) > 1 else [0, y.mean()]
    future_x = np.arange(len(hist) + periods_months)
    yhat = np.polyval(coeffs, future_x)
    resid_std = np.std(y - np.polyval(coeffs, x)) if len(x) > 1 else abs(y.mean()) * 0.1
    last_date = hist["ds"].max()
    dates = list(hist["ds"]) + [last_date + pd.DateOffset(months=i) for i in range(1, periods_months + 1)]
    return pd.DataFrame({
        "ds": dates, "yhat": yhat,
        "yhat_lower": yhat - 1.5 * resid_std, "yhat_upper": yhat + 1.5 * resid_std,
    })


def render_forecast_chart(forecast_df, hist_df, metric_label, currency):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df["month"], y=hist_df[hist_df.columns[1]],
        mode="lines+markers", name="Actual", line=dict(color=COLORS["primary"], width=3),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["yhat"],
        mode="lines", name="Forecast", line=dict(color=COLORS["accent"], width=3, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=list(forecast_df["ds"]) + list(forecast_df["ds"])[::-1],
        y=list(forecast_df["yhat_upper"]) + list(forecast_df["yhat_lower"])[::-1],
        fill="toself", fillcolor="rgba(0,200,150,0.12)", line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band", showlegend=True,
    ))
    fig.update_layout(**plotly_theme(), title=f"{metric_label} Forecast", height=400)
    st.plotly_chart(fig, width='stretch')


####################################
# Groq Client
####################################
def get_effective_api_key(session_key: str, env_var: str):
    """
    Resolve an API key: a key pasted into Settings for this session takes
    priority over the environment variable. Nothing is ever written to disk.
    """
    session_val = st.session_state.get(session_key, "").strip()
    if session_val:
        return session_val
    return os.environ.get(env_var, "")


@st.cache_resource(show_spinner=False)
def _build_groq_client(api_key: str):
    if not api_key or not GROQ_SDK_AVAILABLE:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None


def get_groq_client():
    api_key = get_effective_api_key("groq_api_key_override", "GROQ_API_KEY")
    return _build_groq_client(api_key)


def build_financial_context(user_id, df, summary, health, forecast_note, currency):
    """
    Build ONE structured, numbers-grounded snapshot of the user's entire financial
    profile so the AI never has to guess or hallucinate values. This is reused by
    the AI Summary card, the AI Advisor chat, and the Reports generator.
    """
    networth = compute_net_worth(user_id, summary)
    emergency = analyze_emergency_fund(user_id, df, summary)
    goals = analyze_goals(user_id, summary)
    investments = analyze_investments(user_id)
    loans = analyze_loans(user_id, summary)
    budget_util = compute_budget_utilization(user_id, df)
    debt_to_income = compute_debt_to_income(summary, loans)
    risk_score = compute_risk_score(debt_to_income, emergency, health)

    top_categories = (
        df[df["type"] == "expense"].groupby("category")["amount"].sum().sort_values(ascending=False).head(5)
        if not df.empty else pd.Series(dtype=float)
    )
    top_merchants = (
        df[(df["type"] == "expense") & (df["merchant"] != "")].groupby("merchant")["amount"].sum()
        .sort_values(ascending=False).head(5) if not df.empty and "merchant" in df.columns else pd.Series(dtype=float)
    )

    goals_txt = "; ".join(
        f"{g['name']} ({g['progress_pct']:.0f}% complete, target {format_currency(g['target'], currency)})"
        for g in goals
    ) or "No goals set."

    context = f"""
FINANCIAL PROFILE (all figures are the user's real, current data — never invent numbers not shown here)

Income & Cash Flow (based on {summary['months_tracked']} month(s) of transaction history):
- Average Monthly Income: {format_currency(summary['avg_monthly_income'], currency)}
- Average Monthly Expenses: {format_currency(summary['avg_monthly_expenses'], currency)}
- Average Monthly Net Cash Flow: {format_currency(summary['avg_monthly_savings'], currency)}
- All-Time Total Income: {format_currency(summary['income'], currency)}
- All-Time Total Expenses: {format_currency(summary['expenses'], currency)}
- All-Time Net Savings: {format_currency(summary['savings'], currency)}
- Savings Rate: {summary['savings_rate']:.1f}%

Net Worth:
- Assets: {format_currency(networth['assets'], currency)}
- Liabilities: {format_currency(networth['liabilities'], currency)}
- Investments (current value): {format_currency(networth['investments'], currency)}
- Net Worth: {format_currency(networth['net_worth'], currency)}

Emergency Fund:
- Current Fund: {format_currency(emergency['current_fund'], currency)}
- Recommended (6 months expenses): {format_currency(emergency['recommended'], currency)}
- Months Covered: {emergency['months_covered']} | Risk: {emergency['risk']}

Investments:
- Total Invested: {format_currency(investments['total_invested'], currency)}
- Current Value: {format_currency(investments['total_value'], currency)}
- Growth: {investments['growth_pct']}% | Diversification Score: {investments['diversification_score']}/100

Loans & Debt:
- Total EMI Outstanding: {format_currency(loans['total_emi'], currency)}
- Debt-to-Income Ratio: {debt_to_income}%
- Loan Risk Level: {loans['risk']} | Safe Additional EMI Capacity: {format_currency(loans['safe_emi'], currency)}

Budget:
- This Month's Budget Utilization: {budget_util.get('utilization_pct', 0)}%

Goals: {goals_txt}

Financial Health Score: {health['score']}/100 ({health['status']})
Risk Score (higher = riskier): {risk_score}/100

Top Spending Categories: {", ".join(f"{k} ({format_currency(v, currency)})" for k, v in top_categories.items()) or "N/A"}
Top Merchants: {", ".join(f"{k} ({format_currency(v, currency)})" for k, v in top_merchants.items()) or "N/A"}

Forecast Note: {forecast_note}
""".strip()
    return context


def ask_ai_advisor(question, context, model_name):
    client = get_groq_client()
    if client is None:
        return ("\u26a0\ufe0f Groq API key not found or SDK not installed. Paste a key in "
                "Settings \u2192 API Keys, or set the `GROQ_API_KEY` environment variable, "
                "to enable the AI Advisor.")
    system_prompt = (
        "You are FinPilot AI, a professional, encouraging, and precise personal financial advisor "
        "for an Indian banking context. Use ONLY the numbers in the provided financial profile — "
        "never invent or estimate values that are not present. Give specific, actionable, "
        "numbers-grounded advice. Keep responses concise and structured with short headings or "
        "bullet points where useful. Never give definitive legal/tax advice — recommend consulting "
        "a professional for those matters."
    )
    user_prompt = f"{context}\n\nUser question: {question}"
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=700,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"\u26a0\ufe0f AI Advisor request failed: {e}"


####################################
# Mistral Client
####################################
@st.cache_resource(show_spinner=False)
def _build_mistral_client(api_key: str):
    try:
        return Mistral(api_key=api_key), None
    except Exception as e:
        return None, str(e)


def get_mistral_client():
    if not MISTRAL_SDK_AVAILABLE:
        return None, "missing_sdk", None
    api_key = get_effective_api_key("mistral_api_key_override", "MISTRAL_API_KEY")
    if not api_key:
        return None, "missing_key", None
    client, init_error = _build_mistral_client(api_key)
    if client is None:
        return None, "init_failed", init_error
    return client, None, None


def generate_ai_narrative(context, loan_info, forecast_note):
    """Ask Mistral for a narrative overlay (tone, framing, prioritized advice) on top of the
    deterministic, template-based report — so the report never depends on AI being configured."""
    client, reason, detail = get_mistral_client()
    if client is None:
        if reason == "missing_sdk":
            return ("\u26a0\ufe0f The `mistralai` Python package is not installed in this "
                    "environment. Run `pip install mistralai` and restart the app.")
        if reason == "missing_key":
            return ("\u26a0\ufe0f No Mistral API key found. Paste one in Settings \u2192 API Keys "
                    "(and click **Use these keys for this session**), or set the "
                    "`MISTRAL_API_KEY` environment variable.")
        return f"\u26a0\ufe0f Mistral client failed to initialize: {detail}"
    prompt = f"""
You are a senior financial advisor writing the narrative overlay for a client's financial audit report.
Using ONLY the data below (never invent numbers), write these sections, each 3-6 sentences:
Executive Summary, Behavioral Spending Analysis, Strengths, Weaknesses, Opportunities, Threats,
Top 10 Personalized Recommendations (as a numbered list).

{context}

Loan Analysis Data: {loan_info}
Forecast Note: {forecast_note}

Keep it professional, specific, and grounded strictly in the numbers given.
"""
    try:
        resp = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"\u26a0\ufe0f AI narrative generation failed: {e}"


####################################
# Full Report Generator (deterministic, always available — no AI required)
####################################
def _recommendation_bank(summary, health, networth, emergency, goals, investments, loans, budget_util, currency):
    """Generate a pool of personalized, numbers-grounded recommendations. Always returns >= 25 items
    by layering conditional advice with evergreen best-practice items scaled to the user's numbers."""
    recs = []
    if summary["savings_rate"] < 20:
        recs.append(f"Raise your savings rate from {summary['savings_rate']:.1f}% toward the recommended 20%+ "
                     f"by trimming discretionary categories (Shopping, Entertainment, Travel).")
    else:
        recs.append(f"Maintain your strong {summary['savings_rate']:.1f}% savings rate and consider directing the "
                     f"surplus into higher-growth investment instruments.")
    if emergency["months_covered"] < 6:
        recs.append(f"Grow your emergency fund from {emergency['months_covered']} months of coverage to the "
                     f"recommended 6 months — a gap of {format_currency(emergency['gap'], currency)}.")
    else:
        recs.append("Your emergency fund already covers 6+ months of expenses; consider allocating any further "
                     "surplus toward long-term investments rather than idle cash.")
    if investments["diversification_score"] < 60:
        recs.append(f"Diversify your investment portfolio (currently {investments['diversification_score']}/100 "
                     f"diversification score) across more asset classes such as equities, gold, and fixed income.")
    if investments["growth_pct"] < 5 and investments["total_invested"] > 0:
        recs.append(f"Your portfolio has grown {investments['growth_pct']}% — review underperforming holdings "
                     f"and rebalance toward better-performing asset classes.")
    if loans["debt_ratio"] > 35:
        recs.append(f"Your debt-to-income ratio of {loans['debt_ratio']}% is elevated; prioritize paying down "
                     f"high-interest liabilities before taking on new debt.")
    else:
        recs.append(f"Your debt-to-income ratio of {loans['debt_ratio']}% is healthy, leaving "
                     f"{format_currency(loans['safe_emi'], currency)} of safe additional EMI capacity if needed.")
    if budget_util.get("total_budget"):
        if budget_util["utilization_pct"] > 100:
            recs.append(f"You are {budget_util['utilization_pct']-100:.0f}% over your planned budget this month — "
                        f"review the categories driving the overage.")
        else:
            recs.append(f"You are tracking at {budget_util['utilization_pct']:.0f}% of your monthly budget — "
                        f"keep monitoring to stay within plan.")
    else:
        recs.append("Set up a monthly budget in the Budget Planner so spending can be tracked against a plan.")
    for g in goals[:5]:
        if g["completion_probability"] < 50:
            recs.append(f"Your '{g['name']}' goal is at {g['completion_probability']}% completion probability — "
                        f"increase monthly contributions to roughly {format_currency(g['required_monthly'], currency)} "
                        f"to stay on track for {g['target_date']}.")
        else:
            recs.append(f"Your '{g['name']}' goal is on track ({g['completion_probability']}% completion "
                        f"probability) — continue current contributions.")
    if networth["net_worth"] < 0:
        recs.append("Your net worth is currently negative; focus on debt reduction before increasing discretionary "
                     "spending or new large purchases.")
    else:
        recs.append(f"Your net worth of {format_currency(networth['net_worth'], currency)} is positive — "
                     f"continue building assets and paying down liabilities to accelerate growth.")

    evergreen = [
        "Automate transfers to savings and investment accounts on salary day to enforce consistency.",
        "Review recurring subscriptions quarterly and cancel unused ones.",
        "Maintain a credit utilization ratio below 30% to protect your credit score.",
        "Rebalance your investment portfolio at least once a year.",
        "Keep 3-6 months of expenses in a liquid, low-risk instrument for emergencies.",
        "Increase term life and health insurance coverage in line with income growth.",
        "Use tax-saving instruments (PPF, ELSS, NPS) to optimize your tax outgo before the financial year ends.",
        "Track every expense category monthly to catch lifestyle inflation early.",
        "Negotiate lower interest rates on existing loans if your credit score has improved.",
        "Set specific, time-bound targets for each financial goal rather than open-ended saving.",
        "Avoid new EMIs that push your debt-to-income ratio above 40%.",
        "Build a separate sinking fund for irregular annual expenses (insurance premiums, travel).",
        "Review your investment asset allocation against your risk tolerance and time horizon.",
        "Consolidate high-interest debts (credit cards, personal loans) into lower-interest options where possible.",
        "Revisit your budget allocations every quarter as income or expenses change.",
        "Keep a written record of financial goals and revisit progress monthly.",
        "Consider a systematic investment plan (SIP) to average out market volatility.",
        "Set up alerts for bill due dates to avoid late fees and credit score damage.",
    ]
    recs.extend(evergreen)
    seen, unique = set(), []
    for r in recs:
        if r not in seen:
            unique.append(r)
            seen.add(r)
    return unique[:25] if len(unique) >= 25 else unique


def generate_full_report_text(user_name, df, summary, health, networth, emergency, goals, investments,
                               loans, budget_util, forecast_note, currency, ai_narrative=""):
    """Deterministic, template-driven 1000-1500+ word financial audit report built entirely from the
    user's real computed data. Works even with no AI configured; AI narrative is layered on top if present."""
    today_str = datetime.now().strftime("%d %b %Y")
    top_categories = (
        df[df["type"] == "expense"].groupby("category")["amount"].sum().sort_values(ascending=False)
        if not df.empty else pd.Series(dtype=float)
    )
    debt_to_income = compute_debt_to_income(summary, loans)
    risk_score = compute_risk_score(debt_to_income, emergency, health)
    stability_index = compute_financial_stability_index(health, networth, emergency, budget_util)
    recs = _recommendation_bank(summary, health, networth, emergency, goals, investments, loans, budget_util, currency)

    swot_strengths, swot_weaknesses = [], []
    if summary["savings_rate"] >= 20:
        swot_strengths.append(f"A healthy savings rate of {summary['savings_rate']:.1f}%.")
    else:
        swot_weaknesses.append(f"A below-target savings rate of {summary['savings_rate']:.1f}%.")
    if emergency["months_covered"] >= 6:
        swot_strengths.append("A fully-funded emergency reserve.")
    else:
        swot_weaknesses.append(f"An emergency fund covering only {emergency['months_covered']} months of expenses.")
    if loans["debt_ratio"] < 35:
        swot_strengths.append(f"A manageable debt-to-income ratio of {loans['debt_ratio']}%.")
    else:
        swot_weaknesses.append(f"A high debt-to-income ratio of {loans['debt_ratio']}%.")
    if investments["diversification_score"] >= 60:
        swot_strengths.append("A well-diversified investment portfolio.")
    else:
        swot_weaknesses.append("Limited diversification across investment asset classes.")

    goals_section = "\n".join(
        f"- **{g['name']}** ({g['category']}): {format_currency(g['current'], currency)} of "
        f"{format_currency(g['target'], currency)} saved ({g['progress_pct']:.0f}%), target date {g['target_date']}, "
        f"completion probability {g['completion_probability']}%. Requires ~{format_currency(g['required_monthly'], currency)}/month."
        for g in goals
    ) or "No goals have been set yet. Consider adding specific savings goals in the Goals page."

    alloc_lines = "\n".join(
        f"- {k}: {format_currency(v, currency)}" for k, v in investments["allocation"].items()
    ) or "No investments recorded yet."

    cat_lines = "\n".join(
        f"- {cat}: {format_currency(amt, currency)} ({amt / summary['expenses'] * 100:.1f}% of total expenses)"
        for cat, amt in top_categories.items()
    ) if not top_categories.empty and summary["expenses"] > 0 else "No expense data available."

    report = f"""
# {APP_NAME} — Comprehensive Financial Audit Report

**Prepared for:** {user_name}
**Date:** {today_str}
**Financial Health Score:** {health['score']}/100 ({health['status']})
**Financial Stability Index:** {stability_index}/100
**Risk Score:** {risk_score}/100 (higher = riskier)

---

## 1. Executive Summary
This report presents a complete audit of {user_name}'s financial position based on {len(df)} recorded
transactions, current investment holdings, liabilities, and stated goals. Overall financial health is rated
**{health['status']}** with a composite score of **{health['score']}/100**. Net income stands at
{format_currency(summary['income'], currency)} against total expenses of {format_currency(summary['expenses'], currency)},
producing net savings of {format_currency(summary['savings'], currency)} and a savings rate of
{summary['savings_rate']:.1f}%. Net worth is currently {format_currency(networth['net_worth'], currency)}, and the
emergency fund covers {emergency['months_covered']} months of expenses against a 6-month target.

## 2. Overall Financial Health
The financial health score blends savings rate, debt burden, investment ratio, emergency fund coverage, and the
stability of both income and expenses over time. Component scores: Savings {health['components'].get('savings', 0)},
Debt {health['components'].get('debt', 0)}, Investment {health['components'].get('invest', 0)},
Emergency Fund {health['components'].get('emergency', 0)}, Expense Stability {health['components'].get('exp_stability', 0)},
Income Stability {health['components'].get('inc_stability', 0)}. A score above 75 is considered excellent; the current
score of {health['score']} places this profile in the **{health['status']}** band.

## 3. Income Analysis
Total recorded income is {format_currency(summary['income'], currency)}. Income stability contributes
{health['components'].get('inc_stability', 0)}/100 to the overall score, reflecting how consistent monthly income has
been across the transaction history.

## 4. Expense Analysis
Total expenses are {format_currency(summary['expenses'], currency)}. The leading spending categories are:
{cat_lines}
Expense stability contributes {health['components'].get('exp_stability', 0)}/100 to the overall score.

## 5. Cash Flow Analysis
Net cash flow (income minus expenses) is {format_currency(summary['savings'], currency)}. A positive and growing cash
flow is the foundation for both the emergency fund and long-term investment goals described below.

## 6. Savings Analysis
The current savings rate of {summary['savings_rate']:.1f}% {"meets" if summary['savings_rate'] >= 20 else "falls short of"}
the commonly recommended 20% benchmark. {"Maintaining this discipline will compound significantly over time." if summary['savings_rate'] >= 20 else "Incremental cuts to discretionary categories can close this gap."}

## 7. Investment Analysis
Total amount invested: {format_currency(investments['total_invested'], currency)}. Current portfolio value:
{format_currency(investments['total_value'], currency)} ({investments['growth_pct']}% growth). Diversification score:
{investments['diversification_score']}/100. Allocation by type:
{alloc_lines}

## 8. Loan & Debt Analysis
Total outstanding EMI: {format_currency(loans['total_emi'], currency)} per month against total principal of
{format_currency(loans['total_principal'], currency)}. Debt-to-income ratio: {debt_to_income}% (Risk: {loans['risk']}).
Safe additional EMI capacity: {format_currency(loans['safe_emi'], currency)}. Estimated maximum affordable new loan at
current safe EMI: {format_currency(loans['max_loan'], currency)}.

## 9. Budget Analysis
{"Budget utilization this month stands at " + str(budget_util.get('utilization_pct', 0)) + "% of a planned " + format_currency(budget_util.get('total_budget', 0), currency) + " budget, with " + format_currency(budget_util.get('remaining', 0), currency) + " remaining." if budget_util.get('total_budget') else "No monthly budget has been configured yet — set one up in the Budget Planner to unlock this analysis."}

## 10. Emergency Fund Analysis
Current fund: {format_currency(emergency['current_fund'], currency)}. Recommended fund (6 months of expenses):
{format_currency(emergency['recommended'], currency)}. Coverage: {emergency['months_covered']} months. Gap:
{format_currency(emergency['gap'], currency)}. Risk level: **{emergency['risk']}**.

## 11. Goal Tracking
{goals_section}

## 12. Net Worth Analysis
Assets: {format_currency(networth['assets'], currency)} | Liabilities: {format_currency(networth['liabilities'], currency)} |
Investments: {format_currency(networth['investments'], currency)} | Cash/Savings: {format_currency(networth['balance'], currency)}
**Net Worth: {format_currency(networth['net_worth'], currency)}**

## 13. Behavioral Spending Analysis
{"Spending is concentrated in " + str(top_categories.index[0]) + ", which represents " + f"{(top_categories.iloc[0] / summary['expenses'] * 100):.1f}%" + " of total expenses — a pattern worth monitoring for lifestyle creep." if not top_categories.empty and summary['expenses'] > 0 else "Insufficient expense history to identify behavioral spending patterns yet."}

## 14. Financial Risks
Key risk indicators: Risk Score {risk_score}/100, Debt-to-Income {debt_to_income}%, Emergency Fund Risk
**{emergency['risk']}**. {"These figures indicate elevated financial risk that should be addressed proactively." if risk_score > 50 else "Overall risk exposure is currently within an acceptable range."}

## 15. SWOT Analysis
**Strengths:** {"; ".join(swot_strengths) or "None identified from current data."}
**Weaknesses:** {"; ".join(swot_weaknesses) or "None identified from current data."}
**Opportunities:** Increasing SIP contributions, consolidating high-interest debt, and improving diversification.
**Threats:** Income volatility, unplanned large expenses, and insufficient insurance coverage.

## 16. Forecast Summary
{forecast_note}

## 17. Insurance Suggestions
Maintain term life coverage of at least 10-15x annual income and adequate health coverage for all dependents.
Review policy renewal dates annually to avoid coverage lapses.

## 18. Tax Saving Suggestions
Utilize Section 80C instruments (PPF, ELSS, EPF) up to the annual limit, and consider NPS for additional
tax-advantaged retirement savings.

## 19. Investment Suggestions
{"Given the current diversification score of " + str(investments['diversification_score']) + "/100, consider allocating fresh investments toward under-represented asset classes." if investments['diversification_score'] < 70 else "The current portfolio is well diversified; continue periodic rebalancing."}

## 20. Loan Suggestions
{"Consider consolidating or refinancing existing debt given a debt-to-income ratio of " + str(debt_to_income) + "%." if debt_to_income > 35 else "Current loan obligations are manageable relative to income."}

## 21. Emergency Fund Suggestions
{"Prioritize building the emergency fund by " + format_currency(emergency['gap'], currency) + " to reach the 6-month target." if emergency['gap'] > 0 else "Emergency fund target has been met — maintain current reserve levels."}

## 22. Top 25 Personalized Recommendations
{chr(10).join(f"{i+1}. {r}" for i, r in enumerate(recs))}

## 23. 30-Day Action Plan
1. Review and categorize all uncategorized transactions.
2. Set up (or revisit) a monthly budget covering every major category.
3. Automate a transfer of at least {summary['savings_rate']:.0f}% of income to savings on salary day.

## 24. 90-Day Action Plan
1. Close the emergency fund gap of {format_currency(emergency['gap'], currency)} through consistent monthly transfers.
2. Reassess investment allocation and rebalance toward target diversification.
3. Renegotiate or consolidate any high-interest debt identified in Section 8.

## 25. 180-Day Action Plan
1. Reach at least 4 months of emergency fund coverage.
2. Increase SIP/investment contributions in line with income growth.
3. Review insurance coverage against current income and dependents.

## 26. 365-Day Financial Roadmap
1. Achieve the full 6-month emergency fund target.
2. Bring debt-to-income ratio below 30%.
3. Reach or exceed a Financial Health Score of 75.
4. Make measurable progress on all active goals listed in Section 11.

## 27. Conclusion
{user_name}'s financial profile shows a **{health['status'].lower()}** overall health score of {health['score']}/100 and a
net worth of {format_currency(networth['net_worth'], currency)}. The priority actions are closing the emergency fund gap,
maintaining or improving the savings rate, and steadily reducing the debt-to-income ratio. Consistent execution of the
action plans above should measurably improve the Financial Health Score and Financial Stability Index over the coming
year.

## 28. Appendix — Summary Table
| Metric | Value |
|---|---|
| Income | {format_currency(summary['income'], currency)} |
| Expenses | {format_currency(summary['expenses'], currency)} |
| Savings Rate | {summary['savings_rate']:.1f}% |
| Net Worth | {format_currency(networth['net_worth'], currency)} |
| Emergency Fund Coverage | {emergency['months_covered']} months |
| Debt-to-Income Ratio | {debt_to_income}% |
| Financial Health Score | {health['score']}/100 |
| Financial Stability Index | {stability_index}/100 |
| Risk Score | {risk_score}/100 |
""".strip()

    if ai_narrative and not ai_narrative.startswith("\u26a0\ufe0f"):
        report += f"\n\n---\n\n## AI Advisor Narrative\n\n{ai_narrative}"

    return report


####################################
# PDF Generator
####################################
def _render_chart_image(fig):
    """Best-effort PNG export of a Plotly figure for PDF embedding; returns None if kaleido is unavailable."""
    if not KALEIDO_AVAILABLE:
        return None
    try:
        return pio.to_image(fig, format="png", width=900, height=420, scale=2)
    except Exception:
        return None


def _markdown_line_to_paragraph(line, styles):
    line = line.strip()
    if not line:
        return None
    line = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
    if line.startswith("# "):
        return Paragraph(line[2:], styles["h1"])
    if line.startswith("## "):
        return Paragraph(line[3:], styles["h2"])
    if line.startswith("|") or line.startswith("---"):
        return None  # tables are rendered separately via the summary Table below
    if line.startswith("- "):
        return Paragraph(f"&bull; {line[2:]}", styles["body"])
    return Paragraph(line, styles["body"])


def generate_pdf_report(user_name, report_text, summary, health, networth, currency, chart_fig=None):
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    base_styles = getSampleStyleSheet()
    styles = {
        "h1": ParagraphStyle("H1", parent=base_styles["Title"], textColor=rl_colors.HexColor(COLORS["primary"]), fontSize=20),
        "h2": ParagraphStyle("H2", parent=base_styles["Heading2"], textColor=rl_colors.HexColor(COLORS["secondary"]), spaceBefore=10),
        "body": base_styles["BodyText"],
    }

    story = [
        Paragraph(f"{APP_NAME} \u2014 Financial Audit Report", styles["h1"]),
        Paragraph(f"Prepared for: {user_name} | Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", styles["body"]),
        Spacer(1, 14),
    ]

    table_data = [
        ["Metric", "Value"],
        ["Income", format_currency(summary["income"], currency)],
        ["Expenses", format_currency(summary["expenses"], currency)],
        ["Savings Rate", f"{summary['savings_rate']:.1f}%"],
        ["Net Worth", format_currency(networth["net_worth"], currency)],
        ["Health Score", f"{health['score']}/100 ({health['status']})"],
    ]
    table = Table(table_data, colWidths=[7 * cm, 7 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(COLORS["primary"])),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#DDDDDD")),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#F5F7FA")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 16))

    if chart_fig is not None:
        img_bytes = _render_chart_image(chart_fig)
        if img_bytes:
            story.append(RLImage(io.BytesIO(img_bytes), width=16 * cm, height=7.5 * cm))
            story.append(Spacer(1, 14))

    for line in report_text.split("\n"):
        para = _markdown_line_to_paragraph(line, styles)
        if para is not None:
            story.append(para)
            story.append(Spacer(1, 4))

    try:
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return None


####################################
# Dashboard UI
####################################
def page_dashboard(user_id, currency):
    st.markdown("### \U0001f4ca Dashboard")
    with st.spinner("Loading your financial data..."):
        df = load_transactions(user_id)
        summary = compute_summary(df)
        health = calculate_health_score(df, summary)

    if df.empty:
        st.info("No transactions yet. Add some in the Transactions page to see your dashboard come alive.")
        return

    networth = compute_net_worth(user_id, summary)
    snapshot_networth(user_id, networth)
    emergency = analyze_emergency_fund(user_id, df, summary)
    goals = analyze_goals(user_id, summary)
    investments = analyze_investments(user_id)
    loans = analyze_loans(user_id, summary)
    budget_util = compute_budget_utilization(user_id, df)
    debt_to_income = compute_debt_to_income(summary, loans)
    stability_index = compute_financial_stability_index(health, networth, emergency, budget_util)
    risk_score = compute_risk_score(debt_to_income, emergency, health)
    credit_util = compute_credit_utilization(user_id)
    goal_completion = compute_goal_completion_pct(goals)

    # --- Row 1: core cash-flow KPIs ---
    # "Monthly" cards use the average-per-month figures (avg_monthly_income / avg_monthly_expenses),
    # NOT the all-time totals — mixing the two previously inflated these numbers by however many
    # months of history existed, which also cascaded into an inflated Loan Eligibility estimate below.
    cols = st.columns(4)
    with cols[0]:
        metric_card("Current Balance", format_currency(summary["balance"], currency))
    with cols[1]:
        metric_card("Monthly Income", format_currency(summary["avg_monthly_income"], currency))
    with cols[2]:
        metric_card("Monthly Expenses", format_currency(summary["avg_monthly_expenses"], currency))
    with cols[3]:
        metric_card("Net Cash Flow", format_currency(summary["avg_monthly_savings"], currency),
                     delta=f"{summary['savings_rate']:.1f}% savings rate", delta_positive=summary["avg_monthly_savings"] >= 0)

    # --- Row 2: health, worth & fund KPIs ---
    cols2 = st.columns(4)
    with cols2[0]:
        metric_card("Financial Health Score", f"{health['score']} / 100")
    with cols2[1]:
        metric_card("Net Worth", format_currency(networth["net_worth"], currency))
    with cols2[2]:
        metric_card("Emergency Fund", format_currency(emergency["current_fund"], currency),
                     delta=f"{emergency['months_covered']} mo covered", delta_positive=emergency["months_covered"] >= 6)
    with cols2[3]:
        metric_card("Investment Value", format_currency(investments["total_value"], currency),
                     delta=f"{investments['growth_pct']}%", delta_positive=investments["growth_pct"] >= 0)

    # --- Row 3: risk & efficiency KPIs ---
    loan_quick = predict_loan_eligibility(35, max(summary["avg_monthly_income"], 1), 700, loans["total_emi"], 1)
    cols3 = st.columns(4)
    with cols3[0]:
        metric_card("Loan Eligibility %", f"{loan_quick.get('probability', 0)}%")
    with cols3[1]:
        metric_card("Debt-to-Income Ratio", f"{debt_to_income}%")
    with cols3[2]:
        metric_card("Budget Utilization", f"{budget_util.get('utilization_pct', 0)}%")
    with cols3[3]:
        metric_card("Remaining Budget", format_currency(budget_util.get("remaining", 0), currency))

    # --- Row 4: forward-looking & composite KPIs ---
    predicted_savings = quick_predict_next_month(df, "savings")
    predicted_expenses = quick_predict_next_month(df, "expenses")
    cols4 = st.columns(4)
    with cols4[0]:
        metric_card("Financial Stability Index", f"{stability_index} / 100")
    with cols4[1]:
        metric_card("Risk Score", f"{risk_score} / 100")
    with cols4[2]:
        metric_card("Credit Utilization", f"{credit_util}%")
    with cols4[3]:
        metric_card("Goal Completion %", f"{goal_completion}%")

    cols5 = st.columns(3)
    with cols5[0]:
        metric_card("Predicted Savings (next mo.)", format_currency(predicted_savings, currency))
    with cols5[1]:
        metric_card("Predicted Expenses (next mo.)", format_currency(predicted_expenses, currency))
    with cols5[2]:
        metric_card("Predicted Cash Flow (next mo.)", format_currency(predicted_savings, currency))

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure(data=[go.Pie(
            labels=["Income", "Expenses"], values=[summary["income"], summary["expenses"]],
            hole=0.55, marker_colors=[COLORS["accent"], COLORS["danger"]],
        )])
        fig.update_layout(**plotly_theme(), title="Income vs Expense", height=340)
        st.plotly_chart(fig, width='stretch')

    with c2:
        cat_df = df[df["type"] == "expense"].groupby("category")["amount"].sum().reset_index()
        if not cat_df.empty:
            fig = px.pie(cat_df, names="category", values="amount", hole=0.5)
            fig.update_layout(**plotly_theme(), title="Expense by Category", height=340)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No expense data to categorize yet.")

    ms = monthly_series(df)
    if not ms.empty:
        c3, c4 = st.columns(2)
        with c3:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ms["month"], y=ms["income"], name="Income", marker_color=COLORS["accent"]))
            fig.add_trace(go.Bar(x=ms["month"], y=ms["expenses"], name="Expenses", marker_color=COLORS["danger"]))
            fig.update_layout(**plotly_theme(), title="Monthly Trend", barmode="group", height=340)
            st.plotly_chart(fig, width='stretch')
        with c4:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ms["month"], y=ms["savings"], mode="lines+markers",
                                      name="Savings", line=dict(color=COLORS["primary"], width=3)))
            fig.update_layout(**plotly_theme(), title="Savings Trend", height=340)
            st.plotly_chart(fig, width='stretch')

    c5, c6 = st.columns(2)
    with c5:
        nw_hist = get_networth_history(user_id)
        if not nw_hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nw_hist["date"], y=nw_hist["net_worth"], mode="lines+markers",
                                      name="Net Worth", line=dict(color=COLORS["primary"], width=3),
                                      fill="tozeroy", fillcolor="rgba(4,106,56,0.08)"))
            fig.update_layout(**plotly_theme(), title="Net Worth Growth", height=340)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Net worth history will build up as you use the app daily.")
    with c6:
        if not investments["rows"].empty:
            inv_df = investments["rows"].copy()
            fig = px.bar(inv_df, x="name", y=["amount", "current_value"], barmode="group",
                         labels={"value": "Amount", "variable": "Type"}, title="Investment Growth by Holding")
            fig.update_layout(**plotly_theme(), height=340)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("Add investments in the Investments page to see growth here.")

    c7, c8 = st.columns(2)
    with c7:
        merchant_df = (df[(df["type"] == "expense") & (df["merchant"] != "")]
                       .groupby("merchant")["amount"].sum().sort_values(ascending=False).head(8).reset_index())
        if not merchant_df.empty:
            fig = px.bar(merchant_df, x="amount", y="merchant", orientation="h", title="Top Merchants")
            fig.update_layout(**plotly_theme(), height=340)
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No merchant data detected yet.")
    with c8:
        heatmap_fig = build_spending_heatmap(df)
        if heatmap_fig is not None:
            st.plotly_chart(heatmap_fig, width='stretch')
        else:
            st.info("Add more transactions to unlock the daily spending heatmap.")

    st.markdown("#### \U0001f3af Goal Progress")
    if goals:
        gcols = st.columns(min(len(goals), 3))
        for i, g in enumerate(goals[:3]):
            with gcols[i % 3]:
                progress_bar(g["name"], g["progress_pct"],
                             f"{format_currency(g['current'], currency)} / {format_currency(g['target'], currency)}")
    else:
        st.caption("No goals set yet — add one in the Goals page.")

    st.markdown("#### Recent Transactions")
    st.dataframe(
        df.head(10)[["date", "category", "description", "merchant", "amount", "type"]],
        width='stretch', hide_index=True,
    )

    render_ai_summary_card(user_id, df, summary, health, currency)


def build_spending_heatmap(df):
    """Daily spending heatmap (week x weekday) for the most recent ~12 weeks of expense data."""
    exp = df[df["type"] == "expense"].copy()
    if exp.empty:
        return None
    exp["week"] = exp["date"].dt.isocalendar().week
    exp["weekday"] = exp["date"].dt.day_name()
    daily = exp.groupby(exp["date"].dt.date)["amount"].sum().reset_index()
    daily.columns = ["date", "amount"]
    if daily.empty:
        return None
    daily["date"] = pd.to_datetime(daily["date"])
    daily["week"] = daily["date"].dt.strftime("%Y-W%U")
    daily["weekday"] = daily["date"].dt.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = daily.pivot_table(index="weekday", columns="week", values="amount", aggfunc="sum").reindex(weekday_order)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0, "#F5F7FA"], [1, COLORS["primary"]]],
    ))
    fig.update_layout(**plotly_theme(), title="Daily Spending Heatmap", height=340)
    return fig


def render_ai_summary_card(user_id, df, summary, health, currency):
    st.markdown("#### \U0001f9e0 AI Summary")
    settings = get_settings(user_id)
    if not settings["enable_ai"]:
        st.info("AI features are disabled in Settings.")
        return
    with st.container(border=True):
        if st.button("Generate AI Summary", key="ai_summary_btn"):
            with st.spinner("Asking FinPilot AI Advisor..."):
                context = build_financial_context(user_id, df, summary, health, "See Forecast page for details.", currency)
                answer = ask_ai_advisor(
                    "Give me a brief 3-4 sentence summary of my current financial standing and one key action item.",
                    context, settings["groq_model"],
                )
            st.markdown(answer)


####################################
# Transactions Page
####################################
def page_transactions(user_id):
    st.markdown("### \U0001f4b3 Transactions")
    df = load_transactions(user_id)

    tab1, tab2, tab3 = st.tabs(["\U0001f4cb View & Manage", "\u2795 Add Manually", "\U0001f4e4 Smart CSV Upload"])

    with tab1:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            search = st.text_input("Search description/category/merchant", "")
        with col_b:
            cat_filter = st.multiselect("Filter by category", TRANSACTION_CATEGORIES)
        with col_c:
            type_filter = st.selectbox("Type", ["All", "income", "expense"])

        filtered = df.copy()
        if search:
            mask = (
                filtered["description"].str.contains(search, case=False, na=False)
                | filtered["category"].str.contains(search, case=False, na=False)
                | filtered.get("merchant", pd.Series(dtype=str)).astype(str).str.contains(search, case=False, na=False)
            )
            filtered = filtered[mask]
        if cat_filter:
            filtered = filtered[filtered["category"].isin(cat_filter)]
        if type_filter != "All":
            filtered = filtered[filtered["type"] == type_filter]

        sort_col = st.selectbox("Sort by", ["date", "amount", "category"], key="sort_col")
        filtered = filtered.sort_values(sort_col, ascending=False)

        with st.expander("\U0001f501 Duplicate Detection"):
            dupes = find_duplicate_transactions(df)
            if dupes.empty:
                st.caption("No likely duplicate transactions found.")
            else:
                st.warning(f"Found {len(dupes)} transactions that look like duplicates (same date, amount, category).")
                st.dataframe(dupes[["id", "date", "category", "description", "amount", "type"]],
                             width='stretch', hide_index=True)
                dupe_ids_to_remove = st.multiselect("Select duplicate IDs to delete", dupes["id"].tolist(),
                                                     key="dupe_remove")
                if st.button("\U0001f5d1\ufe0f Delete Selected Duplicates", disabled=not dupe_ids_to_remove):
                    if bulk_delete_transactions([int(i) for i in dupe_ids_to_remove]):
                        st.success(f"Deleted {len(dupe_ids_to_remove)} duplicate transactions.")
                        st.rerun()

        page_size = 25
        total_rows = len(filtered)
        max_page = max((total_rows - 1) // page_size, 0)
        page_num = st.number_input("Page", min_value=0, max_value=max_page, value=0, step=1) if total_rows > page_size else 0
        page_slice = filtered.iloc[page_num * page_size: (page_num + 1) * page_size]

        st.dataframe(
            page_slice[["id", "date", "category", "description", "merchant", "amount", "type"]],
            width='stretch', hide_index=True,
        )
        st.caption(f"Showing {len(page_slice)} of {total_rows} transactions.")

        st.markdown("##### Bulk Actions")
        bc1, bc2, bc3 = st.columns([2, 1.4, 1])
        with bc1:
            bulk_ids = st.multiselect("Select transaction IDs for bulk action", filtered["id"].tolist(), key="bulk_ids")
        with bc2:
            bulk_category = st.selectbox("New category (bulk edit)", TRANSACTION_CATEGORIES, key="bulk_cat")
        with bc3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            bulk_edit_clicked = st.button("Apply Category", disabled=not bulk_ids, width='stretch')
        if bulk_edit_clicked and bulk_ids:
            if bulk_update_category([int(i) for i in bulk_ids], bulk_category):
                st.success(f"Updated category for {len(bulk_ids)} transactions.")
                st.rerun()
        if st.button("\U0001f5d1\ufe0f Bulk Delete Selected", disabled=not bulk_ids):
            if bulk_delete_transactions([int(i) for i in bulk_ids]):
                st.success(f"Deleted {len(bulk_ids)} transactions.")
                st.rerun()

        st.markdown("##### Edit or Delete a Single Transaction")
        if not filtered.empty:
            txn_id = st.selectbox("Select transaction ID", filtered["id"].tolist())
            selected = filtered[filtered["id"] == txn_id].iloc[0]

            with st.form("edit_txn_form"):
                ec1, ec2 = st.columns(2)
                with ec1:
                    new_date = st.date_input("Date", pd.to_datetime(selected["date"]))
                    new_category = st.selectbox(
                        "Category", TRANSACTION_CATEGORIES,
                        index=TRANSACTION_CATEGORIES.index(selected["category"])
                        if selected["category"] in TRANSACTION_CATEGORIES else 0,
                    )
                with ec2:
                    new_amount = st.number_input("Amount", value=float(selected["amount"]), min_value=0.0)
                    new_type = st.selectbox("Type", ["income", "expense"],
                                             index=0 if selected["type"] == "income" else 1)
                new_desc = st.text_input("Description", value=selected["description"] or "")

                fc1, fc2 = st.columns(2)
                with fc1:
                    update_clicked = st.form_submit_button("\U0001f4be Update", width='stretch')
                with fc2:
                    delete_clicked = st.form_submit_button("\U0001f5d1\ufe0f Delete", width='stretch')

                if update_clicked:
                    if update_transaction(int(txn_id), str(new_date), new_category, new_desc, new_amount, new_type):
                        st.success("Transaction updated.")
                        st.rerun()
                if delete_clicked:
                    if delete_transaction(int(txn_id)):
                        st.success("Transaction deleted.")
                        st.rerun()
        else:
            st.caption("No transactions match your filters.")

    with tab2:
        with st.form("add_txn_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                date = st.date_input("Date", datetime.now())
                category = st.selectbox("Category", TRANSACTION_CATEGORIES, key="add_cat")
            with c2:
                amount = st.number_input("Amount", min_value=0.0, step=100.0)
                ttype = st.selectbox("Type", ["income", "expense"], key="add_type")
            description = st.text_input("Description")
            submitted = st.form_submit_button("Add Transaction", width='stretch')
            if submitted:
                if amount <= 0:
                    st.warning("Amount must be greater than zero.")
                else:
                    with st.spinner("Saving transaction..."):
                        ok = add_transaction(user_id, str(date), category, description, amount, ttype)
                    if ok:
                        st.success("Transaction added.")
                        st.rerun()

    with tab3:
        st.caption("CSV should include: date, description, amount. Category, type, and merchant are "
                    "auto-detected if missing.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                with st.spinner("Parsing, detecting categories/merchants, and checking for duplicates..."):
                    csv_df = pd.read_csv(uploaded)
                    csv_df.columns = [c.lower().strip() for c in csv_df.columns]
                    required_cols = {"date", "amount"}
                    missing = required_cols - set(csv_df.columns)
                    if missing:
                        st.error(f"CSV is missing required columns: {', '.join(missing)}")
                    else:
                        if "description" not in csv_df.columns:
                            csv_df["description"] = ""
                        csv_df = csv_df.dropna(subset=["date", "amount"])
                        csv_df["amount"] = pd.to_numeric(csv_df["amount"], errors="coerce")
                        csv_df = csv_df.dropna(subset=["amount"])
                        csv_df = enrich_csv_rows(csv_df)

                        existing_keys = set(zip(df["date"].dt.strftime("%Y-%m-%d"), df["amount"].round(2))) if not df.empty else set()
                        csv_df["potential_duplicate"] = csv_df.apply(
                            lambda r: (str(r["date"]), round(float(r["amount"]), 2)) in existing_keys, axis=1
                        )

                        st.dataframe(
                            csv_df[["date", "category", "description", "merchant", "amount", "type", "potential_duplicate"]].head(30),
                            width='stretch', hide_index=True,
                        )
                        n_dupes = int(csv_df["potential_duplicate"].sum())
                        if n_dupes:
                            st.warning(f"{n_dupes} row(s) look like they may already exist in your transactions.")
                        skip_dupes = st.checkbox("Skip rows flagged as potential duplicates", value=True)

                        import_df = csv_df[~csv_df["potential_duplicate"]] if skip_dupes else csv_df
                        if st.button(f"Import {len(import_df)} transactions", width='stretch'):
                            with st.spinner("Importing transactions..."):
                                with get_conn() as conn:
                                    rows = [
                                        (user_id, str(r["date"]), r["category"], r["description"],
                                         float(r["amount"]), r["type"], r["merchant"])
                                        for _, r in import_df.iterrows()
                                    ]
                                    conn.executemany(
                                        "INSERT INTO transactions (user_id, date, category, description, amount, type, merchant) "
                                        "VALUES (?,?,?,?,?,?,?)",
                                        rows,
                                    )
                                load_transactions.clear()
                            st.success(f"Imported {len(import_df)} transactions.")
                            st.rerun()
            except pd.errors.ParserError as e:
                st.error(f"Could not parse CSV file: {e}")
            except Exception as e:
                st.error(f"Unexpected error while processing CSV: {e}")


####################################
# Budget Planner Page (persisted — powers the Smart Budget Engine dashboard KPIs)
####################################
def page_budget_planner(user_id, currency):
    st.markdown("### \U0001f9ee Budget Planner")
    st.caption("Plan your monthly budget — saved allocations automatically feed the Dashboard's Budget "
               "Utilization KPI, the Financial Health Score, Loan Eligibility, Forecast, Reports and AI Advisor.")

    month = datetime.now().strftime("%Y-%m")
    existing = get_budget_allocations(user_id, month)

    with st.form("budget_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            salary = st.number_input("Salary", min_value=0.0, value=float(existing.get("Salary", 80000.0)), step=1000.0)
            rent = st.number_input("Rent", min_value=0.0, value=float(existing.get("Rent", 18000.0)), step=500.0)
            food = st.number_input("Food", min_value=0.0, value=float(existing.get("Food", 8000.0)), step=500.0)
        with c2:
            shopping = st.number_input("Shopping", min_value=0.0, value=float(existing.get("Shopping", 5000.0)), step=500.0)
            travel = st.number_input("Travel", min_value=0.0, value=float(existing.get("Travel", 3000.0)), step=500.0)
            utilities = st.number_input("Utilities", min_value=0.0, value=float(existing.get("Utilities", 2500.0)), step=200.0)
        with c3:
            insurance = st.number_input("Insurance", min_value=0.0, value=float(existing.get("Insurance", 2000.0)), step=200.0)
            emis = st.number_input("EMIs", min_value=0.0, value=float(existing.get("EMIs", 9000.0)), step=500.0)
            savings_goal = st.number_input("Savings Goal", min_value=0.0, value=float(existing.get("Savings Goal", 10000.0)), step=500.0)
        submitted = st.form_submit_button("Save & Analyze Budget", width='stretch')

    if submitted or existing:
        expenses = {
            "Rent": rent if submitted else existing.get("Rent", 0),
            "Food": food if submitted else existing.get("Food", 0),
            "Shopping": shopping if submitted else existing.get("Shopping", 0),
            "Travel": travel if submitted else existing.get("Travel", 0),
            "Utilities": utilities if submitted else existing.get("Utilities", 0),
            "Insurance": insurance if submitted else existing.get("Insurance", 0),
            "EMIs": emis if submitted else existing.get("EMIs", 0),
        }
        salary_val = salary if submitted else existing.get("Salary", 0)
        savings_goal_val = savings_goal if submitted else existing.get("Savings Goal", 0)

        if submitted:
            save_budget_allocations(user_id, month, {
                "Salary": salary, "Rent": rent, "Food": food, "Shopping": shopping, "Travel": travel,
                "Utilities": utilities, "Insurance": insurance, "EMIs": emis, "Savings Goal": savings_goal,
            })
            st.success("Budget saved — it now feeds the Dashboard, Financial Health, Forecast, Reports and AI Advisor.")

        total_expenses = sum(expenses.values())
        remaining = salary_val - total_expenses - savings_goal_val

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Total Expenses", format_currency(total_expenses, currency))
        with c2:
            metric_card("Savings Goal", format_currency(savings_goal_val, currency))
        with c3:
            metric_card("Remaining Income", format_currency(remaining, currency),
                        delta=format_currency(remaining, currency), delta_positive=remaining >= 0)

        if remaining < 0:
            st.markdown('<span class="badge-bad">\u26a0 Over Budget</span>', unsafe_allow_html=True)
            st.warning("Your planned expenses and savings goal exceed your salary. Consider reducing "
                       "discretionary spending (Shopping, Travel, Entertainment).")
        elif salary_val and remaining < salary_val * 0.05:
            st.markdown('<span class="badge-warn">Tight Budget</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-good">Healthy Budget</span>', unsafe_allow_html=True)

        fig = px.pie(names=list(expenses.keys()), values=list(expenses.values()), hole=0.5,
                     title="Budget Allocation")
        fig.update_layout(**plotly_theme(), height=380)
        st.plotly_chart(fig, width='stretch')

        recommended = {
            "Rent": 0.30, "Food": 0.12, "Shopping": 0.08, "Travel": 0.05,
            "Utilities": 0.05, "Insurance": 0.05, "EMIs": 0.15,
        }
        rec_df = pd.DataFrame({
            "Category": list(recommended.keys()),
            "Your %": [expenses[k] / salary_val * 100 if salary_val else 0 for k in recommended],
            "Recommended %": [v * 100 for v in recommended.values()],
        })
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=rec_df["Category"], y=rec_df["Your %"], name="Your Budget", marker_color=COLORS["primary"]))
        fig2.add_trace(go.Bar(x=rec_df["Category"], y=rec_df["Recommended %"], name="Recommended", marker_color=COLORS["accent"]))
        fig2.update_layout(**plotly_theme(), title="Your Budget vs Recommended (%)", barmode="group", height=380)
        st.plotly_chart(fig2, width='stretch')

        df = load_transactions(user_id)
        util = compute_budget_utilization(user_id, df)
        if util.get("total_budget"):
            st.markdown("##### This Month's Actual vs Budget")
            for cat, vals in util["by_category"].items():
                pct = (vals["spent"] / vals["budget"] * 100) if vals["budget"] else 0
                progress_bar(cat, pct, f"{format_currency(vals['spent'], currency)} of {format_currency(vals['budget'], currency)}")


####################################
# Loan Eligibility & Analyzer Page
####################################
def page_loan_eligibility(user_id, currency):
    st.markdown("### \U0001f3e6 Loan Eligibility & Analyzer")
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is not installed. Please install it to use this feature.")
        return

    df = load_transactions(user_id)
    summary = compute_summary(df)
    loan_analysis = analyze_loans(user_id, summary)

    st.markdown("#### \U0001f4ca Current Loan Portfolio")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total EMI Outstanding", format_currency(loan_analysis["total_emi"], currency))
    with c2:
        metric_card("Debt-to-Income Ratio", f"{loan_analysis['debt_ratio']}%")
    with c3:
        metric_card("Safe Additional EMI", format_currency(loan_analysis["safe_emi"], currency))
    with c4:
        metric_card("Max Affordable New Loan", format_currency(loan_analysis["max_loan"], currency))

    st.markdown(f'<span class="badge-{"good" if loan_analysis["risk"]=="Low" else "warn" if loan_analysis["risk"]=="Medium" else "bad"}">'
                f'Risk: {loan_analysis["risk"]}</span>', unsafe_allow_html=True)

    with st.expander("\u2795 Add an Existing Loan"):
        with st.form("add_loan_form"):
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                loan_type = st.text_input("Loan Type", "Personal Loan")
                principal = st.number_input("Principal", min_value=0.0, value=100000.0, step=5000.0)
            with lc2:
                interest_rate = st.number_input("Interest Rate (% p.a.)", min_value=0.0, value=10.5, step=0.1)
                tenure_months = st.number_input("Tenure (months)", min_value=1, value=36, step=1)
            with lc3:
                emi = st.number_input("Monthly EMI", min_value=0.0, value=3200.0, step=100.0)
                start_date = st.date_input("Start Date", datetime.now())
            if st.form_submit_button("Save Loan", width='stretch'):
                if add_record("loans", user_id, loan_type=loan_type, principal=principal,
                               interest_rate=interest_rate, tenure_months=int(tenure_months), emi=emi,
                               start_date=str(start_date)):
                    st.success("Loan added.")
                    st.rerun()

    if not loan_analysis["rows"].empty:
        st.dataframe(loan_analysis["rows"][["id", "loan_type", "principal", "interest_rate", "tenure_months", "emi", "start_date"]],
                     width='stretch', hide_index=True)
        del_id = st.selectbox("Remove a loan (select ID)", [None] + loan_analysis["rows"]["id"].tolist())
        if del_id and st.button("\U0001f5d1\ufe0f Delete Selected Loan"):
            delete_record("loans", int(del_id))
            st.rerun()

    st.markdown("---")
    st.markdown("#### \U0001f9ee New Loan Eligibility Check (ML Model)")
    default_salary = max(summary.get("avg_monthly_income", 0) or 60000.0, 5000.0)
    default_existing_loans = min(len(loan_analysis["rows"]), 5) if not loan_analysis["rows"].empty else 1
    with st.form("loan_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 21, 65, 32)
            salary = st.number_input("Monthly Salary", min_value=5000.0, value=float(default_salary), step=1000.0)
            credit_score = st.slider("Credit Score", 300, 900, 720)
        with c2:
            emi = st.number_input("Current Monthly EMI", min_value=0.0, value=loan_analysis["total_emi"] or 5000.0, step=500.0)
            existing_loans = st.slider("Existing Loans", 0, 5, default_existing_loans)
        submitted = st.form_submit_button("Check Eligibility", width='stretch')

    if submitted:
        with st.spinner("Running ML risk model..."):
            result = predict_loan_eligibility(age, salary, credit_score, emi, existing_loans)

        if result.get("error"):
            st.error(result["error"])
            return

        c1, c2, c3 = st.columns(3)
        with c1:
            status = "\u2705 Eligible" if result["eligible"] else "\u274c Not Eligible"
            metric_card("Eligibility", status)
        with c2:
            metric_card("Approval Probability", f"{result['probability']}%")
        with c3:
            metric_card("Risk Score", f"{result['risk_score']}")

        render_gauge(100 - result["risk_score"], title="Approval Confidence")

        st.markdown("##### Suggestions")
        for s in result["suggestions"]:
            st.markdown(f"- {s}")


####################################
# Net Worth Page
####################################
def page_networth(user_id, currency):
    st.markdown("### \U0001f4b0 Net Worth")
    df = load_transactions(user_id)
    summary = compute_summary(df)
    networth = compute_net_worth(user_id, summary)
    snapshot_networth(user_id, networth)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Assets", format_currency(networth["assets"], currency))
    with c2:
        metric_card("Liabilities", format_currency(networth["liabilities"], currency))
    with c3:
        metric_card("Investments", format_currency(networth["investments"], currency))
    with c4:
        metric_card("Net Worth", format_currency(networth["net_worth"], currency))

    nw_hist = get_networth_history(user_id)
    if not nw_hist.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nw_hist["date"], y=nw_hist["net_worth"], mode="lines+markers",
                                  name="Net Worth", line=dict(color=COLORS["primary"], width=3),
                                  fill="tozeroy", fillcolor="rgba(4,106,56,0.08)"))
        fig.update_layout(**plotly_theme(), title="Net Worth Growth Over Time", height=380)
        st.plotly_chart(fig, width='stretch')

    tab1, tab2 = st.tabs(["\U0001f3e0 Assets", "\U0001f4b3 Liabilities"])
    with tab1:
        with st.form("add_asset_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                name = st.text_input("Asset Name", "")
            with c2:
                category = st.selectbox("Category", ASSET_CATEGORIES)
            with c3:
                value = st.number_input("Value", min_value=0.0, step=1000.0)
            if st.form_submit_button("Add Asset", width='stretch'):
                if name and add_record("assets", user_id, name=name, category=category, value=value,
                                        date=datetime.now().strftime("%Y-%m-%d")):
                    st.success("Asset added.")
                    st.rerun()
        assets_df = get_records("assets", user_id)
        if not assets_df.empty:
            st.dataframe(assets_df[["id", "name", "category", "value", "date"]], width='stretch', hide_index=True)
            del_id = st.selectbox("Remove an asset (select ID)", [None] + assets_df["id"].tolist(), key="del_asset")
            if del_id and st.button("\U0001f5d1\ufe0f Delete Selected Asset"):
                delete_record("assets", int(del_id))
                st.rerun()
            fig = px.pie(assets_df, names="category", values="value", hole=0.5, title="Assets by Category")
            fig.update_layout(**plotly_theme(), height=340)
            st.plotly_chart(fig, width='stretch')
        else:
            st.caption("No assets recorded yet.")

    with tab2:
        with st.form("add_liability_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                name = st.text_input("Liability Name", "", key="liab_name")
            with c2:
                category = st.selectbox("Category", LIABILITY_CATEGORIES, key="liab_cat")
            with c3:
                amount = st.number_input("Amount", min_value=0.0, step=1000.0, key="liab_amt")
            if st.form_submit_button("Add Liability", width='stretch'):
                if name and add_record("liabilities", user_id, name=name, category=category, amount=amount,
                                        date=datetime.now().strftime("%Y-%m-%d")):
                    st.success("Liability added.")
                    st.rerun()
        liab_df = get_records("liabilities", user_id)
        if not liab_df.empty:
            st.dataframe(liab_df[["id", "name", "category", "amount", "date"]], width='stretch', hide_index=True)
            del_id = st.selectbox("Remove a liability (select ID)", [None] + liab_df["id"].tolist(), key="del_liab")
            if del_id and st.button("\U0001f5d1\ufe0f Delete Selected Liability"):
                delete_record("liabilities", int(del_id))
                st.rerun()
            fig = px.pie(liab_df, names="category", values="amount", hole=0.5, title="Liabilities by Category")
            fig.update_layout(**plotly_theme(), height=340)
            st.plotly_chart(fig, width='stretch')
        else:
            st.caption("No liabilities recorded yet.")


####################################
# Goals Page
####################################
def page_goals(user_id, currency):
    st.markdown("### \U0001f3af Goal Tracker")
    df = load_transactions(user_id)
    summary = compute_summary(df)

    with st.expander("\u2795 Add a New Goal"):
        with st.form("add_goal_form"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Goal Name", "")
                category = st.selectbox("Category", GOAL_CATEGORIES)
            with c2:
                target_amount = st.number_input("Target Amount", min_value=0.0, step=5000.0)
                current_amount = st.number_input("Current Progress", min_value=0.0, step=1000.0)
            target_date = st.date_input("Target Date", datetime.now() + timedelta(days=365))
            if st.form_submit_button("Add Goal", width='stretch'):
                if name and add_record("goals", user_id, name=name, category=category, target_amount=target_amount,
                                        current_amount=current_amount, target_date=str(target_date)):
                    st.success("Goal added.")
                    st.rerun()

    goals = analyze_goals(user_id, summary)
    if not goals:
        st.info("No goals yet — add one above (Car, House, Vacation, Emergency Fund, Education, Marriage, Retirement...).")
        return

    for g in goals:
        with st.container(border=True):
            gc1, gc2 = st.columns([3, 1])
            with gc1:
                st.markdown(f"**{g['name']}** &nbsp; <span class='badge-good'>{g['category']}</span>",
                            unsafe_allow_html=True)
                progress_bar("Progress", g["progress_pct"],
                             f"{format_currency(g['current'], currency)} of {format_currency(g['target'], currency)} "
                             f"— target {g['target_date']}")
                st.caption(f"Needs ~{format_currency(g['required_monthly'], currency)}/month to stay on track "
                          f"({g['months_remaining']} months remaining). "
                          f"Completion probability: **{g['completion_probability']}%**.")
            with gc2:
                new_progress = st.number_input("Update saved amount", min_value=0.0, value=float(g["current"]),
                                                key=f"goal_upd_{g['id']}", step=1000.0)
                if st.button("Save", key=f"goal_save_{g['id']}", width='stretch'):
                    update_record("goals", g["id"], current_amount=new_progress)
                    st.rerun()
                if st.button("Delete", key=f"goal_del_{g['id']}", width='stretch'):
                    delete_record("goals", g["id"])
                    st.rerun()

    fig = go.Figure()
    fig.add_trace(go.Bar(y=[g["name"] for g in goals], x=[g["progress_pct"] for g in goals],
                         orientation="h", marker_color=COLORS["accent"], name="Progress %"))
    fig.update_layout(**plotly_theme(), title="Goal Progress Overview", height=340, xaxis=dict(range=[0, 100]))
    st.plotly_chart(fig, width='stretch')


####################################
# Investments Page
####################################
def page_investments(user_id, currency):
    st.markdown("### \U0001f4c8 Investment Analyzer")

    with st.expander("\u2795 Add an Investment"):
        with st.form("add_investment_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                inv_type = st.selectbox("Type", INVESTMENT_TYPES)
                name = st.text_input("Name", "")
            with c2:
                amount = st.number_input("Amount Invested", min_value=0.0, step=1000.0)
                current_value = st.number_input("Current Value", min_value=0.0, step=1000.0)
            with c3:
                date = st.date_input("Date", datetime.now())
            if st.form_submit_button("Add Investment", width='stretch'):
                if name and add_record("investments", user_id, inv_type=inv_type, name=name, amount=amount,
                                        current_value=current_value, date=str(date)):
                    st.success("Investment added.")
                    st.rerun()

    analysis = analyze_investments(user_id)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Invested", format_currency(analysis["total_invested"], currency))
    with c2:
        metric_card("Current Value", format_currency(analysis["total_value"], currency))
    with c3:
        metric_card("Growth", f"{analysis['growth_pct']}%")
    with c4:
        metric_card("Diversification Score", f"{analysis['diversification_score']} / 100")

    if not analysis["rows"].empty:
        c5, c6 = st.columns(2)
        with c5:
            fig = px.pie(names=list(analysis["allocation"].keys()), values=list(analysis["allocation"].values()),
                        hole=0.5, title="Allocation by Type")
            fig.update_layout(**plotly_theme(), height=340)
            st.plotly_chart(fig, width='stretch')
        with c6:
            inv_df = analysis["rows"]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=inv_df["name"], y=inv_df["amount"], name="Invested", marker_color=COLORS["secondary"]))
            fig.add_trace(go.Bar(x=inv_df["name"], y=inv_df["current_value"], name="Current Value", marker_color=COLORS["accent"]))
            fig.update_layout(**plotly_theme(), title="Investment Growth", barmode="group", height=340)
            st.plotly_chart(fig, width='stretch')

        st.dataframe(analysis["rows"][["id", "inv_type", "name", "amount", "current_value", "date"]],
                     width='stretch', hide_index=True)
        del_id = st.selectbox("Remove an investment (select ID)", [None] + analysis["rows"]["id"].tolist())
        if del_id and st.button("\U0001f5d1\ufe0f Delete Selected Investment"):
            delete_record("investments", int(del_id))
            st.rerun()
    else:
        st.info("No investments recorded yet. Add mutual funds, stocks, gold, FDs, PPF, NPS or crypto above.")


####################################
# Forecast Page
####################################
def page_forecast(user_id, currency, default_months=6):
    st.markdown("### \U0001f4c8 Forecast")
    df = load_transactions(user_id)
    if df.empty:
        st.info("Add transactions first to generate a forecast.")
        return

    months = st.slider("Forecast horizon (months)", 1, 12, default_months)
    metric = st.selectbox("Metric to forecast", ["income", "expenses", "savings"])
    scenario = st.selectbox("Scenario", ["Realistic", "Optimistic", "Worst Case"])

    with st.spinner("Running forecast model..."):
        forecast_df, note = run_forecast(df, metric, months)

    if forecast_df is None:
        st.warning(note)
        return
    if note:
        st.caption(f"\u2139\ufe0f {note}")

    ms = monthly_series(df)
    last_actual_date = ms["month"].max()

    # Only scale genuinely FUTURE periods by the scenario factor — scaling the historical fit
    # too (the old behaviour) made the forecast line diverge from real history even in the past.
    adj = {"Optimistic": 1.10, "Realistic": 1.0, "Worst Case": 0.85}[scenario]
    scenario_df = forecast_df.copy()
    future_mask = scenario_df["ds"] > last_actual_date
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        scenario_df.loc[future_mask, col] = scenario_df.loc[future_mask, col] * adj

    # Pass only the selected metric's historical column so the "Actual" trace matches what was
    # actually chosen above (previously this always plotted Income regardless of selection).
    render_forecast_chart(scenario_df, ms[["month", metric]], f"{metric.capitalize()} ({scenario})", currency)

    st.session_state["last_forecast_note"] = (
        f"{metric.capitalize()} projected ({scenario} scenario) to reach "
        f"{format_currency(scenario_df['yhat'].iloc[-1], currency)} in {months} months."
    )
    st.info(st.session_state["last_forecast_note"])


####################################
# AI Advisor Page
####################################
def page_ai_advisor(user_id, currency):
    st.markdown("### \U0001f9e0 AI Advisor")
    settings = get_settings(user_id)
    if not settings["enable_ai"]:
        st.warning("AI features are disabled. Enable them in Settings.")
        return

    df = load_transactions(user_id)
    summary = compute_summary(df)
    health = calculate_health_score(df, summary)
    forecast_note = st.session_state.get("last_forecast_note", "No forecast generated yet \u2014 visit the Forecast page.")
    context = build_financial_context(user_id, df, summary, health, forecast_note, currency)

    st.markdown("##### Quick Questions")
    quick_questions = [
        "Can I buy a car?", "Can I buy a house?", "How do I save more?",
        "Should I invest?", "Should I take a loan?", "How can I improve my score?",
        "Can I retire early?", "Should I increase my SIP?", "How much emergency fund should I keep?",
    ]
    cols = st.columns(3)
    clicked_question = None
    for i, q in enumerate(quick_questions):
        with cols[i % 3]:
            if st.button(q, key=f"qq_{i}", width='stretch'):
                clicked_question = q

    custom_q = st.chat_input("Or ask your own financial question...")
    question = custom_q or clicked_question

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if question:
        with st.spinner("FinPilot AI is thinking..."):
            answer = ask_ai_advisor(question, context, settings["groq_model"])
        st.session_state.chat_history.append((question, answer))

    for q, a in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)


####################################
# Reports Page
####################################
def page_reports(user_id, currency):
    st.markdown("### \U0001f4c4 Reports")
    df = load_transactions(user_id)
    summary = compute_summary(df)
    health = calculate_health_score(df, summary)
    networth = compute_net_worth(user_id, summary)
    emergency = analyze_emergency_fund(user_id, df, summary)
    goals = analyze_goals(user_id, summary)
    investments = analyze_investments(user_id)
    loans = analyze_loans(user_id, summary)
    budget_util = compute_budget_utilization(user_id, df)
    forecast_note = st.session_state.get("last_forecast_note", "No forecast generated yet — visit the Forecast page.")
    context = build_financial_context(user_id, df, summary, health, forecast_note, currency)

    use_ai = st.checkbox("Include AI narrative overlay (requires Mistral API key)", value=False)
    ai_narrative = ""
    if st.button("\U0001f4c4 Generate Full Financial Audit Report", width='stretch'):
        if use_ai:
            with st.spinner("Generating AI narrative overlay..."):
                ai_narrative = generate_ai_narrative(context, loans, forecast_note)
        with st.spinner("Compiling your report..."):
            report_text = generate_full_report_text(
                st.session_state.user.get("name", "User"), df, summary, health, networth, emergency,
                goals, investments, loans, budget_util, forecast_note, currency, ai_narrative,
            )
        st.session_state["report_text"] = report_text
        word_count = len(report_text.split())
        st.success(f"Report generated ({word_count} words).")

    report_text = st.session_state.get("report_text", "")
    if report_text:
        st.markdown(report_text)

        ms = monthly_series(df)
        chart_fig = None
        if not ms.empty:
            chart_fig = go.Figure()
            chart_fig.add_trace(go.Scatter(x=ms["month"], y=ms["savings"], mode="lines+markers",
                                           name="Savings", line=dict(color=COLORS["primary"], width=3)))
            chart_fig.update_layout(**plotly_theme(), title="Savings Trend", height=350)

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.download_button("\u2b07\ufe0f Download Markdown", data=report_text.encode("utf-8"),
                                file_name=f"finpilot_report_{datetime.now().strftime('%Y%m%d')}.md",
                                mime="text/markdown", width='stretch')
        with dc2:
            plain_text = re.sub(r"[#*|]", "", report_text)
            st.download_button("\u2b07\ufe0f Download Text", data=plain_text.encode("utf-8"),
                                file_name=f"finpilot_report_{datetime.now().strftime('%Y%m%d')}.txt",
                                mime="text/plain", width='stretch')
        with dc3:
            if REPORTLAB_AVAILABLE:
                with st.spinner("Preparing PDF..."):
                    user_name = st.session_state.user.get("name", "User")
                    pdf_buffer = generate_pdf_report(user_name, report_text, summary, health, networth, currency, chart_fig)
                if pdf_buffer:
                    st.download_button(
                        "\u2b07\ufe0f Download PDF", data=pdf_buffer,
                        file_name=f"finpilot_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf", width='stretch',
                    )
            else:
                st.warning("ReportLab is not installed; PDF export unavailable.")

        try:
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO reports (user_id, title, content, created_at) VALUES (?,?,?,?)",
                    (user_id, f"Financial Audit Report {datetime.now().strftime('%Y-%m-%d')}",
                     report_text[:2000], datetime.now().isoformat()),
                )
        except sqlite3.Error:
            pass

    st.markdown("#### Past Reports")
    try:
        with get_conn() as conn:
            reports_df = pd.read_sql_query(
                "SELECT title, created_at FROM reports WHERE user_id=? ORDER BY created_at DESC",
                conn, params=(user_id,),
            )
        if reports_df.empty:
            st.caption("No saved reports yet.")
        else:
            st.dataframe(reports_df, width='stretch', hide_index=True)
    except sqlite3.Error as e:
        st.error(f"Could not load past reports: {e}")


####################################
# Settings Page
####################################
def page_settings(user_id):
    st.markdown("### \u2699\ufe0f Settings")
    settings = get_settings(user_id)

    with st.form("settings_form"):
        c1, c2 = st.columns(2)
        with c1:
            theme = st.selectbox("Theme", ["Light", "Dark"],
                                  index=0 if settings["theme"] == "Light" else 1)
            currency = st.selectbox("Currency", list(CURRENCY_SYMBOLS.keys()),
                                     index=list(CURRENCY_SYMBOLS.keys()).index(settings["currency"])
                                     if settings["currency"] in CURRENCY_SYMBOLS else 0)
            forecast_months = st.slider("Forecast Months", 1, 12, int(settings["forecast_months"]))
        with c2:
            groq_model = st.text_input("Groq Model", value=settings["groq_model"])
            enable_ai = st.checkbox("Enable AI Features", value=settings["enable_ai"])

        saved = st.form_submit_button("Save Settings", width='stretch')
        if saved:
            with st.spinner("Saving settings..."):
                ok = save_settings(user_id, theme, currency, forecast_months, groq_model, enable_ai)
            if ok:
                st.success("Settings saved.")
                st.rerun()

    st.markdown("---")
    st.markdown("#### API Keys")
    st.caption(
        "Paste a key here to enable AI features for **this session only** — it is kept in "
        "memory, never written to disk or the database, and is cleared when you close or "
        "refresh the browser tab. If an environment variable is also set, the key entered "
        "here takes priority."
    )

    if "groq_api_key_override" not in st.session_state:
        st.session_state.groq_api_key_override = ""
    if "mistral_api_key_override" not in st.session_state:
        st.session_state.mistral_api_key_override = ""

    c1, c2 = st.columns(2)
    with c1:
        groq_key_input = st.text_input(
            "Groq API Key", value=st.session_state.groq_api_key_override,
            type="password", placeholder="gsk_...", key="groq_key_input",
        )
        has_key = bool(groq_key_input.strip()) or bool(os.environ.get("GROQ_API_KEY"))
        if not GROQ_SDK_AVAILABLE:
            st.markdown('<span class="badge-bad">Groq: `groq` package not installed</span>', unsafe_allow_html=True)
        elif has_key:
            st.markdown('<span class="badge-good">Groq: Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-bad">Groq: No key set</span>', unsafe_allow_html=True)
    with c2:
        mistral_key_input = st.text_input(
            "Mistral API Key", value=st.session_state.mistral_api_key_override,
            type="password", placeholder="...", key="mistral_key_input",
        )
        has_key = bool(mistral_key_input.strip()) or bool(os.environ.get("MISTRAL_API_KEY"))
        if not MISTRAL_SDK_AVAILABLE:
            st.markdown('<span class="badge-bad">Mistral: `mistralai` package not installed</span>', unsafe_allow_html=True)
        elif has_key:
            st.markdown('<span class="badge-good">Mistral: Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-bad">Mistral: No key set</span>', unsafe_allow_html=True)

    kc1, kc2 = st.columns(2)
    with kc1:
        if st.button("\U0001f4be Use these keys for this session", width='stretch'):
            st.session_state.groq_api_key_override = groq_key_input.strip()
            st.session_state.mistral_api_key_override = mistral_key_input.strip()
            _build_groq_client.clear()
            _build_mistral_client.clear()
            st.success("Session API keys updated.")
            st.rerun()
    with kc2:
        if st.button("\U0001f9f9 Clear session keys", width='stretch'):
            st.session_state.groq_api_key_override = ""
            st.session_state.mistral_api_key_override = ""
            _build_groq_client.clear()
            _build_mistral_client.clear()
            st.info("Session API keys cleared.")
            st.rerun()

    st.markdown("---")
    st.markdown("#### Data Management")
    c1, c2, c3 = st.columns(3)
    with c1:
        df = load_transactions(user_id)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("\u2b07\ufe0f Export Transactions (CSV)", data=csv_data,
                            file_name="finpilot_transactions.csv", mime="text/csv",
                            width='stretch')
    with c2:
        if os.path.exists(DB_PATH):
            with open(DB_PATH, "rb") as f:
                db_bytes = f.read()
            st.download_button("\u2b07\ufe0f Export Full Database", data=db_bytes,
                                file_name="finpilot.db", mime="application/octet-stream",
                                width='stretch')
    with c3:
        if st.button("\U0001f9f9 Clear AI Chat History", width='stretch'):
            st.session_state.chat_history = []
            st.session_state.pop("report_text", None)
            st.success("AI chat history and cached report cleared.")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    confirm = st.checkbox("I understand this will permanently delete all data")
    if st.button("\U0001f5d1\ufe0f Reset Database", width='stretch', disabled=not confirm):
        with st.spinner("Resetting database..."):
            ok = reset_database()
        if ok:
            st.success("Database reset. Reloading...")
            logout()


####################################
# Sidebar Navigation
####################################
def render_sidebar():
    user = st.session_state.user
    with st.sidebar:
        st.markdown(f"### \U0001f4a0 {APP_NAME}")
        st.caption(f"Welcome, **{user.get('name', 'User')}**")
        st.markdown("---")

        nav_items = [
            "Dashboard", "Transactions", "Budget Planner", "Net Worth", "Goals", "Investments",
            "Financial Health", "Loan Eligibility", "Forecast", "AI Advisor", "Reports", "Settings",
        ]
        icons = {
            "Dashboard": "\U0001f4ca", "Transactions": "\U0001f4b3", "Budget Planner": "\U0001f9ee",
            "Net Worth": "\U0001f4b0", "Goals": "\U0001f3af", "Investments": "\U0001f4c8",
            "Financial Health": "\U0001f49a", "Loan Eligibility": "\U0001f3e6", "Forecast": "\U0001f4c8",
            "AI Advisor": "\U0001f9e0", "Reports": "\U0001f4c4", "Settings": "\u2699\ufe0f",
        }
        current_page = st.session_state.get("page", "Dashboard")
        for item in nav_items:
            label = f"{icons[item]}  {item}"
            if st.button(label, key=f"nav_{item}", width='stretch',
                         type="primary" if current_page == item else "secondary"):
                st.session_state.page = item
                st.rerun()

        st.markdown("---")
        if st.button("\U0001f6aa  Logout", key="nav_logout", width='stretch'):
            logout()


####################################
# Main App
####################################
def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not init_database():
        st.stop()

    if not st.session_state.authenticated:
        login_page()
        return

    inject_custom_css()
    user_id = st.session_state.user["id"]
    settings = get_settings(user_id)
    currency = settings["currency"]

    render_sidebar()
    page = st.session_state.get("page", "Dashboard")

    try:
        if page == "Dashboard":
            page_dashboard(user_id, currency)
        elif page == "Transactions":
            page_transactions(user_id)
        elif page == "Budget Planner":
            page_budget_planner(user_id, currency)
        elif page == "Net Worth":
            page_networth(user_id, currency)
        elif page == "Goals":
            page_goals(user_id, currency)
        elif page == "Investments":
            page_investments(user_id, currency)
        elif page == "Financial Health":
            df = load_transactions(user_id)
            summary = compute_summary(df)
            health = calculate_health_score(df, summary)
            st.markdown("### \U0001f49a Financial Health")
            if df.empty:
                st.info("Add transactions to compute your financial health score.")
            else:
                render_gauge(health["score"])
                st.markdown(f"**Status:** {health['status']}")
                comp_df = pd.DataFrame({
                    "Pillar": list(health["components"].keys()),
                    "Score": list(health["components"].values()),
                })
                fig = px.bar(comp_df, x="Pillar", y="Score", range_y=[0, 100])
                fig.update_layout(**plotly_theme(), height=340)
                st.plotly_chart(fig, width='stretch')
                st.markdown("##### Suggestions")
                for s in health["suggestions"]:
                    st.markdown(f"- {s}")

                st.markdown("---")
                st.markdown("#### \U0001f6a8 Emergency Fund Analysis")
                emergency = analyze_emergency_fund(user_id, df, summary)
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    metric_card("Current Fund", format_currency(emergency["current_fund"], currency))
                with ec2:
                    metric_card("Recommended Fund", format_currency(emergency["recommended"], currency))
                with ec3:
                    metric_card("Months Covered", f"{emergency['months_covered']}")
                progress_bar("Emergency Fund Progress", emergency["progress_pct"],
                             f"Gap: {format_currency(emergency['gap'], currency)} | Risk: {emergency['risk']}")
                with st.expander("\u2795 Log Emergency Fund Contribution"):
                    amt = st.number_input("Amount", min_value=0.0, step=1000.0, key="ef_amt")
                    if st.button("Add to Emergency Fund"):
                        if add_record("emergency_fund", user_id, amount=amt, date=datetime.now().strftime("%Y-%m-%d")):
                            st.success("Logged.")
                            st.rerun()

                st.markdown("---")
                st.markdown("#### \u26a0\ufe0f Risk & Stability")
                networth = compute_net_worth(user_id, summary)
                loans = analyze_loans(user_id, summary)
                budget_util = compute_budget_utilization(user_id, df)
                debt_to_income = compute_debt_to_income(summary, loans)
                stability_index = compute_financial_stability_index(health, networth, emergency, budget_util)
                risk_score = compute_risk_score(debt_to_income, emergency, health)
                rc1, rc2 = st.columns(2)
                with rc1:
                    render_gauge(stability_index, title="Financial Stability Index")
                with rc2:
                    render_gauge(100 - risk_score, title="Safety Score (inverse of risk)")
        elif page == "Loan Eligibility":
            page_loan_eligibility(user_id, currency)
        elif page == "Forecast":
            page_forecast(user_id, currency, settings["forecast_months"])
        elif page == "AI Advisor":
            page_ai_advisor(user_id, currency)
        elif page == "Reports":
            page_reports(user_id, currency)
        elif page == "Settings":
            page_settings(user_id)
    except Exception as e:
        st.error(f"Something went wrong while rendering this page: {e}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
