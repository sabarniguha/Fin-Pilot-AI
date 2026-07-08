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
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


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
        }}
        .glass-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 14px 40px rgba(4, 106, 56, 0.16);
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
    for m in range(9):
        month_date = start + timedelta(days=30 * m)
        rows.append((user_id, month_date.strftime("%Y-%m-%d"), "Salary", "Monthly Salary",
                     float(rng.normal(85000, 2000)), "income"))
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
                        float(max(amt, 200)), "expense"))
    c.executemany(
        "INSERT INTO transactions (user_id, date, category, description, amount, type) VALUES (?,?,?,?,?,?)",
        rows,
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
        return df
    except sqlite3.Error as e:
        st.error(f"Could not load transactions: {e}")
        return pd.DataFrame(columns=["id", "user_id", "date", "category", "description", "amount", "type"])


def add_transaction(user_id, date, category, description, amount, ttype):
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO transactions (user_id, date, category, description, amount, type) VALUES (?,?,?,?,?,?)",
                (user_id, date, category, description, amount, ttype),
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
                "UPDATE transactions SET date=?, category=?, description=?, amount=?, type=? WHERE id=?",
                (date, category, description, amount, ttype, txn_id),
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
        load_transactions.clear()
        init_database()
        return True
    except OSError as e:
        st.error(f"Could not reset database: {e}")
        return False


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
            login_clicked = st.button("Log in", use_container_width=True)

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
    """Single source of truth for all headline financial numbers (avoids duplicate calc)."""
    if df.empty:
        return dict(income=0.0, expenses=0.0, savings=0.0, balance=0.0,
                     investments=0.0, emergency_fund=0.0, savings_rate=0.0)

    income = df.loc[df["type"] == "income", "amount"].sum()
    expenses = df.loc[df["type"] == "expense", "amount"].sum()
    investments = df.loc[(df["type"] == "expense") & (df["category"] == "Investment"), "amount"].sum()
    savings = income - expenses
    savings_rate = (savings / income * 100) if income > 0 else 0.0
    emergency_fund = max(savings, 0) * 0.6
    balance = savings

    return dict(
        income=float(income), expenses=float(expenses), savings=float(savings),
        balance=float(balance), investments=float(investments),
        emergency_fund=float(emergency_fund), savings_rate=float(savings_rate),
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
    st.plotly_chart(fig, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)


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


def build_financial_context(summary, health, forecast_note, currency):
    return f"""
Financial Summary:
- Income: {format_currency(summary['income'], currency)}
- Expenses: {format_currency(summary['expenses'], currency)}
- Savings: {format_currency(summary['savings'], currency)}
- Savings Rate: {summary['savings_rate']:.1f}%
- Investments: {format_currency(summary['investments'], currency)}
- Emergency Fund: {format_currency(summary['emergency_fund'], currency)}
- Financial Health Score: {health['score']}/100 ({health['status']})
- Forecast Note: {forecast_note}
""".strip()


def ask_ai_advisor(question, context, model_name):
    client = get_groq_client()
    if client is None:
        return ("\u26a0\ufe0f Groq API key not found or SDK not installed. Paste a key in "
                "Settings \u2192 API Keys, or set the `GROQ_API_KEY` environment variable, "
                "to enable the AI Advisor.")
    system_prompt = (
        "You are FinPilot AI, a professional, encouraging, and precise personal financial advisor "
        "for an Indian banking context. Use the provided financial context to give specific, "
        "actionable, numbers-grounded advice. Keep responses concise and structured with short "
        "headings or bullet points where useful. Never give definitive legal/tax advice \u2014 "
        "recommend consulting a professional for those matters."
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


def generate_ai_report(context, loan_info, forecast_note):
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
Write a professional financial report for a bank customer using the data below.
Structure the report with these exact section headings:
Executive Summary, Income Analysis, Expense Analysis, Budget Analysis,
Financial Health, Investment Suggestions, Loan Analysis, Forecast Summary, Action Plan.

{context}

Loan Analysis Data: {loan_info}
Forecast Note: {forecast_note}

Keep it well-organized, specific, and under 700 words.
"""
    try:
        resp = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"\u26a0\ufe0f Report generation failed: {e}"


####################################
# PDF Generator
####################################
def generate_pdf_report(user_name, summary, health, loan_info, forecast_note, ai_report_text, currency):
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], textColor=rl_colors.HexColor(COLORS["primary"]))
    h2_style = ParagraphStyle("H2Style", parent=styles["Heading2"], textColor=rl_colors.HexColor(COLORS["secondary"]))
    body_style = styles["BodyText"]

    story = [
        Paragraph(f"{APP_NAME} \u2014 Financial Report", title_style),
        Paragraph(f"Prepared for: {user_name} | Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", body_style),
        Spacer(1, 16),
        Paragraph("Financial Snapshot", h2_style),
    ]

    table_data = [
        ["Metric", "Value"],
        ["Income", format_currency(summary["income"], currency)],
        ["Expenses", format_currency(summary["expenses"], currency)],
        ["Savings", format_currency(summary["savings"], currency)],
        ["Savings Rate", f"{summary['savings_rate']:.1f}%"],
        ["Investments", format_currency(summary["investments"], currency)],
        ["Emergency Fund", format_currency(summary["emergency_fund"], currency)],
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

    story.append(Paragraph("Loan Eligibility", h2_style))
    if loan_info.get("error"):
        story.append(Paragraph(loan_info["error"], body_style))
    else:
        story.append(Paragraph(
            f"Eligible: {'Yes' if loan_info['eligible'] else 'No'} | "
            f"Approval Probability: {loan_info['probability']}% | Risk Score: {loan_info['risk_score']}",
            body_style,
        ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Forecast Summary", h2_style))
    story.append(Paragraph(forecast_note or "No forecast available.", body_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("AI-Generated Report", h2_style))
    for para in (ai_report_text or "No AI report generated.").split("\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), body_style))
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

    cols = st.columns(4)
    with cols[0]:
        metric_card("Current Balance", format_currency(summary["balance"], currency))
    with cols[1]:
        metric_card("Income", format_currency(summary["income"], currency))
    with cols[2]:
        metric_card("Expenses", format_currency(summary["expenses"], currency))
    with cols[3]:
        metric_card("Savings Rate", f"{summary['savings_rate']:.1f}%",
                     delta=f"{summary['savings_rate']:.1f}%", delta_positive=summary["savings_rate"] >= 0)

    cols2 = st.columns(4)
    with cols2[0]:
        metric_card("Investments", format_currency(summary["investments"], currency))
    with cols2[1]:
        metric_card("Emergency Fund", format_currency(summary["emergency_fund"], currency))
    with cols2[2]:
        metric_card("Health Score", f"{health['score']} / 100")
    loan_quick = predict_loan_eligibility(35, max(summary["income"], 1), 700, summary["expenses"] * 0.1, 1)
    with cols2[3]:
        metric_card("Loan Risk (est.)", f"{loan_quick.get('risk_score', 0)}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fig = go.Figure(data=[go.Pie(
            labels=["Income", "Expenses"], values=[summary["income"], summary["expenses"]],
            hole=0.55, marker_colors=[COLORS["accent"], COLORS["danger"]],
        )])
        fig.update_layout(**plotly_theme(), title="Income vs Expense", height=340)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cat_df = df[df["type"] == "expense"].groupby("category")["amount"].sum().reset_index()
        if not cat_df.empty:
            fig = px.pie(cat_df, names="category", values="amount", hole=0.5)
            fig.update_layout(**plotly_theme(), title="Expense by Category", height=340)
            st.plotly_chart(fig, use_container_width=True)
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
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ms["month"], y=ms["savings"], mode="lines+markers",
                                      name="Savings", line=dict(color=COLORS["primary"], width=3)))
            fig.update_layout(**plotly_theme(), title="Savings Trend", height=340)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Recent Transactions")
    st.dataframe(
        df.head(10)[["date", "category", "description", "amount", "type"]],
        use_container_width=True, hide_index=True,
    )

    render_ai_summary_card(user_id, summary, health, currency)


def render_ai_summary_card(user_id, summary, health, currency):
    st.markdown("#### \U0001f9e0 AI Summary")
    settings = get_settings(user_id)
    if not settings["enable_ai"]:
        st.info("AI features are disabled in Settings.")
        return
    with st.container(border=True):
        if st.button("Generate AI Summary", key="ai_summary_btn"):
            with st.spinner("Asking FinPilot AI Advisor..."):
                context = build_financial_context(summary, health, "See Forecast page for details.", currency)
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

    tab1, tab2, tab3 = st.tabs(["\U0001f4cb View & Manage", "\u2795 Add Manually", "\U0001f4e4 Upload CSV"])

    with tab1:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            search = st.text_input("Search description/category", "")
        with col_b:
            cat_filter = st.multiselect("Filter by category", TRANSACTION_CATEGORIES)
        with col_c:
            type_filter = st.selectbox("Type", ["All", "income", "expense"])

        filtered = df.copy()
        if search:
            mask = (
                filtered["description"].str.contains(search, case=False, na=False)
                | filtered["category"].str.contains(search, case=False, na=False)
            )
            filtered = filtered[mask]
        if cat_filter:
            filtered = filtered[filtered["category"].isin(cat_filter)]
        if type_filter != "All":
            filtered = filtered[filtered["type"] == type_filter]

        sort_col = st.selectbox("Sort by", ["date", "amount", "category"], key="sort_col")
        filtered = filtered.sort_values(sort_col, ascending=False)

        st.dataframe(
            filtered[["id", "date", "category", "description", "amount", "type"]],
            use_container_width=True, hide_index=True,
        )

        st.markdown("##### Edit or Delete a Transaction")
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
                    update_clicked = st.form_submit_button("\U0001f4be Update", use_container_width=True)
                with fc2:
                    delete_clicked = st.form_submit_button("\U0001f5d1\ufe0f Delete", use_container_width=True)

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
            submitted = st.form_submit_button("Add Transaction", use_container_width=True)
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
        st.caption("CSV must include columns: date, category, description, amount, type")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            try:
                with st.spinner("Parsing and validating CSV..."):
                    csv_df = pd.read_csv(uploaded)
                    required_cols = {"date", "category", "description", "amount", "type"}
                    missing = required_cols - set(c.lower() for c in csv_df.columns)
                    if missing:
                        st.error(f"CSV is missing required columns: {', '.join(missing)}")
                    else:
                        csv_df.columns = [c.lower() for c in csv_df.columns]
                        csv_df = csv_df.dropna(subset=["date", "amount"])
                        csv_df["amount"] = pd.to_numeric(csv_df["amount"], errors="coerce")
                        csv_df = csv_df.dropna(subset=["amount"])

                        st.dataframe(csv_df.head(20), use_container_width=True, hide_index=True)
                        if st.button(f"Import {len(csv_df)} transactions", use_container_width=True):
                            with st.spinner("Importing transactions..."):
                                with get_conn() as conn:
                                    rows = [
                                        (user_id, str(r["date"]), r.get("category", "Other"),
                                         r.get("description", ""), float(r["amount"]),
                                         r.get("type", "expense"))
                                        for _, r in csv_df.iterrows()
                                    ]
                                    conn.executemany(
                                        "INSERT INTO transactions (user_id, date, category, description, amount, type) VALUES (?,?,?,?,?,?)",
                                        rows,
                                    )
                                load_transactions.clear()
                            st.success(f"Imported {len(csv_df)} transactions.")
                            st.rerun()
            except pd.errors.ParserError as e:
                st.error(f"Could not parse CSV file: {e}")
            except Exception as e:
                st.error(f"Unexpected error while processing CSV: {e}")


####################################
# Budget Planner Page
####################################
def page_budget_planner(user_id, currency):
    st.markdown("### \U0001f9ee Budget Planner")
    st.caption("Plan your monthly budget across key categories.")

    with st.form("budget_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            salary = st.number_input("Salary", min_value=0.0, value=80000.0, step=1000.0)
            rent = st.number_input("Rent", min_value=0.0, value=18000.0, step=500.0)
            food = st.number_input("Food", min_value=0.0, value=8000.0, step=500.0)
        with c2:
            shopping = st.number_input("Shopping", min_value=0.0, value=5000.0, step=500.0)
            travel = st.number_input("Travel", min_value=0.0, value=3000.0, step=500.0)
            utilities = st.number_input("Utilities", min_value=0.0, value=2500.0, step=200.0)
        with c3:
            insurance = st.number_input("Insurance", min_value=0.0, value=2000.0, step=200.0)
            emis = st.number_input("EMIs", min_value=0.0, value=9000.0, step=500.0)
            savings_goal = st.number_input("Savings Goal", min_value=0.0, value=10000.0, step=500.0)
        submitted = st.form_submit_button("Analyze Budget", use_container_width=True)

    if submitted:
        expenses = {
            "Rent": rent, "Food": food, "Shopping": shopping, "Travel": travel,
            "Utilities": utilities, "Insurance": insurance, "EMIs": emis,
        }
        total_expenses = sum(expenses.values())
        remaining = salary - total_expenses - savings_goal

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Total Expenses", format_currency(total_expenses, currency))
        with c2:
            metric_card("Savings Goal", format_currency(savings_goal, currency))
        with c3:
            metric_card("Remaining Income", format_currency(remaining, currency),
                        delta=format_currency(remaining, currency), delta_positive=remaining >= 0)

        if remaining < 0:
            st.markdown('<span class="badge-bad">\u26a0 Over Budget</span>', unsafe_allow_html=True)
            st.warning("Your planned expenses and savings goal exceed your salary. Consider reducing "
                       "discretionary spending (Shopping, Travel, Entertainment).")
        elif remaining < salary * 0.05:
            st.markdown('<span class="badge-warn">Tight Budget</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-good">Healthy Budget</span>', unsafe_allow_html=True)

        fig = px.pie(names=list(expenses.keys()), values=list(expenses.values()), hole=0.5,
                     title="Budget Allocation")
        fig.update_layout(**plotly_theme(), height=380)
        st.plotly_chart(fig, use_container_width=True)

        recommended = {
            "Rent": 0.30, "Food": 0.12, "Shopping": 0.08, "Travel": 0.05,
            "Utilities": 0.05, "Insurance": 0.05, "EMIs": 0.15,
        }
        rec_df = pd.DataFrame({
            "Category": list(recommended.keys()),
            "Your %": [expenses[k] / salary * 100 if salary else 0 for k in recommended],
            "Recommended %": [v * 100 for v in recommended.values()],
        })
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=rec_df["Category"], y=rec_df["Your %"], name="Your Budget", marker_color=COLORS["primary"]))
        fig2.add_trace(go.Bar(x=rec_df["Category"], y=rec_df["Recommended %"], name="Recommended", marker_color=COLORS["accent"]))
        fig2.update_layout(**plotly_theme(), title="Your Budget vs Recommended (%)", barmode="group", height=380)
        st.plotly_chart(fig2, use_container_width=True)


####################################
# Loan Eligibility Page
####################################
def page_loan_eligibility(currency):
    st.markdown("### \U0001f3e6 Loan Eligibility")
    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is not installed. Please install it to use this feature.")
        return

    with st.form("loan_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 21, 65, 32)
            salary = st.number_input("Monthly Salary", min_value=5000.0, value=60000.0, step=1000.0)
            credit_score = st.slider("Credit Score", 300, 900, 720)
        with c2:
            emi = st.number_input("Current Monthly EMI", min_value=0.0, value=5000.0, step=500.0)
            existing_loans = st.slider("Existing Loans", 0, 5, 1)
        submitted = st.form_submit_button("Check Eligibility", use_container_width=True)

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

    with st.spinner("Running forecast model..."):
        forecast_df, note = run_forecast(df, metric, months)

    if forecast_df is None:
        st.warning(note)
        return
    if note:
        st.caption(f"\u2139\ufe0f {note}")

    ms = monthly_series(df)
    render_forecast_chart(forecast_df, ms, metric.capitalize(), currency)

    st.session_state["last_forecast_note"] = (
        f"{metric.capitalize()} projected to reach "
        f"{format_currency(forecast_df['yhat'].iloc[-1], currency)} in {months} months."
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
    context = build_financial_context(summary, health, forecast_note, currency)

    st.markdown("##### Quick Questions")
    quick_questions = [
        "Can I buy a car?", "Can I buy a house?", "How do I save more?",
        "Should I invest?", "Should I take a loan?", "How can I improve my score?",
    ]
    cols = st.columns(3)
    clicked_question = None
    for i, q in enumerate(quick_questions):
        with cols[i % 3]:
            if st.button(q, key=f"qq_{i}", use_container_width=True):
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
    loan_info = predict_loan_eligibility(35, max(summary["income"], 1), 700, summary["expenses"] * 0.1, 1)
    forecast_note = st.session_state.get("last_forecast_note", "No forecast generated yet.")
    context = build_financial_context(summary, health, forecast_note, currency)

    if st.button("\U0001f9e0 Generate AI Report (Mistral)", use_container_width=True):
        with st.spinner("Generating professional report..."):
            ai_report = generate_ai_report(context, loan_info, forecast_note)
        st.session_state["ai_report_text"] = ai_report

    ai_report_text = st.session_state.get("ai_report_text", "")
    if ai_report_text:
        st.markdown("#### AI-Generated Report")
        st.markdown(ai_report_text)

        if REPORTLAB_AVAILABLE:
            with st.spinner("Preparing PDF..."):
                user_name = st.session_state.user.get("name", "User")
                pdf_buffer = generate_pdf_report(
                    user_name, summary, health, loan_info, forecast_note, ai_report_text, currency,
                )
            if pdf_buffer:
                st.download_button(
                    "\u2b07\ufe0f Download PDF Report", data=pdf_buffer,
                    file_name=f"finpilot_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf", use_container_width=True,
                )
        else:
            st.warning("ReportLab is not installed; PDF export unavailable.")

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
            st.dataframe(reports_df, use_container_width=True, hide_index=True)
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

        saved = st.form_submit_button("Save Settings", use_container_width=True)
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
        if st.button("💾 Use these keys for this session", use_container_width=True):
            st.session_state.groq_api_key_override = groq_key_input.strip()
            st.session_state.mistral_api_key_override = mistral_key_input.strip()
            # Clear cached clients so the new key takes effect immediately
            _build_groq_client.clear()
            _build_mistral_client.clear()
            st.success("Session API keys updated.")
            st.rerun()
    with kc2:
        if st.button("🧹 Clear session keys", use_container_width=True):
            st.session_state.groq_api_key_override = ""
            st.session_state.mistral_api_key_override = ""
            _build_groq_client.clear()
            _build_mistral_client.clear()
            st.info("Session API keys cleared.")
            st.rerun()

    st.markdown("---")
    st.markdown("#### Data Management")
    c1, c2 = st.columns(2)
    with c1:
        df = load_transactions(user_id)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("\u2b07\ufe0f Export Transactions (CSV)", data=csv_data,
                            file_name="finpilot_transactions.csv", mime="text/csv",
                            use_container_width=True)
    with c2:
        confirm = st.checkbox("I understand this will permanently delete all data")
        if st.button("\U0001f5d1\ufe0f Reset Database", use_container_width=True, disabled=not confirm):
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
            "Dashboard", "Transactions", "Budget Planner", "Financial Health",
            "Loan Eligibility", "Forecast", "AI Advisor", "Reports", "Settings",
        ]
        icons = {
            "Dashboard": "\U0001f4ca", "Transactions": "\U0001f4b3", "Budget Planner": "\U0001f9ee",
            "Financial Health": "\U0001f49a", "Loan Eligibility": "\U0001f3e6", "Forecast": "\U0001f4c8",
            "AI Advisor": "\U0001f9e0", "Reports": "\U0001f4c4", "Settings": "\u2699\ufe0f",
        }
        current_page = st.session_state.get("page", "Dashboard")
        for item in nav_items:
            label = f"{icons[item]}  {item}"
            if st.button(label, key=f"nav_{item}", use_container_width=True,
                         type="primary" if current_page == item else "secondary"):
                st.session_state.page = item
                st.rerun()

        st.markdown("---")
        if st.button("\U0001f6aa  Logout", key="nav_logout", use_container_width=True):
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
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("##### Suggestions")
                for s in health["suggestions"]:
                    st.markdown(f"- {s}")
        elif page == "Loan Eligibility":
            page_loan_eligibility(currency)
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
