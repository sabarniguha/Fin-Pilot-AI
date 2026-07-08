# 💠 FinPilot AI

AI-powered banking analytics platform built for the **IDBI Innovate Hackathon**.
Single-file Streamlit app — no folders, no modules, just `app.py`.

---

## Features

- 🔐 **Login** with session-based auth (SQLite-backed, SHA-256 password hashing)
- 📊 **Dashboard** — balance, income, expenses, savings, investments, emergency fund, health score, loan risk, and charts (income vs expense, category breakdown, monthly trend, savings trend)
- 💳 **Transactions** — manual entry, CSV upload, search/filter/sort, inline edit & delete
- 🧮 **Budget Planner** — salary/expense inputs vs recommended allocation, budget health badges
- 💚 **Financial Health Score** — weighted 0–100 score (savings rate, debt ratio, investment ratio, emergency fund, income/expense stability) with an animated gauge and suggestions
- 🏦 **Loan Eligibility** — scikit-learn logistic regression model estimating approval probability and risk score
- 📈 **Forecast** — Prophet-based projections for income, expenses, and savings (falls back to a linear trend if Prophet isn't installed)
- 🧠 **AI Advisor** — Groq-powered chat for financial questions ("Can I buy a car?", "Should I invest?", etc.), grounded in your real transaction data
- 📄 **Reports** — Mistral-generated professional financial report + downloadable PDF (ReportLab)
- ⚙️ **Settings** — theme, currency, forecast horizon, Groq model, AI on/off, data export, database reset

---

## Requirements

```bash
pip install streamlit pandas numpy plotly scikit-learn prophet groq mistralai reportlab
```

All AI/ML packages are **soft dependencies** — if one isn't installed, that specific feature shows a friendly notice and the rest of the app keeps working normally:

| Package | Powers | Fallback if missing |
|---|---|---|
| `scikit-learn` | Loan Eligibility | Feature disabled with a clear message |
| `prophet` | Forecast | Automatic linear-trend fallback |
| `groq` | AI Advisor | Notice to set `GROQ_API_KEY` |
| `mistralai` | AI Reports | Notice to set `MISTRAL_API_KEY` |
| `reportlab` | PDF export | PDF button hidden, rest of Reports page still works |

---

## Setup

1. **Install dependencies** (see above).
2. **Set API keys** (optional, only needed for AI features):

   ```bash
   export GROQ_API_KEY="your_groq_key"
   export MISTRAL_API_KEY="your_mistral_key"
   ```

3. **Run the app**:

   ```bash
   streamlit run app.py
   ```

4. The SQLite database (`finpilot.db`) is created automatically on first launch, along with a demo user and ~9 months of synthetic transaction history.

---

## Demo Login

```
Email:    admin@example.com
Password: password123
```

---

## Project Structure (single file)

`app.py` is organized into clearly commented sections:

```
Imports → Configuration → Custom CSS → Database Functions → Authentication
→ Financial Analytics → Financial Health Score → Loan Prediction → Forecasting
→ Groq Client → Mistral Client → PDF Generator → Dashboard UI
→ Sidebar Navigation → Main App
```

---

## Notes

- Data resets are irreversible — the Settings page requires an explicit confirmation checkbox before wiping the database.
- CSV uploads must include the columns: `date, category, description, amount, type`.
- All monetary values respect the currency selected in Settings (INR, USD, EUR, GBP).






https://utzoudatf7q7v2nrfskmyk.streamlit.app/- The link of the app.
