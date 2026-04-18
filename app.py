import streamlit as st
import pandas as pd
import sqlite3
from datetime import date
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Finance Dashboard", layout="wide")

# ---------- UI STYLE ----------
st.markdown("""
<style>
body {background-color: #0E1117;}
.block-container {padding-top: 2rem;}
[data-testid="metric-container"] {
    background-color: #1C1F26;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #2A2D34;
}
.stButton>button {
    border-radius: 8px;
    background-color: #4CAF50;
    color: white;
}
section[data-testid="stSidebar"] {
    background-color: #161A23;
}
h1, h2, h3 {color: #EAECEF;}
</style>
""", unsafe_allow_html=True)

# ---------- DATABASE ----------
conn = sqlite3.connect("finance.db", check_same_thread=False)

# ---------- CACHE ----------
@st.cache_data
def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

@st.cache_data
def load_db():
    df = pd.read_sql_query("SELECT * FROM transactions", conn)
    if not df.empty:
        df["trans_date"] = pd.to_datetime(df["trans_date"])
    return df

# ---------- CLEAN ----------
def clean_data(df):
    df["trans_date"] = pd.to_datetime(df["trans_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["type"] = df["type"].astype(str).str.strip().str.capitalize()
    df = df.dropna(subset=["trans_date", "amount"])
    return df

# ---------- PREDICTION ----------
def predict_spending(df):
    exp = df[df["type"] == "Expense"].copy()
    if len(exp) < 5:
        return None

    exp = exp.sort_values("trans_date")
    exp["days"] = (exp["trans_date"] - exp["trans_date"].min()).dt.days

    model = LinearRegression()
    model.fit(exp[["days"]], exp["amount"])

    future_days = np.arange(exp["days"].max(), exp["days"].max() + 30)
    preds = model.predict(future_days.reshape(-1, 1))

    future_dates = pd.date_range(exp["trans_date"].max(), periods=30)

    return pd.DataFrame({"date": future_dates, "predicted": preds})

# ---------- ANOMALY ----------
def detect_anomalies(df):
    exp = df[df["type"] == "Expense"].copy()
    if exp.empty:
        return None

    mean = exp["amount"].mean()
    std = exp["amount"].std()

    exp["z"] = (exp["amount"] - mean) / std
    return exp[exp["z"].abs() > 2]

# ---------- SIDEBAR ----------
st.sidebar.title("Finance App")
st.sidebar.caption("Smart financial tracking")
page = st.sidebar.radio("Navigation", ["Dashboard", "Add Transaction", "Settings"])

# =========================
# DASHBOARD
# =========================
if page == "Dashboard":
    st.markdown("## Finance Dashboard")
    st.caption("Track. Analyze. Improve your financial decisions.")

    uploaded_file = st.file_uploader("Upload Excel/CSV", type=["xlsx", "csv"])

    if uploaded_file:
        st.session_state["file"] = uploaded_file

    if "file" in st.session_state:
        df = load_file(st.session_state["file"])
    else:
        df = load_db()

    if df is not None and not df.empty:

        df = clean_data(df)

        # FILTERS
        st.sidebar.divider()
        st.sidebar.subheader("Filters")

        start, end = st.sidebar.date_input(
            "Date Range",
            [df["trans_date"].min(), df["trans_date"].max()]
        )

        t_type = st.sidebar.selectbox("Type", ["All", "Expense", "Income"])

        df = df[(df["trans_date"] >= pd.to_datetime(start)) &
                (df["trans_date"] <= pd.to_datetime(end))]

        if t_type != "All":
            df = df[df["type"] == t_type]

        # KPIs
        income = df[df["type"] == "Income"]["amount"].sum()
        expense = df[df["type"] == "Expense"]["amount"].sum()
        balance = income - expense

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Income", f"₹{income:,.0f}")
        col2.metric("Expense", f"₹{expense:,.0f}")
        col3.metric("Balance", f"₹{balance:,.0f}")

        if income > 0:
            savings = ((income - expense) / income) * 100
            col4.metric("Savings Rate", f"{savings:.1f}%")

        # TABS
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Overview", "Trends", "AI Insights", "Data"]
        )

        # OVERVIEW
        with tab1:
            st.markdown("### Financial Overview")
            c1, c2 = st.columns(2)

            with c1:
                exp = df[df["type"] == "Expense"]
                if not exp.empty:
                    cat = exp.groupby("category")["amount"].sum()
                    st.bar_chart(cat)

            with c2:
                df["month"] = df["trans_date"].dt.to_period("M")
                trend = df.groupby("month")["amount"].sum()
                st.line_chart(trend)

        # TRENDS + PREDICTION
        with tab2:
            st.markdown("### Future Spending Prediction")
            pred = predict_spending(df)

            if pred is not None:
                st.line_chart(pred.set_index("date"))
            else:
                st.info("Not enough data")

        # AI INSIGHTS
        with tab3:
            st.markdown("### Anomaly Detection")
            anomalies = detect_anomalies(df)

            if anomalies is not None and not anomalies.empty:
                st.warning(f"{len(anomalies)} unusual transactions found")
                st.dataframe(anomalies[["trans_date", "amount", "category"]])
            else:
                st.success("No anomalies detected")

        # DATA
        with tab4:
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Data", csv, "data.csv")

    else:
        st.info("Upload data or add transactions")

# =========================
# ADD TRANSACTION
# =========================
elif page == "Add Transaction":
    st.title("Add Transaction")

    with st.form("form"):
        d = st.date_input("Date", value=date.today())
        amt = st.number_input("Amount", min_value=0.0)
        cat = st.text_input("Category")
        t = st.selectbox("Type", ["Expense", "Income"])
        note = st.text_input("Note")

        submit = st.form_submit_button("Add")

        if submit and amt > 0:
            conn.execute(
                "INSERT INTO transactions VALUES (NULL, ?, ?, ?, ?, ?)",
                (str(d), amt, cat, t, note)
            )
            conn.commit()
            st.success("Transaction added")

# =========================
# SETTINGS
# =========================
elif page == "Settings":
    st.title("Settings")

    if st.button("Clear Database"):
        conn.execute("DELETE FROM transactions")
        conn.commit()
        st.success("Database cleared")

conn.close()