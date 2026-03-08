"""
WFO Risk Prediction Dashboard
Streamlit UI for HR teams to visualise risk distribution and predict individual employees.
Run: streamlit run ui.py
"""
import json
from pathlib import Path

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="WFO Risk Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1e3a5f, #0d2137);
    padding: 1.5rem 2rem; border-radius: 12px;
    margin-bottom: 1.5rem; color: white;
}
.metric-card {
    background: #f8fafc; border-radius: 10px;
    padding: 1rem; text-align: center;
    border: 1px solid #e2e8f0;
}
.risk-high { color: #e53e3e; font-weight: 700; }
.risk-med  { color: #dd6b20; font-weight: 700; }
.risk-low  { color: #38a169; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h2 style="margin:0">🏢 WFO Risk Prediction Dashboard</h2>
  <p style="margin:.3rem 0 0;opacity:.8;font-size:.9rem">
  ML-powered employee return-to-office risk scoring · HR Analytics Tool · 2022
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Navigation")
    page = st.radio("", ["📊 Dataset Overview", "🔮 Predict Employee", "📈 Model Comparison"])
    st.divider()

    # API health
    try:
        h = httpx.get(f"{API_BASE}/health", timeout=3)
        hdata = h.json()
        if hdata.get("model_ready"):
            st.success("✅ API Connected · Model Ready")
        else:
            st.warning("⚠️ API Connected · Model not trained\nRun: python train.py")
    except Exception:
        st.error("❌ API not reachable\nRun: python run.py")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    path = Path("data/employee_wfo_data.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_metrics():
    path = Path("models/metrics.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

df = load_dataset()
metrics = load_metrics()

# ══════════════════════════════════════════════════════════
# PAGE 1: DATASET OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "📊 Dataset Overview":
    if df is None:
        st.warning("No dataset found. Run: `python train.py` first.")
        st.stop()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", f"{len(df):,}")
    col2.metric("High Risk", f"{(df['risk_category']=='High').sum():,}",
                f"{(df['risk_category']=='High').mean()*100:.1f}%")
    col3.metric("Medium Risk", f"{(df['risk_category']=='Medium').sum():,}",
                f"{(df['risk_category']=='Medium').mean()*100:.1f}%")
    col4.metric("Low Risk", f"{(df['risk_category']=='Low').sum():,}",
                f"{(df['risk_category']=='Low').mean()*100:.1f}%")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        # Risk distribution pie
        risk_counts = df["risk_category"].value_counts().reset_index()
        risk_counts.columns = ["category", "count"]
        fig_pie = px.pie(
            risk_counts, values="count", names="category",
            title="Employee Risk Distribution",
            color="category",
            color_discrete_map={"High": "#e53e3e", "Medium": "#dd6b20", "Low": "#38a169"},
            hole=0.4,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        # Risk score histogram
        fig_hist = px.histogram(
            df, x="risk_score", color="risk_category",
            title="Risk Score Distribution",
            color_discrete_map={"High": "#e53e3e", "Medium": "#dd6b20", "Low": "#38a169"},
            nbins=40, barmode="overlay", opacity=0.75,
        )
        fig_hist.update_xaxes(title="Risk Score (0-1)")
        st.plotly_chart(fig_hist, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        # Commute vs risk score scatter
        fig_scatter = px.scatter(
            df.sample(300), x="commute_distance_km", y="risk_score",
            color="risk_category",
            title="Commute Distance vs Risk Score",
            color_discrete_map={"High": "#e53e3e", "Medium": "#dd6b20", "Low": "#38a169"},
            opacity=0.7,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_d:
        # Children vs risk
        children_risk = df.groupby(["has_children_under_5", "risk_category"]).size().reset_index(name="count")
        children_risk["has_children"] = children_risk["has_children_under_5"].map({0: "No Children", 1: "Has Children <5"})
        fig_bar = px.bar(
            children_risk, x="has_children", y="count", color="risk_category",
            title="Children Under 5 vs Risk Category",
            color_discrete_map={"High": "#e53e3e", "Medium": "#dd6b20", "Low": "#38a169"},
            barmode="group",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("📋 Sample Data")
    st.dataframe(
        df[["employee_id", "age", "commute_distance_km", "has_children_under_5",
            "anxiety_score", "productivity_wfh_score", "risk_score", "risk_category"]].head(20),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════
# PAGE 2: PREDICT EMPLOYEE
# ══════════════════════════════════════════════════════════
elif page == "🔮 Predict Employee":
    st.subheader("🔮 Predict WFO Risk for an Employee")
    st.caption("Enter employee details to get an ML-powered risk score")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            emp_id = st.text_input("Employee ID", value="EMP0042")
            age = st.slider("Age", 18, 70, 34)
            commute = st.number_input("Commute Distance (km)", 0.0, 200.0, 35.0, step=0.5)
            has_children = st.selectbox("Has Children Under 5?", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with col2:
            vaccinated = st.selectbox("Vaccination Status", [1, 0], format_func=lambda x: "Vaccinated" if x else "Not Vaccinated")
            prior_wfo = st.slider("Pre-Pandemic WFO Days/Week", 0, 5, 3)
            internet = st.slider("Home Internet Quality (1-10)", 1, 10, 7)
            team_size = st.number_input("Team Size", 1, 200, 12)

        with col3:
            manager_wfo = st.selectbox("Manager Plans to WFO?", [1, 0], format_func=lambda x: "Yes" if x else "No")
            anxiety = st.slider("Anxiety Score (1-10)", 1, 10, 5)
            productivity = st.slider("WFH Productivity Score (1-10)", 1, 10, 7)

        submitted = st.form_submit_button("🔮 Predict Risk", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "employee_id": emp_id,
            "age": age,
            "commute_distance_km": commute,
            "has_children_under_5": has_children,
            "vaccination_status": vaccinated,
            "prior_wfo_days_per_week": prior_wfo,
            "home_internet_quality": internet,
            "team_size": team_size,
            "manager_wfo": manager_wfo,
            "anxiety_score": anxiety,
            "productivity_wfh_score": productivity,
        }
        try:
            resp = httpx.post(f"{API_BASE}/predict", json=payload, timeout=10)
            result = resp.json()

            st.divider()
            cat = result["risk_category"]
            css_class = {"High": "risk-high", "Medium": "risk-med", "Low": "risk-low"}[cat]

            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Score", f"{result['risk_score_pct']}%")
            col2.metric("Risk Category", cat)
            col3.metric("Latency", f"{result['latency_ms']}ms")

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["risk_score_pct"],
                title={"text": f"WFO Risk Score — {emp_id}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e53e3e" if cat == "High" else "#dd6b20" if cat == "Medium" else "#38a169"},
                    "steps": [
                        {"range": [0, 33], "color": "#c6f6d5"},
                        {"range": [33, 66], "color": "#feebc8"},
                        {"range": [66, 100], "color": "#fed7d7"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 3}, "value": result["risk_score_pct"]},
                },
                number={"suffix": "%"},
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.info(f"💡 **Recommendation:** {result['recommendation']}")

        except httpx.ConnectError:
            st.error("❌ Cannot connect to API. Start with: python run.py")
        except Exception as e:
            st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════
# PAGE 3: MODEL COMPARISON
# ══════════════════════════════════════════════════════════
elif page == "📈 Model Comparison":
    if metrics is None:
        st.warning("No model metrics found. Run: `python train.py` first.")
        st.stop()

    st.subheader("📈 Model Comparison — Logistic Regression vs Random Forest")

    lr = metrics["logistic_regression"]
    rf = metrics["random_forest"]
    best = metrics["best_model"]

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LR Accuracy", f"{lr['accuracy']*100:.1f}%")
    col2.metric("LR ROC-AUC", f"{lr['roc_auc']:.4f}")
    col3.metric("RF Accuracy", f"{rf['accuracy']*100:.1f}%")
    col4.metric("RF ROC-AUC", f"{rf['roc_auc']:.4f}")

    st.success(f"✅ Best Model Selected: **{best.replace('_', ' ').title()}** (by CV ROC-AUC)")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        # Model comparison bar
        compare_df = pd.DataFrame({
            "Metric": ["Accuracy", "ROC-AUC", "CV ROC-AUC"],
            "Logistic Regression": [lr["accuracy"], lr["roc_auc"], lr["cv_roc_auc_mean"]],
            "Random Forest": [rf["accuracy"], rf["roc_auc"], rf["cv_roc_auc_mean"]],
        })
        fig_compare = px.bar(
            compare_df.melt(id_vars="Metric", var_name="Model", value_name="Score"),
            x="Metric", y="Score", color="Model", barmode="group",
            title="Model Performance Comparison",
            color_discrete_map={"Logistic Regression": "#4299e1", "Random Forest": "#48bb78"},
        )
        fig_compare.update_yaxes(range=[0.5, 1.0])
        st.plotly_chart(fig_compare, use_container_width=True)

    with col_b:
        # Feature importances (RF only)
        if "feature_importances" in rf:
            fi = pd.DataFrame(
                list(rf["feature_importances"].items()),
                columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)
            fig_fi = px.bar(
                fi, x="Importance", y="Feature", orientation="h",
                title="Random Forest — Feature Importances",
                color="Importance", color_continuous_scale="Greens",
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    # Confusion matrices
    st.subheader("Confusion Matrices")
    col_c, col_d = st.columns(2)

    for col, name, m in [(col_c, "Logistic Regression", lr), (col_d, "Random Forest", rf)]:
        with col:
            cm = m["confusion_matrix"]
            fig_cm = px.imshow(
                cm, text_auto=True,
                title=f"{name} — Confusion Matrix",
                labels={"x": "Predicted", "y": "Actual"},
                x=["Low Risk", "High Risk"], y=["Low Risk", "High Risk"],
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig_cm, use_container_width=True)
