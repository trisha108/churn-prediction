"""
dashboard/app.py — Churn Prediction Intelligence Platform
Panels: KPIs, dept attrition, risk distribution, SHAP importances,
        6-month forecast, high-risk table, LLM retention recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path
import dash
from dash import dcc, html, dash_table, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import json

ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data"
OUTPUT_DIR   = ROOT / "outputs"

RAW_CSV      = DATA_DIR  / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
PREDS_CSV    = OUTPUT_DIR / "predictions.csv"
FI_CSV       = OUTPUT_DIR / "feature_importances.csv"
SHAP_CSV     = OUTPUT_DIR / "shap_values.csv"
FORECAST_CSV = OUTPUT_DIR / "forecast_data.csv"
RECO_CSV     = OUTPUT_DIR / "retention_recommendations.csv"

def load_data():
    raw      = pd.read_csv(RAW_CSV)      if RAW_CSV.exists()      else pd.DataFrame()
    preds    = pd.read_csv(PREDS_CSV)    if PREDS_CSV.exists()    else pd.DataFrame()
    fi       = pd.read_csv(FI_CSV)       if FI_CSV.exists()       else pd.DataFrame()
    shap     = pd.read_csv(SHAP_CSV)     if SHAP_CSV.exists()     else pd.DataFrame()
    forecast = pd.read_csv(FORECAST_CSV) if FORECAST_CSV.exists() else pd.DataFrame()
    reco     = pd.read_csv(RECO_CSV)     if RECO_CSV.exists()     else pd.DataFrame()
    return raw, preds, fi, shap, forecast, reco

raw_df, pred_df, fi_df, shap_df, forecast_df, reco_df = load_data()

COLORS = {
    "primary": "#2563EB", "success": "#16A34A", "danger": "#DC2626",
    "warning": "#D97706", "bg": "#0F172A", "card": "#1E293B",
    "border": "#334155", "text": "#F1F5F9", "muted": "#94A3B8",
    "purple": "#7C3AED",
}
CARD = {"background": COLORS["card"], "borderRadius": "12px", "padding": "20px",
        "border": f"1px solid {COLORS['border']}", "marginBottom": "16px"}

def kpis():
    total         = len(raw_df)
    attr_rate     = (raw_df["Attrition"] == "Yes").mean() * 100 if not raw_df.empty else 0
    high_risk     = int((pred_df["ChurnProbability"] >= 0.65).sum()) if not pred_df.empty else 0
    has_reco      = len(reco_df) if not reco_df.empty else 0
    return total, attr_rate, high_risk, has_reco

total, attr_rate, high_risk, has_reco = kpis()

# ── Figures ────────────────────────────────────────────────────────────────
def dept_fig():
    if raw_df.empty: return go.Figure()
    d = (raw_df.groupby("Department")["Attrition"]
         .apply(lambda x: (x=="Yes").mean()*100).reset_index())
    d.columns = ["Department","Rate"]
    d = d.sort_values("Rate", ascending=True)
    fig = go.Figure(go.Bar(
        x=d["Rate"], y=d["Department"], orientation="h",
        marker_color=[COLORS["danger"] if v>18 else COLORS["primary"] for v in d["Rate"]],
        text=[f"{v:.1f}%" for v in d["Rate"]], textposition="outside"))
    fig.update_layout(title="Attrition Rate by Department", xaxis_title="Rate (%)",
                      plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"],
                      font_color=COLORS["text"], height=280,
                      margin=dict(l=10,r=40,t=40,b=10))
    return fig

def dist_fig():
    if pred_df.empty: return go.Figure()
    fig = go.Figure(go.Histogram(x=pred_df["ChurnProbability"], nbinsx=30,
                                  marker_color=COLORS["primary"], opacity=0.85))
    fig.add_vline(x=0.35, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="Medium")
    fig.add_vline(x=0.65, line_dash="dash", line_color=COLORS["danger"],
                  annotation_text="High")
    fig.update_layout(title="Churn Probability Distribution",
                      xaxis_title="Probability", yaxis_title="Count",
                      plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"],
                      font_color=COLORS["text"], height=280,
                      margin=dict(l=10,r=10,t=40,b=10))
    return fig

def shap_fig():
    # Prefer SHAP values; fall back to XGBoost importances
    source = shap_df if not shap_df.empty else fi_df
    if source.empty: return go.Figure()
    if not shap_df.empty:
        top = shap_df.head(12).reset_index()
        top.columns = [c for c in top.columns]
        top = top[[top.columns[1], top.columns[2]]] if len(top.columns) > 2 else top.iloc[:, :2]
        top.columns = ["Feature","Value"]
        top["Value"] = pd.to_numeric(top["Value"], errors="coerce").fillna(0)
        title = "Top 12 SHAP Feature Importances (Mean |SHAP|)"
    else:
        top = fi_df.sort_values("Importance").tail(12)
        top = top.rename(columns={"Importance":"Value"})
        title = "Top 12 Feature Importances (XGBoost Gain)"
    fig = go.Figure(go.Bar(
        x=top["Value"], y=top["Feature"], orientation="h",
        marker=dict(color=top["Value"], colorscale="Blues", showscale=False),
        text=[f"{float(v):.4f}" for v in top["Value"]], textposition="outside"))
    fig.update_layout(title=title, xaxis_title="Importance",
                      plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"],
                      font_color=COLORS["text"], height=380,
                      margin=dict(l=10,r=60,t=40,b=10))
    return fig

def forecast_fig():
    if forecast_df.empty: return go.Figure()
    fig = go.Figure()
    dept_colors = {"Sales": COLORS["primary"],
                   "Research & Development": COLORS["success"],
                   "Human Resources": COLORS["danger"]}
    for dept in forecast_df["Department"].unique():
        ddf   = forecast_df[forecast_df["Department"]==dept]
        color = dept_colors.get(dept, COLORS["purple"])
        fig.add_trace(go.Scatter(x=ddf["ForecastMonth"], y=ddf["ForecastRate"],
                                  mode="lines+markers", name=dept,
                                  line=dict(color=color, width=2.5), marker=dict(size=7)))
        fig.add_trace(go.Scatter(
            x=list(ddf["ForecastMonth"])+list(ddf["ForecastMonth"])[::-1],
            y=list(ddf["CI_Upper"])+list(ddf["CI_Lower"])[::-1],
            fill="toself", fillcolor=color.replace(")", ",0.12)").replace("rgb","rgba"),
            line=dict(color="rgba(0,0,0,0)"), showlegend=False))
    fig.update_layout(title="6-Month Churn Rate Forecast by Department",
                      xaxis_title="Month", yaxis_title="Rate (%)",
                      plot_bgcolor=COLORS["card"], paper_bgcolor=COLORS["card"],
                      font_color=COLORS["text"], legend=dict(bgcolor=COLORS["card"]),
                      height=380, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def high_risk_table():
    if raw_df.empty or pred_df.empty: return []
    raw_s  = raw_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n      = min(len(pred_df), len(raw_s))
    merged = raw_s.iloc[:n].copy()
    merged["ChurnProbability"] = pred_df["ChurnProbability"].values[:n]
    merged["ChurnRisk"] = merged["ChurnProbability"].apply(
        lambda p: "🔴 High" if p>=0.65 else ("🟡 Medium" if p>=0.35 else "🟢 Low"))
    hr = (merged[merged["ChurnProbability"]>=0.5]
          [["EmployeeNumber","Age","Department","JobRole",
            "MonthlyIncome","YearsAtCompany","ChurnProbability","ChurnRisk"]]
          .sort_values("ChurnProbability", ascending=False).head(15))
    hr["ChurnProbability"] = hr["ChurnProbability"].apply(lambda x: f"{x:.2%}")
    hr["MonthlyIncome"]    = hr["MonthlyIncome"].apply(lambda x: f"${x:,.0f}")
    return hr.to_dict("records")

def reco_table():
    if reco_df.empty: return []
    cols = ["EmployeeNumber","JobRole","Department","ChurnProbability",
            "RetentionLever","TopRiskFactors","ImmediateActions"]
    avail = [c for c in cols if c in reco_df.columns]
    df = reco_df[avail].copy()
    if "ChurnProbability" in df.columns:
        df["ChurnProbability"] = df["ChurnProbability"].apply(lambda x: f"{x:.0%}")
    return df.to_dict("records")

# ── Table style helpers ────────────────────────────────────────────────────
TABLE_CELL = {"backgroundColor": COLORS["card"], "color": COLORS["text"],
              "border": f"1px solid {COLORS['border']}", "padding": "10px 14px",
              "fontSize": "13px", "fontFamily": "'Inter', sans-serif"}
TABLE_HDR  = {"backgroundColor": COLORS["bg"], "fontWeight": "600",
              "fontSize": "12px", "color": COLORS["muted"],
              "textTransform": "uppercase", "letterSpacing": "0.05em"}

# ── Layout ─────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Churn Intelligence Platform",
                suppress_callback_exceptions=True)

app.layout = html.Div(
    style={"backgroundColor": COLORS["bg"], "minHeight": "100vh",
           "fontFamily": "'Inter','Segoe UI',sans-serif", "color": COLORS["text"]},
    children=[
        # Header
        html.Div(
            style={"background": "linear-gradient(135deg,#1E40AF 0%,#1E293B 100%)",
                   "padding": "28px 40px", "borderBottom": f"1px solid {COLORS['border']}"},
            children=[
                html.H1("Churn Prediction Intelligence Platform",
                        style={"margin":0,"fontSize":"26px","fontWeight":700}),
                html.P("IBM HR Analytics · XGBoost (Tuned) · SHAP · MLflow · FastAPI · Holt-Winters · Claude AI",
                       style={"margin":"4px 0 0","color":COLORS["muted"],"fontSize":"13px"}),
            ]),

        html.Div(style={"padding": "28px 40px"}, children=[

            # KPI cards
            html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(200px,1fr))",
                            "gap":"16px","marginBottom":"24px"}, children=[
                html.Div(style={**CARD,"borderLeft":f"4px solid {COLORS['primary']}"},children=[
                    html.P("Total Employees",style={"margin":0,"color":COLORS["muted"],"fontSize":"12px"}),
                    html.H2(f"{total:,}",style={"margin":"4px 0 0","fontSize":"32px","fontWeight":700})]),
                html.Div(style={**CARD,"borderLeft":f"4px solid {COLORS['danger']}"},children=[
                    html.P("Overall Attrition Rate",style={"margin":0,"color":COLORS["muted"],"fontSize":"12px"}),
                    html.H2(f"{attr_rate:.1f}%",style={"margin":"4px 0 0","fontSize":"32px","fontWeight":700,"color":COLORS["danger"]})]),
                html.Div(style={**CARD,"borderLeft":f"4px solid {COLORS['warning']}"},children=[
                    html.P("High-Risk Employees (≥65%)",style={"margin":0,"color":COLORS["muted"],"fontSize":"12px"}),
                    html.H2(f"{high_risk:,}",style={"margin":"4px 0 0","fontSize":"32px","fontWeight":700,"color":COLORS["warning"]})]),
                html.Div(style={**CARD,"borderLeft":f"4px solid {COLORS['success']}"},children=[
                    html.P("AI Retention Plans Generated",style={"margin":0,"color":COLORS["muted"],"fontSize":"12px"}),
                    html.H2(f"{has_reco}",style={"margin":"4px 0 0","fontSize":"32px","fontWeight":700,"color":COLORS["success"]}),
                    html.P("via Claude API",style={"margin":"2px 0 0","color":COLORS["muted"],"fontSize":"11px"})]),
            ]),

            # Row 1: dept + dist
            html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(400px,1fr))",
                            "gap":"16px","marginBottom":"16px"}, children=[
                html.Div(style=CARD,children=[dcc.Graph(figure=dept_fig(),config={"displayModeBar":False})]),
                html.Div(style=CARD,children=[dcc.Graph(figure=dist_fig(),config={"displayModeBar":False})]),
            ]),

            # Row 2: SHAP + forecast
            html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(400px,1fr))",
                            "gap":"16px","marginBottom":"16px"}, children=[
                html.Div(style=CARD,children=[dcc.Graph(figure=shap_fig(),config={"displayModeBar":False})]),
                html.Div(style=CARD,children=[dcc.Graph(figure=forecast_fig(),config={"displayModeBar":False})]),
            ]),

            # High-risk table
            html.Div(style=CARD, children=[
                html.H3("High-Risk Employee Segments (Predicted Churn ≥ 50%)",
                        style={"margin":"0 0 16px","fontSize":"15px","fontWeight":600}),
                dash_table.DataTable(
                    data=high_risk_table(),
                    columns=[{"name":n,"id":i} for n,i in [
                        ("Emp #","EmployeeNumber"),("Age","Age"),("Department","Department"),
                        ("Job Role","JobRole"),("Income","MonthlyIncome"),
                        ("Yrs @ Co.","YearsAtCompany"),("Churn Prob","ChurnProbability"),
                        ("Risk","ChurnRisk")]],
                    style_table={"overflowX":"auto"},
                    style_cell=TABLE_CELL, style_header=TABLE_HDR,
                    style_data_conditional=[
                        {"if":{"filter_query":'{ChurnRisk} contains "High"'},"color":COLORS["danger"]},
                        {"if":{"filter_query":'{ChurnRisk} contains "Medium"'},"color":COLORS["warning"]},
                    ],
                    page_size=8, sort_action="native", filter_action="native")
            ]),

            # AI Retention Recommendations
            html.Div(style=CARD, children=[
                html.Div(style={"display":"flex","alignItems":"center","marginBottom":"16px"}, children=[
                    html.H3("🤖 AI-Powered Retention Recommendations",
                            style={"margin":0,"fontSize":"15px","fontWeight":600}),
                    html.Span("Powered by Claude API",
                              style={"marginLeft":"12px","padding":"2px 10px",
                                     "backgroundColor":"#1E40AF","borderRadius":"12px",
                                     "fontSize":"11px","color":"#93C5FD"}),
                ]),
                html.P(
                    "Run 'python scripts/05_retention_recommendations.py' to generate AI action plans for high-risk employees."
                    if reco_df.empty else
                    f"Showing {len(reco_df)} personalized retention plans generated by Claude.",
                    style={"color":COLORS["muted"],"fontSize":"13px","margin":"0 0 12px"}),
                dash_table.DataTable(
                    data=reco_table(),
                    columns=[{"name":n,"id":i} for n,i in [
                        ("Emp #","EmployeeNumber"),("Role","JobRole"),("Dept","Department"),
                        ("Churn %","ChurnProbability"),("Key Lever","RetentionLever"),
                        ("Top Risk Factors","TopRiskFactors"),("Immediate Actions","ImmediateActions")]]
                    if not reco_df.empty else [],
                    style_table={"overflowX":"auto"},
                    style_cell={**TABLE_CELL,"maxWidth":"300px","overflow":"hidden",
                                "textOverflow":"ellipsis","whiteSpace":"nowrap"},
                    style_header=TABLE_HDR,
                    style_data_conditional=[
                        {"if":{"filter_query":'{RetentionLever} = "compensation"'},"color":COLORS["danger"]},
                        {"if":{"filter_query":'{RetentionLever} = "growth"'},"color":COLORS["warning"]},
                        {"if":{"filter_query":'{RetentionLever} = "flexibility"'},"color":COLORS["success"]},
                    ],
                    page_size=5, sort_action="native", tooltip_data=[
                        {col: {"value": str(row.get(col,"")), "type":"markdown"}
                         for col in ["ImmediateActions","TopRiskFactors"]}
                        for row in reco_table()
                    ] if not reco_df.empty else [],
                    tooltip_duration=None,
                ) if not reco_df.empty else html.P(
                    "No recommendations yet. Add your ANTHROPIC_API_KEY to .env and run script 05.",
                    style={"color":COLORS["muted"],"fontStyle":"italic","padding":"20px 0"})
            ]),

            html.P("IBM HR Analytics · XGBoost+SMOTE (Tuned) · SHAP · Holt-Winters · MLflow · FastAPI+Docker · Claude AI",
                   style={"textAlign":"center","color":COLORS["muted"],
                          "fontSize":"12px","marginTop":"8px"}),
        ]),
    ])

if __name__ == "__main__":
    import os; app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
