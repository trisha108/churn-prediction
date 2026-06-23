"""
05_retention_recommendations.py
================================
LLM-powered retention recommendation engine using Anthropic Claude API.
Reads high-risk employees from predictions + raw data, generates
personalized manager action plans for each high-risk employee.

Requires: ANTHROPIC_API_KEY in .env file
Outputs:  outputs/retention_recommendations.csv
          outputs/retention_recommendations.json
"""

import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
OUTPUT_DIR  = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_CSV     = DATA_DIR  / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
PREDS_CSV   = OUTPUT_DIR / "predictions.csv"
FI_CSV      = OUTPUT_DIR / "feature_importances.csv"
OUT_CSV     = OUTPUT_DIR / "retention_recommendations.csv"
OUT_JSON    = OUTPUT_DIR / "retention_recommendations.json"


def load_env():
    """Load API key from .env file."""
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "\n[!] ANTHROPIC_API_KEY not found.\n"
            "    1. Get your free API key from https://console.anthropic.com\n"
            "    2. Create a file called .env in the churn-platform/ folder\n"
            "    3. Add this line: ANTHROPIC_API_KEY=sk-ant-your-key-here\n"
            "    4. Re-run this script."
        )
    return api_key


def build_high_risk_profiles(n: int = 10) -> pd.DataFrame:
    """Merge predictions with raw employee data to get high-risk profiles."""
    if not RAW_CSV.exists() or not PREDS_CSV.exists():
        raise FileNotFoundError("[!] Run scripts 01 and 02 first.")

    raw   = pd.read_csv(RAW_CSV)
    preds = pd.read_csv(PREDS_CSV)

    # Align by index (same train/test split seed)
    raw_sample = raw.sample(frac=1, random_state=42).reset_index(drop=True)
    n_preds    = len(preds)
    merged     = raw_sample.iloc[:n_preds].copy()
    merged["ChurnProbability"] = preds["ChurnProbability"].values
    merged["PredictedChurn"]   = preds["PredictedChurn"].values

    # Top N highest-risk employees
    high_risk = (
        merged[merged["ChurnProbability"] >= 0.5]
        .sort_values("ChurnProbability", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    print(f"[✓] Selected {len(high_risk)} high-risk employees for LLM analysis")
    return high_risk


def get_top_risk_factors() -> list[str]:
    """Load top feature importances to inform LLM context."""
    if not FI_CSV.exists():
        return ["OverTime", "JobSatisfaction", "MonthlyIncome",
                "YearsAtCompany", "MaritalStatus"]
    fi = pd.read_csv(FI_CSV)
    return fi["Feature"].tolist()[:5]


def build_prompt(employee: pd.Series, top_factors: list[str]) -> str:
    """Build a structured prompt for each high-risk employee."""
    churn_pct = f"{employee['ChurnProbability']:.0%}"

    return f"""You are an expert HR retention strategist. A predictive model has flagged an employee as high attrition risk.

EMPLOYEE PROFILE:
- Age: {employee['Age']} | Gender: {employee['Gender']} | Marital Status: {employee['MaritalStatus']}
- Department: {employee['Department']} | Job Role: {employee['JobRole']} | Job Level: {employee['JobLevel']}
- Monthly Income: ${employee['MonthlyIncome']:,} | Years at Company: {employee['YearsAtCompany']}
- Years in Current Role: {employee['YearsInCurrentRole']} | Years Since Last Promotion: {employee['YearsSinceLastPromotion']}
- Job Satisfaction: {employee['JobSatisfaction']}/4 | Work-Life Balance: {employee['WorkLifeBalance']}/4
- Environment Satisfaction: {employee['EnvironmentSatisfaction']}/4
- Overtime: {employee['OverTime']} | Business Travel: {employee['BusinessTravel']}
- Distance From Home: {employee['DistanceFromHome']} miles | Stock Option Level: {employee['StockOptionLevel']}
- Training Times Last Year: {employee['TrainingTimesLastYear']}

MODEL PREDICTION: {churn_pct} probability of leaving

TOP MODEL RISK SIGNALS (dataset-wide): {', '.join(top_factors)}

Based on this specific employee's profile, provide a concise retention action plan in this exact JSON format:
{{
  "risk_summary": "2-sentence summary of why this employee is likely to leave based on their specific profile",
  "top_3_risk_factors": ["factor1", "factor2", "factor3"],
  "immediate_actions": [
    "Specific action 1 the manager should take this week",
    "Specific action 2 the manager should take this week"
  ],
  "30_day_plan": [
    "Action 1 within 30 days",
    "Action 2 within 30 days",
    "Action 3 within 30 days"
  ],
  "retention_lever": "The single most important lever (compensation/growth/flexibility/recognition/workload)",
  "success_metric": "How to measure if retention effort is working in 60 days"
}}

Respond with valid JSON only. No preamble, no explanation outside the JSON."""


def get_recommendation(client: anthropic.Anthropic,
                        employee: pd.Series,
                        top_factors: list[str]) -> dict:
    """Call Claude API for a single employee recommendation."""
    prompt = build_prompt(employee, top_factors)

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Fast + cheap for batch processing
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_text = message.content[0].text.strip()

    # Parse JSON response
    try:
        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        return {
            "risk_summary": raw_text[:200],
            "top_3_risk_factors": top_factors[:3],
            "immediate_actions": ["Review compensation", "Schedule 1:1"],
            "30_day_plan": ["Career development discussion"],
            "retention_lever": "unknown",
            "success_metric": "Employee sentiment survey"
        }


def main():
    print("\n── Step 5: LLM Retention Recommendations ────────────────────")

    # Load API key
    api_key = load_env()
    client  = anthropic.Anthropic(api_key=api_key)
    print("[✓] Anthropic client initialized")

    # Load high-risk profiles
    high_risk    = build_high_risk_profiles(n=10)
    top_factors  = get_top_risk_factors()
    print(f"[✓] Top model risk factors: {top_factors}")

    # Generate recommendations
    all_results = []
    print(f"\n[→] Generating recommendations for {len(high_risk)} employees...\n")

    for idx, row in high_risk.iterrows():
        print(f"  [{idx+1}/{len(high_risk)}] Emp #{row['EmployeeNumber']} "
              f"| {row['JobRole']} | {row['Department']} "
              f"| Churn: {row['ChurnProbability']:.0%}")

        try:
            rec = get_recommendation(client, row, top_factors)
            result = {
                "EmployeeNumber":    int(row["EmployeeNumber"]),
                "JobRole":           row["JobRole"],
                "Department":        row["Department"],
                "Age":               int(row["Age"]),
                "MonthlyIncome":     int(row["MonthlyIncome"]),
                "YearsAtCompany":    int(row["YearsAtCompany"]),
                "ChurnProbability":  round(float(row["ChurnProbability"]), 4),
                "RiskSummary":       rec.get("risk_summary", ""),
                "TopRiskFactors":    "; ".join(rec.get("top_3_risk_factors", [])),
                "ImmediateActions":  " | ".join(rec.get("immediate_actions", [])),
                "Plan30Day":         " | ".join(rec.get("30_day_plan", [])),
                "RetentionLever":    rec.get("retention_lever", ""),
                "SuccessMetric":     rec.get("success_metric", ""),
                "FullRecommendation": json.dumps(rec),
            }
            all_results.append(result)
            print(f"     → Lever: {rec.get('retention_lever', 'N/A')} | "
                  f"Risk: {rec.get('top_3_risk_factors', ['N/A'])[0]}")

        except Exception as e:
            print(f"     [!] Error for employee {row['EmployeeNumber']}: {e}")
            continue

        # Rate limit: ~1 req/sec to be safe on free tier
        time.sleep(1.0)

    # Save outputs
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_CSV, index=False)
    print(f"\n[✓] Recommendations CSV saved → {OUT_CSV}")

    with open(OUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[✓] Recommendations JSON saved → {OUT_JSON}")

    # Summary
    print("\n── Retention Lever Distribution ─────────────────────────────")
    lever_counts = results_df["RetentionLever"].value_counts()
    print(lever_counts.to_string())

    print(f"\n[✓] Done — {len(all_results)} retention plans generated")
    print("    Next: Run 'python dashboard/app.py' to see them in the dashboard")


if __name__ == "__main__":
    main()
