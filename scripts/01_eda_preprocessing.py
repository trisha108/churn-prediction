"""
01_eda_preprocessing.py
=======================
IBM HR Analytics — EDA, feature engineering, class imbalance handling.
Outputs: data/processed_features.csv, outputs/eda_summary.txt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
OUTPUT_DIR  = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_CSV     = DATA_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
PROCESSED   = DATA_DIR / "processed_features.csv"


def load_data() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"\n[!] Dataset not found at {RAW_CSV}\n"
            "    Download from: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset\n"
            "    Place 'WA_Fn-UseC_-HR-Employee-Attrition.csv' in the data/ folder."
        )
    df = pd.read_csv(RAW_CSV)
    print(f"[✓] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def run_eda(df: pd.DataFrame) -> None:
    lines = []
    lines.append("=" * 60)
    lines.append("IBM HR ATTRITION — EDA SUMMARY")
    lines.append("=" * 60)
    lines.append(f"\nShape: {df.shape}")
    lines.append(f"\nAttrition distribution:\n{df['Attrition'].value_counts(normalize=True).round(4).to_string()}")
    lines.append(f"\nMissing values: {df.isnull().sum().sum()}")
    lines.append(f"\nNumeric summary:\n{df.describe().round(2).to_string()}")

    summary_path = OUTPUT_DIR / "eda_summary.txt"
    summary_path.write_text("\n".join(lines))
    print(f"[✓] EDA summary saved → {summary_path}")

    # Attrition by department
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    dept_attrition = df.groupby("Department")["Attrition"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index()
    dept_attrition.columns = ["Department", "AttritionRate"]
    axes[0].bar(dept_attrition["Department"], dept_attrition["AttritionRate"], color="#2563EB")
    axes[0].set_title("Attrition Rate by Department (%)")
    axes[0].set_ylabel("Attrition Rate (%)")
    axes[0].tick_params(axis="x", rotation=15)

    df["AttritionBinary"] = (df["Attrition"] == "Yes").astype(int)
    corr_cols = ["Age", "MonthlyIncome", "YearsAtCompany", "JobSatisfaction",
                 "WorkLifeBalance", "OverTime", "AttritionBinary"]
    df_corr = df[corr_cols].copy()
    df_corr["OverTime"] = (df_corr["OverTime"] == "Yes").astype(int)
    sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
    axes[1].set_title("Correlation Heatmap (key features)")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "eda_plots.png", dpi=150)
    plt.close()
    print("[✓] EDA plots saved → outputs/eda_plots.png")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer 8 new features on top of the raw IBM dataset."""
    df = df.copy()

    # Binary target
    df["AttritionBinary"] = (df["Attrition"] == "Yes").astype(int)

    # ── Feature 1: Salary-to-Age Ratio ─────────────────────────────────────
    df["SalaryAgeRatio"] = df["MonthlyIncome"] / df["Age"].replace(0, np.nan)

    # ── Feature 2: Tenure per Role (years in current role / total tenure) ──
    df["TenurePerRole"] = np.where(
        df["YearsAtCompany"] > 0,
        df["YearsInCurrentRole"] / df["YearsAtCompany"],
        0
    )

    # ── Feature 3: Satisfaction Composite Score ─────────────────────────────
    #    Average of 4 satisfaction dimensions, scaled 1-4
    df["SatisfactionComposite"] = (
        df["JobSatisfaction"] +
        df["EnvironmentSatisfaction"] +
        df["RelationshipSatisfaction"] +
        df["WorkLifeBalance"]
    ) / 4.0

    # ── Feature 4: Promotion Lag (years since last promotion) ───────────────
    df["PromotionLag"] = df["YearsAtCompany"] - df["YearsSinceLastPromotion"]

    # ── Feature 5: Income vs Role Average (relative salary standing) ────────
    role_avg = df.groupby("JobRole")["MonthlyIncome"].transform("mean")
    df["IncomeVsRoleAvg"] = df["MonthlyIncome"] / role_avg

    # ── Feature 6: Overtime Flag ─────────────────────────────────────────────
    df["OvertimeFlag"] = (df["OverTime"] == "Yes").astype(int)

    # ── Feature 7: Career Velocity (job level / total working years) ─────────
    df["CareerVelocity"] = np.where(
        df["TotalWorkingYears"] > 0,
        df["JobLevel"] / df["TotalWorkingYears"],
        0
    )

    # ── Feature 8: Manager Tenure Ratio ──────────────────────────────────────
    df["ManagerTenureRatio"] = np.where(
        df["YearsAtCompany"] > 0,
        df["YearsWithCurrManager"] / df["YearsAtCompany"],
        0
    )

    print("[✓] Engineered 8 new features:")
    new_feats = ["SalaryAgeRatio", "TenurePerRole", "SatisfactionComposite",
                 "PromotionLag", "IncomeVsRoleAvg", "OvertimeFlag",
                 "CareerVelocity", "ManagerTenureRatio"]
    for f in new_feats:
        print(f"      • {f}")

    return df


def encode_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categoricals, drop leakage/constant columns, return model-ready df."""
    # Drop columns with single value or direct leakage
    drop_cols = ["Attrition", "EmployeeCount", "EmployeeNumber",
                 "Over18", "StandardHours"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode binary categoricals
    binary_map = {"Yes": 1, "No": 0}
    for col in ["OverTime"]:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    # One-hot encode remaining object columns
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    print(f"[✓] Final feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def main():
    print("\n── Step 1: Load ─────────────────────────────────────────────")
    df = load_data()

    print("\n── Step 2: EDA ──────────────────────────────────────────────")
    run_eda(df)

    print("\n── Step 3: Feature Engineering ──────────────────────────────")
    df = engineer_features(df)

    print("\n── Step 4: Encode & Clean ───────────────────────────────────")
    df = encode_and_clean(df)

    df.to_csv(PROCESSED, index=False)
    print(f"\n[✓] Processed dataset saved → {PROCESSED}")
    print(f"    Target column: 'AttritionBinary'  |  Positive rate: "
          f"{df['AttritionBinary'].mean():.2%}")


if __name__ == "__main__":
    main()
