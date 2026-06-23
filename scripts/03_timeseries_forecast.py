"""
03_timeseries_forecast.py
=========================
Holt-Winters exponential smoothing to forecast 6-month churn rate by department.
Simulates monthly attrition from YearsAtCompany / tenure fields.
Outputs: outputs/ts_forecast.png, outputs/forecast_data.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_CSV    = DATA_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"


def simulate_monthly_attrition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate a hire date using YearsAtCompany, then assign each employee
    who left to a simulated departure month. Aggregate by Department + Month.
    """
    np.random.seed(42)
    base_date = pd.Timestamp("2024-01-01")

    df = df.copy()
    df["AttritionBinary"] = (df["Attrition"] == "Yes").astype(int)

    # Simulate hire date: base_date minus YearsAtCompany (with small jitter)
    jitter_days = np.random.randint(-30, 30, size=len(df))
    df["HireDate"] = base_date - pd.to_timedelta(
        df["YearsAtCompany"] * 365 + jitter_days, unit="D"
    )

    # Simulate departure month for those who left
    # Uniform random month between hire date and base_date
    def random_departure(row):
        if row["AttritionBinary"] == 0:
            return pd.NaT
        total_months = max(int(row["YearsAtCompany"] * 12), 1)
        offset = np.random.randint(0, total_months)
        return row["HireDate"] + pd.DateOffset(months=offset)

    df["DepartureDate"] = df.apply(random_departure, axis=1)

    # Keep only those who left
    departed = df[df["AttritionBinary"] == 1].copy()
    departed["DepartureMonth"] = departed["DepartureDate"].dt.to_period("M")

    # Headcount by department (denominator)
    headcount = df.groupby("Department").size().rename("Headcount")

    # Departures by dept / month
    monthly = (
        departed.groupby(["Department", "DepartureMonth"])
        .size()
        .reset_index(name="Departures")
    )
    monthly = monthly.merge(headcount, on="Department")
    monthly["AttritionRate"] = monthly["Departures"] / monthly["Headcount"]
    monthly["Date"] = monthly["DepartureMonth"].dt.to_timestamp()

    return monthly


def build_dept_series(monthly: pd.DataFrame, dept: str,
                      n_months: int = 24) -> pd.Series:
    """
    Pull the last n_months of monthly attrition rate for a department.
    Fill gaps with interpolation.
    """
    dept_df = monthly[monthly["Department"] == dept].copy()
    dept_df = dept_df.set_index("Date")["AttritionRate"].sort_index()

    # Build complete monthly index
    full_idx = pd.date_range(
        start=dept_df.index.min(),
        end=dept_df.index.max(),
        freq="MS"
    )
    dept_df = dept_df.reindex(full_idx).interpolate(method="linear").fillna(0)

    # Take last n_months
    return dept_df.iloc[-n_months:]


def forecast_department(series: pd.Series, horizon: int = 6) -> tuple:
    """Fit Holt-Winters and return forecast + confidence interval."""
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add" if len(series) >= 24 else None,
        seasonal_periods=12 if len(series) >= 24 else None,
        initialization_method="estimated",
    ).fit(optimized=True)

    forecast     = model.forecast(horizon)
    fitted_vals  = model.fittedvalues

    # Simulate confidence interval from residual std
    resid_std    = (series - fitted_vals).std()
    ci_lower     = forecast - 1.96 * resid_std
    ci_upper     = forecast + 1.96 * resid_std
    ci_lower     = ci_lower.clip(lower=0)

    return forecast, ci_lower, ci_upper


def plot_forecasts(monthly: pd.DataFrame, departments: list,
                   horizon: int = 6) -> pd.DataFrame:
    fig, axes = plt.subplots(len(departments), 1,
                             figsize=(12, 4 * len(departments)),
                             sharex=False)
    if len(departments) == 1:
        axes = [axes]

    colors      = {"Sales": "#2563EB", "Research & Development": "#16A34A",
                   "Human Resources": "#DC2626"}
    all_forecasts = []

    for ax, dept in zip(axes, departments):
        series = build_dept_series(monthly, dept, n_months=24)
        if len(series) < 6:
            print(f"  [!] Not enough data for {dept} — skipping")
            continue

        forecast, ci_lower, ci_upper = forecast_department(series, horizon)

        last_date     = series.index[-1]
        future_dates  = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon, freq="MS"
        )
        forecast.index  = future_dates
        ci_lower.index  = future_dates
        ci_upper.index  = future_dates

        color = colors.get(dept, "#7C3AED")

        # Historical
        ax.plot(series.index, series.values * 100,
                color=color, lw=2, label="Historical")
        # Forecast
        ax.plot(future_dates, forecast.values * 100,
                color=color, lw=2.5, linestyle="--", label="Forecast (6mo)")
        # CI
        ax.fill_between(future_dates,
                        ci_lower.values * 100,
                        ci_upper.values * 100,
                        alpha=0.2, color=color, label="95% CI")
        # Vertical divider
        ax.axvline(x=last_date, color="gray", linestyle=":", lw=1.5, alpha=0.7)

        ax.set_title(f"{dept} — Monthly Attrition Rate (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Attrition Rate (%)")
        ax.legend(loc="upper left", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)

        # Collect for CSV
        for dt, f, lo, hi in zip(future_dates, forecast.values,
                                  ci_lower.values, ci_upper.values):
            all_forecasts.append({
                "Department":      dept,
                "ForecastMonth":   dt.strftime("%Y-%m"),
                "ForecastRate":    round(f * 100, 3),
                "CI_Lower":        round(lo * 100, 3),
                "CI_Upper":        round(hi * 100, 3),
            })

    plt.suptitle("6-Month Churn Rate Forecast by Department (Holt-Winters)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "ts_forecast.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Forecast plot saved → outputs/ts_forecast.png")

    return pd.DataFrame(all_forecasts)


def main():
    if not RAW_CSV.exists():
        raise FileNotFoundError(
            "[!] Raw dataset not found. Run 01_eda_preprocessing.py first."
        )

    df      = pd.read_csv(RAW_CSV)
    print(f"[✓] Loaded {len(df)} records for time-series simulation")

    monthly = simulate_monthly_attrition(df)
    print(f"[✓] Simulated monthly attrition across "
          f"{monthly['Department'].nunique()} departments, "
          f"{monthly['DepartureMonth'].nunique()} months")

    departments = sorted(df["Department"].unique().tolist())
    print(f"    Departments: {departments}")

    forecast_df = plot_forecasts(monthly, departments)
    forecast_df.to_csv(OUTPUT_DIR / "forecast_data.csv", index=False)
    print(f"[✓] Forecast data saved → outputs/forecast_data.csv")

    # Print summary
    print("\n── 6-Month Forecast Summary ─────────────────────────────────")
    summary = (
        forecast_df.groupby("Department")["ForecastRate"]
        .agg(["mean", "max"])
        .rename(columns={"mean": "Avg Monthly Rate (%)", "max": "Peak Rate (%)"})
        .round(3)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
