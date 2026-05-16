from __future__ import annotations

import calendar

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Backtest Trade Dashboard jQuant", layout="wide")


REQUIRED_COLUMNS = {"Exit Date", "PnL"}


def load_trade_data(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required column(s): {missing}")

    trades = df.loc[:, ["Exit Date", "PnL"]].copy()
    trades["Exit Date"] = pd.to_datetime(trades["Exit Date"], errors="coerce")
    trades["PnL"] = pd.to_numeric(trades["PnL"], errors="coerce")
    trades = trades.dropna(subset=["Exit Date", "PnL"]).sort_values("Exit Date")

    if trades.empty:
        raise ValueError("No valid rows were found after parsing 'Exit Date' and 'PnL'.")

    trades["Equity Curve"] = trades["PnL"].cumsum()
    trades["Running Peak"] = trades["Equity Curve"].cummax()
    trades["Drawdown"] = trades["Equity Curve"] - trades["Running Peak"]
    trades["Year"] = trades["Exit Date"].dt.year.astype(str)
    trades["Month Num"] = trades["Exit Date"].dt.month
    trades["Month"] = trades["Month Num"].map(lambda month: calendar.month_abbr[month])

    return trades.reset_index(drop=True)


def calculate_metrics(trades: pd.DataFrame) -> dict[str, float]:
    pnl = trades["PnL"]
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = wins.sum()
    gross_loss = losses.sum()
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = losses.mean() if not losses.empty else 0.0

    win_rate = (len(wins) / len(pnl) * 100) if len(pnl) else 0.0
    loss_rate = (len(losses) / len(pnl) * 100) if len(pnl) else 0.0
    risk_reward = (avg_win / abs(avg_loss)) if avg_loss != 0 else None
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else None
    expectancy = pnl.mean() if len(pnl) else 0.0

    return {
        "Total Trades": len(pnl),
        "Winning Trades": len(wins),
        "Losing Trades": len(losses),
        "Winning Probability": win_rate,
        "Losing Probability": loss_rate,
        "Net PnL": pnl.sum(),
        "Average Trade": expectancy,
        "Average Win": avg_win,
        "Average Loss": avg_loss,
        "Risk Reward": risk_reward,
        "Profit Factor": profit_factor,
        "Max Profit": pnl.max(),
        "Max Loss": pnl.min(),
        "Max Drawdown": trades["Drawdown"].min(),
    }


def format_metric_value(label: str, value: float | int | None) -> str:
    if value is None:
        return "N/A"

    if label in {"Winning Probability", "Losing Probability"}:
        return f"{value:.2f}%"

    if label in {"Risk Reward", "Profit Factor"}:
        return f"{value:.2f}"

    if isinstance(value, int):
        return f"{value}"

    return f"{value:,.2f}"


def render_metrics(metrics: dict[str, float]) -> None:
    metric_order = [
        "Total Trades",
        "Winning Trades",
        "Losing Trades",
        "Winning Probability",
        "Net PnL",
        "Average Trade",
        "Average Win",
        "Average Loss",
        "Risk Reward",
        "Profit Factor",
        "Max Profit",
        "Max Loss",
        "Max Drawdown",
    ]

    columns = st.columns(4)
    for index, label in enumerate(metric_order):
        with columns[index % 4]:
            st.metric(label, format_metric_value(label, metrics[label]))


def render_charts(trades: pd.DataFrame) -> None:
    equity_chart = trades.set_index("Exit Date")[["Equity Curve"]]
    drawdown_chart = trades.set_index("Exit Date")[["Drawdown"]]

    yearly_pnl = (
        trades.groupby("Year", as_index=False)["PnL"]
        .sum()
        .rename(columns={"PnL": "Yearly PnL"})
        .sort_values("Year")
    )

    monthly_pnl = (
        trades.groupby(["Month Num", "Month"], as_index=False)["PnL"]
        .sum()
        .rename(columns={"PnL": "Monthly PnL"})
        .sort_values("Month Num")
        .set_index("Month")
    )

    st.subheader("Equity Curve")
    st.line_chart(equity_chart, use_container_width=True)

    st.subheader("Drawdown Curve")
    st.area_chart(drawdown_chart, use_container_width=True)

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Yearly PnL Distribution")
        st.bar_chart(yearly_pnl.set_index("Year"), use_container_width=True)

    with right_col:
        st.subheader("Monthly PnL Distribution")
        st.bar_chart(monthly_pnl[["Monthly PnL"]], use_container_width=True)


def render_trade_table(trades: pd.DataFrame) -> None:
    st.subheader("Trade Data")
    display_df = trades[["Exit Date", "PnL", "Equity Curve", "Drawdown"]].copy()
    display_df["Exit Date"] = display_df["Exit Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Backtested Trade Dashboard")
    st.write(
        "Upload a CSV with `Exit Date` and `PnL` columns to view your equity curve, "
        "drawdown, PnL distributions, and key backtesting metrics."
    )

    uploaded_file = st.file_uploader("Upload trade CSV", type=["csv"])

    if not uploaded_file:
        st.info("Upload your CSV file to begin.")
        return

    try:
        trades = load_trade_data(uploaded_file)
    except ValueError as error:
        st.error(str(error))
        return
    except Exception as error:  # pragma: no cover
        st.error(f"Unable to read the file: {error}")
        return

    metrics = calculate_metrics(trades)
    render_metrics(metrics)
    st.divider()
    render_charts(trades)
    st.divider()
    render_trade_table(trades)


if __name__ == "__main__":
    main()
