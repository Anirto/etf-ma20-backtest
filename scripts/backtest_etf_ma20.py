#!/usr/bin/env python3
"""
ETF MA20 仓位管理策略 - 量化回测
====================================

策略规则:
  - 价格 > MA20（20日均线）→ 卖出当前仓位的 20%
  - 价格 < MA20（20日均线）→ 加仓（买入）当前资金的 20%

数据来源: AkShare（东方财富接口）
回测区间: 2024-01-01 ~ 2026-05-01
"""

import sys
import os
import json
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# ===================== 配置 =====================

ETF_LIST = [
    ("510050", "上证50ETF"),
    ("510300", "沪深300ETF"),
    ("510500", "中证500ETF"),
    ("159915", "创业板ETF"),
    ("588000", "科创50ETF"),
    ("510880", "红利ETF"),
    ("512880", "证券ETF"),
    ("512660", "军工ETF"),
]

INITIAL_CAPITAL = 1_000_000  # 初始资金
MA_PERIOD = 20               # 均线周期
ADJUST_PCT = 0.20            # 每次调整比例
START_DATE = "20240101"
END_DATE = "20260501"
COMMISSION = 0.0003          # 佣金万分之三
STAMP_DUTY = 0.001           # 印花税千分之一（卖出）
MIN_COMMISSION = 5           # 最低佣金（元）


# ===================== 数据获取 =====================

def fetch_etf_data(code: str, start: str, end: str) -> pd.DataFrame:
    """从东方财富获取 ETF 日线数据"""
    import akshare as ak

    df = ak.fund_etf_hist_em(
        symbol=code,
        period="daily",
        start_date=start,
        end_date=end,
        adjust="qfq",
    )
    if df is None or df.empty:
        return None

    df = df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ===================== 回测引擎 =====================

def run_backtest(df: pd.DataFrame) -> dict:
    """执行 MA20 仓位管理策略回测"""
    if df is None or len(df) < MA_PERIOD + 5:
        return None

    df["ma20"] = df["close"].rolling(window=MA_PERIOD).mean()

    cash = INITIAL_CAPITAL
    shares = 0
    trades = []
    nav = []

    start_idx = MA_PERIOD - 1  # 第一个有效 MA20

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        ma20 = float(row["ma20"])
        date = row["date"]

        if price > ma20 and shares > 0:
            # ── 高于 MA20：卖出 20% 仓位 ──
            sell_shares = max(100, (int(shares * ADJUST_PCT) // 100) * 100)
            sell_shares = min(sell_shares, shares)

            if sell_shares >= 100:
                sell_value = sell_shares * price
                fee = max(COMMISSION * sell_value, MIN_COMMISSION)
                tax = sell_value * STAMP_DUTY
                cash += sell_value - fee - tax
                shares -= sell_shares

                trades.append(
                    {
                        "date": str(date.date()),
                        "action": "SELL",
                        "price": round(price, 3),
                        "shares": sell_shares,
                        "value": round(sell_value, 2),
                        "reason": f"价{price:.2f}>均{ma20:.2f}",
                    }
                )

        elif price < ma20:
            # ── 低于 MA20：加仓 20% ──
            if shares == 0:
                buy_value = cash * ADJUST_PCT  # 空仓，建底仓
            else:
                buy_value = cash * ADJUST_PCT  # 有仓位，加仓

            buy_shares = max(100, (int(buy_value / price) // 100) * 100)

            if buy_shares >= 100:
                fee = max(COMMISSION * buy_shares * price, MIN_COMMISSION)
                cash -= buy_shares * price + fee
                shares += buy_shares

                trades.append(
                    {
                        "date": str(date.date()),
                        "action": "BUY",
                        "price": round(price, 3),
                        "shares": buy_shares,
                        "value": round(buy_shares * price, 2),
                        "reason": f"价{price:.2f}<均{ma20:.2f}",
                    }
                )

        # 记录每日净值
        nav.append(
            {
                "date": str(date.date()),
                "total_value": round(cash + shares * price, 2),
                "cash": round(cash, 2),
                "shares": shares,
                "price": round(price, 3),
                "ma20": round(ma20, 3),
            }
        )

    nav_df = pd.DataFrame(nav)
    final_value = cash + shares * float(df.iloc[-1]["close"])
    total_return = ((final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    # ── 年化收益率 ──
    days = (df.iloc[-1]["date"] - df.iloc[start_idx]["date"]).days
    years = days / 365.0
    if years > 0 and INITIAL_CAPITAL > 0:
        annual_return = ((final_value / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
    else:
        annual_return = 0

    # ── 最大回撤 ──
    nav_df["peak"] = nav_df["total_value"].cummax()
    nav_df["drawdown"] = (
        (nav_df["total_value"] - nav_df["peak"]) / nav_df["peak"] * 100
    )
    max_drawdown = nav_df["drawdown"].min()

    # ── 胜率（卖出价比最近买入价高就算赢） ──
    buy_trades = [t for t in trades if t["action"] == "BUY"]
    sell_trades = [t for t in trades if t["action"] == "SELL"]
    wins = 0
    for s in sell_trades:
        buys_before = [t for t in buy_trades if t["date"] < s["date"]]
        if buys_before and s["price"] > buys_before[-1]["price"]:
            wins += 1
    win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0

    # ── 夏普比率（年化） ──
    nav_df["daily_return"] = nav_df["total_value"].pct_change()
    avg_ret = nav_df["daily_return"].mean()
    std_ret = nav_df["daily_return"].std()
    sharpe = (avg_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    # ── 卡玛比率 ──
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        "total_return_pct": round(total_return, 2),
        "annual_return_pct": round(annual_return, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "win_rate_pct": round(win_rate, 2),
        "sharpe_ratio": round(sharpe, 2),
        "calmar_ratio": round(calmar, 2),
        "total_trades": len(trades),
        "final_value": round(final_value, 2),
        "trades": trades,
        "nav": nav,
    }


# ===================== 输出 =====================

def print_header(code: str, name: str):
    print(f"\n{'=' * 60}")
    print(f"  回测标的: {code} ({name})")
    print(f"{'=' * 60}")


def print_result(code: str, name: str, result: dict):
    if result is None:
        print(f"  ❌ {code} 回测无结果")
        return

    r = result
    print(f"\n  📊 {code} ({name})")
    print(f"  ┌──────────────────────────────────────┐")
    print(f"  │  总收益率        {r['total_return_pct']:>8.2f}%               │")
    print(f"  │  年化收益率      {r['annual_return_pct']:>8.2f}%               │")
    print(f"  │  最大回撤        {r['max_drawdown_pct']:>8.2f}%               │")
    print(f"  │  胜率            {r['win_rate_pct']:>8.2f}%               │")
    print(f"  │  夏普比率        {r['sharpe_ratio']:>8.2f}                │")
    print(f"  │  卡玛比率        {r['calmar_ratio']:>8.2f}                │")
    print(f"  │  总交易次数      {r['total_trades']:>8d}                │")
    print(f"  │  初始资金        {INITIAL_CAPITAL:>10,.0f}              │")
    print(f"  │  最终资产        {r['final_value']:>10,.0f}              │")
    print(f"  │  净收益          {r['final_value'] - INITIAL_CAPITAL:>10,.0f}              │")
    print(f"  └──────────────────────────────────────┘")

    recent = r["trades"][-10:]
    if recent:
        print(f"  📝 最近 10 笔交易:")
        for t in recent:
            emoji = "🟢 BUY" if t["action"] == "BUY" else "🔴 SELL"
            print(f"    {emoji}  {t['date']}  {t['shares']:>6d}股 @ {t['price']:>7.3f}  | {t['reason']}")


# ===================== 主流程 =====================

def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(PROJECT_ROOT)

    print("=" * 60)
    print("  ETF MA20 仓位管理策略 · 量化回测")
    print(f"  区间: {START_DATE} ~ {END_DATE}")
    print(f"  初始资金: {INITIAL_CAPITAL:,.0f}")
    print(f"  策略: 价格>MA{MA_PERIOD}→卖{ADJUST_PCT*100:.0f}% | 价格<MA{MA_PERIOD}→加仓{ADJUST_PCT*100:.0f}%")
    print("=" * 60)

    all_results = []
    failed = []

    for code, name in ETF_LIST:
        print_header(code, name)

        print(f"  📡 获取数据...", end=" ", flush=True)
        df = fetch_etf_data(code, START_DATE, END_DATE)

        if df is None:
            print("❌ 失败")
            failed.append(code)
            continue

        print(f"{len(df)} 条日线 ✅")
        result = run_backtest(df)

        if result:
            print_result(code, name, result)
            all_results.append(
                {
                    "code": code,
                    "name": name,
                    **{k: v for k, v in result.items() if k not in ("trades", "nav")},
                }
            )
        else:
            print(f"  ⚠️ 数据不足")
            failed.append(code)

    # ── 汇总报告 ──
    if all_results:
        print(f"\n\n{'=' * 60}")
        print(f"  📊 ETF MA20 仓位管理策略 · 汇总")
        print(f"{'=' * 60}")
        header = f"  {'ETF':<8} {'名称':<12} {'总收益率':>8} {'年化':>8} {'最大回撤':>8} {'夏普':>6} {'卡玛':>6} {'交易':>5}"
        print(header)
        print(f"  {'─' * 63}")
        for r in all_results:
            print(
                f"  {r['code']:<6} {r['name']:<12} {r['total_return_pct']:>7.1f}% "
                f"{r['annual_return_pct']:>7.1f}% {r['max_drawdown_pct']:>7.1f}% "
                f"{r['sharpe_ratio']:>5.1f} {r['calmar_ratio']:>5.1f} {r['total_trades']:>4d}"
            )

        avg_annual = np.mean([r["annual_return_pct"] for r in all_results])
        avg_dd = np.mean([r["max_drawdown_pct"] for r in all_results])
        avg_sharpe = np.mean([r["sharpe_ratio"] for r in all_results])
        avg_ret = np.mean([r["total_return_pct"] for r in all_results])
        avg_win = np.mean([r["win_rate_pct"] for r in all_results])

        print(f"  {'─' * 63}")
        print(
            f"  {'平均':<19} {avg_ret:>7.1f}% {avg_annual:>7.1f}% "
            f"{avg_dd:>7.1f}% {avg_sharpe:>5.1f}   {avg_win:>5.1f}%"
        )

        best = max(all_results, key=lambda x: x["annual_return_pct"])
        worst = min(all_results, key=lambda x: x["annual_return_pct"])
        print(f"\n  🏆 最佳: {best['code']} ({best['name']})  年化 {best['annual_return_pct']:.1f}%")
        print(f"  💀 最差: {worst['code']} ({worst['name']})  年化 {worst['annual_return_pct']:.1f}%")

    if failed:
        print(f"\n  ⚠️ 失败: {', '.join(failed)}")

    # ── 保存结果 ──
    output = {
        "strategy": "ETF MA20 仓位管理",
        "period": f"{START_DATE} - {END_DATE}",
        "initial_capital": INITIAL_CAPITAL,
        "params": {
            "ma_period": MA_PERIOD,
            "adjust_pct": ADJUST_PCT,
            "commission": COMMISSION,
            "stamp_duty": STAMP_DUTY,
        },
        "results": all_results,
        "summary": {
            "avg_annual_return_pct": round(avg_annual, 2),
            "avg_max_drawdown_pct": round(avg_dd, 2),
            "avg_sharpe_ratio": round(avg_sharpe, 2),
            "best": best["code"] if all_results else None,
            "worst": worst["code"] if all_results else None,
        },
    }

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "etf_ma20_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 结果已保存到 {out_path}")
    print()


if __name__ == "__main__":
    main()
