"""Read-only MT5 collector. Run on Windows with a logged-in demo terminal."""
import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

TIMEFRAMES = {"M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D1": "TIMEFRAME_D1"}
SECONDS = {"M1": 60, "M5": 300, "M15": 900, "H1": 3600, "H4": 14400, "D1": 86400}


def quality(rows, seconds):
    timestamps = [int(row["time"]) for row in rows]
    diffs = [b-a for a,b in zip(timestamps, timestamps[1:])]
    invalid = sum(1 for r in rows if not (r["low"] <= min(r["open"], r["close"]) <= max(r["open"], r["close"]) <= r["high"]))
    return {"count": len(rows), "first_utc": datetime.fromtimestamp(timestamps[0], timezone.utc).isoformat() if timestamps else None,
            "last_utc": datetime.fromtimestamp(timestamps[-1], timezone.utc).isoformat() if timestamps else None,
            "duplicate_or_unordered": sum(d <= 0 for d in diffs),
            "gaps": [{"from_epoch": a, "to_epoch": b, "missing_intervals": (b-a)//seconds-1} for a,b in zip(timestamps,timestamps[1:]) if b-a > seconds],
            "invalid_ohlc": invalid}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--ticks-days", type=int, default=2)
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()
    if args.days < 1 or args.ticks_days < 0 or args.ticks_days > args.days:
        parser.error("Require days >= 1 and 0 <= ticks-days <= days")
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    try:
        account = mt5.account_info()
        if account is None:
            raise RuntimeError("No MT5 account connected")
        if getattr(account, "trade_mode", None) != mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise RuntimeError("Refusing collection: connect a DEMO account first")
        symbol = mt5.symbol_info(args.symbol)
        if symbol is None:
            raise RuntimeError(f"Exact symbol {args.symbol!r} not found; do not substitute automatically")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        metadata = {"symbol": args.symbol, "server": account.server, "currency": account.currency,
                    "digits": symbol.digits, "point": symbol.point, "trade_contract_size": symbol.trade_contract_size,
                    "volume_min": symbol.volume_min, "volume_step": symbol.volume_step, "trade_tick_size": symbol.trade_tick_size,
                    "trade_tick_value": symbol.trade_tick_value, "spread_points_snapshot": symbol.spread,
                    "start_utc": (now-timedelta(days=args.days)).isoformat(), "end_utc": now.isoformat(),
                    "note": "Bars may be bid-based; spread snapshots do not reconstruct historical execution costs."}
        report = {"metadata": metadata, "timeframes": {}}
        for name, attr in TIMEFRAMES.items():
            rates = mt5.copy_rates_range(args.symbol, getattr(mt5, attr), now-timedelta(days=args.days), now)
            if rates is None:
                raise RuntimeError(f"Failed {name}: {mt5.last_error()}")
            rows = [{k: (v.item() if hasattr(v, "item") else v) for k,v in zip(rates.dtype.names, row)} for row in rates]
            with (out/f"{name.lower()}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rates.dtype.names)); writer.writeheader(); writer.writerows(rows)
            report["timeframes"][name] = quality(rows, SECONDS[name])
            # Do not silently present requested coverage as downloaded coverage.
            report["timeframes"][name]["requested_days"] = args.days
            if rows:
                report["timeframes"][name]["actual_span_days"] = round(
                    (int(rows[-1]["time"]) - int(rows[0]["time"])) / 86400, 2)
                report["timeframes"][name]["may_be_history_limited"] = (
                    report["timeframes"][name]["actual_span_days"] < args.days - 7)
            else:
                report["timeframes"][name]["actual_span_days"] = 0
                report["timeframes"][name]["may_be_history_limited"] = True
        if args.ticks_days:
            ticks = mt5.copy_ticks_range(args.symbol, now-timedelta(days=args.ticks_days), now, mt5.COPY_TICKS_ALL)
            if ticks is None:
                raise RuntimeError(f"Failed ticks: {mt5.last_error()}")
            with (out/"ticks.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(ticks.dtype.names)); writer.writeheader()
                for row in ticks:
                    writer.writerow({k: (v.item() if hasattr(v, "item") else v) for k,v in zip(ticks.dtype.names,row)})
            report["ticks_count"] = len(ticks)
        (out/"quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(out), "bars": {k:v["count"] for k,v in report["timeframes"].items()}, "ticks": report.get("ticks_count", 0)}, indent=2))
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
