"""Continuous autonomous DEMO scanner: generates signals and journals decisions only.

Research mode: NEVER sends an order. Running this module does not trade.
"""
import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .agent import HybridAgent
from .signals import breakout


def scan_once(mt5, symbol: str, last_closed: int | None = None, lookback=20):
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 1, lookback+1)
    if rates is None:
        raise RuntimeError(f"MT5 rates unavailable: {mt5.last_error()}")
    rows = [{k: (v.item() if hasattr(v, "item") else v)
             for k, v in zip(rates.dtype.names, row)} for row in rates]
    rows.sort(key=lambda b: int(b["time"]))
    if len(rows) < lookback+1:
        return last_closed, {"status":"insufficient_history"}
    timestamp=int(rows[-1]["time"])
    if timestamp==last_closed:
        return last_closed, None
    if last_closed is not None and timestamp<last_closed:
        raise RuntimeError("Terminal supplied older completed bar")
    if any(b["time"]-a["time"]!=300 for a,b in zip(rows,rows[1:])):
        return timestamp, {"status":"gap_detected","closed_bar":timestamp}
    signal=HybridAgent().decide(breakout(rows,lookback=lookback))
    if signal is None:
        return timestamp, {"status":"no_signal","closed_bar":timestamp}
    return timestamp, {"status":"candidate_only","closed_bar":timestamp,
                       "side":signal.side,"reference":signal.reference_price,
                       "stop":signal.stop,"target":signal.target,
                       "strategy":signal.strategy,
                       "warning":"No price execution, sizing or FTMO compliance decision made"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--symbol",default="XAUUSD")
    p.add_argument("--interval",type=int,default=10)
    p.add_argument("--out",default="data/paper/decisions.jsonl")
    p.add_argument("--once",action="store_true")
    args=p.parse_args()
    if args.interval<5:
        p.error("Minimum interval is 5 seconds")
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")
    try:
        account=mt5.account_info()
        if account is None or account.trade_mode!=mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise RuntimeError("Only MT5 DEMO accounts permitted")
        if mt5.symbol_info(args.symbol) is None:
            raise RuntimeError("Symbol missing; check exact MT5 name")
        out=Path(args.out)
        out.parent.mkdir(parents=True,exist_ok=True)
        previous=None
        while True:
            try:
                previous,event=scan_once(mt5,args.symbol,previous)
                if event is not None:
                    event["observed_utc"]=datetime.now(timezone.utc).isoformat()
                    event["symbol"]=args.symbol
                    with out.open("a",encoding="utf-8") as f:
                        f.write(json.dumps(event)+"\n")
                    print(json.dumps(event),flush=True)
            except Exception as exc:
                # Connection issues must remain visible. Never infer missed candles.
                error={"status":"scanner_error","error":str(exc),
                       "observed_utc":datetime.now(timezone.utc).isoformat()}
                print(json.dumps(error),flush=True)
                if args.once:
                    raise
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        mt5.shutdown()


if __name__=="__main__":
    main()
