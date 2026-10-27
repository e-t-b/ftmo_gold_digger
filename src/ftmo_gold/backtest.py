"""Research-only bar backtest. Bid OHLC approximation; NEVER a tick-quality execution claim."""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from .signals import breakout
from .risk import RiskGate, RiskPolicy

PRAGUE = ZoneInfo("Europe/Prague")


def load_bars(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = [{k: float(v) if k in ("open", "high", "low", "close") else int(v)
                 for k, v in row.items() if k in ("time", "open", "high", "low", "close")}
                for row in csv.DictReader(f)]
    for i, b in enumerate(rows):
        if not (math.isfinite(b["open"]) and math.isfinite(b["close"])
                and math.isfinite(b["high"]) and math.isfinite(b["low"])
                and b["low"] <= min(b["open"], b["close"]) <= max(b["open"], b["close"]) <= b["high"]):
            raise ValueError(f"Invalid OHLC on row {i}")
        if i and b["time"] <= rows[i-1]["time"]:
            raise ValueError(f"Duplicate/unordered timestamp at row {i}")
    return rows


def round_down_lots(raw, minimum, step):
    if raw < minimum or step <= 0:
        return 0.0
    return round(minimum + math.floor((raw-minimum)/step+1e-9)*step, 8)


def simulate(bars, initial=10000., spread=0.25, slippage=0.05,
             contract_size=100., minimum_lot=0.01, lot_step=0.01,
             commission_per_lot_roundtrip=0., max_hold_bars=12,
             risk_pct=.0035):
    """One position at a time; worst-case if SL and TP touched in same bar.

    All OHLC prices assumed BID. Synthetic ASK=bid+constant spread.
    Commission charged at entry; adverse slippage at entry/exit.
    No swaps, actual ticks, variable spreads, partial fills, or margin model.
    """
    if len(bars)<22 or any(x < 0 for x in (spread, slippage, commission_per_lot_roundtrip)):
        raise ValueError("Insufficient bars or invalid costs")
    policy=RiskPolicy(initial_balance=initial, risk_pct=risk_pct)
    gate=RiskGate(policy)
    balance=float(initial)
    pending=None
    position=None
    trades=[]
    equity_curve=[]
    active_day=None
    for i,bar in enumerate(bars):
        day=datetime.fromtimestamp(bar["time"], timezone.utc).astimezone(PRAGUE).date()
        if day != active_day:
            if position is not None:
                # No position may cross a Prague calendar day in this approximate simulator.
                raise ValueError("Position crossed Prague midnight: backtest segment not supported")
            gate.reset_day(datetime.fromtimestamp(bar["time"], timezone.utc),balance)
            active_day=day
            pending=None
        if pending is not None and position is None:
            sig=pending
            pending=None
            entry=bar["open"]+(spread+slippage if sig.side=="buy" else -slippage)
            distance=abs(entry-sig.stop)
            stop=sig.stop
            target=entry+2*distance*(1 if sig.side=="buy" else -1)
            budget=initial*risk_pct
            lot=round_down_lots(budget/(distance*contract_size+commission_per_lot_roundtrip)
                                if distance>0 else 0,minimum_lot,lot_step)
            risk=lot*(distance*contract_size+commission_per_lot_roundtrip)
            ok,_=gate.evaluate(balance,risk,positions_open=0)
            if ok and lot>0:
                position=dict(side=sig.side,entry=entry,stop=stop,target=target,
                              lot=lot,index=i,opened=bar["time"],fee=lot*commission_per_lot_roundtrip)
                balance-=position["fee"]
                gate.record_entry()
        if position is not None:
            p=position
            # Bid-based stops for BUY; Ask-based stops for SELL.
            low=bar["low"] if p["side"]=="buy" else bar["low"]+spread
            high=bar["high"] if p["side"]=="buy" else bar["high"]+spread
            stop_hit=low<=p["stop"] if p["side"]=="buy" else high>=p["stop"]
            target_hit=high>=p["target"] if p["side"]=="buy" else low<=p["target"]
            next_day = (i+1==len(bars) or
                datetime.fromtimestamp(bars[i+1]["time"],timezone.utc).astimezone(PRAGUE).date()!=day)
            timed=i-p["index"]>=max_hold_bars or next_day
            if stop_hit or target_hit or timed:
                reason="stop" if stop_hit else ("target" if target_hit else "session_close" if next_day else "timeout")
                exit_price=(p["stop"] if stop_hit else p["target"] if target_hit else
                            bar["close"] if p["side"]=="buy" else bar["close"]+spread)
                exit_price+=-slippage if p["side"]=="buy" else slippage
                pnl=(exit_price-p["entry"])*(1 if p["side"]=="buy" else -1)*p["lot"]*contract_size
                balance+=pnl
                trades.append(dict(opened=p["opened"],closed=bar["time"],side=p["side"],
                                   entry=p["entry"],exit=exit_price,lots=p["lot"],
                                   reason=reason,net_pnl=round(pnl-p["fee"],4),balance=round(balance,4)))
                position=None
        if position is not None:
            p=position
            mark=bar["close"] if p["side"]=="buy" else bar["close"]+spread
            unreal=(mark-p["entry"])*(1 if p["side"]=="buy" else -1)*p["lot"]*contract_size
        else:
            unreal=0
        equity=balance+unreal
        if gate.day is not None and equity<=max(gate.floors()):
            gate.halted=True
        equity_curve.append(equity)
        if position is None and pending is None and not gate.halted and i>=20 and i+1<len(bars):
            window=bars[i-20:i+1]
            if all(b["time"]-a["time"]==300 for a,b in zip(window,window[1:])) and (
                datetime.fromtimestamp(bars[i+1]["time"], timezone.utc).astimezone(PRAGUE).date()==day
            ):
                pending=breakout(window)
    if position:
        raise ValueError("Open position at end of sample; extend test input")
    peak=initial
    max_dd=0.
    for equity in equity_curve:
        peak=max(peak,equity)
        max_dd=max(max_dd,(peak-equity)/peak)
    wins=sum(t["net_pnl"]>0 for t in trades)
    profits=sum(max(t["net_pnl"],0) for t in trades)
    losses=-sum(min(t["net_pnl"],0) for t in trades)
    return dict(initial=initial,final=round(balance,2),net=round(balance-initial,2),
                trades=len(trades),wins=wins,win_rate=wins/len(trades) if trades else None,
                profit_factor=profits/losses if losses else None,
                max_drawdown_pct=round(100*max_dd,3),
                assumptions="Synthetic fixed spread and slippage; bid OHLC; conservative intrabar SL priority; no margin, swaps, FTMO pass probability or real-tick execution",
                transactions=trades)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--csv", default="data/raw/m5.csv")
    p.add_argument("--out", default="reports/baseline.json")
    p.add_argument("--spread", type=float, required=True,
                   help="ASK minus BID in XAUUSD price dollars; explicit assumption")
    p.add_argument("--slippage", type=float, default=.05)
    p.add_argument("--contract-size",type=float, required=True)
    p.add_argument("--min-lot",type=float,default=.01)
    p.add_argument("--lot-step",type=float,default=.01)
    p.add_argument("--commission",type=float,default=0.,
                   help="Round-trip commission in account currency per 1.0 lot")
    args=p.parse_args()
    result=simulate(load_bars(args.csv),spread=args.spread,slippage=args.slippage,
                    contract_size=args.contract_size,minimum_lot=args.min_lot,
                    lot_step=args.lot_step,commission_per_lot_roundtrip=args.commission)
    output=Path(args.out)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="transactions"},indent=2))


if __name__=="__main__":
    main()
