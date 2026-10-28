"""Persistent virtual fills and performance accounting for Sol paper trading."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


PRAGUE=ZoneInfo("Europe/Prague")


def _day(timestamp):
    return datetime.fromtimestamp(timestamp,timezone.utc).astimezone(PRAGUE).date().isoformat()


def _round_down(value, minimum, step, maximum):
    if not all(math.isfinite(x) and x>0 for x in (value,minimum,step,maximum)):
        return 0.
    rounded=math.floor((value+1e-12)/step)*step
    if rounded<minimum:
        return 0.
    return round(min(rounded,maximum),8)


class PaperPortfolio:
    def __init__(self,path,symbol,starting_balance,*,risk_fraction=.0025,
                 max_daily_entries=3,commission_per_lot_roundtrip=0.,max_hold_hours=24.,
                 min_reward_risk=1.,max_reward_risk=4.):
        self.path=Path(path)
        self.symbol=symbol
        self.risk_fraction=float(risk_fraction)
        self.max_daily_entries=int(max_daily_entries)
        self.commission=float(commission_per_lot_roundtrip)
        self.max_hold_seconds=float(max_hold_hours)*3600
        self.min_reward_risk=float(min_reward_risk)
        self.max_reward_risk=float(max_reward_risk)
        if self.path.exists():
            self.state=json.loads(self.path.read_text(encoding="utf-8"))
            if self.state.get("symbol")!=symbol or self.state.get("version")!=1:
                raise RuntimeError("Incompatible paper portfolio state")
        else:
            balance=float(starting_balance)
            self.state={"version":1,"symbol":symbol,"initial_balance":balance,
                        "balance":balance,"equity":balance,"peak_equity":balance,
                        "max_drawdown_pct":0.,"position":None,"trades":[],
                        "trading_day":None,"day_start_equity":balance,
                        "entries_today":0,"last_processed_m15":None}

    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary=self.path.with_suffix(self.path.suffix+".tmp")
        temporary.write_text(json.dumps(self.state,indent=2),encoding="utf-8")
        temporary.replace(self.path)

    def _roll_day(self,timestamp):
        current=_day(timestamp)
        if self.state["trading_day"]!=current:
            self.state["trading_day"]=current
            self.state["day_start_equity"]=float(self.state["equity"])
            self.state["entries_today"]=0

    def _update_equity(self,bid,ask,contract_size):
        position=self.state["position"]
        unrealized=0.
        if position:
            mark=bid if position["side"]=="BUY" else ask
            direction=1 if position["side"]=="BUY" else -1
            unrealized=(mark-position["entry"])*direction*position["lots"]*contract_size
        equity=self.state["balance"]+unrealized
        self.state["equity"]=round(equity,2)
        self.state["peak_equity"]=round(max(self.state["peak_equity"],equity),2)
        peak=self.state["peak_equity"]
        drawdown=(peak-equity)/peak if peak>0 else 0.
        self.state["max_drawdown_pct"]=round(max(self.state["max_drawdown_pct"],100*drawdown),4)

    def _close(self,price,timestamp,reason,contract_size):
        position=self.state["position"]
        if not position:
            return None
        direction=1 if position["side"]=="BUY" else -1
        gross=(price-position["entry"])*direction*position["lots"]*contract_size
        fee=self.commission*position["lots"]
        net=gross-fee
        trade={**position,"closed_at":int(timestamp),"exit":round(float(price),6),
               "exit_reason":reason,"gross_pnl":round(gross,2),
               "commission":round(fee,2),"net_pnl":round(net,2)}
        self.state["balance"]=round(self.state["balance"]+net,2)
        self.state["position"]=None
        self.state["trades"].append(trade)
        return trade

    def process_closed_bar(self,bar,bar_open,bid,ask,contract_size,spread,bar_seconds=900):
        """Apply SL/TP conservatively once for a newly closed bar."""
        timestamp=int(bar_open)+int(bar_seconds)
        self._roll_day(timestamp)
        trade=None
        position=self.state["position"]
        already=self.state.get("last_processed_bar",self.state.get("last_processed_m15"))
        opened_bar=(position.get("opened_bar",position.get("opened_m15")) if position else None)
        if position and already!=bar_open and int(bar_open)>opened_bar:
            high=float(bar["high"]);low=float(bar["low"])
            if position["side"]=="BUY":
                if low<=position["stop"]:
                    trade=self._close(position["stop"],timestamp,"stop",contract_size)
                elif high>=position["target"]:
                    trade=self._close(position["target"],timestamp,"target",contract_size)
            else:
                ask_high=high+spread;ask_low=low+spread
                if ask_high>=position["stop"]:
                    trade=self._close(position["stop"],timestamp,"stop",contract_size)
                elif ask_low<=position["target"]:
                    trade=self._close(position["target"],timestamp,"target",contract_size)
            position=self.state["position"]
            if position and timestamp-position["opened_at"]>=self.max_hold_seconds:
                trade=self._close(ask if position["side"]=="SELL" else bid,
                                  timestamp,"max_hold",contract_size)
        self.state["last_processed_bar"]=int(bar_open)
        self.state["last_processed_m15"]=int(bar_open)
        self._update_equity(bid,ask,contract_size)
        return trade

    def apply_decision(self,decision,bid,ask,timestamp,signal_bar,symbol_info,
                       snapshot_id=None):
        """Apply a validated model decision to the virtual portfolio only."""
        contract=float(symbol_info.trade_contract_size)
        self._roll_day(timestamp)
        fill=None
        if decision.action=="CLOSE" and self.state["position"]:
            position=self.state["position"]
            fill=self._close(bid if position["side"]=="BUY" else ask,
                             timestamp,"model_close",contract)
        elif decision.action in ("BUY","SELL") and not self.state["position"]:
            entry=ask if decision.action=="BUY" else bid
            stop=float(decision.stop);target=float(decision.target)
            valid=(stop<entry<target if decision.action=="BUY" else target<entry<stop)
            risk_distance=abs(entry-stop)
            reward_distance=abs(target-entry)
            reward_risk=reward_distance/risk_distance if risk_distance>0 else 0.
            rr_valid=self.min_reward_risk<=reward_risk<=self.max_reward_risk
            initial=self.state["initial_balance"]
            total_floor=initial*.90
            daily_floor=self.state["day_start_equity"]-initial*.05
            rejection=("invalid_execution_levels" if not valid else
                       "reward_risk_out_of_range" if not rr_valid else
                       "daily_entry_limit" if self.state["entries_today"]>=self.max_daily_entries else
                       "drawdown_gate" if self.state["equity"]<=max(total_floor,daily_floor) else None)
            if rejection:
                fill={"type":"virtual_reject","reason":rejection,
                      "reward_risk":round(reward_risk,4)}
            else:
                risk_budget=self.state["equity"]*self.risk_fraction
                per_lot=abs(entry-stop)*contract+self.commission
                raw=risk_budget/per_lot if per_lot>0 else 0.
                lots=_round_down(raw,float(symbol_info.volume_min),
                                 float(symbol_info.volume_step),float(symbol_info.volume_max))
                if lots>0:
                    self.state["position"]={"side":decision.action,"entry":round(entry,6),
                        "stop":round(stop,6),"target":round(target,6),"lots":lots,
                        "opened_at":int(timestamp),"opened_bar":int(signal_bar),
                        "opened_m15":int(signal_bar),
                        "snapshot_id":snapshot_id,"risk_budget":round(risk_budget,2)}
                    self.state["entries_today"]+=1
                    fill={"type":"virtual_open","side":decision.action,
                          "entry":round(entry,6),"stop":round(stop,6),
                          "target":round(target,6),"lots":lots,
                          "reward_risk":round(reward_risk,4),
                          "risk_budget":round(risk_budget,2)}
                else:
                    fill={"type":"virtual_reject","reason":"below_minimum_volume",
                          "reward_risk":round(reward_risk,4)}
        self._update_equity(bid,ask,contract)
        return fill

    def context(self):
        return {"balance":self.state["balance"],"equity":self.state["equity"],
                "position":self.state["position"],"entries_today":self.state["entries_today"],
                "max_drawdown_pct":self.state["max_drawdown_pct"]}

    def metrics(self):
        trades=self.state["trades"]
        wins=sum(1 for trade in trades if trade["net_pnl"]>0)
        losses=sum(1 for trade in trades if trade["net_pnl"]<0)
        breakeven=len(trades)-wins-losses
        gross_profit=sum(max(trade["net_pnl"],0) for trade in trades)
        gross_loss=-sum(min(trade["net_pnl"],0) for trade in trades)
        return {"balance":self.state["balance"],"equity":self.state["equity"],
                "closed_trades":len(trades),"wins":wins,"losses":losses,
                "breakeven":breakeven,
                "win_rate":round(wins/len(trades),4) if trades else None,
                "net_pnl":round(self.state["balance"]-self.state["initial_balance"],2),
                "profit_factor":round(gross_profit/gross_loss,4) if gross_loss else None,
                "max_drawdown_pct":self.state["max_drawdown_pct"],
                "position_open":bool(self.state["position"])}
