from ftmo_gold.paper import scan_once


class MT5:
    TIMEFRAME_M5=5
    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        return [dict(time=i*300,open=100.,high=101.,low=99.,close=100.)
                for i in range(count-1)]+[
                dict(time=(count-1)*300,open=100.,high=103.,low=99.,close=102.)]
    def last_error(self):
        return "test error"


def test_paper_scanner_is_idempotent():
    mt5=MT5()
    timestamp,event=scan_once(mt5,"XAUUSD")
    assert event["status"]=="candidate_only"
    assert event["side"]=="buy"
    assert scan_once(mt5,"XAUUSD",timestamp)==(timestamp,None)


def test_scanner_refuses_incomplete_history():
    class Empty(MT5):
        def copy_rates_from_pos(self,*args):
            return []
    _,event=scan_once(Empty(),"XAUUSD")
    assert event["status"]=="insufficient_history"
