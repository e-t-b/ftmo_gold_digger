from ftmo_gold.collector import quality

def test_detects_gap_and_invalid_ohlc():
    rows = [dict(time=60,open=10,high=12,low=9,close=11), dict(time=180,open=10,high=9,low=8,close=11)]
    result = quality(rows, 60)
    assert result['count'] == 2
    assert result['invalid_ohlc'] == 1
    assert result['gaps'][0]['missing_intervals'] == 1

def test_detects_duplicates():
    rows = [dict(time=60,open=10,high=11,low=9,close=10)] * 2
    assert quality(rows,60)['duplicate_or_unordered'] == 1
