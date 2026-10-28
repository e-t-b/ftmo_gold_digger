from ftmo_gold.plan_history import candidate_times, historical_observation


ASOF=86400*100


def frames():
    result={}
    for name,step in (("M5",300),("M15",900),("H1",3600)):
        result[name]=[{"time":ASOF-(70-index)*step,"open":3000.,"high":3001.,
                       "low":2999.,"close":3000.,"tick_volume":100}
                      for index in range(70)]
    return result


def test_historical_observation_is_closed_and_causal():
    observation=historical_observation(frames(),ASOF,.2)
    assert observation["asof_utc"]==ASOF
    assert all(value["last_closed_at_utc"]==ASOF
               for value in observation["timeframes"].values())


def test_candidate_times_are_h1_closes_inside_bounds():
    values=candidate_times(frames(),ASOF-7200,ASOF)
    assert values and all(ASOF-7200<=value<=ASOF for value in values)
