from ftmo_gold.sol_plan_paper import refresh_due


def state(mode,valid_until=1000,source=600):
    return {"plan":{"mode":mode,"valid_until":valid_until},
            "consumed":False,"source_h1_closed_at":source}


def test_wait_refreshes_after_fifteen_minutes_inside_same_h1():
    assert refresh_due(state("WAIT"),600,1000)
    assert not refresh_due(state("WAIT"),600,999)


def test_conditional_plan_waits_for_new_h1_and_no_refresh_in_position():
    assert not refresh_due(state("CONDITIONAL"),600,1000)
    assert refresh_due(state("CONDITIONAL"),900,1000)
    assert not refresh_due(state("WAIT"),600,1000,position_open=True)
