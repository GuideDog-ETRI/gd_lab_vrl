from gd_lab.gast_nav.supervisor import HoldReasonMetrics


def test_executed_hold_metrics_denominators_and_snapshot():
    metrics = HoldReasonMetrics()
    row = dict(kind='nav_door', requested_mode='NAV', mode='LEGACY',
               phase='APPROACH', handshake='STOPPING', controller='HOLD_3906',
               reason='SUPERVISOR_RISK', clearance_risk=True, unknown_risk=True,
               effective_go=False)
    metrics.add([row, dict(row, controller='NAV_POLICY', reason='', unknown_risk=False)])
    group = metrics.snapshot()[0]
    assert group['ticks'] == 2 and group['hold'] == 1
    assert group['sums']['unknown_risk'] == 1
    assert group['reasons'] == {'HOLD_3906:SUPERVISOR_RISK': 1, 'NAV_POLICY:OK': 1}
    group['reasons'].clear()
    assert len(metrics.snapshot()[0]['reasons']) == 2
    metrics.add([dict(row, kind='nav_turn_room', requested_mode='TURN')])
    assert len(metrics.snapshot()) == 2


def test_fsm_reason_does_not_count_as_controller_hold():
    metrics = HoldReasonMetrics()
    metrics.add([dict(kind='nav_stair_landing', requested_mode='NAV', mode='NAV',
                      phase='DWELL', handshake='ACTIVE', controller='NAV_POLICY',
                      reason='NO_DESCENT_VIEW', effective_go=False)])
    group = metrics.snapshot()[0]
    assert group['ticks'] == 1 and group['hold'] == 0
    assert group['reasons'] == {'NAV_POLICY:NO_DESCENT_VIEW': 1}
