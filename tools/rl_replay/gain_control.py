"""Keep seeded initial actuator gains fixed throughout matched UPDATE110 rollouts."""
import copy

def no_gain_resample(env,env_ids,**kwargs):
    return None

def freeze_gain_resampling(base):
    if not getattr(base.scene['robot'],'actuators',{}):return {'mode':'mock_no_actuators'}
    term=copy.copy(base.event_manager.get_term_cfg('randomize_actuator_gains'))
    if term.mode!='reset':raise ValueError('unexpected gain event mode')
    original=str(term.func)
    term.func=no_gain_resample
    base.event_manager.set_term_cfg('randomize_actuator_gains',term)
    print('[UPDATE110] seeded initial gains frozen across all later episode resets',flush=True)
    return {'mode':'seeded_initial_gains_frozen_after_initial_reset','disabled_term':'randomize_actuator_gains','original':original}
