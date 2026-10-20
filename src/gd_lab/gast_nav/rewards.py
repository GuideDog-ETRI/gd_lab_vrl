"""Nav reward terms; privileged feasibility/SDF are reward+critic only."""
import torch
from . import schema

def reward_terms(*,mode,go,phase,feasible,observed,v_path,v_cap,speed,yaw_rate,turn_rate,
                 positive_yaw,clearance,margin,offset_error,yaw_error,turn_drift,contact,entered_forbidden,
                 cfg, stair_mask=None, support_stable=None, entered_unknown=None, supervisor_certified=None, supervisor_mode='enforce', stair_end_wait=None):
    nav=mode==schema.MODES.index('NAV');turn=mode==schema.MODES.index('TURN')
    legacy=mode==schema.MODES.index('LEGACY')
    if bool((v_cap<=0).any()):raise ValueError('v_cap must be positive')
    certified=observed.bool() if supervisor_certified is None else supervisor_certified.bool()
    moving=go.bool() if supervisor_mode=='shadow' else go.bool() & feasible.bool() & certified
    if supervisor_mode not in ('shadow','enforce'):raise ValueError('supervisor_mode')
    # No desired-speed squared tracking penalty: slow feasible motion stays positive.
    progress=torch.minimum(v_path,v_cap)/v_cap
    stopped=speed<cfg['stopped_speed_mps']
    blocked=(~feasible.bool())
    known_feasible=feasible.bool() & observed.bool()
    if stair_mask is None: stair_mask=torch.zeros_like(go,dtype=torch.bool)
    unknown_feasible=feasible.bool() & ~observed.bool() & ~legacy
    # UPDATE167: do not substitute speed alone for measured support, or a
    # current UNKNOWN flag for actual swept entry. Nonzero weights require
    # the caller to provide those geometric/contact measurements.
    if cfg.get('unknown_stop',0.) and support_stable is None:
        raise ValueError('unknown_stop requires measured support_stable')
    if cfg.get('unknown_entry',0.) and entered_unknown is None:
        raise ValueError('unknown_entry requires measured swept entry')
    dwell=phase==schema.PHASES.index('DWELL')
    terms={
        'progress':torch.where(nav & moving,progress,0.),
        'overspeed':torch.where(nav & go,(v_path-v_cap).clamp_min(0),0.),
        'clearance':torch.where(~legacy,(margin-clearance).clamp_min(0),0.),
        'offset':torch.where(nav & go,offset_error.abs(),0.),
        'stop_success':torch.where(~legacy & blocked,stopped.float(),0.),
        'false_stop':torch.where((nav | turn) & go & known_feasible,stopped.float(),0.),
        'forbidden_entry':torch.where(~legacy & blocked,entered_forbidden.float(),0.),
        'turn_tracking':torch.where(turn & go & (torch.ones_like(certified) if supervisor_mode=='shadow' else certified),(yaw_rate-turn_rate).square(),0.),
        'reverse_rate':torch.where(turn,yaw_rate.clamp_min(0),0.),
        'reverse_accumulation':torch.where(turn,positive_yaw,0.),
        'turn_drift':torch.where(turn,turn_drift,0.),
        'stair_alignment':torch.where(nav & go & stair_mask,yaw_error.abs(),0.),
        'dwell_motion':torch.where(dwell,speed,0.),
        'contact':torch.where(~legacy,contact.float(),0.),
    }
    if 'unknown_stop' in cfg:
        stable=torch.zeros_like(go,dtype=torch.bool) if support_stable is None else support_stable.bool()
        terms['unknown_stop']=torch.where(unknown_feasible & stopped & stable & (clearance>=margin) & ~contact.bool() & (yaw_rate.abs()<cfg.get('stopped_yaw_rate_radps',.05)),1.,0.)
    if 'unknown_entry' in cfg:
        entry=torch.zeros_like(go,dtype=torch.bool) if entered_unknown is None else entered_unknown.bool()
        terms['unknown_entry']=torch.where(unknown_feasible & entry,1.,0.)
    if stair_end_wait is not None:
        wait=stair_end_wait.bool() & nav
        stable=torch.zeros_like(wait) if support_stable is None else support_stable.bool()
        terms['stair_end_stop']=torch.where(wait & stopped & stable & (yaw_rate.abs()<cfg.get('stopped_yaw_rate_radps',.05)),1.,0.)
        terms['stair_end_motion']=torch.where(wait,speed+yaw_rate.abs(),0.)
        # 2026-10-21 (RL replay): go stays 1 during the stair-end wait, so progress paid for moving on
        # against stair_end_motion. No progress while the wait holds.
        terms['progress']=torch.where(wait,0.,terms['progress'])
    return terms,{name:value*cfg[name] for name,value in terms.items()}

def legacy_reward_masks(mode,phase,go,hold):
    legacy=mode==schema.MODES.index('LEGACY')
    blocked=(~go.bool())|hold.bool()
    for p in ('PRE_BRAKE','LANDING_ENTERED','DWELL'):blocked |= phase==schema.PHASES.index(p)
    return {'xy_tracking':legacy,'yaw_tracking':legacy,'stair_stall':~blocked}

def compose_total(legacy_terms,nav_weighted,*,mode,phase,go,hold,xy_names,yaw_names,stall_names):
    masks=legacy_reward_masks(mode,phase,go,hold)
    result={}
    for name,value in legacy_terms.items():
        mask=masks['xy_tracking'] if name in xy_names else masks['yaw_tracking'] if name in yaw_names else masks['stair_stall'] if name in stall_names else torch.ones_like(go,dtype=torch.bool)
        result[name]=value*mask
    result.update({'nav_'+k:v for k,v in nav_weighted.items()})
    return sum(result.values()),result
