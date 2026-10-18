"""Nav reward terms; privileged feasibility/SDF are reward+critic only."""
import torch
from . import schema

def reward_terms(*,mode,go,phase,feasible,observed,v_path,v_cap,speed,yaw_rate,turn_rate,
                 positive_yaw,clearance,margin,offset_error,yaw_error,turn_drift,contact,entered_forbidden,
                 cfg, stair_mask=None):
    nav=mode==schema.MODES.index('NAV');turn=mode==schema.MODES.index('TURN')
    legacy=mode==schema.MODES.index('LEGACY')
    if bool((v_cap<=0).any()):raise ValueError('v_cap must be positive')
    moving=go.bool() & feasible.bool() & observed.bool()
    # No desired-speed squared tracking penalty: slow feasible motion stays positive.
    progress=torch.minimum(v_path,v_cap)/v_cap
    stopped=speed<cfg['stopped_speed_mps']
    blocked=(~feasible.bool())
    known_feasible=feasible.bool() & observed.bool()
    if stair_mask is None: stair_mask=torch.zeros_like(go,dtype=torch.bool)
    dwell=phase==schema.PHASES.index('DWELL')
    terms={
        'progress':torch.where(nav & moving,progress,0.),
        'overspeed':torch.where(nav & go,(v_path-v_cap).clamp_min(0),0.),
        'clearance':torch.where(~legacy,(margin-clearance).clamp_min(0),0.),
        'offset':torch.where(nav & go,offset_error.abs(),0.),
        'stop_success':torch.where(~legacy & blocked,stopped.float(),0.),
        'false_stop':torch.where((nav | turn) & go & known_feasible,stopped.float(),0.),
        'forbidden_entry':torch.where(~legacy & blocked,entered_forbidden.float(),0.),
        'turn_tracking':torch.where(turn & go,(yaw_rate-turn_rate).square(),0.),
        'reverse_rate':torch.where(turn,yaw_rate.clamp_min(0),0.),
        'reverse_accumulation':torch.where(turn,positive_yaw,0.),
        'turn_drift':torch.where(turn,turn_drift,0.),
        'stair_alignment':torch.where(nav & go & stair_mask,yaw_error.abs(),0.),
        'dwell_motion':torch.where(dwell,speed,0.),
        'contact':torch.where(~legacy,contact.float(),0.),
    }
    return terms,{name:value*cfg[name] for name,value in terms.items()}

def legacy_reward_masks(mode,phase,go,hold):
    legacy=mode==schema.MODES.index('LEGACY')
    blocked=(~go.bool())|hold.bool()
    for p in ('PRE_BRAKE','LANDING_ENTERED','DWELL'):blocked |= phase==schema.PHASES.index(p)
    return {'xy_tracking':legacy,'yaw_tracking':legacy,'stair_stall':~blocked}

def intervention_penalty(controller,previous,coefficient):
    """Use either this env term OR GastNavPPO adjustment, never both."""
    hold=schema.CONTROLLERS.index('HOLD_3906')
    return ((controller==hold)&(previous!=hold)).float()*coefficient

def compose_total(legacy_terms,nav_weighted,*,mode,phase,go,hold,xy_names,yaw_names,stall_names):
    masks=legacy_reward_masks(mode,phase,go,hold)
    result={}
    for name,value in legacy_terms.items():
        mask=masks['xy_tracking'] if name in xy_names else masks['yaw_tracking'] if name in yaw_names else masks['stair_stall'] if name in stall_names else torch.ones_like(go,dtype=torch.bool)
        result[name]=value*mask
    result.update({'nav_'+k:v for k,v in nav_weighted.items()})
    return sum(result.values()),result
