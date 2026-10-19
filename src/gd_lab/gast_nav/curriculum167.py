"""Explicit preparation helpers; no implicit activation of C or GT map edits."""
from copy import deepcopy


def easy_scene_settings(settings, limits):
    result=deepcopy(settings)
    minimum=limits['passage_min_width_m']
    if minimum < 1.0 or limits['turn_min_delta_m'] < .2 or not 0 < limits['v_cap_max_mps'] <= .6:
        raise ValueError('UPDATE167 C limits')
    for key in ('corridor','door'):
        result[key]['width']=[max(minimum,v) for v in result[key]['width']]
    # Side boxes and pillars can reduce the nominal width below the contract.
    result['corridor']['box_count']=[0,0]
    result['corridor']['pillar_probability']=0.
    result['turn_room']['delta_easy_m']=max(result['turn_room']['delta_easy_m'],limits['turn_min_delta_m'])
    result['turn_room']['delta_hard_m']=max(result['turn_room']['delta_hard_m'],limits['turn_min_delta_m'])
    result['stairs']['width']=[max(limits['stairs_min_width_m'],v) for v in result['stairs']['width']]
    result['stairs']['box_count']=list(limits['stairs_box_count'])
    if result['stairs']['box_count'] != [0,0]:raise ValueError('C stairs must be obstacle-free')
    return result


def easy_command_settings(settings, limits):
    result=deepcopy(settings)
    result['v_cap_range'][1]=min(result['v_cap_range'][1],limits['v_cap_max_mps'])
    if result['v_cap_range'][0]>result['v_cap_range'][1]:raise ValueError('empty C speed range')
    return result
