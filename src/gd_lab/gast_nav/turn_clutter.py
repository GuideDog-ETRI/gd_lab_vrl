"""UPDATE161 turn workspace construction without privileged actor shortcuts.

The turn centre is chosen before clutter. An anchor OBB has EXACT specified
surface clearance; all other clutter is farther away. A reserved tangent path
near the entry leaves room for the C cylinder and its bounded sway.
"""
import math
import numpy as np
from .scene_types import StaticBox
from .scene_geometry import point_box_distance, line_box_interval


def point_segment_distance(point, a, b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    sq=dx*dx+dy*dy
    u=0. if sq==0 else max(0.,min(1.,((point[0]-a[0])*dx+(point[1]-a[1])*dy)/sq))
    return math.hypot(point[0]-a[0]-u*dx,point[1]-a[1]-u*dy)


def cylinder_slot(meta, config):
    """Reserved nominal cylinder endpoints from existing route/spot metadata.

    No SceneTileMeta/schema change. route[0] is slot centre; its radial direction
    determines the perpendicular segment shared by generator and sampler.
    """
    centre=meta.route_polyline[0]; spot=meta.turn_spot[0]
    dx,dy=centre[0]-spot[0],centre[1]-spot[1]; length=math.hypot(dx,dy)
    eps=config['terrain']['geometry_epsilon']
    if (length<=eps or math.dist(meta.route_polyline[-1],spot)>eps
            or abs(length-config['turn_room']['cylinder_slot_distance_m'])>eps): return None
    half=config['turn_room']['cylinder_path_half_length_m']
    tangent=(-dy/length,dx/length)
    return ((centre[0]-half*tangent[0],centre[1]-half*tangent[1]),
            (centre[0]+half*tangent[0],centre[1]+half*tangent[1]))


def segment_clear(a,b,obstacle,radius,eps):
    """Conservative swept disk vs OBB (box Minkowski proxy at corners)."""
    if point_box_distance(a,obstacle)<=radius or point_box_distance(b,obstacle)<=radius: return False
    length=math.dist(a,b)
    if length<=eps: return True
    inflated=StaticBox(obstacle.center,(obstacle.half[0]+radius,obstacle.half[1]+radius,obstacle.half[2]),obstacle.yaw,obstacle.kind)
    interval=line_box_interval(a,((b[0]-a[0])/length,(b[1]-a[1])/length),inflated,eps)
    return not (interval and interval[1]>=0 and interval[0]<=length)


def build_turn_clutter(difficulty,level,rng,settings,wall_height):
    """Return (OBBs, route, nominal span, target radius) in env-local coordinates."""
    cfg=settings['turn_room']; terrain=settings['terrain']; oracle=settings['oracle']; cyl=settings['cylinder']
    width=float(rng.uniform(*cfg['width'])); length=float(rng.uniform(*cfg['length']))
    spot=tuple(float(x) for x in rng.uniform(-cfg['spot_jitter_m'],cfg['spot_jitter_m'],2))
    delta=cfg['delta_easy_m']+(cfg['delta_hard_m']-cfg['delta_easy_m'])*level/(terrain['levels']-1)
    target=oracle['turn_swept_radius_m']+oracle['margin_m']+delta
    if target<=0: raise ValueError('nonpositive target turn clearance')
    # A diagonal sector fits the reserved disk even in a 3x3 workspace.
    slot_angle=float(rng.choice([math.pi/4,3*math.pi/4,5*math.pi/4,7*math.pi/4]))
    direction=(math.cos(slot_angle),math.sin(slot_angle)); tangent=(-direction[1],direction[0])
    centre=(spot[0]+cfg['cylinder_slot_distance_m']*direction[0],spot[1]+cfg['cylinder_slot_distance_m']*direction[1])
    half=cfg['cylinder_path_half_length_m']
    slot_a=(centre[0]-half*tangent[0],centre[1]-half*tangent[1]); slot_b=(centre[0]+half*tangent[0],centre[1]+half*tangent[1])
    reserve=max(cyl['radius'])+cyl['clearance_m']+cyl['sway_max_offset_m']
    eps=terrain['geometry_epsilon']; objects=[]
    count=int(rng.integers(cfg['obstacle_count'][0],cfg['obstacle_count'][1]+1))
    if count<4: raise ValueError('turn clutter requires at least four obstacles')
    has_outer_walls=rng.random()<cfg['outer_wall_probability']
    # If perimeter walls are present, seat rows run parallel to a wall.
    if has_outer_walls:
        anchor_angle=(math.pi if direction[0]>0 else 0.) if int(rng.integers(0,2))==0 else (3*math.pi/2 if direction[1]>0 else math.pi/2)
    else:
        anchor_angle=slot_angle+math.pi+float(rng.uniform(-cfg['row_jitter_rad'],cfg['row_jitter_rad']))
    for i in range(count):
        category='chair' if i==0 or i%3==0 else ('pillar' if i%3==1 else 'block')
        for attempt in range(cfg['placement_attempts']):
            if category=='chair':
                radial=float(rng.uniform(*cfg['chair_depth']))/2; lateral=float(rng.uniform(*cfg['chair_length']))/2; height=float(rng.uniform(*cfg['chair_height'])); kind='box'
            elif category=='pillar':
                radial=lateral=float(rng.uniform(*cfg['pillar_width']))/2; height=float(rng.uniform(*cfg['pillar_height'])); kind='pillar'
            else:
                radial=float(rng.uniform(*cfg['block_depth']))/2; lateral=float(rng.uniform(*cfg['block_width']))/2; height=float(rng.uniform(*cfg['block_height'])); kind='box'
            # Row chairs share yaw; others sample arbitrary radial face angles.
            angle=anchor_angle if category=='chair' else float(rng.uniform(0.,2*math.pi))
            outward=0. if i==0 else float(rng.uniform(*cfg['other_clearance_m']))
            shift=0. if i==0 or category!='chair' else float(rng.uniform(*cfg['row_offset_m']))
            ux,uy=math.cos(angle),math.sin(angle)
            x=spot[0]+(target+radial+outward)*ux-shift*uy
            y=spot[1]+(target+radial+outward)*uy+shift*ux
            candidate=StaticBox((x,y,height/2),(radial,lateral,height/2),angle,kind)
            ex=abs(ux)*radial+abs(uy)*lateral; ey=abs(uy)*radial+abs(ux)*lateral
            if abs(x)+ex>length/2-eps or abs(y)+ey>width/2-eps: continue
            clearance=point_box_distance(spot,candidate)
            if clearance<target-eps: continue
            if not segment_clear(slot_a,slot_b,candidate,reserve,eps): continue
            if not segment_clear(centre,spot,candidate,cfg['route_clearance_m'],eps): continue
            objects.append(candidate); break
        else:
            raise ValueError(f'TURN_CLUTTER_PLACEMENT_FAILED level={level} obstacle={i}')
    if has_outer_walls:
        thickness=terrain['wall_thickness']
        for sign in (-1,1):
            objects.append(StaticBox((sign*(length+thickness)/2,0.,wall_height/2),(thickness/2,(width+2*thickness)/2,wall_height/2),0.,'wall'))
            objects.append(StaticBox((0.,sign*(width+thickness)/2,wall_height/2),(length/2,thickness/2,wall_height/2),0.,'wall'))
    actual=min(point_box_distance(spot,b) for b in objects)
    if abs(actual-target)>eps: raise ValueError('nearest obstacle missed target turn clearance')
    return tuple(objects),(centre,spot),max(width,length),actual


def turn_free_polygon(meta, *, config, samples=None):
    """Angular visible-free polygon approximation; not the full connected free set.

    Rays terminate at OBBs or a declared visualization radius. Add each obstacle's
    closest-point direction so the minimum clearance is represented exactly.
    This is critic/debug metadata, not a sensor observation.
    """
    samples=config['turn_room']['polygon_samples'] if samples is None else samples
    if samples<3 or meta.turn_spot is None: raise ValueError('polygon arguments')
    origin=meta.turn_spot[0]; limit=max(*config['turn_room']['width'],*config['turn_room']['length'])/2
    angles=list(np.linspace(0.,2*math.pi,samples,endpoint=False))
    for box in meta.obstacles:
        c,s=math.cos(box.yaw),math.sin(box.yaw); dx,dy=origin[0]-box.center[0],origin[1]-box.center[1]
        lx=max(-box.half[0],min(box.half[0],c*dx+s*dy)); ly=max(-box.half[1],min(box.half[1],-s*dx+c*dy))
        px,py=box.center[0]+c*lx-s*ly,box.center[1]+s*lx+c*ly
        angles.append(math.atan2(py-origin[1],px-origin[0])%(2*math.pi))
    points=[]; eps=config['terrain']['geometry_epsilon']
    for angle in sorted(set(float(a) for a in angles)):
        direction=(math.cos(angle),math.sin(angle)); distance=limit
        for box in meta.obstacles:
            interval=line_box_interval(origin,direction,box,eps)
            if interval and interval[1]>=0: distance=min(distance,max(0.,interval[0]))
        points.append((origin[0]+distance*direction[0],origin[1]+distance*direction[1]))
    return tuple(points)
