"""CPU turn-clutter review artifacts, no simulator/optimizer imports."""
from dataclasses import asdict
import json
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Circle
import trimesh
import torch
from gd_lab.gast_nav.terrains import build_scene
from gd_lab.gast_nav.scene_config import load_scene_config
from gd_lab.gast_nav.scene_types import SceneIndex
from gd_lab.gast_nav.scene_geometry import box_corners_xy
from gd_lab.gast_nav.turn_clutter import turn_free_polygon,cylinder_slot
from gd_lab.gast_nav.moving_cylinder import MovingCylinder


def main():
    directory=Path(__file__).with_name('artifacts161'); directory.mkdir(exist_ok=True)
    config=load_scene_config(); fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for level,ax in zip((0,5,9),axes):
        meshes,origin,meta=build_scene('nav_turn_room',level/9,seed=161)
        poly=turn_free_polygon(meta,config=config); path=cylinder_slot(meta,config)
        mesh=trimesh.util.concatenate(meshes); mesh.apply_translation(-origin)
        mesh.export(directory/f'turn_level{level}.obj')
        payload={'scene_meta':asdict(meta),'visible_free_polygon':poly,'reserved_cylinder_segment':path}
        (directory/f'turn_level{level}.json').write_text(json.dumps(payload,indent=2)+'\n')
        ax.add_patch(Polygon(poly,closed=True,facecolor='#cae6cf',edgecolor='#4d995b',alpha=.6))
        for box in meta.obstacles:
            color='#777777' if box.kind=='wall' else ('#337da1' if box.kind=='pillar' else '#bc7950')
            ax.add_patch(Polygon(box_corners_xy(box),closed=True,facecolor=color,edgecolor='black',linewidth=.5))
        spot,radius=meta.turn_spot
        ax.add_patch(Circle(spot,radius,fill=False,edgecolor='#1166aa',linewidth=1.8))
        required=config['oracle']['turn_swept_radius_m']+config['oracle']['margin_m']
        ax.add_patch(Circle(spot,required,fill=False,edgecolor='#aa3333',linestyle='--'))
        ax.plot(*np.array(meta.route_polyline).T,color='#1166aa',marker='o',markersize=4)
        ax.plot(*np.array(path).T,color='#8d5bad',linewidth=3,label='reserved cylinder path')
        for endpoint in path:
            ax.add_patch(Circle(endpoint,max(config['cylinder']['radius'])+config['cylinder']['clearance_m']+config['cylinder']['sway_max_offset_m'],facecolor='none',edgecolor='#8d5bad',linestyle=':'))
        ax.set_title(f'Level {level}: available {radius:.3f}m / required {required:.2f}m')
        ax.set_aspect('equal'); ax.set_xlim(-2.8,2.8); ax.set_ylim(-2.8,2.8); ax.grid(alpha=.2)
        ax.set_xlabel('vehicle-floor x (m)'); ax.set_ylabel('vehicle-floor y (m)')
    fig.savefig(directory/'turn_clutter_overview.png',dpi=140); plt.close(fig)
    # Filter fixture, not a physical simulation: zero nominal speed, far robot.
    _,_,meta=build_scene('nav_turn_room',.5,seed=161)
    person=MovingCylinder(SceneIndex([meta]),stage='C',seed=1); person.speed[:]=0.
    start=person.position.clone(); history=[]; dt=.01
    for i in range(600):
        t=i*dt; accel=1.5 if t<2 else (-3. if t<4 else 0.)
        person.update(dt,torch.tensor([[accel,0.]])); person.step(dt,torch.tensor([[10.,10.,.3]]))
        history.append([t,accel,float(person.a_train_lp[0,0]),float(person.sway_offset[0,0]),float(person.position[0,0]-start[0,0]),bool(person.blocked[0])])
    (directory/'sway_fixture.json').write_text(json.dumps({'dt_s':dt,'columns':['time_s','a_train_x','a_lp_x','desired_offset_x','actual_displacement_x','blocked'],'samples':history},indent=2)+'\n')
    x=np.array(history,dtype=float); fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True,layout='constrained')
    axes[0].plot(x[:,0],x[:,1],label='train acceleration'); axes[0].plot(x[:,0],x[:,2],label='LPF'); axes[0].set_ylabel('m/s2'); axes[0].legend()
    axes[1].plot(x[:,0],x[:,3],label='desired -k*a_lp (clipped)'); axes[1].plot(x[:,0],x[:,4],'--',label='actual collider displacement')
    axes[1].set_ylabel('metres'); axes[1].set_xlabel('seconds'); axes[1].legend()
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(directory/'sway_fixture.png',dpi=140); plt.close(fig)


if __name__=='__main__': main()
