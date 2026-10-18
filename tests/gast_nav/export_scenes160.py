"""CPU review artifacts only; no simulator/optimizer imports. Run via staging overlay."""
from dataclasses import asdict
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle
import trimesh
from gd_lab.gast_nav.terrains import build_scene,KINDS
from gd_lab.gast_nav.scene_geometry import box_corners_xy
from gd_lab.gast_nav.feasibility import passage_feasible,turn_feasible


def main():
    output=Path(__file__).with_name('artifacts160'); output.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    summary={}
    for ax,kind in zip(axes.flat,KINDS):
        meshes,origin,meta=build_scene(kind,.5,seed=160)
        local=trimesh.util.concatenate(meshes); local.apply_translation(-origin)
        local.export(output/f'{kind}.obj')
        (output/f'{kind}.json').write_text(json.dumps(asdict(meta),indent=2)+'\n')
        for box in meta.obstacles:
            color={'wall':'#777777','frame':'#777777','step':'#b6d9ef','rail':'#af7a3b','box':'#d88045','pillar':'#b34f4f'}[box.kind]
            ax.add_patch(Polygon(box_corners_xy(box),closed=True,facecolor=color,edgecolor='black',linewidth=.4,alpha=.65))
        route=meta.route_polyline
        ax.plot([x for x,y in route],[y for x,y in route],'-o',color='#1a6c9a',markersize=3,label='route')
        if meta.landing:
            (x0,y0),(x1,y1)=meta.landing
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='green',linewidth=2))
        if meta.turn_spot:
            ax.add_patch(Circle(meta.turn_spot[0],meta.turn_spot[1],fill=False,color='green',linewidth=1.5))
            outcome=turn_feasible(meta)
        else: outcome=passage_feasible(meta)
        points=[p for box in meta.obstacles for p in box_corners_xy(box)]+list(route)
        ax.set_xlim(min(p[0] for p in points)-.4,max(p[0] for p in points)+.4)
        ax.set_ylim(min(p[1] for p in points)-.4,max(p[1] for p in points)+.4)
        ax.set_aspect('equal'); ax.set_title(f'{kind} | available={outcome.available:.2f} required={outcome.required:.2f}')
        ax.set_xlabel('env-local x (m)'); ax.set_ylabel('env-local y (m)'); ax.grid(alpha=.2)
        summary[kind]={'difficulty':.5,'seed':160,'mesh_count':len(meshes),'obstacles':len(meta.obstacles),
                       'all_components_watertight':all(m.is_watertight for m in meshes),
                       'available':outcome.available,'required':outcome.required,'feasible':outcome.feasible,
                       'origin':origin.tolist(),'route_length':meta.route_length}
    fig.savefig(output/'scene_overview.png',dpi=140); plt.close(fig)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__': main()
