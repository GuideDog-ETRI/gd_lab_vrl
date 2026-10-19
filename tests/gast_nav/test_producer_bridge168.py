import numpy as np
from gd_lab.gast_nav.terrains import build_scene

def test_sensor_inventory_is_exact_generated_boxes_including_floor():
    for kind in ('nav_corridor','nav_door','nav_turn_room','nav_stair_landing'):
        meshes,origin,meta=build_scene(kind,0.,seed=168)
        assert len(meshes)>len(meta.obstacles)  # floor slabs are not oracle obstacles
        for mesh in meshes:
            b=mesh.metadata['gast_nav_sensor_solid']
            c,s=np.cos(b.yaw),np.sin(b.yaw)
            rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            local=(np.asarray(mesh.vertices)-origin-np.asarray(b.center))@rotation
            np.testing.assert_allclose(np.abs(local),np.broadcast_to(b.half,local.shape),atol=1e-8)
