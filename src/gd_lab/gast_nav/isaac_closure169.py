"""Live USD audit for the opt-in primitive-only C169 diagnostic."""
import numpy as np
from .inventory169 import primitive_c_fixture169,fixture_meshes169

def audit_closed_scene169(env):
    import omni.usd
    from pxr import Usd,UsdGeom,UsdPhysics
    stage=omni.usd.get_context().get_stage()
    enabled=[];unexpected=[];grounds=[];disabled_cylinders=0;boxes174=[]
    gen=env.scene.terrain.cfg.terrain_generator
    mixed=getattr(gen,'coverage174_mixed',False)
    for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        path=str(prim.GetPath());on=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if on is False:
            if mixed and path=='/World/ground/terrain/mesh':grounds.append(prim)
            if '/NavCylinder/' in path:disabled_cylinders+=1
            continue
        enabled.append(path)
        if path.startswith('/World/ground/') and prim.IsA(UsdGeom.Mesh):grounds.append(prim)
        elif mixed and path.startswith('/World/ground/solids/box_') and prim.IsA(UsdGeom.Cube):boxes174.append(prim)
        elif path.startswith('/World/envs/env_') and '/Robot/' in path:pass
        else:unexpected.append(path)
    if unexpected:raise RuntimeError(f'C169 unexpected enabled colliders: {unexpected[:20]}')
    if len(grounds)!=1:raise RuntimeError(f'C169 expected one terrain mesh, got {len(grounds)}')
    mesh=UsdGeom.Mesh(grounds[0]);points=np.array(mesh.GetPointsAttr().Get(),dtype=np.float64)
    matrix=np.array(UsdGeom.Xformable(grounds[0]).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    points=(np.concatenate((points,np.ones((len(points),1))),axis=1)@matrix)[:,:3]
    faces=np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    if not np.all(np.array(mesh.GetFaceVertexCountsAttr().Get())==3):raise RuntimeError('nontriangle ground')
    expected=[]
    local,_=fixture_meshes169(primitive_c_fixture169(),positive_coordinates=False)
    origins=env.scene.terrain.terrain_origins.detach().cpu().numpy().reshape(-1,3)
    gen=env.scene.terrain.cfg.terrain_generator
    for index,origin in enumerate(origins):
        if getattr(gen,'coverage174_mixed',False):
            from .inventory169 import PrimitiveCFixture169
            key=(index//gen.num_cols,index%gen.num_cols)
            local,_=fixture_meshes169(PrimitiveCFixture169(tuple(gen.size),gen.sensor_solids169[key],gen.nav_scene_grid[key]),positive_coordinates=False)
        for m in local:expected.append(np.asarray(m.vertices)[np.asarray(m.faces)]+origin)
    expected=np.concatenate(expected)
    if mixed:
        source=[(b,origin) for i,origin in enumerate(origins) for b in gen.sensor_solids169[(i//gen.num_cols,i%gen.num_cols)]]
        if len(boxes174)!=len(source):raise RuntimeError('physical source-box inventory mismatch')
        for prim in boxes174:
            index=int(str(prim.GetPath()).rsplit('_',1)[1]);b,origin=source[index]
            matrix=np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            c,s=np.cos(b.yaw),np.sin(b.yaw);target=np.eye(4)
            target[:3,:3]=np.diag(b.half)@np.array([[c,s,0],[-s,c,0],[0,0,1]])
            target[3,:3]=np.asarray(b.center)+origin
            if not np.allclose(matrix,target,atol=3e-5,rtol=0) or UsdGeom.Cube(prim).GetSizeAttr().Get()!=2.:raise RuntimeError('physical source-box geometry mismatch')

    # Triangle vertex/order invariant, exact geometry within USD float32 precision.
    def canonical(tri):
        tri=np.round(tri,4)
        tri=np.sort(tri.view([('x','f8'),('y','f8'),('z','f8')]).reshape(-1,3),axis=1).view('f8').reshape(-1,9)
        return tri[np.lexsort(tri.T[::-1])]
    actual=points[faces]
    if getattr(gen,'coverage174_mixed',False):
        # Rotated float32 vertices at +/-100m can cross a decimal rounding edge.
        # Compare complete triangles in both directions, 9D Euclidean tolerance 0.3mm after rounding to 0.1mm; no box-only audit.
        from scipy.spatial import cKDTree
        aa=canonical(actual);ee=canonical(expected)
        max_error=max(cKDTree(aa).query(ee)[0].max(),cKDTree(ee).query(aa)[0].max()) if aa.shape==ee.shape else float('inf')
        if max_error>3e-4:raise RuntimeError(f'C174 triangle mismatch maximum 9D error {max_error}')
    elif actual.shape!=expected.shape or not np.array_equal(canonical(actual),canonical(expected)):
        raise RuntimeError(f'C169 live terrain triangle mismatch {actual.shape} vs {expected.shape}')
    if bool(env.nav.cylinder.enabled.any()) if getattr(env,'nav',None) else False:
        raise RuntimeError('C169 dynamic cylinders enabled')
    return dict(closed=True,enabled_colliders=len(enabled),terrain_path=str(grounds[0].GetPath()),
                terrain_triangles=len(faces),tile_count=len(origins),disabled_cylinder_colliders=disabled_cylinders,
                geometry_check=('bidirectional complete-triangle comparison; rounded 1e-4 m; 9D tolerance 3e-4 m' if getattr(gen,'coverage174_mixed',False) else 'all live world triangles equal same-source primitive mesh (round 1e-4 m)'),
                triangle_max_9d_error_m=(float(max_error) if getattr(gen,'coverage174_mixed',False) else 0.))
