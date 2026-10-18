"""Pair-filtered contact classification, CPU-testable and independent of Isaac.

A geometric proximity/blocked flag is NEVER evidence of engine invalidity.
Invalid causes must be supplied explicitly by the reset/engine diagnostics.
"""
import torch

INVALID_CODES={'INITIAL_OVERLAP':1,'NONFINITE_STATE':2,'ENGINE_FAILURE':3,'PERSON_MODEL_INTRUSION':4,'NONFINITE_CONTACT':5}

class PersonContactTracker:
    def __init__(self,num_envs,threshold_n,*,device='cpu'):
        if not threshold_n>0:raise ValueError('positive contact threshold required')
        self.threshold_n=threshold_n
        self.contact=torch.zeros(num_envs,dtype=torch.bool,device=device)
        self.invalid_code=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.person_contact=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.invalid=torch.zeros_like(self.person_contact)
        self.episodes=torch.zeros_like(self.person_contact)
        self._pre_action=torch.ones_like(self.contact)
        self.invalid_by_reason={key:0 for key in INVALID_CODES}
        self._counted=torch.zeros_like(self.contact)
        self.last_contact=self.contact.clone();self.last_invalid_code=self.invalid_code.clone()

    def reset(self,ids,*,initial_overlap=None):
        ids=torch.as_tensor(ids,device=self.contact.device,dtype=torch.long)
        self.contact[ids]=False;self.invalid_code[ids]=0;self._counted[ids]=False;self._pre_action[ids]=True
        self.episodes[ids]+=1
        if initial_overlap is not None:
            self.mark_invalid(ids[torch.as_tensor(initial_overlap,device=ids.device,dtype=torch.bool)],'INITIAL_OVERLAP')

    def mark_invalid(self,ids,reason):
        code=INVALID_CODES[reason]
        ids=torch.as_tensor(ids,device=self.contact.device,dtype=torch.long)
        if reason=='INITIAL_OVERLAP' and bool((~self._pre_action[ids]).any()):
            raise RuntimeError('initial overlap is reset-only, before first action')
        if bool(self._counted[ids].any()):raise RuntimeError('cannot retroactively reclassify a recorded episode')
        # Preserve the first diagnosed reason (e.g. reset overlap before NaN).
        self.invalid_code[ids]=torch.where(self.invalid_code[ids]==0,code,self.invalid_code[ids])

    def begin_action(self):
        self._pre_action[:]=False

    def observe(self,force_history,enabled):
        """[N,...,3] per-link/collider-pair histories; NEVER sum links before norm."""
        if force_history.shape[0]!=len(self.contact) or force_history.shape[-1]!=3:
            raise ValueError('contact history shape')
        finite=torch.isfinite(force_history).flatten(1).all(-1)
        bad=(~finite)&enabled&~self._counted
        self.mark_invalid(bad.nonzero().flatten(),'NONFINITE_CONTACT')
        hit=(torch.nan_to_num(force_history).norm(dim=-1)>self.threshold_n).flatten(1).any(-1)
        self.contact |= hit & enabled

    def classify(self):
        """Pure classification; used by either termination term in any order."""
        invalid=self.invalid_code!=0
        return self.contact & ~invalid,invalid

    def commit(self):
        """Once per episode at termination evaluation, before automatic reset."""
        contact,invalid=self.classify()
        fresh=(contact|invalid)&~self._counted
        self.person_contact+=(fresh&contact).long();self.invalid+=(fresh&invalid).long()
        for reason,code in INVALID_CODES.items():
            self.invalid_by_reason[reason]+=int((fresh&(self.invalid_code==code)).sum())
        self._counted|=fresh
        self.last_contact.copy_(contact);self.last_invalid_code.copy_(self.invalid_code)
        return contact,invalid

    def summary(self):
        total=int(self.episodes.sum());invalid=int(self.invalid.sum())
        return {'person_contact':int(self.person_contact.sum()),'invalid':invalid,'episodes_started':total,
                'valid_episode_denominator':total-invalid,
                'invalid_by_reason':dict(self.invalid_by_reason),
                'invalid_fraction':invalid/max(total,1)}


def sample_person_contacts(env):
    """Read ONLY per-link forces filtered to NavCylinder, including short contacts."""
    nav=getattr(env,'nav',None)
    if nav is None:return
    cfg=nav.cfg['person_contact']
    for name in cfg['robot_links']:
        data=env.scene['person_contact_'+name].data
        forces=data.force_matrix_w_history
        if forces is None:raise RuntimeError('person contact sensor requires collider filter')
        nav.person_contacts.observe(forces,nav.cylinder.enabled)
    finite=(torch.isfinite(nav.robot.data.root_state_w).all(-1)&
            torch.isfinite(nav.robot.data.joint_pos).all(-1)&torch.isfinite(nav.robot.data.joint_vel).all(-1))
    ids=((~finite)&~nav.person_contacts._counted).nonzero().flatten()
    nav.person_contacts.mark_invalid(ids,'NONFINITE_STATE')


def person_contact_termination(env):
    if getattr(env,'nav',None) is None:return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    sample_person_contacts(env)
    return env.nav.person_contacts.commit()[0]


def person_invalid_termination(env):
    if getattr(env,'nav',None) is None:return torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    sample_person_contacts(env)
    return env.nav.person_contacts.commit()[1]


def reset_overlap_mask(cylinder,ids,query_shape):
    """Exact-shape query adapter, called only after reset poses, before actions.

    query_shape(position_w,radius,height,robot_prefix) returns overlap with the
    actual robot colliders, not an inflated robot footprint or obstacle proxy.
    """
    ids=[int(i) for i in ids]
    result=torch.zeros(len(ids),dtype=torch.bool,device=cylinder.position.device)
    for j,i in enumerate(ids):
        if bool(cylinder.enabled[i]):
            result[j]=query_shape(cylinder.position[i].tolist(),float(cylinder.radius[i]),float(cylinder.height[i]),
                                  f'/World/envs/env_{i}/Robot/')
    return result


class IsaacResetOverlapQuery:
    """Non-colliding USD query primitive; never changes the physical cylinder.

    PhysX overlap_shape consumes a USD GPrim transform. A separate query shape
    avoids relying on USD write-back of tensor-written kinematic body poses.
    """
    def __init__(self,stage):
        from pxr import UsdGeom,Sdf,PhysicsSchemaTools
        from omni.physx import get_physx_scene_query_interface
        self.shape=UsdGeom.Cylinder.Define(stage,'/World/GastNavResetOverlapQuery')
        self.shape.CreateAxisAttr('Z');self.shape.MakeInvisible()
        self.translate=UsdGeom.Xformable(self.shape).AddTranslateOp()
        self.encoded=PhysicsSchemaTools.encodeSdfPath(Sdf.Path('/World/GastNavResetOverlapQuery'))
        self.query=get_physx_scene_query_interface()

    def __call__(self,position,radius,height,robot_prefix):
        from pxr import Gf
        self.shape.GetRadiusAttr().Set(radius);self.shape.GetHeightAttr().Set(height)
        self.translate.Set(Gf.Vec3d(*position))
        found=False
        def hit(record):
            nonlocal found
            if str(record.collision).startswith(robot_prefix):found=True;return False
            return True
        self.query.overlap_shape(*self.encoded,hit,False)
        return found
