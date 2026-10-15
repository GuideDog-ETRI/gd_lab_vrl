"""Training-only phase/contact/terminal recorder. Never passed to student forward."""
import types
import torch
class Labels:
    def __init__(self,b,gap):
        self.b=b;self.gap=gap;self.robot=b.scene['robot'];self.sensor=b.scene['contact_forces']
        self.feet=[self.sensor.body_names.index(n) for n in self.robot.body_names if n.endswith('_foot')]
        if len(self.feet)!=4:raise ValueError('four foot contacts required')
        self.air=torch.zeros(b.num_envs,dtype=torch.bool,device=b.device)
        self.landed=torch.zeros(b.num_envs,dtype=torch.long,device=b.device)
        self.terminal={};self.original=b._reset_idx
        def reset(base,ids,*args,**kwargs):
            snapshot=self.read(update=False)
            tm=b.termination_manager
            for i in ids.tolist():
                reasons=[name for name in tm.active_terms if bool(tm.get_term(name)[i])]
                if 'b3_approach' in reasons:
                    reasons.append('sampler_post_gap_cutoff' if snapshot['far_distance'][i]<-.8 else 'sampler_stall_cutoff')
                self.terminal[i]={'reasons':reasons,'timeout':bool(tm.time_outs[i]),'contact':snapshot['contacts'][i].cpu().tolist(),'phase':snapshot['phase'][i]}
            result=self.original(ids,*args,**kwargs)
            self.air[ids]=False;self.landed[ids]=0
            return result
        b._reset_idx=types.MethodType(reset,b)
    def read(self,update=True):
        from gd_lab.mdp.terrain_families import family_column_masks
        b=self.b;d=self.robot.data;contacts=self.sensor.data.net_forces_w[:,self.feet].norm(dim=-1)>15
        airborne=~contacts.any(-1);landing=self.air & ~airborne
        if update:
            self.landed=torch.where(landing,torch.full_like(self.landed,30),(self.landed-1).clamp(min=0));self.air=airborne
        slots,_=self.gap.for_envs(b);distance=slots[:,1,0]-(d.root_pos_w[:,0]-b.scene.env_origins[:,0])
        far_distance=slots[:,1,1]-(d.root_pos_w[:,0]-b.scene.env_origins[:,0])
        families=['unknown']*b.num_envs
        for name,cols in family_column_masks(b).items():
            for i in torch.isin(b.scene.terrain.terrain_types,cols).nonzero().flatten().tolist():families[i]=name
        phase=[]
        for i,f in enumerate(families):
            if airborne[i]:p='takeoff' if d.root_lin_vel_w[i,2]>.15 else 'airborne'
            elif landing[i]:p='landing'
            elif self.landed[i]>0:p='recovery'
            elif f=='platform_gap':p='post_gap' if far_distance[i]<0 else ('prep' if .3<=distance[i]<=1 else 'approach')
            else:p='stairs' if 'stairs' in f else ('flat' if f=='v3_flat' else 'other')
            phase.append(p)
        forces={}
        for name in b.reward_manager.active_terms:
            fn=b.reward_manager.get_term_cfg(name).func
            if hasattr(fn,'force_w'):forces[name]=fn.force_w.detach().cpu().clone()
        return {'phase_version':'v3_contact15N_v1_heuristic','phase':phase,'family':families,'contacts':contacts.detach().cpu(),'vertical_velocity':d.root_lin_vel_w[:,2].detach().cpu().clone(),'near_distance':distance.detach().cpu(),'far_distance':far_distance.detach().cpu(),'force_world':forces,'terminal':dict(self.terminal)}
    def close(self):self.b._reset_idx=self.original
