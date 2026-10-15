"""E1 interventions: CPU-testable, no simulator import or optimizer."""
import torch
CONDITIONS = ('p_teacher_packet','a_student_gate','b_student_gate1','c_teacher_full','d_teacher_early','e_teacher_late','f_teacher_student_gate')

class LatentSwitch:
    def __init__(self,n,device):
        self.injected=torch.zeros(n,dtype=torch.bool,device=device)
        self.first_step=torch.full((n,),-1,dtype=torch.long,device=device)
    def reset(self,done):
        self.injected[done]=False;self.first_step[done]=-1
    @torch.inference_mode()
    def select(self,condition,raw,delivered,delivery_gate,ready,age,teacher,distance,step,packet_teacher=None):
        if condition not in CONDITIONS:raise ValueError(condition)
        fresh=ready & (age>=0) & (age<.3)
        actual_gate=delivery_gate*fresh[:,None]
        baseline=delivered*fresh[:,None]  # delivered is ALREADY gated
        if condition=='a_student_gate':out=baseline
        elif condition=='b_student_gate1':out=raw*ready[:,None]  # remove quality AND age gate; still need a received frame
        elif condition=='p_teacher_packet':
            if packet_teacher is None:raise ValueError('missing capture-time teacher latent')
            out=packet_teacher*actual_gate
        elif condition=='c_teacher_full':out=teacher
        elif condition=='f_teacher_student_gate':out=teacher*actual_gate
        else:
            threshold=1.0 if condition=='d_teacher_early' else .3
            trigger=torch.isfinite(distance)&(distance<=threshold)
            first=trigger & ~self.injected
            self.first_step[first]=step
            self.injected |= trigger  # from first approach through episode end
            out=torch.where(self.injected[:,None],teacher,baseline)
        if not torch.isfinite(out).all():raise ValueError('nonfinite actor latent')
        return out,actual_gate

@torch.inference_mode()
def action_at_boundary(teacher,obs,switch,condition,raw,delivered,gate,ready,age,distance,step,packet_teacher=None):
    # One teacher latent from this environment's CURRENT clean history, every policy step.
    good=teacher.terrain_latent(obs)
    latent,effective=switch.select(condition,raw,delivered,gate,ready,age,good,distance,step,packet_teacher)
    actions=teacher.act_with_terrain_latent(obs,latent)
    return actions,latent,effective
