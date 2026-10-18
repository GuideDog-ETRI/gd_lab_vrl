"""Vehicle-frame translational inertia. Pure Torch core; one force writer for all sources."""
import math
import torch

def limited_sine_amplitude(amplitude, frequency, jerk_max):
    if bool((frequency <= 0).any()): raise ValueError('frequency must be positive')
    return torch.minimum(amplitude, jerk_max / (2 * math.pi * frequency))

def sum_external_wrenches(masses, acceleration, handle_force=None, handle_torque=None):
    """All inputs world frame; [N,B], [N,3], optional [N,B,3]. Force applied at link COM.
    Callers turn off-COM handle force into torque r×F before supplying it.
    """
    if masses.ndim != 2 or acceleration.shape != (masses.shape[0], 3): raise ValueError('mass/acceleration shape')
    if not bool(torch.isfinite(masses).all()) or bool((masses <= 0).any()): raise ValueError('invalid actual link masses')
    force = -masses[..., None] * acceleration[:, None, :]
    torque = torch.zeros_like(force)
    if handle_force is not None:
        if handle_force.shape != force.shape: raise ValueError('handle force shape')
        force = force + handle_force
    if handle_torque is not None:
        if handle_torque.shape != torque.shape: raise ValueError('handle torque shape')
        torque = torque + handle_torque
    return force, torque

class TrainInertia:
    def __init__(self, num_envs, cfg, *, device='cpu', seed=42):
        self.cfg, self.device = cfg, torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.n = num_envs
        self.time = torch.zeros(num_envs, device=self.device)
        self.acceleration = torch.zeros(num_envs, 3, device=self.device)
        self.enabled = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.parameters = {k:torch.zeros(num_envs,device=self.device) for k in ('peak','rise','hold','delay','axis','amp','frequency','phase')}
        self.bias_direction = torch.zeros(num_envs,3,device=self.device)

    def reset(self, env_ids, enabled, *, scale=1.0):
        ids = torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        m = len(ids)
        def unit(): return torch.rand(m,generator=self.generator,device=self.device)
        def uniform(bounds): return bounds[0] + (bounds[1]-bounds[0])*unit()
        p,c = self.parameters,self.cfg
        peak = torch.where(unit()<c['rare_probability'],uniform(c['rare_acceleration_mps2']),uniform(c['acceleration_mps2'])) * scale
        p['peak'][ids] = peak * torch.where(unit()<.5,-1.,1.)
        p['rise'][ids] = torch.maximum(uniform(c['rise_s']),peak/c['jerk_max_mps3'])
        p['hold'][ids],p['delay'][ids] = uniform(c['hold_s']),uniform(c['delay_s'])
        p['axis'][ids] = (unit()*2-1)*math.pi
        p['frequency'][ids] = uniform(c['lateral_frequency_hz'])
        p['amp'][ids] = limited_sine_amplitude(uniform(c['lateral_amplitude_mps2'])*scale,p['frequency'][ids],c['jerk_max_mps3'])
        p['phase'][ids] = unit()*2*math.pi
        self.bias_direction[ids] = torch.rand((m,3),generator=self.generator,device=self.device)*2-1
        self.enabled[ids] = torch.as_tensor(enabled,device=self.device).bool()
        self.time[ids] = 0.; self.acceleration[ids] = 0.

    def step(self, dt):
        if dt <= 0: raise ValueError('physics dt')
        p = self.parameters
        self.time += dt
        t=(self.time-p['delay']).clamp_min(0)
        rise=p['rise'].clamp_min(torch.finfo(self.time.dtype).eps)
        trapezoid=torch.minimum((t/rise).clamp(0,1),((2*rise+p['hold']-t)/rise).clamp(0,1))
        x=p['peak']*trapezoid
        y=p['amp']*torch.sin(2*math.pi*p['frequency']*t+p['phase'])*trapezoid
        yaw=p['axis']
        target=torch.stack((yaw.cos()*x-yaw.sin()*y,yaw.sin()*x+yaw.cos()*y,torch.zeros_like(x)),-1)
        target *= self.enabled[:,None]
        # Bound the actual vector jerk, including envelope transitions and reset phase.
        delta=target-self.acceleration
        norm=delta.norm(dim=-1,keepdim=True)
        scale=(self.cfg['jerk_max_mps3']*dt/norm.clamp_min(torch.finfo(norm.dtype).eps)).clamp_max(1)
        self.acceleration += delta*scale
        return self.acceleration

    def gravity_bias(self, projected_gravity):
        amplitude=(self.acceleration.norm(dim=-1)*self.cfg['gravity_bias_per_mps2']).clamp_max(self.cfg['gravity_bias_limit'])
        biased=projected_gravity + self.bias_direction*amplitude[:,None]
        return biased/biased.norm(dim=-1,keepdim=True).clamp_min(torch.finfo(biased.dtype).eps)

class ExternalForceAccumulator:
    """Handle producers only submit buffers. apply() is the sole Isaac force write per physics step."""
    def __init__(self, num_envs, num_bodies, *, device):
        self.force=torch.zeros(num_envs,num_bodies,3,device=device)
        self.torque=torch.zeros_like(self.force)
        self.last_force=torch.zeros_like(self.force)
        self.last_torque=torch.zeros_like(self.force)
        self.write_count=0

    def submit_handle(self, force_w, torque_w):
        if force_w.shape != self.force.shape or torque_w.shape != self.torque.shape: raise ValueError('wrench shape')
        self.force.copy_(force_w);self.torque.copy_(torque_w)

    def reset(self, env_ids):
        self.force[env_ids]=0;self.torque[env_ids]=0
        self.last_force[env_ids]=0;self.last_torque[env_ids]=0

    def apply(self, robot, acceleration):
        masses=robot.root_physx_view.get_masses().to(acceleration.device)
        self.last_force,self.last_torque=sum_external_wrenches(masses,acceleration,self.force,self.torque)
        robot.set_external_force_and_torque(self.last_force,self.last_torque,is_global=True)
        self.write_count += 1
        return self.last_force,self.last_torque
