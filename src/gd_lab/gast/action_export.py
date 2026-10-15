"""Deploy wrappers. Residual explicitly requires runtime-v2 age/gate inputs."""
import torch
from gd_lab.deploy.export_vrl import DreamwaqVrlDeployPolicy


class ResidualDeployPolicy(DreamwaqVrlDeployPolicy):
    def __init__(self, teacher, term_dims, student_policy):
        super().__init__(teacher, term_dims)
        self.student_policy = student_policy.cpu().eval()

    def forward(self, direct_obs, cenet_obs, terrain_latent, age_seconds, quality_gate):
        mean = self.normalizer._mean[0, self.latest_idx]
        std = self.normalizer._std[0, self.latest_idx]
        latest = (direct_obs - mean) / (std + self.normalizer.eps)
        norm_history = self.normalizer(cenet_obs[..., self.time_major_to_term_major_idx])
        out = self.cenet(norm_history, deterministic=True)
        code = self.cenet.get_code(out)
        frames = torch.cat((latest, norm_history[..., self.older_frames_idx]), -1)
        base = torch.cat((frames, code), -1)
        return self.student_policy(base, terrain_latent, age_seconds, quality_gate), code


def export_action_wrapper(module, directory, residual=False):
    from pathlib import Path
    output = Path(directory)
    inputs = ['direct_obs', 'cenet_obs', 'terrain_latent']
    example = (torch.zeros(1, 46), torch.zeros(1, 230), torch.zeros(1, 32))
    if residual:
        inputs += ['age_seconds', 'quality_gate']
        example += (torch.zeros(1), torch.ones(1))
    jit, onnx = output/'policy_vrl.pt', output/'policy_vrl.onnx'
    torch.jit.trace(module, example).save(str(jit))
    torch.onnx.export(module, example, str(onnx), opset_version=18, dynamo=False,
                      input_names=inputs, output_names=['actions', 'z_t'],
                      dynamic_axes={name:{0:'batch'} for name in inputs+['actions','z_t']})
    return str(jit), str(onnx)
