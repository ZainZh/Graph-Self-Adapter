"""Two training-only engineering rollouts; no success-based candidate selection."""
import json
import torch
from prove.fast_adapter.skill_adapter import SkillAdapter
from prove.fast_adapter.geometry_insert import geometry
from prove.fast_adapter.visual_alignment import DEFAULT_MODEL
from tools.fast_adapter.p24_common import R, load_base, weight_hash
from tools.fast_adapter.p24_rollout import SceneBatch, support, query, visual_assets
from tools.fast_adapter.p18_run import save_json, digest


def main():
    torch.set_num_threads(2); torch.backends.cudnn.enabled = False; torch.manual_seed(24999)
    out = R/'engineering'; out.mkdir(exist_ok=True)
    net = SkillAdapter().cuda().eval().requires_grad_(False)
    base = load_base(); weights = (weight_hash(net), weight_hash(base))
    encoder, pca = visual_assets(DEFAULT_MODEL)
    scene = SceneBatch([geometry(2201321+i) for i in range(2)], [2202321, 2202322], encoder, pca)
    trace = support(scene, net, base, [30000321, 30000322])
    replay, _ = net.adapt_path(trace['visual'].cuda(), trace['proprio'].cuda(), trace['actions'].cuda())
    error = float((torch.stack(replay, 1).cpu()-trace['meta_paths']).abs().max())
    assert error < 1e-4, error
    raw = query(scene, base, [31000321, 31000322], net, replay[3], labels=True)
    assert raw['visual'].shape == (2, 49, 64)
    assert raw['expert_label'].shape == (2, 48, 3)
    assert torch.isfinite(raw['visual']).all()
    assert (weight_hash(net), weight_hash(base)) == weights
    torch.save({'support': trace, 'query': raw}, out/'SMOKE_TRAJECTORY.pt')
    scene.close()
    result = {'passed': True, 'queries': 2, 'scope': 'Training geometry only; scratch model discarded; no success gate or held-out observation',
              'live_vs_replayed_latent_max_error': error, 'weights_unchanged': True,
              'trajectory_sha256': digest(out/'SMOKE_TRAJECTORY.pt')}
    save_json(out/'LIVE_SMOKE.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
