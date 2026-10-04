import copy
import torch
from prove.task_graph.p52_stage_policy import Config, StagePolicy, StageRuntime


def fixture():
    torch.manual_seed(52)
    c=Config(raw_dim=24,appearance_dim=8,role_dim=8,hidden=16,latent=4)
    m=StagePolicy(c)
    def obs():
        return dict(raw=torch.randn(2,2,3,24),boxes=torch.rand(2,2,3,4),
                    visible=torch.ones(2,2,3),proprio=torch.randn(2,18))
    return m,obs(),obs(),obs()


def test_outer_bc_trains_predictor_through_z_update():
    m,o,n,q=fixture()
    loss,_=m.outer_loss((o,torch.randn(2,6),n),q,torch.randn(2,6),prediction_weight=0)
    loss.backward()
    # This is only possible through differentiable inner adaptation, not aux loss.
    assert sum(p.grad.abs().sum() for p in m.predictor.parameters())>0
    assert sum(p.grad.abs().sum() for p in m.policy.parameters())>0
    assert m.z0.grad.abs().sum()>0


def test_runtime_only_z_updates_and_closed_command():
    m,o,n,q=fixture()
    before=copy.deepcopy(m.state_dict())
    r=StageRuntime(m);r.enter(o)
    command=r.command(q);z=r.z.clone()
    r.feedback(o,torch.zeros(2,6),n)
    assert not torch.equal(z,r.z)
    assert not torch.equal(command,r.command(q))
    assert torch.equal(r.command(q)[:,-1],torch.ones(2))
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    assert all(not p.requires_grad for p in m.parameters())
    r.exit();assert r.z is None


def test_fixed_z_still_uses_current_observation():
    m,o,n,q=fixture();r=StageRuntime(m,online=False);r.enter(o)
    z=r.z.clone();a=r.command(o)
    r.feedback(o,torch.zeros(2,6),n)
    assert torch.equal(z,r.z) and r.updates==0
    assert not torch.equal(a,r.command(q))


def test_missing_roles_do_not_leak_raw_features():
    m,o,_,_=fixture();o['visible'].zero_();a=m.encode(o)
    o['raw']=torch.randn_like(o['raw'])*100
    assert torch.equal(a,m.encode(o))


def test_inner_step_is_batch_size_independent():
    m,o,n,_=fixture();a=torch.randn(2,6)
    z=m.initial(o);zb,_=m.adapt(o,a,n,z,z,create_graph=True)
    single=lambda d:{k:v[:1] for k,v in d.items()}
    z1=m.initial(single(o));zs,_=m.adapt(single(o),a[:1],single(n),z1,z1,create_graph=True)
    assert torch.allclose(zb[:1],zs,atol=1e-7,rtol=1e-5)


def test_runtime_window_does_not_reset_z_after_eight():
    m,o,n,q=fixture();r=StageRuntime(m);r.enter(o)
    z0=r.z.clone()
    for _ in range(12):r.feedback(o,torch.zeros(2,6),n)
    assert r.updates==12 and len(r.history)==8
    assert not torch.equal(r.z,z0)
    assert torch.equal(r.z0,z0)


def test_native_segment_training_smoke_and_chronology():
    from tools.task_graph.p52_train import train,segment_loss
    m,o,n,q=fixture()
    f={k:torch.stack([v[0]+.01*t if k!='visible' else v[0] for t in range(17)]) for k,v in o.items()}
    e=dict(episode_id='synthetic_schema_only',split='train',features=f,
           executed_action=torch.zeros(16,6),expert_action=torch.ones(16,6)*.1)
    trained,last=train([e],m.config,seed=11,warmup_steps=1,joint_steps=1,batch_size=1,lr=3e-4,prediction_weight=.1)
    assert last['loss']>=0 and torch.isfinite(trained.z0).all()
    # Query labels before time8 must not affect meta BC.
    fb={k:v[None] for k,v in f.items()};a=e['executed_action'][None];labels=e['expert_action'][None].clone()
    l1,_=segment_loss(trained,fb,a,labels,meta=True)
    labels[:,:8]+=10
    l2,_=segment_loss(trained,fb,a,labels,meta=True)
    assert torch.equal(l1,l2)
