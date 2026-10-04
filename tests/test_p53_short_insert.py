"""Software-only gradients/lifecycle checks; no new physical toy or training run."""
import copy
import pytest
import torch
from prove.task_graph.p52_stage_policy import Config,StageRuntime
from prove.task_graph.p53_short_insert import ShortInsertPolicy,full_stage_loss


def sample(length=10):
    torch.manual_seed(53)
    model=ShortInsertPolicy(Config(raw_dim=24,appearance_dim=8,role_dim=8,hidden=16,latent=4))
    features=dict(raw=torch.randn(length+1,2,3,24),boxes=torch.rand(length+1,2,3,4),
                  visible=torch.ones(length+1,2,3),proprio=torch.randn(length+1,18))
    return model,features,torch.randn(length,6)*.1,torch.randn(length,6)*.1


def test_direct_auxiliary_has_no_control_or_z0_gradient():
    m,f,a,_=sample();at=lambda t:{k:v[t] for k,v in f.items()}
    loss=m.isolated_auxiliary(at(0),a[0],at(1),m.initial(at(0)))
    loss.backward()
    assert all(p.grad is None for module in (m.role,m.encoder,m.policy) for p in module.parameters())
    assert m.z0.grad is None
    assert sum(p.grad.abs().sum() for p in m.predictor.parameters())>0


def test_online_action_bc_keeps_second_order_predictor_path():
    m,f,a,y=sample()
    loss,_=full_stage_loss(m,f,a,y,horizon=10,fixed_bc_weight=0,auxiliary_weight=0)
    loss.backward()
    assert sum(p.grad.abs().sum() for p in m.predictor.parameters())>0
    assert sum(p.grad.abs().sum() for p in m.encoder.parameters())>0
    assert m.z0.grad.abs().sum()>0


def test_first_stage_action_is_supervised():
    m,f,a,y=sample();_,before=full_stage_loss(m,f,a,y,horizon=10)
    changed=y.clone();changed[0]+=1
    _,after=full_stage_loss(m,f,a,changed,horizon=10)
    assert after['fixed_bc']>before['fixed_bc'] and after['online_bc']>before['online_bc']


def test_full_stage_latents_match_deployment_without_midstage_reset():
    m,f,a,y=sample(12)
    _,parts=full_stage_loss(m,f,a,y,horizon=12)
    expected=parts['action_latents'].detach();end=parts['final_z'].detach()
    runtime=StageRuntime(copy.deepcopy(m));at=lambda t:{k:v[t] for k,v in f.items()};runtime.enter(at(0))
    for t in range(12):
        assert torch.allclose(runtime.z,expected[t],atol=1e-7)
        runtime.command(at(t));runtime.feedback(at(t),a[t],at(t+1))
    assert runtime.updates==12 and len(runtime.history)==8
    assert torch.allclose(runtime.z,end,atol=1e-7)
    with pytest.raises(ValueError,match='horizon'):full_stage_loss(m,f,a,y,horizon=11)


def test_variable_length_minibatch_uses_full_stages_without_padding():
    from tools.task_graph.p53_train_short import minibatch_loss
    m,f,a,y=sample(3);_,f2,a2,y2=sample(7)
    episodes=[dict(features=f,executed_action=a,expert_action=y),dict(features=f2,executed_action=a2,expert_action=y2)]
    actual,_=minibatch_loss(m,episodes,horizon=7,branch='joint')
    first,_=full_stage_loss(m,f,a,y,horizon=7);second,_=full_stage_loss(m,f2,a2,y2,horizon=7)
    assert torch.allclose(actual,(first+second)/2)


def test_two_branches_load_identical_common_weights_rng_optimizer(tmp_path):
    from dataclasses import asdict
    from tools.task_graph.p53_train_short import fork_common
    m,_,_,_=sample();opt=torch.optim.Adam(m.parameters(),lr=3e-4);g=torch.Generator().manual_seed(11)
    path=tmp_path/'synthetic_common_state.pt'
    # Format/lifecycle test, not an optimized or performance-qualified checkpoint.
    torch.save(dict(schema='p53_common_warmup_v01',common_outer_updates=1000,model=m.state_dict(),optimizer=opt.state_dict(),
                    config=asdict(m.config),generator_state=g.get_state(),torch_rng_state=torch.get_rng_state()),path)
    a,oa,ga,_=fork_common(path);b,ob,gb,_=fork_common(path)
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
    assert oa.state_dict()==ob.state_dict()
    assert torch.equal(torch.randint(100,(8,),generator=ga),torch.randint(100,(8,),generator=gb))
