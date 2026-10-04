import torch
from prove.fast_adapter.structured_response import nominal_z,prediction,adapt,control,least_squares,MetaGradient

A=torch.tensor([[.25,0.],[0.,.25],[-.25,0.]])

def hidden():
    return torch.tensor([[[1.12,-.13,.12],[.11,.91,-.08]],[[.8,.12,-.1],[-.14,1.13,.14]]])

def test_gradient_identifies_affine_response_and_agrees_with_ls():
    truth=hidden();d=prediction(truth,A);prior=nominal_z(2)
    z,trace=adapt(prior,d,A,steps=150)
    assert torch.allclose(z,truth,atol=1e-5)
    assert torch.allclose(z,least_squares(d,A),atol=1e-5)
    assert trace[-1]<trace[0]*1e-6

def test_independent_domain_batch_scaling():
    d=prediction(hidden(),A)
    batch,_=adapt(nominal_z(2),d,A,steps=3)
    single,_=adapt(nominal_z(1),d[:1],A,steps=3)
    assert torch.allclose(single,batch[:1],atol=1e-7)

def test_updates_change_control_and_swap_hurts_prediction():
    truth=hidden();d=prediction(truth,A);z,_=adapt(nominal_z(2),d,A,steps=20)
    state=torch.zeros(2,2);goal=torch.full((2,2),.1)
    before,_=control(state,goal,nominal_z(2));after,_=control(state,goal,z)
    assert (after-before).abs().max()>.01
    good=(prediction(z,A)-d).square().mean()
    wrong=(prediction(z.roll(1,0),A)-d).square().mean()
    assert wrong>100*good

def test_freeze_and_meta_training_gradient_paths():
    model=MetaGradient();saved={k:v.detach().clone() for k,v in model.state_dict().items()}
    d=prediction(hidden(),A);z=model(d,A,create_graph=True)
    loss=(prediction(z,A)-d).square().mean();loss.backward()
    assert model.z0.grad.abs().sum()>0
    assert model.rate_logits.grad.abs().sum()>0
    for k,v in model.state_dict().items():assert torch.equal(v,saved[k])
    fixed=model(d,A,steps=0)
    assert torch.equal(fixed,model.z0.expand(2,-1,-1))

def test_singular_estimate_falls_back_to_nominal_without_true_physics():
    state=torch.zeros(2,2);goal=torch.ones(2,2)*.2
    action,invalid=control(state,goal,torch.zeros(2,2,3))
    assert invalid.all();assert torch.allclose(action,.5*(goal-state))
