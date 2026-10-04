"""Offline derivatives only. Never used to choose an executed robot action."""
import torch


def causal_pairs(record):
    """An update after action[t] may influence action[t+1], not action[t].

    Omit the last update because no subsequent action was executed in the record.
    """
    steps = record['actions'].shape[1]
    assert record['visual'].shape[1] == steps+1
    assert record['proprio'].shape[1] == steps+1
    assert record['latent_before_action'].shape[1] == steps
    return dict(visual=record['visual'][:,1:steps], proprio=record['proprio'][:,1:steps],
                previous=record['actions'][:,:-1], actual_next=record['actions'][:,1:],
                before=record['latent_before_action'][:,:-1], after=record['latent_before_action'][:,1:],
                active=record['active'][:,1:], source_active=record['active'][:,:-1],
                update=(record['counts_before_action'][:,1:]-record['counts_before_action'][:,:-1])>0,
                update_number=record['counts_before_action'][:,1:])


def action_direction(net, visual, proprio, previous, base, before, after, teacher, active, anchor):
    """Hold next observation, previous action, base and stage gate FIXED.

    Positive benefit means less error to the privileged state-based teacher;
    this is a local action proxy, never an observed task-success improvement.
    """
    with torch.enable_grad():
        z = before.detach().requires_grad_(True)
        potential_before = net.action(visual, proprio, previous, base, z)
        applied_before = torch.where(active[...,None], potential_before, base)
        loss_before = ((applied_before-teacher)/.5).square().mean(-1)
        grad, = torch.autograd.grad(loss_before.sum(), z)
    with torch.no_grad():
        potential_after = net.action(visual, proprio, previous, base, after)
        applied_after = torch.where(active[...,None], potential_after, base)
        frozen3 = torch.where(active[...,None], net.action(visual,proprio,previous,base,anchor), base)
        eb=((applied_before-teacher)/.5).square();ea=((applied_after-teacher)/.5).square()
        delta=after-before;denom=grad.norm(dim=-1)*delta.norm(dim=-1)
        cosine=-(grad*delta).sum(-1)/denom.clamp_min(1e-20)
        loss_base=((base-teacher)/.5).square().mean(-1)
        return {k:v.detach() for k,v in {
            'action_loss_before':eb.mean(-1),'action_loss_after':ea.mean(-1),
            'action_benefit':(eb-ea).mean(-1),'translation_benefit':(eb[...,:3]-ea[...,:3]).mean(-1),
            'gripper_benefit':eb[...,3]-ea[...,3],
            'task_descent_cosine':cosine,'direction_defined':denom>1e-12,
            'first_order_benefit':-(grad*delta).sum(-1),
            'z_delta_norm':delta.norm(dim=-1),'action_change_norm':(applied_after-applied_before).norm(dim=-1),
            'current_minus_base_loss':ea.mean(-1)-loss_base,
            'current_minus_first3_loss':ea.mean(-1)-((frozen3-teacher)/.5).square().mean(-1),
            'actual_after':applied_after,'teacher_grip':teacher[...,3],
            'before_grip':applied_before[...,3],'after_grip':applied_after[...,3],'base_grip':base[...,3],
            'close_flip_harm':(teacher[...,3]>0)&(base[...,3]>0)&(applied_after[...,3]<=0),
            'open_flip_harm':(teacher[...,3]<0)&(base[...,3]<0)&(applied_after[...,3]>=0),
        }.items()}
