"""P53 draft: native short-Insert full-stage BC and isolated prediction auxiliary.

No experiment budget/horizon is fixed by this module. P52 remains immutable.
Only the direct auxiliary branch detaches control representation and latent;
inner adaptation retains the differentiable chain needed by outer action BC.
"""
import torch
from prove.task_graph.p52_stage_policy import StagePolicy


class ShortInsertPolicy(StagePolicy):
    def isolated_auxiliary(self, obs, executed_action, next_obs, z):
        """Train predictor, not role/interaction encoder, policy, or latent init."""
        c=self.config
        inputs=torch.cat((self.encode(obs).detach(),z.detach(),executed_action.detach()),-1)
        pred=self.predictor(inputs)
        cut=c.views*3*c.appearance_dim
        appearance=pred[...,:cut].reshape(*pred.shape[:-1],c.views,3,c.appearance_dim)
        boxes=pred[...,cut:cut+c.views*12].reshape(*pred.shape[:-1],c.views,3,4)
        proprio=pred[...,cut+c.views*12:]
        with torch.no_grad():
            target_appearance=((next_obs['raw']-self.raw_mean)/self.raw_scale)[...,:c.appearance_dim]
            target_boxes=(next_obs['boxes']-self.box_mean)/self.box_scale
            target_proprio=(next_obs['proprio']-self.proprio_mean)/self.proprio_scale
            mask=next_obs['visible']
        per_role=(appearance-target_appearance).square().mean(-1)+(boxes-target_boxes).square().mean(-1)
        return ((per_role*mask).sum((-1,-2))/mask.sum((-1,-2)).clamp_min(1)
                +(proprio-target_proprio).square().mean(-1)).mean()


def full_stage_loss(model, features, executed_actions, expert_actions, *, horizon,
                    meta=True, fixed_bc_weight=.5, auxiliary_weight=.1):
    """One actual stage, t0 included; no teacher phase/GT relation input.

    Explicit horizon is supplied by the future frozen experiment. Variable-length
    episodes can be averaged as independent trajectories, without artificial padded
    transitions or restarting z halfway through a physical stage.
    """
    length=len(executed_actions)
    if not 1<=length<=horizon:raise ValueError('Stage exceeds declared training horizon')
    if executed_actions.shape!=(length,6) or expert_actions.shape!=(length,6):
        raise ValueError('Expected real executed/expert OSC6 sequences')
    if set(features)!={'raw','boxes','visible','proprio'} or any(len(x)!=length+1 for x in features.values()):
        raise ValueError('Expected exactly T+1 public feature observations')
    at=lambda t:{k:v[t] for k,v in features.items()}
    z0=model.initial(at(0));z=z0;history=[];online_bc=[];fixed_bc=[];aux=[];latents=[]
    for t in range(length):
        obs,nxt=at(t),at(t+1)
        # Act on o_t first; expert supervision includes the first stage action.
        online_bc.append((model.action(obs,z)-expert_actions[t]).square().mean())
        fixed_bc.append((model.action(obs,z0)-expert_actions[t]).square().mean())
        aux.append(model.isolated_auxiliary(obs,executed_actions[t],nxt,z))
        latents.append(z)
        # Feedback is available only after the executed action and next observation.
        history.append((obs,executed_actions[t],nxt));history=history[-model.config.history:]
        if meta:
            z,_=model.adapt_history(history,z,z0,create_graph=True)
    online_bc=torch.stack(online_bc).mean();fixed_bc=torch.stack(fixed_bc).mean();aux=torch.stack(aux).mean()
    control=(1-fixed_bc_weight)*online_bc+fixed_bc_weight*fixed_bc if meta else fixed_bc
    return control+auxiliary_weight*aux,dict(control=control,online_bc=online_bc,fixed_bc=fixed_bc,
                                            auxiliary=aux,action_latents=torch.stack(latents),final_z=z)
