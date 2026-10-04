import torch,json
from pathlib import Path
r=Path('/home/zheng/github/Interaction-RSI/evaluations/fast_adapter_p18_visual_v01')
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
b=torch.load(r/'train_bank/bank.pt',map_location='cpu',weights_only=False)
# Only already-training data; never influences checkpoint selection or source.
x=b['query'][:8192].flatten(0,1).flatten(1).float().cuda();y=(2*b['query_base'][:8192]).flatten(0,1).cuda()
a=4096*4;mu=x[:a].mean(0);sd=x[:a].std(0).clamp(min=.05);x=(x-mu)/sd
x=torch.cat([x,torch.ones(x.shape[0],1,device='cuda')],1)
w=torch.linalg.solve(x[:a].T@x[:a]+torch.eye(x.shape[1],device='cuda')*10,x[:a].T@y[:a])
p=x[a:]@w;err=p-y[a:];mean=y[:a].mean(0)
d={'purpose':'training-bank visual observability diagnostic only; no protocol changes or checkpoint selection','train_domains':4096,'heldout_training_bank_domains':4096,'regularizer':10,'relative_position_rmse':err.square().mean(0).sqrt().tolist(),'relative_position_mae':err.abs().mean(0).tolist(),'r2':(1-err.square().sum(0)/(y[a:]-y[a:].mean(0)).square().sum(0)).tolist(),'constant_mean_mae':(y[a:]-mean).abs().mean(0).tolist(),'success_threshold_world':.05,'bias_abs_max':.04,'no_test_data_used':True}
(r/'preflight/TRAIN_BANK_VISUAL_DIAGNOSTIC.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
