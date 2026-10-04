import torch,json
from pathlib import Path
from tools.fast_adapter.p18_run import model,records_from_features,physics_data
from tools.fast_adapter.p18_experiment import prepare_latent_for_arm

torch.set_num_threads(2)
r=Path('evaluations/fast_adapter_p18_visual_v01');bank=torch.load(r/'train_bank/bank.pt',map_location='cpu',weights_only=False)
ids=torch.randperm(bank['n'],generator=torch.Generator().manual_seed(90007111))[:1024]
sh,*_=physics_data(bank['n'],bank['seed'],'train',train=True)
y=torch.cat([sh.M.flatten(1),sh.b],1)[ids]
f=bank['support'][ids,:4].float();rows=[]
for ck in sorted((r/'train').glob('*/*/model.pt')):
 cp=torch.load(ck,map_location='cpu',weights_only=False);torch.manual_seed(cp['seed']);init=model(cp['family']);sys=model(cp['family']);sys.load_state_dict(cp['state']);sys.freeze_for_eval()
 rec=records_from_features(sys,f);z,_=prepare_latent_for_arm(sys,cp['family'],rec,batch_size=len(ids),k=3,create_graph=False,detach_initial=True);z=z.detach().double();yy=y.double()
 mu=z[:512].mean(0);sd=z[:512].std(0).clamp(min=1e-10);xx=(z-mu)/sd;xx=torch.cat([xx,torch.ones(len(xx),1,dtype=torch.float64)],1)
 w=torch.linalg.solve(xx[:512].T@xx[:512]+10*torch.eye(33,dtype=torch.float64),xx[:512].T@yy[:512]);pred=xx[512:]@w
 r2=1-(pred-yy[512:]).square().sum(0)/(yy[512:]-yy[512:].mean(0)).square().sum(0)
 groups={}
 for name in ['projector','actor','dynamics','history']:
  keys=[k for k in cp['state'] if k.startswith(name+'.')]
  if not keys:continue
  a=torch.cat([cp['state'][k].flatten().float() for k in keys]);b=torch.cat([init.state_dict()[k].flatten().float() for k in keys])
  groups[name]={'relative_weight_change':float((a-b).norm()/b.norm()),'original_norm':float(b.norm()),'trained_norm':float(a.norm())}
 rows.append({'family':cp['family'],'seed':cp['seed'],'latent_linear_decode_physics_r2':r2.tolist(),'physics_order':['M00','M01','M10','M11','bx','by'],'weight_changes':groups})
print(json.dumps({'mean_bias_decode_r2':{fam:[sum(x['latent_linear_decode_physics_r2'][i] for x in rows if x['family']==fam)/5 for i in [4,5]] for fam in ['gradient_z','history','hybrid']}}))
(r/'preflight/LATENT_TRAINING_DIAGNOSTIC.json').write_text(json.dumps({'scope':'read-only training-bank diagnostic;512 domains fit ridge,512 diagnose; NO fresh final inputs; hidden M,b used ONLY as diagnostic readout targets, never adaptation or action inputs','ridge_lambda':10,'rows':rows},indent=2)+'\n')
