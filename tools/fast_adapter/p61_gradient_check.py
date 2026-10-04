import json
import numpy as np
import torch
from prove.fast_adapter.p61_adaptation import adapt_z
from prove.fast_adapter.p60_contour import ContourMeasurement, _uniform_closed_polyline

torch.set_num_threads(1)
x=torch.tensor([.011,.013,-1.01,.041,.053,.027,np.log(.081),np.log(.043)],dtype=torch.double,requires_grad=True)
T=torch.eye(4,dtype=torch.double); K=np.array([[200.,0,224],[0,200,224],[0,0,1.]])
hull=np.array([[206.3,214.7],[241.5,215.1],[241.1,233.9],[206.1,233.5]],dtype=np.float32)
m=ContourMeasurement(_uniform_closed_polyline(hull,32),hull,K,np.zeros(3),np.eye(3),(448,448),0)
def objective(x):
    z,_=adapt_z(x,torch.zeros(8,dtype=x.dtype),T,[m],create_graph=True)
    return (x+z*.07).square().sum()
y=objective(x);g,=torch.autograd.grad(y,x,create_graph=True);h,=torch.autograd.grad(g.square().sum(),x)
checks=[]
for index in (0,4,6):
    epsilon=1e-6;d=torch.zeros_like(x);d[index]=epsilon
    finite=float((objective(x+d)-objective(x-d))/(2*epsilon));auto=float(g[index])
    checks.append(dict(coordinate=index,autograd=auto,finite_difference=finite,
                       absolute_error=abs(finite-auto),relative_error=abs(finite-auto)/max(1.,abs(finite),abs(auto))))
result=dict(fixture='synthetic fixed public contour; no training/test data',steps=3,
            first_derivative_finite=bool(g.isfinite().all()),second_derivative_finite=bool(h.isfinite().all()),
            finite_difference=checks,passed=all(row['relative_error']<1e-4 for row in checks))
print(json.dumps(result,indent=2))
open('evaluations/fast_adapter_p61_task_visual_v01/engineering/GN_GRADIENT_CHECK.json','w').write(json.dumps(result,indent=2)+'\n')
assert result['passed']
