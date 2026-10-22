"""Audit saved scalar records and current source; no performance benchmark reruns."""
from pathlib import Path
import sys, csv, json, argparse, math, hashlib
import numpy as np
import torch
from .gp_fit import Operator,make_samples,exact_data_for_samples,logdet_and_gradient,parameter_prior
from .optimizer import adam_update
from .kernels import lanczos
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',type=Path,default=ROOT/'results/paper/affine_gpu')
parser.add_argument('--output',type=Path,default=ROOT/'artifacts/affine_gpu/verification')
args=parser.parse_args()
OUT=args.output
OUT.mkdir(parents=True,exist_ok=True)
DATA=args.results
FILES={128:'mps_hadamard_matern_gp_fit_positive_m128_cheb128_all.csv',384:'mps_hadamard_matern_gp_fit_positive_m384_cheb384_initial_all.csv',768:'mps_hadamard_matern_gp_fit_positive_m768_cheb768_initial_nofull.csv',1536:'mps_hadamard_matern_gp_fit_positive_m1536_far_highscale_cheb.csv'}
a=argparse.Namespace(operator='hadamard-clusters',n=32768,samples=2,clusters=128,condition=1e10,cluster_width=1e-8,seed=0,sample_seed=123,true_mass=1e-6,true_scale=1.,true_alpha=.5,logdet_scale=1.,mass_min=1e-12,mass_max=10.,scale_min=1e-6,scale_max=10.,alpha_min=.05,alpha_max=2.)
op=Operator(a,torch.device('cpu'),torch.float32)
samples=make_samples(op,a,torch.device('cpu'),torch.float32)
center=np.asarray([math.log(1e-6),0.,.5])
fields=['mass','scale','alpha']
metrics=[];summaries=[];finals=[];checks={};datasets={}
for m,name in FILES.items():
 source=DATA/name
 rows=list(csv.DictReader(source.open()));datasets[m]=rows
 max_ref_diff=0.;max_metric_diff=0.;max_adam_diff=0.
 for r in rows:
  p=np.array([float(r['log_mass']),float(r['log_scale']),float(r['alpha'])])
  dv,dg=exact_data_for_samples(op,samples,p,a)
  lv,lg=logdet_and_gradient(op,p,2)
  pv,pg=parameter_prior(p,center,1e-4)
  eg=dg+lg+pg
  stored_eg=np.array([float(r['exact_grad_'+k]) for k in fields])
  ag=np.array([float(r['grad_'+k]) for k in fields])
  delta=ag-stored_eg
  max_ref_diff=max(max_ref_diff,abs(dv-float(r['exact_data_value'])),abs(lv-float(r['logdet_term'])),abs(dv+lv+pv-float(r['exact_nll_at_params'])),float(np.max(abs(eg-stored_eg))))
  ge=np.linalg.norm(delta)/max(np.linalg.norm(stored_eg),1.)
  ve=abs(float(r['data_value'])-dv)/max(abs(dv),1.)
  max_metric_diff=max(max_metric_diff,abs(ge-float(r['rel_grad_vs_exact'])),abs(ve-float(r['rel_data_value_vs_exact'])))
  metrics.append({'m':m,'method':r['method'],'iteration':int(r['iteration']),'abs_error_log_mu':abs(delta[0]),'abs_error_log_s':abs(delta[1]),'abs_error_alpha':abs(delta[2]),'data_gradient_norm':float(np.linalg.norm(dg)),'relative_data_gradient_error':float(np.linalg.norm(delta)/np.linalg.norm(dg)),'scaled_objective_gradient_error':ge})
 for method in dict.fromkeys(r['method'] for r in rows):
  group=[r for r in rows if r['method']==method]
  assert [int(r['iteration']) for r in group]==list(range(9 if m in (128,1536) else 1))
  p=np.array([float(group[0]['log_mass']),float(group[0]['log_scale']),float(group[0]['alpha'])]);first=np.zeros(3);second=np.zeros(3)
  assert np.array_equal(p,np.array([math.log(1e-5),math.log(5),1.5]))
  for r,nxt in zip(group,group[1:]):
   g=np.array([float(r['grad_'+k]) for k in fields])
   p,first,second=adam_update(p,g,first,second,int(r['iteration'])+1,.02,a)
   stored=np.array([float(nxt['log_mass']),float(nxt['log_scale']),float(nxt['alpha'])])
   max_adam_diff=max(max_adam_diff,float(np.max(abs(p-stored))))
  last=group[-1]
  finals.append({'m':m,'method':method,'iteration':int(last['iteration']),'mu':float(last['mass']),'s':float(last['scale']),'alpha':float(last['alpha'])})
  if m in (128,1536):
   sub=[r for r in metrics if r['m']==m and r['method']==method]
   summaries.append({'m':m,'method':method,'evaluations':len(group),'mean_seconds':np.mean([float(r['seconds']) for r in group]),'max_epsilon_g':max(float(r['rel_grad_vs_exact']) for r in group),'max_epsilon_D':max(float(r['rel_data_value_vs_exact']) for r in group),**{'max_'+key:max(r[key] for r in sub) for key in ['abs_error_log_mu','abs_error_log_s','abs_error_alpha','relative_data_gradient_error']}})
 checks[name]={'rows':len(rows),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'max_reference_absolute_difference':max_ref_diff,'max_recomputed_metric_absolute_difference':max_metric_diff,'max_adam_parameter_difference':max_adam_diff}
 assert max_ref_diff<1e-8 and max_metric_diff<1e-12 and max_adam_diff<1e-12,checks[name]
for name,rows in [('gradient_diagnostics',metrics),('trajectory_summary',summaries),('final_parameters',finals)]:
 with (OUT/(name+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Inspect a tiny reachable AD graph, rather than rerun an archived benchmark.
n,m=64,6
small_args=argparse.Namespace(operator='hadamard-clusters',n=n,clusters=16,condition=1e3,cluster_width=1e-8,seed=0)
small_op=Operator(small_args,torch.device('cpu'),torch.float32)
v=torch.randn(n,generator=torch.Generator().manual_seed(123))
lm=torch.tensor(-3.,requires_grad=True);ls=torch.tensor(.2,requires_grad=True)
tri,*_=lanczos(small_op,v,lm,ls,m,full_reorth=True,full_reorth_passes=1,beta_breakdown_tolerance=1e-30,store_basis=False,check_orthogonality=False)
queue=[tri.grad_fn];seen=set();blocks={}
while queue:
 fn=queue.pop()
 if fn is None or fn in seen:continue
 seen.add(fn)
 for attr in dir(fn):
  if not attr.startswith('_saved_'):continue
  try:t=getattr(fn,attr)
  except Exception:continue
  if isinstance(t,torch.Tensor) and t.ndim==2 and n in t.shape and min(t.shape)<=m:
   key=t.untyped_storage().data_ptr()
   block=blocks.setdefault(key,{'bytes':t.untyped_storage().nbytes(),'j':min(t.shape),'views':[]})
   block['views'].append({'node':type(fn).__name__,'field':attr,'shape':list(t.shape)})
 queue.extend(a for a,_ in fn.next_functions)
block_list=sorted(blocks.values(),key=lambda r:r['j'])
expected_saved_stack_pattern = ([r['j'] for r in block_list]==list(range(1,m)) and sum(r['bytes'] for r in block_list)==2*n*m*(m-1))
source_files=['experiments/affine_gpu/gp_fit.py','experiments/affine_gpu/kernels.py','experiments/affine_gpu/optimizer.py']
report={'audit_torch_version':torch.__version__,'configuration':vars(a),'prior_center_coordinates':center.tolist(),'prior_strength':1e-4,'saved_csv_checks':checks,'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in source_files},'tiny_cpu_saved_tensor_check':{'n':n,'m':m,'matches_audited_saved_stack_pattern':expected_saved_stack_pattern,'distinct_basis_blocks':block_list,'total_distinct_bytes':sum(r['bytes'] for r in block_list),'scope':'Reachable graph after forward completion in current CPU implementation, not historical MPS peak memory. The final m-column stack is transient and is not reachable from T.'},'no_benchmark_reruns':True}
(OUT/'artifact_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(checks,indent=2))
print('m=1536 supplemental diagnostics:')
for r in summaries:
 if r['m']==1536:print(r)
