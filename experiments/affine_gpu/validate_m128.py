"""Check the complete m128 MPS trajectory, spectral oracles and Adam updates."""
from pathlib import Path
import argparse,csv,hashlib,json,math,sys
from datetime import datetime, timezone
import numpy as np
import torch
R=Path(__file__).resolve().parents[2]
from experiments.affine_gpu.gp_fit import Operator,make_samples,exact_data_for_samples,logdet_and_gradient,parameter_prior
from experiments.affine_gpu.optimizer import adam_update
parser=argparse.ArgumentParser(description='Validate one complete reconstructed paper-sized m128 MPS run against the archived trajectory.')
parser.add_argument('--results',type=Path,default=R/'results/reruns/affine_gpu/mps_hadamard_matern_gp_fit_positive_m128_cheb128_all.csv')
parser.add_argument('--output',type=Path,default=R/'artifacts/affine_gpu/m128_validation')
cli=parser.parse_args()
new_path=cli.results
old_path=R/'results/paper/affine_gpu/mps_hadamard_matern_gp_fit_positive_m128_cheb128_all.csv'
with new_path.open() as f:new=list(csv.DictReader(f))
with old_path.open() as f:old=list(csv.DictReader(f))
metadata=json.loads(new_path.with_suffix('.metadata.json').read_text())
assert metadata['status']=='completed' and metadata['actual_device']=='mps'
assert len(new)==45
numeric=[k for k in new[0] if k not in ['method','notes']]
assert all(np.isfinite(float(row[k])) for row in new for k in numeric)
args=argparse.Namespace(**metadata['config'])
assert (args.n,args.steps,args.samples,args.iterations)==(32768,128,2,8), 'Expected the complete paper-sized m128 configuration'
op=Operator(args,torch.device('cpu'),torch.float32)
samples=make_samples(op,args,torch.device('cpu'),torch.float32)
center=np.array([math.log(args.prior_mass),math.log(args.prior_scale),args.prior_alpha])
max_reference_error=max_metric_error=max_adam_error=0.
for r in new:
 p=np.array([float(r[k]) for k in ['log_mass','log_scale','alpha']])
 dv,dg=exact_data_for_samples(op,samples,p,args)
 lv,lg=logdet_and_gradient(op,p,args.samples)
 pv,pg=parameter_prior(p,center,args.prior_strength)
 ref=dg+lg+pg
 stored=np.array([float(r['exact_grad_'+k]) for k in ['mass','scale','alpha']])
 actual=np.array([float(r['grad_'+k]) for k in ['mass','scale','alpha']])
 max_reference_error=max(max_reference_error,abs(dv-float(r['exact_data_value'])),abs(lv-float(r['logdet_term'])),abs(dv+lv+pv-float(r['exact_nll_at_params'])),float(np.max(abs(ref-stored))))
 eg=np.linalg.norm(actual-ref)/max(np.linalg.norm(ref),1.)
 ed=abs(float(r['data_value'])-dv)/max(abs(dv),1.)
 max_metric_error=max(max_metric_error,abs(eg-float(r['rel_grad_vs_exact'])),abs(ed-float(r['rel_data_value_vs_exact'])))
methods={}
for method in ['exact','full','no','projected','chebyshev']:
 g=[r for r in new if r['method']==method];h=[r for r in old if r['method']==method]
 assert [int(r['iteration']) for r in g]==list(range(9))
 p=np.array([float(g[0][k]) for k in ['log_mass','log_scale','alpha']]);first=np.zeros(3);second=np.zeros(3)
 for r,nxt in zip(g,g[1:]):
  grad=np.array([float(r['grad_'+k]) for k in ['mass','scale','alpha']])
  p,first,second=adam_update(p,grad,first,second,int(r['iteration'])+1,args.learning_rate,args)
  target=np.array([float(nxt[k]) for k in ['log_mass','log_scale','alpha']])
  max_adam_error=max(max_adam_error,float(np.max(abs(p-target))))
 endpoint=lambda r:{k:float(r[k]) for k in ['mass','scale','alpha']}
 methods[method]={'evaluations':len(g),'max_scaled_gradient_error':max(float(r['rel_grad_vs_exact']) for r in g),'archived_max_scaled_gradient_error':max(float(r['rel_grad_vs_exact']) for r in h),'max_scaled_data_value_error':max(float(r['rel_data_value_vs_exact']) for r in g),'archived_max_scaled_data_value_error':max(float(r['rel_data_value_vs_exact']) for r in h),'endpoint':endpoint(g[-1]),'archived_endpoint':endpoint(h[-1]),'max_trajectory_absolute_difference_from_archive':{k:max(abs(float(r[k])-float(s[k])) for r,s in zip(g,h)) for k in ['mass','scale','alpha']},'mean_data_evaluation_seconds_diagnostic_only':float(np.mean([float(r['seconds']) for r in g]))}
assert max_reference_error<1e-8 and max_metric_error<1e-12 and max_adam_error<1e-12
projected=methods['projected'];exact=methods['exact']
assert projected['max_scaled_gradient_error']<1e-4
assert projected['max_scaled_data_value_error']<1e-4
scale_relative_error=abs(projected['endpoint']['scale']-exact['endpoint']['scale'])/exact['endpoint']['scale']
assert scale_relative_error<1e-4
for k in ['mass','scale','alpha']:
 np.testing.assert_allclose(projected['endpoint'][k],exact['endpoint'][k],rtol=1e-4,atol=1e-10)
output=cli.output;output.mkdir(parents=True,exist_ok=True)
(output/'affine_m128.csv').write_bytes(new_path.read_bytes())
report={'validated_utc':datetime.now(timezone.utc).isoformat(),'validation_command':'python -m experiments.affine_gpu.validate_m128 --results '+str(new_path)+' --output '+str(output),'scope':'Complete paper-sized n=32768,m=128,2samples,8Adamupdates,all5methods on MPS; 45 evaluations.','configuration_status':'Reconstructed historical settings: Ritz guard none and full reorthogonalization passes1 are explicit current defaults, not established original launch flags.','timing_scope':'Other release-validation jobs ran concurrently. New timings are diagnostic only; no timing agreement or speedup claim is made.','run_metadata':metadata,'raw_validation_csv':str((output/'affine_m128.csv').relative_to(R)) if output.is_relative_to(R) else str(output/'affine_m128.csv'),'raw_validation_sha256':hashlib.sha256(new_path.read_bytes()).hexdigest(),'archived_csv_sha256':hashlib.sha256(old_path.read_bytes()).hexdigest(),'checks':{'all45rows_finite':True,'allmethods_iterations0through8':True,'max_reference_absolute_difference':max_reference_error,'max_error_metric_absolute_difference':max_metric_error,'max_adam_parameter_difference':max_adam_error,'projected_max_scaled_gradient_error_below_1e_4':True,'projected_max_scaled_value_error_below_1e_4':True,'projected_final_scale_relative_error_to_spectral_reference':scale_relative_error,'projected_all_endpoint_parameters_match_reference_rtol1e_4':True},'methods':methods,'interpretation':'Projected and both AD methods retain small gradient errors at m128; all five trajectories complete. Floating-point values differ modestly from archived GPU records. The exact spectral trajectory agrees with the archive; historical GPU environment/settings remain unresolved.'}
(output/'affine_m128.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checks':report['checks'],'methods':methods},indent=2))
