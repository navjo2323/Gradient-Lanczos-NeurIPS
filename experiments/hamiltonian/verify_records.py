"""Recompute the paper initialization and gradient diagnostics (no training)."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import scipy
from .train import make_dataset, dense_loss_and_grad, gm_loss_and_grad
ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results/paper/hamiltonian/results.json')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/hamiltonian/verification.json')
    args=parser.parse_args()
    saved=json.loads(args.results.read_text());c=saved['config'];start=time.perf_counter()
    basis,_,data=make_dataset(c['sites'],c['samples'],c['t_min'],c['t_max'],c['seed'],c['disorder'],c['regime'])
    theta=data.theta_true+c['init_scale']*np.random.default_rng(c['seed']+999).normal(size=len(data.theta_true))
    np.testing.assert_allclose(data.theta_true,saved['theta_true'],rtol=0,atol=0)
    np.testing.assert_allclose(theta,saved['theta0'],rtol=0,atol=0)
    loss,dense=dense_loss_and_grad(theta,basis,data)
    checks=[]
    for depth,run in saved['gm_runs'].items():
        projected_loss,gradient=gm_loss_and_grad(theta,basis,data,int(depth))
        relative_error=np.linalg.norm(gradient-dense)/max(np.linalg.norm(dense),1e-14)
        original=run['initial_gradient_diagnostic']
        np.testing.assert_allclose(loss,original['loss_dense'],rtol=2e-12,atol=1e-12)
        np.testing.assert_allclose(projected_loss,original['loss_gm'],rtol=2e-12,atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(dense),original['grad_norm_dense'],rtol=2e-11,atol=1e-11)
        checks.append({'depth':int(depth),'relative_gradient_error':float(relative_error),
                       'recorded_relative_gradient_error':original['grad_rel_error'],
                       'absolute_difference':float(abs(relative_error-original['grad_rel_error'])),
                       'matches_table_2_three_significant_digits':f'{relative_error:.2e}' == f"{original['grad_rel_error']:.2e}"})
        print(checks[-1],flush=True)
    report={'checked_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'numpy':np.__version__,
            'scipy':scipy.__version__,'platform':platform.platform(),'seconds':time.perf_counter()-start,
            'checks':checks,'scope':'Full L=8, 30-sample initialization at m=30,40,50; no 400-step retraining or timing replication.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if not all(c['matches_table_2_three_significant_digits'] for c in checks):
        raise RuntimeError('A recomputed gradient error differs at published precision; inspect the saved verification report and environment.')


if __name__=='__main__':main()
