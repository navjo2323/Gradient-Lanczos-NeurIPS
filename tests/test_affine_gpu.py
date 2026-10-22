"""CPU spectral oracles for the affine specialization and saved paper tables."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from experiments.affine_gpu.gp_fit import (Operator, fwht, make_samples, exact_positive_reference,
    evaluate_projected_positive, evaluate_lanczos_positive, evaluate_chebyshev_positive)
from experiments.affine_gpu.tables import FILES, read_csv

ROOT = Path(__file__).resolve().parents[1]


def test_fwht_involution_and_affine_gradients_against_spectral_reference():
    cfg = json.loads((ROOT/'configs/affine_gpu_quick.json').read_text())
    args = argparse.Namespace(**cfg)
    args.mass=args.init_mass;args.scale=args.init_scale
    device=torch.device('cpu')
    op=Operator(args,device,torch.float64)
    initial=make_samples(op,args,device,torch.float64)[0]
    np.testing.assert_allclose((fwht(fwht(initial))/args.n).numpy(),initial.numpy(),atol=2e-14)
    params=np.array([np.log(args.init_mass),np.log(args.init_scale),args.init_alpha])
    exact_value,exact_grad=exact_positive_reference(op,initial,params,args)
    for index in range(3):
        direction=np.eye(3)[index]*1e-6
        plus=exact_positive_reference(op,initial,params+direction,args)[0]
        minus=exact_positive_reference(op,initial,params-direction,args)[0]
        np.testing.assert_allclose(exact_grad[index],(plus-minus)/2e-6,rtol=2e-7,atol=1e-7)
    for method in [evaluate_projected_positive,lambda *x:evaluate_lanczos_positive(*x,full_reorth=True),evaluate_chebyshev_positive]:
        value,grad=method(op,initial,params,args)
        np.testing.assert_allclose(value,exact_value,rtol=3e-6,atol=2e-6)
        np.testing.assert_allclose(grad,exact_grad,rtol=3e-6,atol=2e-6)


def test_saved_affine_csv_metrics_and_table_5_rounding():
    rows={m:read_csv(ROOT/'results/paper/affine_gpu'/f) for m,f in FILES.items()}
    assert [len(rows[m]) for m in FILES] == [45,5,4,36]
    times={384:['9.065','2.124','1.146','0.899'],768:['4.627','2.422','1.793']}
    for m in [384,768]:
        assert [f"{float(r['seconds']):.3f}" for r in rows[m] if r['method']!='exact']==times[m]
    for group in rows.values():
        for r in group:
            got=np.array([float(r['grad_'+k]) for k in ['mass','scale','alpha']])
            ref=np.array([float(r['exact_grad_'+k]) for k in ['mass','scale','alpha']])
            np.testing.assert_allclose(np.linalg.norm(got-ref)/max(np.linalg.norm(ref),1),float(r['rel_grad_vs_exact']),rtol=1e-13,atol=1e-15)
            np.testing.assert_allclose(abs(float(r['data_value'])-float(r['exact_data_value']))/max(abs(float(r['exact_data_value'])),1),float(r['rel_data_value_vs_exact']),rtol=1e-13,atol=1e-15)
