"""Generate the affine appendix tables from the archived CSV measurements."""
from __future__ import annotations
import csv
import math
from pathlib import Path
import numpy as np

FILES = {128:'mps_hadamard_matern_gp_fit_positive_m128_cheb128_all.csv',
         384:'mps_hadamard_matern_gp_fit_positive_m384_cheb384_initial_all.csv',
         768:'mps_hadamard_matern_gp_fit_positive_m768_cheb768_initial_nofull.csv',
         1536:'mps_hadamard_matern_gp_fit_positive_m1536_far_highscale_cheb.csv'}
LABELS = {'full':'Full-reorth. AD','no':'No-reorth. AD','projected':'Projected (ours)',
          'chebyshev':'Chebyshev','exact':'Spectral reference'}


def read_csv(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def emit(output, name, rows, headers, formatters):
    with (output / f'{name}.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    keys = list(rows[0])
    lines = [r'\begin{tabular}{' + 'l' * len(keys) + '}', r'\toprule',
             ' & '.join(headers) + r' \\', r'\midrule']
    for row in rows:
        lines.append(' & '.join(formatters[k](row[k]) for k in keys) + r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (output / f'{name}.tex').write_text('\n'.join(lines)+'\n')


def sci(x):
    mantissa, exponent = f'{float(x):.2e}'.split('e')
    return f'${mantissa}\\times10^{{{int(exponent)}}}$'


def write_tables(data: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    datasets = {m:read_csv(data / f) for m,f in FILES.items()}
    rows=[]
    for m in [384,768]:
        for r in datasets[m]:
            if r['method']=='exact':continue
            rows.append({'depth':m,'method':LABELS[r['method']], 'seconds':float(r['seconds']),
                         'epsilon_g':float(r['rel_grad_vs_exact']),'epsilon_D':float(r['rel_data_value_vs_exact'])})
    emit(output,'table_5',rows,['$m$','Method','Time (s)',r'$\varepsilon_g$',r'$\varepsilon_D$'],
         {'depth':str,'method':str,'seconds':lambda x:f'{x:.3f}','epsilon_g':sci,'epsilon_D':sci})
    table_path = output / 'table_5.tex'
    table_text = table_path.read_text()
    annotation = r'768 & Full-reorth. AD & \multicolumn{3}{l}{OOM reported; no completed measurement} \\' + '\n'
    table_path.write_text(table_text.replace('768 & ', annotation + '768 & ', 1))
    (output / 'table_5_notes.txt').write_text('Full-reorthogonalized AD at m=768: OOM reported; no completed measurement. No failure log was recovered.\n')
    summary=[];final=[]
    for m in [128,1536]:
        for method in ['full','no','projected','chebyshev']:
            group=[r for r in datasets[m] if r['method']==method]
            if not group:continue
            summary.append({'depth':m,'method':LABELS[method],'mean_seconds':float(np.mean([float(r['seconds']) for r in group])),
                            'max_epsilon_g':max(float(r['rel_grad_vs_exact']) for r in group),
                            'max_epsilon_D':max(float(r['rel_data_value_vs_exact']) for r in group)})
            last=group[-1]
            final.append({'depth':m,'method':LABELS[method],'mass_times_1e5':1e5*float(last['mass']),
                          'scale':float(last['scale']),'alpha':float(last['alpha'])})
    last=[r for r in datasets[1536] if r['method']=='exact'][-1]
    final.append({'depth':'--','method':LABELS['exact'],'mass_times_1e5':1e5*float(last['mass']),
                  'scale':float(last['scale']),'alpha':float(last['alpha'])})
    emit(output,'table_6a',summary,['$m$','Method','Mean (s)',r'Max $\varepsilon_g$',r'Max $\varepsilon_D$'],
         {'depth':str,'method':str,'mean_seconds':lambda x:f'{x:.3f}','max_epsilon_g':sci,'max_epsilon_D':sci})
    emit(output,'table_6b_rounded',final,['$m$','Method',r'$10^5\mu$','$s$',r'$\alpha$'],
         {'depth':str,'method':str,**{k:lambda x:f'{x:.3f}' for k in ['mass_times_1e5','scale','alpha']}})
    emit(output,'table_6b',final,['$m$','Method',r'$10^5\mu$','$s$',r'$\alpha$'],
         {'depth':str,'method':str,**{k:lambda x:f'{math.trunc(x*1000)/1000:.3f}' for k in ['mass_times_1e5','scale','alpha']}})
    n=32768
    memory=[{'depth':m,'basis_MiB':4*n*m/2**20,'growing_stacks_GiB':2*n*m*(m+1)/2**30,'small_tensor_MiB':8*m*m/2**20} for m in FILES]
    emit(output,'table_7',memory,['$m$','One basis (MiB)','Growing stacks (GiB)','One small tensor (MiB)'],
         {'depth':str,'basis_MiB':lambda x:f'{x:.0f}','growing_stacks_GiB':lambda x:f'{x:.3f}','small_tensor_MiB':lambda x:f'{x:.3f}'})
    diagnostics=read_csv(data/'gradient_diagnostics.csv')
    extra=[]
    for method in ['no','projected','chebyshev']:
        group=[r for r in diagnostics if int(r['m'])==1536 and r['method']==method]
        extra.append({'method':LABELS[method],**{k:max(float(r[k]) for r in group) for k in
                     ['abs_error_log_mu','abs_error_log_s','abs_error_alpha','relative_data_gradient_error']}})
    emit(output,'table_8',extra,['Method',r'$|\Delta g_{\log\mu}|$',r'$|\Delta g_{\log s}|$',r'$|\Delta g_\alpha|$',r'$\eta_D$'],
         {'method':str,**{k:sci for k in ['abs_error_log_mu','abs_error_log_s','abs_error_alpha','relative_data_gradient_error']}})
    print(f'Wrote Tables 5–8 to {output}')
