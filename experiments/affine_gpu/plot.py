"""Replot the original saved rows, changing labels rather than measurements."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator,NullFormatter
import argparse
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, default=ROOT/'results/paper/affine_gpu')
parser.add_argument('--output', type=Path, default=ROOT/'artifacts/affine_gpu')
args=parser.parse_args()
OUT=args.output
DATA=args.results
(OUT/'figures').mkdir(parents=True,exist_ok=True)
FILES={128:'mps_hadamard_matern_gp_fit_positive_m128_cheb128_all.csv',384:'mps_hadamard_matern_gp_fit_positive_m384_cheb384_initial_all.csv',768:'mps_hadamard_matern_gp_fit_positive_m768_cheb768_initial_nofull.csv',1536:'mps_hadamard_matern_gp_fit_positive_m1536_far_highscale_cheb.csv'}
D={m:list(csv.DictReader((DATA/f).open())) for m,f in FILES.items()}
LABELS={'full':'Full-reorth. AD','no':'No-reorth. AD','projected':'Projected (ours)','chebyshev':'Chebyshev','exact':'Spectral reference'}
COLORS={'full':'#7B4AB5','no':'#C63D32','projected':'#007A87','chebyshev':'#CB8A00','exact':'#333333'}
MARKERS={'full':'^','no':'s','projected':'o','chebyshev':'D','exact':None}
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(OUT/'figures'/(name+'.'+ext),bbox_inches='tight',dpi=220)
 plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(9.3,3.05))
for method in ['full','no','projected','chebyshev']:
 pairs=[(m,r) for m in FILES for r in D[m] if r['method']==method and int(r['iteration'])==0]
 for ax,key in zip(axs,['rel_data_value_vs_exact','rel_grad_vs_exact','seconds']):
  ax.plot([m for m,r in pairs],[float(r[key]) for m,r in pairs],label=LABELS[method],color=COLORS[method],marker=MARKERS[method],markersize=4,linewidth=1.4)
for ax,title,ylabel in zip(axs,['(a) Quadratic value','(b) Objective gradient','(c) Evaluation time'],[r'$\varepsilon_D$ (scaled error)',r'$\varepsilon_g$ (scaled error)','Data-term evaluation time (s)']):
 ax.set(title=title,xlabel='Lanczos depth $m$',ylabel=ylabel,xscale='log',yscale='log')
 ax.set_xticks(list(FILES),[str(m) for m in FILES]);ax.xaxis.set_minor_locator(NullLocator());ax.xaxis.set_minor_formatter(NullFormatter());ax.tick_params(axis='x',labelsize=8);ax.grid(True,alpha=.2)
axs[1].text(.035,.96,'Full AD: OOM reported at $m=768$',transform=axs[1].transAxes,va='top',fontsize=7)
h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,ncol=4,loc='lower center',frameon=False,bbox_to_anchor=(.5,-.015))
fig.tight_layout(rect=(0,.11,1,1));save(fig,'lanczos_regimes_depth')
fig,axs=plt.subplots(1,2,figsize=(8.2,3.0))
for method in ['exact','no','chebyshev','projected']:
 rows=[r for r in D[1536] if r['method']==method];assert [int(r['iteration']) for r in rows]==list(range(9))
 x=[int(r['iteration']) for r in rows]
 axs[0].plot(x,[float(r['scale']) for r in rows],label=LABELS[method],color=COLORS[method],marker=MARKERS[method],markersize=4,linewidth=1.6 if method=='exact' else 1.2,linestyle='--' if method=='exact' else '-',markerfacecolor='none')
 if method!='exact':axs[1].semilogy(x,[float(r['rel_grad_vs_exact']) for r in rows],color=COLORS[method],marker=MARKERS[method],markersize=4,linewidth=1.4)
for ax,title,yl in zip(axs,['(a) Scale trajectory','(b) Gradient fidelity'],['Scale $s$',r'$\varepsilon_g$ (scaled error)']):
 ax.set(title=title,xlabel='Adam updates',ylabel=yl);ax.set_xticks(range(0,9,2));ax.grid(True,alpha=.2)
axs[0].annotate('Reference, projected and Chebyshev overlap',xy=(6,5.474),xytext=(.2,5.63),fontsize=7,arrowprops={'arrowstyle':'-','lw':.7});axs[0].set_ylim(4.54,5.72)
h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,ncol=4,loc='lower center',frameon=False,bbox_to_anchor=(.5,-.015));fig.tight_layout(rect=(0,.1,1,1));save(fig,'lanczos_regimes_fit')
print('Rebuilt two vector figures from original rows only.')

from .tables import write_tables
write_tables(DATA, OUT)
