"""Render recorded Stage II metrics; no checkpoint loading or rescoring."""
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/topo_stage2_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent
ARMS = ('coord_gen_full', 'coord_gen_frozen')
COLORS = ('#0072B2', '#D55E00')
ROWS = {a: [json.loads(s) for s in (ROOT/a/'metrics.jsonl').read_text().splitlines()] for a in ARMS}
SELECTED = {a: json.loads((ROOT/a/'test_report.json').read_text())['arm']['selected_epoch'] for a in ARMS}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'axes.titleweight':'bold','svg.fonttype':'none','pdf.fonttype':42})

def frame(ax, arm):
    ax.axvline(SELECTED[arm], color='#555555', ls='--', lw=1.2, zorder=0)
    ax.grid(axis='y', alpha=.18)
    ax.set_xlabel('Epoch')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.set_xlim(.6, len(ROWS[arm])+.4)

def save(fig, name):
    for ext in ('png','svg','pdf'):
        fig.savefig(ROOT/f'{name}.{ext}', dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)

fig, axes = plt.subplots(2,4,figsize=(15,6.5),layout='constrained')
for r,arm in enumerate(ARMS):
    rows=ROWS[arm]; x=[d['epoch'] for d in rows]
    for c,(key,title) in enumerate([('train_loss','Training composite'),('val_coord_loss',r'Validation $L_{coord}$'),('val_task_loss',r'Validation $L_{BCE}$'),('val_kd_loss',r'Validation $L_{KD}$ (unscaled)')]):
        ax=axes[r,c]; y=[d[key] for d in rows]; ax.plot(x,y,'o-',ms=3,lw=1.8,color=COLORS[r]); frame(ax,arm)
        k=min(range(len(y)),key=y.__getitem__)
        ax.scatter(x[k],y[k],marker='*',s=120,color='black',zorder=3)
        ax.set_title(title); ax.margins(y=.22)
        ax.set_ylabel(('FULL' if r==0 else 'FROZEN')+' reader lane\nLoss')
        ax.text(.98,.97,f'min: {y[k]:.3f} (ep {x[k]})',ha='right',va='top',transform=ax.transAxes,fontsize=9)
fig.suptitle('Stage II loss trajectories — validation components, training total only',fontsize=15)
fig.supxlabel('Dashed vertical: selected checkpoint (full 6; frozen 4). Star: minimum logged loss. No smoothing.\nValidation KD is plotted before its 0.1 weight; validation components are not a reconstructed training loss.',fontsize=10)
save(fig,'loss_curves')

fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
fields=[('endpoint','#0072B2','o'),('relation','#D55E00','s'),('context','#009E73','^')]
for r,arm in enumerate(ARMS):
    rows=ROWS[arm];x=[d['epoch'] for d in rows]
    ax=axes[r,0]
    for field,color,marker in fields:ax.plot(x,[d['val_coord_r2_'+field] for d in rows],label=field,color=color,marker=marker,ms=3,lw=1.8)
    ax.axhline(0,color='black',lw=1);frame(ax,arm);ax.set_ylim(-1.05,.35);ax.set_ylabel(('FULL' if r==0 else 'FROZEN')+' reader lane\nField-level validation R²');ax.legend(ncol=3,fontsize=9,loc='lower right')
    ax=axes[r,1];ax.plot(x,[d['val_coord_dist_acc'] for d in rows],color=COLORS[r],marker='o',ms=3,lw=1.8);frame(ax,arm);ax.set_ylim(0,1);ax.set_ylabel('5-class distance accuracy')
axes[0,0].set_title('Continuous-coordinate prediction');axes[0,1].set_title('Distance-class prediction')
fig.suptitle('Coordinate generator — generalization to node-held-out V_val',fontsize=15)
fig.supxlabel('R² = 0: per-coordinate validation-mean reference; negative means larger squared error.\nDashed vertical: selected checkpoint. No distance majority-class baseline was logged.',fontsize=10)
save(fig,'generator_fit')

fig,axes=plt.subplots(2,3,figsize=(12,6.5),layout='constrained')
metrics=[('val_auprc','Validation AUPRC ↑'),('val_gs_bfs','BFS-macro GS ↑'),('val_rd_bfs','Arithmetic RD → 1'),('val_degree_mmd_ratio','Degree MMD ratio ↓'),('val_clustering_mmd_ratio','Clustering MMD ratio ↓'),('val_spectral_mmd_ratio','Spectral MMD ratio ↓')]
for ax,(key,title) in zip(axes.flat,metrics):
    for i,arm in enumerate(ARMS):
        rows=ROWS[arm];x=[d['epoch'] for d in rows];y=[d[key] for d in rows]
        ax.plot(x,y,color=COLORS[i],marker='o' if i==0 else 's',ms=3,lw=1.6,label=arm)
        ep=SELECTED[arm];ax.scatter(ep,y[ep-1],color=COLORS[i],edgecolor='black',marker='*',s=140,zorder=5)
    if key=='val_rd_bfs':ax.axhline(1,color='black',ls=':',lw=1)
    ax.set_title(title);ax.grid(axis='y',alpha=.18);ax.set_xlabel('Epoch');ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=6))
axes[0,0].legend(fontsize=9)
fig.suptitle('Validation edge and topology metrics beside the losses',fontsize=15)
fig.supxlabel('Stars: selected checkpoints. Every epoch uses its own validation-selected topology threshold; no test curves.',fontsize=10)
save(fig,'validation_topology')

keys=list(ROWS[ARMS[0]][0])
with (ROOT/'learning_curves.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['arm',*keys]);w.writeheader()
    for arm in ARMS:
        for row in ROWS[arm]:w.writerow({'arm':arm,**row})
lines=['| Lane | Logged metric | Epoch 1 | Minimum (epoch) | Selected | Last |','|---|---|---:|---:|---:|---:|']
for arm in ARMS:
    rows=ROWS[arm];ep=SELECTED[arm]
    for key in ['train_loss','val_coord_loss','val_task_loss','val_kd_loss']:
        best=min(rows,key=lambda d:d[key]);lines.append(f'| {arm} | `{key}` | {rows[0][key]:.4f} | {best[key]:.4f} ({best["epoch"]}) | {rows[ep-1][key]:.4f} | {rows[-1][key]:.4f} |')
(ROOT/'summary_table.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))

# Derived unweighted validation composite, using logged component diagnostics.
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
composite_rows=[]
for i,arm in enumerate(ARMS):
    rows=ROWS[arm];x=[d['epoch'] for d in rows]
    y=[d['val_task_loss']+d['val_coord_loss']+.1*d['val_kd_loss'] for d in rows]
    ax=axes[i];ax.plot(x,y,'o-',color=COLORS[i],ms=4,lw=2);frame(ax,arm)
    k=min(range(len(y)),key=y.__getitem__);ax.scatter(x[k],y[k],marker='*',color='black',s=130,zorder=5)
    ax.set_title(arm);ax.set_ylabel('Unweighted validation composite');ax.margins(y=.2)
    ax.text(.98,.97,f'min {y[k]:.3f} at epoch {x[k]}',ha='right',va='top',transform=ax.transAxes)
    for d,value in zip(rows,y):composite_rows.append({'arm':arm,'epoch':d['epoch'],'val_composite_unweighted':value})
    print(arm,'composite: first',round(y[0],4),'minimum',round(y[k],4),'epoch',x[k],'selected',round(y[SELECTED[arm]-1],4),'last',round(y[-1],4))
fig.suptitle(r'Validation composite: $L_{BCE}+L_{coord}+0.1L_{KD}$',fontsize=15)
fig.supxlabel('Derived from logged validation means; not the 5:1 row-weighted training objective.\nDashed: selected checkpoint. Star: minimum composite. No smoothing.',fontsize=10)
save(fig,'validation_composite')
with (ROOT/'validation_composite.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(composite_rows[0]));w.writeheader();w.writerows(composite_rows)
