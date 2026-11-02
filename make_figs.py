import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
R='results/'; O='figures/'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.edgecolor':'#8a8a85','axes.labelcolor':'#3d3d3a',
 'xtick.color':'#5f5e5a','ytick.color':'#3d3d3a','axes.spines.top':False,'axes.spines.right':False})
DS=[('wisdm','WISDM'),('ucihar','UCI-HAR'),('pamap2','PAMAP2')]
ORDER=[('SVM','SVM'),('RF','RF'),('CNN','1D-CNN'),('LSTM','LSTM'),('SAE','SAE (untuned)'),('Random','Random-SAE'),
       ('PSO','PSO-SAE'),('GA','GA-SAE'),('GWO','GWO-SAE'),('SO','SOA-SAE')]
T={d:json.load(open(R+d+'/tables.json')) for d,_ in DS}
# Fig 2
fig,axs=plt.subplots(1,3,figsize=(10,3.6),sharey=True)
for ax,(d,name) in zip(axs,DS):
    t=T[d]['table3']; y=np.arange(len(ORDER))[::-1]
    v=[t[k]['f1_mean'] for k,_ in ORDER]; e=[t[k]['f1_std'] for k,_ in ORDER]
    col=['#eb6834' if k=='SO' else '#2a78d6' for k,_ in ORDER]
    ax.barh(y,v,height=0.62,color=col,xerr=e if max(e)>0 else None,error_kw=dict(ecolor='#3d3d3a',lw=0.8,capsize=2))
    lo=min(v)-6; ax.set_xlim(lo,max(v)+4)
    for yi,vi,ei in zip(y,v,e): ax.text(vi+ei+0.3,yi,f'{vi:.1f}',va='center',fontsize=7.5,color='#3d3d3a')
    ax.set_yticks(y); ax.set_yticklabels([n for _,n in ORDER]); ax.set_title(name,fontsize=10,color='#1f1f1d')
    ax.set_xlabel('Macro-F1 (%)'); ax.grid(axis='x',color='#e6e5e0',lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(O+'fig2_macroF1.png',dpi=300,facecolor='white')
# Fig 3
cm=np.array(T['ucihar']['table5_confusion_pct']); lab=['WALK','UPST','DNST','SIT','STAND','LAY']
cmap=LinearSegmentedColormap.from_list('b',['#ffffff','#cde2fb','#86b6ef','#3987e5','#256abf','#104281'])
fig,ax=plt.subplots(figsize=(4.6,4))
ax.imshow(cm,cmap=cmap,vmin=0,vmax=100)
for i in range(6):
    for j in range(6):
        ax.text(j,i,f'{cm[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if cm[i,j]>55 else '#1f1f1d')
ax.set_xticks(range(6)); ax.set_xticklabels(lab); ax.set_yticks(range(6)); ax.set_yticklabels(lab)
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(O+'fig3_confusion_ucihar.png',dpi=300,facecolor='white')
# Fig 4
OP=[('SO','SOA','#eb6834','-','o'),('PSO','PSO','#2a78d6','--','s'),('GA','GA','#1baf7a','-.','^'),
    ('GWO','GWO','#eda100',':','D'),('Random','Random search','#e87ba4',(0,(5,1,1,1)),'v')]
fig,axs=plt.subplots(1,3,figsize=(10,3.2))
for ax,(d,name) in zip(axs,DS):
    c=T[d]['convergence']
    for k,n,col,ls,mk in OP:
        ax.plot(np.arange(21),100*(1-np.array(c[k])),color=col,ls=ls,lw=1.8,marker=mk,ms=3.5,markevery=4,label=n)
    ax.set_title(name,fontsize=10,color='#1f1f1d'); ax.set_xlabel('Iteration (block of 10 evaluations)')
    ax.set_ylabel('Best inner-validation macro-F1 (%)'); ax.grid(color='#e6e5e0',lw=0.6)
axs[0].legend(frameon=False,fontsize=8,loc='lower right')
fig.tight_layout(); fig.savefig(O+'fig4_convergence.png',dpi=300,facecolor='white')
print('ok')
