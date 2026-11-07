from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
OUT=ROOT/'11_Manuscript_Consolidation'/'Phase20_Publication_Outputs'/'Figures_Corrected'
OUT.mkdir(parents=True,exist_ok=True)
source=ROOT/'11_Manuscript_Consolidation'/'Phase20_Publication_Outputs'/'Tables'/'Table_S3_per_OPO_performance.csv'
if not source.exists():
    source=ROOT/'11_Manuscript_Consolidation'/'Source_Copies'/'07_04_per_OPO_model_C.csv'
if not source.exists():
    matches=list(ROOT.rglob('Table_S3_per_OPO_performance.csv'))
    if not matches: raise FileNotFoundError('Cannot find Table_S3_per_OPO_performance.csv; provide its location.')
    source=matches[0]
df=pd.read_csv(source)
g=df[df['Cohort'].eq('Authorized_OPO_Grouped')].copy()
if len(g)!=6: raise ValueError(f'Expected six OPO-grouped rows, found {len(g)}')
g=g.sort_values('OPO')
fig,ax=plt.subplots(figsize=(9,5.2))
ax.bar(g['OPO'],g['ROC_AUC'])
ax.set(ylim=(0,1),ylabel='ROC–AUC',xlabel='Held-out organ procurement organization',title='Model C discrimination in OPO-grouped validation')
for i,(_,r) in enumerate(g.iterrows()):ax.text(i,r.ROC_AUC+.018,f'{r.ROC_AUC:.3f}',ha='center',fontsize=10)
fig.tight_layout();fig.savefig(OUT/'Figure_3_OPO_grouped_corrected.png',dpi=300);plt.close(fig)
# Study-specific flow: nested authorized -> outcomes, separate subset annotation.
fig,ax=plt.subplots(figsize=(11,7));ax.set_xlim(0,12);ax.set_ylim(0,9);ax.axis('off')
def node(x,y,w,h,text):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.12',linewidth=1.2,edgecolor='#64748b',facecolor='#f1f5f9'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11)
def arrow(x1,y1,x2,y2):ax.annotate('',(x2,y2),(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.4,color='#475569'))
node(3.5,7.4,5,1,'Pediatric donor records\nN = 4,537')
arrow(6,7.4,6,6.4)
node(3.5,5.4,5,1,'Authorized donors (primary cohort)\nN = 819')
arrow(5,5.4,3,4.3);arrow(7,5.4,9,4.3)
node(.5,3.25,5,1,'Heart transplanted\nn = 441')
node(6.5,3.25,5,1,'Heart not transplanted\nn = 378')
ax.text(6,2.5,'Nested secondary analysis subsets (not additional exclusions from the primary cohort)',ha='center',fontsize=10,color='#475569')
node(.7,.8,5,1,'Any-organ procurement recorded\nN = 716 (441 transplanted; 275 not)')
node(6.3,.8,5,1,'Procured + documented brain death\nN = 594 (438 transplanted; 156 not)')
fig.tight_layout();fig.savefig(OUT/'Figure_1_cohort_flow_corrected.png',dpi=300,bbox_inches='tight');plt.close(fig)
print('Saved corrected figures to:',OUT)
