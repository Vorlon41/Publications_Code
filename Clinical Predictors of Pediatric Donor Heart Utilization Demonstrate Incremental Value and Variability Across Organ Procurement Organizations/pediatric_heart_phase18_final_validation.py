from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression

ROOT = Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
SOURCE = ROOT / '10_Final_Incremental_Model' / 'Phase17_Categorical_Incremental_Value' / 'Data' / '02_heldout_predictions.csv'
OUT = ROOT / '10_Final_Incremental_Model' / 'Phase18_Final_Validation'
for sub in ('Tables','Figures','Diagnostics'):
    (OUT/sub).mkdir(parents=True, exist_ok=True)
if not SOURCE.exists():
    raise FileNotFoundError(f'Phase 17 held-out predictions not found: {SOURCE}')
df = pd.read_csv(SOURCE)
required = {'Cohort','Feature_Set','Row_Index','Outcome','Probability','OPO'}
if not required.issubset(df.columns):
    raise ValueError(f'Missing columns: {required-set(df.columns)}')
A = 'A_Clinical'; C = 'C_Clinical_Cause_ABO'
cohorts = ['Authorized_All','Authorized_BrainDeath','Authorized_OPO_Grouped']
rng = np.random.default_rng(20261029)
B = 2000

def metrics(d):
    y=d.Outcome.to_numpy(dtype=int); p=d.Probability.to_numpy(dtype=float)
    if len(np.unique(y))<2: return None
    return {'N':len(y),'Events':int(y.sum()),'Prevalence':float(y.mean()),
            'ROC_AUC':roc_auc_score(y,p),'Average_Precision':average_precision_score(y,p),
            'Brier':brier_score_loss(y,p),'Log_Loss':log_loss(y,np.clip(p,1e-8,1-1e-8))}

def calibration(d):
    y=d.Outcome.to_numpy(dtype=int); p=np.clip(d.Probability.to_numpy(dtype=float),1e-6,1-1e-6)
    if len(np.unique(y))<2 or np.unique(p).size<2: return (np.nan,np.nan)
    z=np.log(p/(1-p)).reshape(-1,1)
    try:
        lr=LogisticRegression(penalty=None,solver='lbfgs',max_iter=2000).fit(z,y)
        return float(lr.intercept_[0]),float(lr.coef_[0,0])
    except Exception: return (np.nan,np.nan)

rows=[]; binrows=[]; comparisons=[]; peropo=[]
for cohort in cohorts:
    for feat in [A,C]:
        d=df[(df.Cohort==cohort)&(df.Feature_Set==feat)].copy()
        if d.empty: continue
        if d.Row_Index.duplicated().any(): raise ValueError(f'Duplicate row identifiers: {cohort} {feat}')
        m=metrics(d); intercept,slope=calibration(d)
        if m is None: continue
        row={'Cohort':cohort,'Feature_Set':feat,**m,'Calibration_Intercept':intercept,'Calibration_Slope':slope}
        boot=[]
        for _ in range(B):
            ix=rng.integers(0,len(d),len(d)); q=metrics(d.iloc[ix])
            if q is not None: boot.append(q)
        for k in ('ROC_AUC','Average_Precision','Brier'):
            vals=[b[k] for b in boot]
            row[k+'_CI_Low'],row[k+'_CI_High']=np.percentile(vals,[2.5,97.5])
        rows.append(row)
        d['Bin']=pd.qcut(d.Probability.rank(method='first'),q=min(10,len(d)),labels=False)
        bins=d.groupby('Bin').agg(N=('Outcome','size'),Predicted=('Probability','mean'),Observed=('Outcome','mean')).reset_index()
        bins.insert(0,'Feature_Set',feat);bins.insert(0,'Cohort',cohort)
        binrows.extend(bins.to_dict('records'))
        if feat==C:
            fig,ax=plt.subplots(figsize=(6.4,5.4))
            ax.plot([0,1],[0,1],ls='--',color='gray',label='Ideal')
            ax.plot(bins.Predicted,bins.Observed,marker='o',label='Observed bins')
            ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed heart utilization',title=cohort+' — Model C calibration')
            ax.legend();fig.tight_layout();fig.savefig(OUT/'Figures'/f'calibration_{cohort}.png',dpi=250);plt.close(fig)
            for opo,g in d.groupby('OPO'):
                mm=metrics(g)
                peropo.append({'Cohort':cohort,'OPO':opo,**(mm or {'N':len(g),'Events':int(g.Outcome.sum()),'ROC_AUC':np.nan,'Average_Precision':np.nan,'Brier':np.nan})})
    a=df[(df.Cohort==cohort)&(df.Feature_Set==A)][['Row_Index','Outcome','OPO','Probability']].rename(columns={'Probability':'p_A'})
    c=df[(df.Cohort==cohort)&(df.Feature_Set==C)][['Row_Index','Outcome','OPO','Probability']].rename(columns={'Probability':'p_C'})
    paired=a.merge(c,on='Row_Index',validate='one_to_one',suffixes=('_a','_c'))
    if len(paired)!=len(a) or len(paired)!=len(c) or not (paired.Outcome_a.eq(paired.Outcome_c)&paired.OPO_a.eq(paired.OPO_c)).all():
        raise ValueError('Predictions are not correctly paired')
    y=paired.Outcome_a.to_numpy();pa=paired.p_A.to_numpy();pc=paired.p_C.to_numpy()
    result={'Cohort':cohort,'N':len(paired),'Delta_ROC_AUC':roc_auc_score(y,pc)-roc_auc_score(y,pa),
            'Delta_Average_Precision':average_precision_score(y,pc)-average_precision_score(y,pa),
            'Delta_Brier':brier_score_loss(y,pc)-brier_score_loss(y,pa)}
    for method in ['donor_bootstrap','OPO_cluster_bootstrap']:
        if method=='OPO_cluster_bootstrap' and cohort!='Authorized_OPO_Grouped':continue
        groups=paired.OPO_a.unique(); deltas=[]
        for _ in range(B):
            if method=='donor_bootstrap': ids=rng.integers(0,len(paired),len(paired))
            else:
                sampled=rng.choice(groups,size=len(groups),replace=True)
                ids=np.concatenate([np.flatnonzero(paired.OPO_a.to_numpy()==g) for g in sampled])
            yy=y[ids]
            if len(np.unique(yy))<2:continue
            deltas.append([roc_auc_score(yy,pc[ids])-roc_auc_score(yy,pa[ids]),
                           average_precision_score(yy,pc[ids])-average_precision_score(yy,pa[ids]),
                           brier_score_loss(yy,pc[ids])-brier_score_loss(yy,pa[ids])])
        q=np.asarray(deltas)
        r={**result,'Resampling':method,'Valid_Resamples':len(q),'OPO_Groups':len(groups)}
        for j,name in enumerate(['Delta_ROC_AUC','Delta_Average_Precision','Delta_Brier']):
            r[name+'_CI_Low'],r[name+'_CI_High']=np.percentile(q[:,j],[2.5,97.5])
        comparisons.append(r)

pd.DataFrame(rows).to_csv(OUT/'Tables'/'01_model_metrics_and_uncertainty.csv',index=False)
pd.DataFrame(comparisons).to_csv(OUT/'Tables'/'02_paired_model_C_vs_A.csv',index=False)
pd.DataFrame(binrows).to_csv(OUT/'Tables'/'03_calibration_bins.csv',index=False)
pd.DataFrame(peropo).to_csv(OUT/'Tables'/'04_per_OPO_model_C.csv',index=False)
notes={'source':str(SOURCE),'bootstrap_iterations':B,'random_seed':20261029,
       'cautions':['Fixed out-of-fold predictions; uncertainty excludes model refitting and model selection',
                   'Six OPOs only; cluster bootstrap unstable and does not establish external validity',
                   'Calibration slope/intercept descriptive; no recalibration applied',
                   'Clinical measurement timing and categorical dictionary mapping unverified',
                   'Retrospective utilization classification, not donor suitability or acceptance']}
(OUT/'Diagnostics'/'phase18_notes.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
print('Phase 18 complete:',OUT)
print(pd.DataFrame(rows)[['Cohort','Feature_Set','N','ROC_AUC','Average_Precision','Brier','Calibration_Slope']].to_string(index=False))
