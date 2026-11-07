from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
SOURCE=ROOT/'07_Clinical_Model_Refinement'/'Data'
OUT=ROOT/'08_Validation_Uncertainty'
for sub in ('Tables','Figures','Diagnostics'):(OUT/sub).mkdir(parents=True,exist_ok=True)
SEED=42; N_BOOT=2000
if not SOURCE.exists():raise FileNotFoundError(f'Phase 8 predictions missing: {SOURCE}')
files=sorted(SOURCE.glob('*_predictions.csv'))
if not files:raise FileNotFoundError('No Phase 8 prediction files found. Run Phase 8 first.')
rng=np.random.default_rng(SEED)
def metrics(y,p):
    y=np.asarray(y,dtype=int);p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6)
    d={'N':len(y),'Events':int(y.sum()),'Prevalence':float(y.mean()),'Brier':brier_score_loss(y,p),'Log_Loss':log_loss(y,p,labels=[0,1])}
    d['ROC_AUC']=roc_auc_score(y,p) if np.unique(y).size==2 else np.nan
    d['Average_Precision']=average_precision_score(y,p) if np.unique(y).size==2 else np.nan
    if np.unique(y).size==2:
        z=np.log(p/(1-p)).reshape(-1,1)
        m=LogisticRegression(C=1e6,max_iter=2000).fit(z,y)
        d['Calibration_Intercept']=float(m.intercept_[0]);d['Calibration_Slope']=float(m.coef_[0,0])
    else:d['Calibration_Intercept']=d['Calibration_Slope']=np.nan
    return d

def boot(y,p,group=None):
    n=len(y);out=[]
    if group is not None:
        groups=np.asarray(group).astype(str); unique=np.unique(groups)
        ix={g:np.where(groups==g)[0] for g in unique}
    for _ in range(N_BOOT):
        if group is None: idx=rng.integers(0,n,n)
        else:idx=np.concatenate([ix[g] for g in rng.choice(unique,size=len(unique),replace=True)])
        if np.unique(y[idx]).size!=2:continue
        out.append(metrics(y[idx],p[idx]))
    return out

results=[];cal=[];group_rows=[];warnings=[];preds={}
for f in files:
    df=pd.read_csv(f)
    if not {'y_true','probability','source_row_index'}.issubset(df):
        warnings.append(f'{f.name}: required columns absent');continue
    if df.source_row_index.duplicated().any():
        warnings.append(f'{f.name}: duplicate row IDs; skipped');continue
    label=f.stem.removesuffix('_predictions')
    bits=label.split('_')
    alg=bits[-1];feature=bits[-2];validation='_'.join(bits[:-2])
    y=df.y_true.to_numpy(dtype=int);p=df.probability.to_numpy(dtype=float)
    if len(y)<20 or np.unique(y).size!=2:continue
    preds[(validation,feature,alg)]=df.set_index('source_row_index')
    point=metrics(y,p)
    # For OPO validation use group bootstrap to account for within-OPO dependence.
    # Six OPOs give highly unstable intervals; report this limitation explicitly.
    is_opo=validation=='OPO_Group_5Fold' and 'OPO' in df
    g=df.OPO.fillna('UNKNOWN').astype(str).to_numpy() if is_opo else None
    samples=boot(y,p,g)
    row={'Validation':validation,'Feature_Set':feature,'Algorithm':alg,**point,'Bootstrap_Type':'OPO cluster' if is_opo else 'Donor','Bootstrap_Valid':len(samples)}
    for k in ('ROC_AUC','Average_Precision','Brier','Calibration_Intercept','Calibration_Slope'):
        vals=np.array([x[k] for x in samples]);vals=vals[np.isfinite(vals)]
        row[k+'_CI_Low']=float(np.quantile(vals,.025)) if len(vals) else np.nan
        row[k+'_CI_High']=float(np.quantile(vals,.975)) if len(vals) else np.nan
    results.append(row)
    bins=pd.qcut(pd.Series(p),q=min(10,len(np.unique(p))),duplicates='drop')
    for _,ids in pd.Series(np.arange(len(y))).groupby(bins,observed=True):
        ix=ids.to_numpy(dtype=int)
        cal.append({'Validation':validation,'Feature_Set':feature,'Algorithm':alg,'N':len(ix),'Mean_Predicted':float(p[ix].mean()),'Observed_Rate':float(y[ix].mean())})
    if is_opo:
        for opo,sub in df.groupby('OPO',dropna=False):
            if len(sub)<10:continue
            group_rows.append({'Feature_Set':feature,'Algorithm':alg,'OPO':str(opo),**metrics(sub.y_true,sub.probability)})

pd.DataFrame(results).to_csv(OUT/'Tables'/'01_validation_metrics_with_bootstrap_CI.csv',index=False)
pd.DataFrame(cal).to_csv(OUT/'Tables'/'02_calibration_bins_by_validation.csv',index=False)
pd.DataFrame(group_rows).to_csv(OUT/'Tables'/'03_per_OPO_performance.csv',index=False)

paired=[]
for (validation,feature,alg),d in preds.items():
    if feature!='Clinical':continue
    baseline=preds.get((validation,'Baseline',alg))
    if baseline is None:continue
    common=d.index.intersection(baseline.index)
    a=d.loc[common];b=baseline.loc[common]
    if not np.array_equal(a.y_true.to_numpy(),b.y_true.to_numpy()):
        warnings.append(f'{validation}/{alg}: target mismatch; skipped paired comparison');continue
    y=a.y_true.to_numpy(dtype=int);p=a.probability.to_numpy();q=b.probability.to_numpy()
    if np.unique(y).size<2:continue
    obs_auc=roc_auc_score(y,p)-roc_auc_score(y,q)
    obs_ap=average_precision_score(y,p)-average_precision_score(y,q)
    vals=[]
    if validation=='OPO_Group_5Fold' and 'OPO' in a:
        group=a.OPO.fillna('UNKNOWN').astype(str).to_numpy(); unique=np.unique(group)
        mapping={g:np.where(group==g)[0] for g in unique}
        draw=lambda:np.concatenate([mapping[g] for g in rng.choice(unique,len(unique),replace=True)])
        kind='OPO cluster'
    else:
        draw=lambda:rng.integers(0,len(y),len(y));kind='Donor'
    for _ in range(N_BOOT):
        idx=draw()
        if np.unique(y[idx]).size<2:continue
        vals.append((roc_auc_score(y[idx],p[idx])-roc_auc_score(y[idx],q[idx]),average_precision_score(y[idx],p[idx])-average_precision_score(y[idx],q[idx])))
    arr=np.array(vals)
    paired.append({'Validation':validation,'Algorithm':alg,'N':len(y),'Bootstrap_Type':kind,'Delta_ROC_AUC':obs_auc,'Delta_AUC_CI_Low':np.quantile(arr[:,0],.025),'Delta_AUC_CI_High':np.quantile(arr[:,0],.975),'Delta_Average_Precision':obs_ap,'Delta_AP_CI_Low':np.quantile(arr[:,1],.025),'Delta_AP_CI_High':np.quantile(arr[:,1],.975)})
pd.DataFrame(paired).to_csv(OUT/'Tables'/'04_paired_clinical_vs_baseline.csv',index=False)

for validation in ('Random_5Fold','Later_Years_Holdout','OPO_Group_5Fold','BrainDeath_Random_5Fold'):
    sub=pd.DataFrame(cal)
    if sub.empty:break
    sub=sub[(sub.Validation==validation)&(sub.Feature_Set=='Clinical')&(sub.Algorithm=='RF')].sort_values('Mean_Predicted')
    if sub.empty:continue
    fig,ax=plt.subplots(figsize=(6,5));ax.plot([0,1],[0,1],'--',color='gray');ax.plot(sub.Mean_Predicted,sub.Observed_Rate,'o-');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed heart utilization',title=f'Clinical RF calibration: {validation}');fig.tight_layout();fig.savefig(OUT/'Figures'/f'RF_calibration_{validation}.png',dpi=250);plt.close(fig)

notes={'input_directory':str(SOURCE),'output_directory':str(OUT),'bootstrap_resamples':N_BOOT,'warnings':warnings,'interpretation_limits':['No models refitted: intervals condition on fixed held-out predictions','Only six OPO groups; cluster-bootstrap confidence intervals can be highly unstable','Temporal holdout is one split, not independent external validation','Calibration intercept and slope are descriptive estimates on held-out predictions','Clinical measurement timestamps and source definitions still unverified','Clinical data missingness may encode donor evaluation processes','No suitability, organ discard, or causal claims are supported']}
(OUT/'Diagnostics'/'phase9_notes.json').write_text(json.dumps(notes,indent=2))
print('Phase 9 complete:',OUT)
print(pd.DataFrame(results).query("Feature_Set == 'Clinical' and Algorithm == 'RF'")[['Validation','N','ROC_AUC','ROC_AUC_CI_Low','ROC_AUC_CI_High','Brier','Calibration_Slope']].to_string(index=False))
