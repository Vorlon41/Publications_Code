from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE=Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
ROOT=Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
OUT=ROOT/'07_Clinical_Model_Refinement'
for s in ('Data','Tables','Figures','Diagnostics'): (OUT/s).mkdir(parents=True,exist_ok=True)
TARGET='outcome_heart_Transplanted'; SEED=42
if not SOURCE.exists(): raise FileNotFoundError(f'Source CSV not found: {SOURCE}')
df=pd.read_csv(SOURCE,low_memory=False)
needed=['Age','authorized',TARGET,'Gender','WeightKg','HeightIn','brain_death']
missing=[c for c in needed if c not in df]
if missing: raise ValueError(f'Missing required columns: {missing}')
for c in needed: df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.loc[df.Age.between(0,17)&df.authorized.eq(1)&df[TARGET].isin([0,1])].copy()
if 'PatientID' in df and df.PatientID.dropna().duplicated().any(): raise ValueError('Duplicate donor IDs')
for c,lo,hi in [('WeightKg',.3,250),('HeightIn',8,90)]:
    df.loc[df[c].notna()&~df[c].between(lo,hi),c]=np.nan
base=['Age','Gender','WeightKg','HeightIn']
clinical=['HR','SBP','DBP','PO2','PCO2','PH','O2SAT','FIO2','PEEP','Rate','TV','Hgb','Hct','Ptl','UOP','Output_Urine']
clinical=[c for c in clinical if c in df]
for c in clinical: df[c]=pd.to_numeric(df[c],errors='coerce')
moderate=[c for c in clinical if df[c].notna().mean()>=.40]
if not moderate: raise ValueError('No clinical variables meet coverage threshold')
feature_sets={'Baseline':base,'Clinical':base+moderate}
algorithms=['LR','RF']
try:
    from xgboost import XGBClassifier
    algorithms.append('XGB')
except ImportError: warnings.warn('XGBoost not installed; evaluating LR and RF')
def pipeline(cols,alg):
    prep=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())]),cols)])
    if alg=='LR': model=LogisticRegression(max_iter=3000,random_state=SEED)
    elif alg=='RF': model=RandomForestClassifier(n_estimators=350,min_samples_leaf=5,random_state=SEED,n_jobs=-1)
    else: model=XGBClassifier(n_estimators=250,max_depth=3,learning_rate=.04,subsample=.85,colsample_bytree=.85,eval_metric='logloss',random_state=SEED,n_jobs=1)
    return Pipeline([('prep',prep),('model',model)])

def scores(y,p):
    out={'N':len(y),'Events':int(np.sum(y)),'Prevalence':float(np.mean(y)),
         'Brier':brier_score_loss(y,p),'Log_Loss':log_loss(y,p,labels=[0,1])}
    out['ROC_AUC']=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan
    out['Average_Precision']=average_precision_score(y,p) if len(np.unique(y))==2 else np.nan
    # calibration intercept/slope via logit of predictions; descriptive only
    from sklearn.linear_model import LogisticRegression as LR
    pp=np.clip(p,1e-6,1-1e-6)
    logits=np.log(pp/(1-pp)).reshape(-1,1)
    if len(np.unique(y))==2:
        fit=LR(C=1e6,max_iter=2000).fit(logits,y)
        out['Calibration_Intercept']=float(fit.intercept_[0]); out['Calibration_Slope']=float(fit.coef_[0,0])
    else: out['Calibration_Intercept']=np.nan;out['Calibration_Slope']=np.nan
    return out

def run_cv(frame,label,splitter,groups=None):
    y=frame[TARGET].astype(int).to_numpy()
    split_list=list(splitter.split(frame,y,groups))
    outputs=[]
    for name,cols in feature_sets.items():
        for alg in algorithms:
            pred=np.full(len(frame),np.nan); fold=np.full(len(frame),-1)
            for k,(tr,te) in enumerate(split_list,1):
                if len(np.unique(y[tr]))<2: continue
                m=pipeline(cols,alg);m.fit(frame.iloc[tr][cols],y[tr])
                pred[te]=m.predict_proba(frame.iloc[te][cols])[:,1];fold[te]=k
            valid=np.isfinite(pred)
            if valid.sum()<20 or len(np.unique(y[valid]))<2: continue
            row={'Validation':label,'Feature_Set':name,'Algorithm':alg,'Tested_N':int(valid.sum()),**scores(y[valid],pred[valid])}
            outputs.append(row)
            pframe=frame.iloc[np.flatnonzero(valid)]
            pd.DataFrame({'source_row_index':pframe.index,'fold':fold[valid],'y_true':y[valid], 'probability':pred[valid],
                          'Age':pframe['Age'].to_numpy(), 'Referral_Year':pframe['Referral_Year'].to_numpy() if 'Referral_Year' in pframe else np.nan,
                          'OPO':pframe['OPO'].astype(str).to_numpy() if 'OPO' in pframe else ''}).to_csv(OUT/'Data'/f'{label}_{name}_{alg}_predictions.csv',index=False)
    return outputs

rows=[]; diagnostics=[]
rows+=run_cv(df,'Random_5Fold',StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED))
# Strictly later-year evaluation, with cutoff fixed before looking at performance.
if 'Referral_Year' in df:
    df['Referral_Year']=pd.to_numeric(df['Referral_Year'],errors='coerce')
    years=sorted(df['Referral_Year'].dropna().unique())
    if len(years)>=3:
        cutoff=years[-2] # reserve last two observed years
        tr=df.loc[df.Referral_Year<cutoff];te=df.loc[df.Referral_Year>=cutoff]
        diagnostics.append({'Check':'Temporal cutoff','Value':str(cutoff),'Train_N':len(tr),'Test_N':len(te)})
        if len(tr)>=100 and len(te)>=50 and tr[TARGET].nunique()==2 and te[TARGET].nunique()==2:
            for name,cols in feature_sets.items():
                for alg in algorithms:
                    m=pipeline(cols,alg);m.fit(tr[cols],tr[TARGET].astype(int))
                    p=m.predict_proba(te[cols])[:,1]
                    rows.append({'Validation':'Later_Years_Holdout','Feature_Set':name,'Algorithm':alg,'Tested_N':len(te),**scores(te[TARGET].astype(int).to_numpy(),p)})
                    pd.DataFrame({'source_row_index':te.index,'y_true':te[TARGET].astype(int),'probability':p,'Age':te.Age,'Referral_Year':te.Referral_Year}).to_csv(OUT/'Data'/f'Later_Years_Holdout_{name}_{alg}_predictions.csv',index=False)
        else: diagnostics.append({'Check':'Temporal validation','Value':'Insufficient class counts or sample sizes'})
    else: diagnostics.append({'Check':'Temporal validation','Value':'Insufficient distinct years'})
else: diagnostics.append({'Check':'Temporal validation','Value':'Referral_Year missing'})
# Group-held-out OPO validation: no OPO predictor in either feature set.
if 'OPO' in df:
    groups=df['OPO'].astype('string').fillna('UNKNOWN').astype(str)
    n_groups=groups.nunique()
    diagnostics.append({'Check':'OPO groups','Value':str(n_groups)})
    if n_groups>=5:
        rows+=run_cv(df,'OPO_Group_5Fold',GroupKFold(n_splits=5),groups.to_numpy())
    else: diagnostics.append({'Check':'OPO validation','Value':'Fewer than 5 OPO groups'})
else: diagnostics.append({'Check':'OPO validation','Value':'OPO missing'})
# Brain-death subgroup with its own CV; do not include brain_death as a predictor.
bd=df.loc[df.brain_death.eq(1)].copy()
if len(bd)>=100 and bd[TARGET].value_counts().min()>=5:
    rows+=run_cv(bd,'BrainDeath_Random_5Fold',StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED))

result=pd.DataFrame(rows)
result.to_csv(OUT/'Tables'/'01_robustness_model_comparison.csv',index=False)
# Age-stratified held-out performance for primary random CV, with no fitting within subgroup.
sub=[]
for name in feature_sets:
    for alg in algorithms:
        f=OUT/'Data'/f'Random_5Fold_{name}_{alg}_predictions.csv'
        if not f.exists(): continue
        p=pd.read_csv(f)
        p['Age_Group']=pd.cut(p.Age,[-.01,.999,4.999,11.999,17.999],labels=['Under_1','1_to_4','5_to_11','12_to_17'])
        for age,g in p.groupby('Age_Group',observed=True):
            if len(g)>=10:
                sub.append({'Feature_Set':name,'Algorithm':alg,'Age_Group':str(age),**scores(g.y_true.to_numpy(),g.probability.to_numpy())})
pd.DataFrame(sub).to_csv(OUT/'Tables'/'02_age_subgroup_performance.csv',index=False)
# Observed data coverage, with separate denominators by utilization.
cov=[]
for c in moderate:
    for outcome in (0,1):
        s=df.loc[df[TARGET].eq(outcome),c]
        cov.append({'Feature':c,'Outcome':outcome,'N':len(s),'Observed_N':int(s.notna().sum()),'Observed_Percent':100*s.notna().mean()})
pd.DataFrame(cov).to_csv(OUT/'Tables'/'03_clinical_coverage_by_outcome.csv',index=False)
# Clinically useful derived variables: descriptive only, not added to prediction without unit/timing verification.
flags=[]
for name,cols in {'PO2_FIO2':['PO2','FIO2'],'TV_per_kg':['TV','WeightKg']}.items():
    if all(c in df for c in cols):
        s=df[cols].apply(pd.to_numeric,errors='coerce')
        valid=s.notna().all(axis=1)&(s[cols[1]]>0)
        flags.append({'Derived':name,'N_Complete_Pairs':int(valid.sum()),'Percent_Complete_Pairs':100*valid.mean(),'Status':'Not computed until FiO2/TV units and measurement timing verified'})
pd.DataFrame(flags).to_csv(OUT/'Diagnostics'/'04_derived_variable_feasibility.csv',index=False)
pd.DataFrame(diagnostics).to_csv(OUT/'Diagnostics'/'05_validation_diagnostics.csv',index=False)
# Reliability diagram, held-out random CV, clinical RF only.
f=OUT/'Data'/'Random_5Fold_Clinical_RF_predictions.csv'
if f.exists():
    p=pd.read_csv(f); p['bin']=pd.qcut(p.probability,q=min(10,p.probability.nunique()),duplicates='drop')
    cal=p.groupby('bin',observed=True).agg(N=('y_true','size'),Mean_Predicted=('probability','mean'),Observed_Rate=('y_true','mean')).reset_index()
    cal['bin']=cal['bin'].astype(str);cal.to_csv(OUT/'Tables'/'04_RF_clinical_calibration_bins.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,5));ax.plot([0,1],[0,1],'--',color='gray');ax.plot(cal.Mean_Predicted,cal.Observed_Rate,'o-');ax.set(xlabel='Mean predicted probability',ylabel='Observed heart utilization',title='Clinical RF: out-of-fold calibration',xlim=(0,1),ylim=(0,1));fig.tight_layout();fig.savefig(OUT/'Figures'/'clinical_RF_calibration.png',dpi=250);plt.close(fig)
notes={'source':str(SOURCE),'authorized_n':len(df),'events':int(df[TARGET].sum()),'feature_sets':feature_sets,'algorithms':algorithms,'seed':SEED,'limitations':['Retrospective classification only; clinical variable timestamps unverified','Pooled OOF calibration is descriptive; no external validation','GroupKFold tests unseen OPOs but fold outcome proportions may differ','Temporal holdout uses last two available referral years; source-year definition requires confirmation','Age subgroups may have imprecise estimates','Clinical missingness may reflect evaluation selection','Do not infer heart suitability or acceptance decisions','No clinical derived ratio used until units verified']}
(OUT/'Diagnostics'/'phase8_notes.json').write_text(json.dumps(notes,indent=2))
print('Phase 8 complete:',OUT)
print(result.to_string(index=False))
