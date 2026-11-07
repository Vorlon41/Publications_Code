from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

SOURCE = Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
ROOT = Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
OUT = ROOT/'06_Model_Interpretation'/'Incremental_Clinical_Value'
for folder in ('Data','Tables','Diagnostics'):
    (OUT/folder).mkdir(parents=True,exist_ok=True)
TARGET='outcome_heart_Transplanted'; SEED=42; FOLDS=5
if not SOURCE.exists(): raise FileNotFoundError(SOURCE)
df=pd.read_csv(SOURCE,low_memory=False)
for col in ('Age','authorized',TARGET,'Gender','WeightKg','HeightIn'):
    if col not in df: raise ValueError(f'Required field absent: {col}')
for col in ('Age','authorized',TARGET,'Gender','WeightKg','HeightIn','Referral_Year','brain_death','procured','approached','transplanted'):
    if col in df and col!='OPO': df[col]=pd.to_numeric(df[col],errors='coerce')
df=df.loc[df['Age'].between(0,17)&df['authorized'].eq(1)&df[TARGET].isin([0,1])].copy()
if 'PatientID' in df and df['PatientID'].notna().any() and df.loc[df['PatientID'].notna(),'PatientID'].duplicated().any():
    raise ValueError('Duplicate PatientID: verify donor independence.')
if (len(df),int(df[TARGET].sum())) != (819,441):
    warnings.warn(f'Cohort differs from previous audit: n={len(df)}, utilized={int(df[TARGET].sum())}')
for col,lo,hi in [('WeightKg',.3,250),('HeightIn',8,90)]:
    df.loc[df[col].notna()&~df[col].between(lo,hi),col]=np.nan
# Deliberately do NOT use other-organ outcomes, timestamps, or derived intervals.
base=['Age','Gender','WeightKg','HeightIn']
clinical_candidates=['HR','SBP','DBP','PO2','PCO2','PH','O2SAT','FIO2','PIP','PEEP','Rate','TV','Hgb','Hct','Ptl','Creatinine','BUN','Lactate','TroponinI','TroponinT','BNP','UOP','Output_Urine']
clinical=[c for c in clinical_candidates if c in df]
for c in clinical: df[c]=pd.to_numeric(df[c],errors='coerce')
coverage=[]
for c in clinical:
    obs=df[c].notna()
    coverage.append({'Feature':c,'Overall_Observed_N':int(obs.sum()),'Overall_Coverage':float(obs.mean()),
        'Transplanted_Coverage':float(obs[df[TARGET].eq(1)].mean()),
        'Not_Transplanted_Coverage':float(obs[df[TARGET].eq(0)].mean())})
coverage_df=pd.DataFrame(coverage).sort_values('Overall_Coverage',ascending=False)
coverage_df.to_csv(OUT/'Tables'/'01_clinical_coverage.csv',index=False)
# Coverage threshold is a sensitivity choice, NOT proof of predecision availability.
moderate=[c for c in clinical if df[c].notna().mean()>=.40]
sets={'A_Baseline':base,'B_Baseline_Plus_Moderate_Coverage_Clinical':base+moderate,
      'C_Baseline_Plus_All_Clinical_Exploratory':base+clinical}
# Missingness-only experiment: mask each clinical value to 0/1 availability.
for c in clinical: df[f'available__{c}']=df[c].notna().astype(int)
sets['D_Baseline_Plus_Clinical_Availability_Only']=base+[f'available__{c}' for c in clinical]
# Deliberate process-information benchmark; NOT a clinically deployable model.
process=[c for c in ['brain_death','procured','approached','transplanted'] if c in df]
if process: sets['E_Process_Inclusive_LEAKAGE_BENCHMARK']=base+process
# Do not silently treat numeric-coded missing as a category; median imputation occurs inside each fold.
for c in set(sum(sets.values(),[])):
    if c!='OPO': df[c]=pd.to_numeric(df[c],errors='coerce')
try:
    from xgboost import XGBClassifier
    algorithms=['LR','RF','XGB']
except ImportError:
    algorithms=['LR','RF']; warnings.warn('xgboost not installed; running LR and RF')

def pipeline(cols,alg):
    prep=ColumnTransformer([('numeric',Pipeline([
        ('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),
        ('scale',StandardScaler())]),cols)],remainder='drop')
    if alg=='LR': model=LogisticRegression(max_iter=3000,random_state=SEED)
    elif alg=='RF': model=RandomForestClassifier(n_estimators=350,min_samples_leaf=5,n_jobs=-1,random_state=SEED)
    else: model=XGBClassifier(n_estimators=250,max_depth=3,learning_rate=.04,subsample=.85,
            colsample_bytree=.85,eval_metric='logloss',random_state=SEED,n_jobs=1)
    return Pipeline([('prep',prep),('model',model)])

all_rows=[]; all_preds={}
def evaluate(frame,cohort):
    y=frame[TARGET].astype(int).to_numpy()
    if min(np.bincount(y,minlength=2))<FOLDS:
        warnings.warn(f'{cohort}: insufficient outcome counts'); return
    splits=list(StratifiedKFold(n_splits=FOLDS,shuffle=True,random_state=SEED).split(np.zeros(len(y)),y))
    for name,cols in sets.items():
        for alg in algorithms:
            probs=np.full(len(frame),np.nan); fold_id=np.zeros(len(frame),dtype=int)
            for k,(train,test) in enumerate(splits,1):
                fitted=pipeline(cols,alg)
                fitted.fit(frame.iloc[train][cols],y[train])
                probs[test]=fitted.predict_proba(frame.iloc[test][cols])[:,1]
                fold_id[test]=k
            assert np.isfinite(probs).all()
            row={'Cohort':cohort,'Feature_Set':name,'Algorithm':alg,'N':len(y),'Events':int(y.sum()),
                'Prevalence':float(y.mean()),'ROC_AUC':roc_auc_score(y,probs),
                'Average_Precision':average_precision_score(y,probs),
                'Brier':brier_score_loss(y,probs),'Log_Loss':log_loss(y,probs,labels=[0,1])}
            all_rows.append(row)
            key=(cohort,name,alg)
            all_preds[key]=(y,probs)
            pd.DataFrame({'source_row_index':frame.index,'fold':fold_id,'y_true':y,'oof_probability':probs}).to_csv(
                OUT/'Data'/f'{cohort}_{name}_{alg}_OOF.csv',index=False)

evaluate(df,'Authorized_All')
if 'brain_death' in df: evaluate(df.loc[df['brain_death'].eq(1)].copy(),'Authorized_BrainDeath_Documented')
summary=pd.DataFrame(all_rows)
summary.to_csv(OUT/'Tables'/'02_incremental_model_comparison.csv',index=False)
# Paired patient-level bootstrap of the difference in OOF metrics. Descriptive, not nested tuning.
rng=np.random.default_rng(SEED); comparisons=[]
for cohort in summary.Cohort.unique():
    for alg in algorithms:
        base_key=(cohort,'A_Baseline',alg)
        if base_key not in all_preds: continue
        y,p0=all_preds[base_key]
        for feature_set in sets:
            key=(cohort,feature_set,alg)
            if feature_set=='A_Baseline' or key not in all_preds: continue
            y2,p1=all_preds[key]
            if not np.array_equal(y,y2): raise RuntimeError('OOF labels not aligned')
            auc_d=[]; ap_d=[]
            for _ in range(1000):
                ix=rng.integers(0,len(y),len(y))
                if np.unique(y[ix]).size<2: continue
                auc_d.append(roc_auc_score(y[ix],p1[ix])-roc_auc_score(y[ix],p0[ix]))
                ap_d.append(average_precision_score(y[ix],p1[ix])-average_precision_score(y[ix],p0[ix]))
            comparisons.append({'Cohort':cohort,'Algorithm':alg,'Comparison':feature_set+' minus A_Baseline',
                'Delta_ROC_AUC':roc_auc_score(y,p1)-roc_auc_score(y,p0),
                'Delta_ROC_AUC_CI_Low':np.percentile(auc_d,2.5),'Delta_ROC_AUC_CI_High':np.percentile(auc_d,97.5),
                'Delta_Average_Precision':average_precision_score(y,p1)-average_precision_score(y,p0),
                'Delta_AP_CI_Low':np.percentile(ap_d,2.5),'Delta_AP_CI_High':np.percentile(ap_d,97.5)})
pd.DataFrame(comparisons).to_csv(OUT/'Tables'/'03_paired_bootstrap_incremental_value.csv',index=False)
with open(OUT/'Diagnostics'/'phase6_notes.json','w',encoding='utf-8') as f:
    json.dump({'source':str(SOURCE),'authorized_cohort_n':len(df),'utilized_n':int(df[TARGET].sum()),
        'feature_sets':sets,'clinical_coverage_threshold':.40,'process_benchmark_features':process,
        'seed':SEED,'folds':FOLDS,'limitations':[
            'Retrospective classification; clinical measurement timing is unknown.',
            'Clinical missingness indicators can encode evaluation selection and care processes.',
            'Process-inclusive model deliberately contains postdecision information and must not be used for prospective prediction.',
            'Paired bootstrap treats out-of-fold predictions as fixed; it does not account for model refitting or tuning.',
            'No external validation; subgroup sample sizes may be small.',
            'Clinical variable units, repeated measurements and source definitions require verification.',
            'No inference of organ suitability or reasons for nonutilization from predicted probabilities.']},f,indent=2)
print('Phase 6 complete:',OUT)
print(summary.to_string(index=False))
