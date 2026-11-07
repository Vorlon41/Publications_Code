from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

SOURCE=Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
ROOT=Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
OUT=ROOT/'10_Final_Incremental_Model'/'Phase17_Categorical_Incremental_Value'
TABLES=OUT/'Tables'; DATA=OUT/'Data'; DIAG=OUT/'Diagnostics'
for p in (TABLES,DATA,DIAG):p.mkdir(parents=True,exist_ok=True)
Y='outcome_heart_Transplanted'; SEED=42
BASE=['Age','Gender','WeightKg','HeightIn']
CLINICAL=['HR','SBP','DBP','PO2','PCO2','PH','O2SAT','FIO2','PEEP','Rate','TV','Hgb','Hct','Ptl','UOP','Output_Urine']
FAMILIES={'Cause':'Cause_of_Death_UNOS','ABO':'ABO_BloodType','Mechanism':'Mechanism_of_Death'}
FEATURE_SETS={
 'A_Clinical':BASE+CLINICAL,
 'B_Clinical_Cause':BASE+CLINICAL+['Cause'],
 'C_Clinical_Cause_ABO':BASE+CLINICAL+['Cause','ABO'],
 'D_Clinical_Cause_ABO_Mechanism':BASE+CLINICAL+['Cause','ABO','Mechanism']
}
raw=pd.read_csv(SOURCE,low_memory=False)
for c in ['Age','authorized','brain_death',Y]:
 if c not in raw:raise ValueError('Required source column absent: '+c)
 raw[c]=pd.to_numeric(raw[c],errors='coerce')
raw=raw.loc[(raw.Age>=0)&(raw.Age<18)&(raw.authorized==1)&raw[Y].isin([0,1])].copy()
if 'PatientID' in raw and raw.PatientID.duplicated().any():raise ValueError('Duplicate PatientID; resolve before CV')

# A family is resolved only if exactly one indicator is positive and none are missing.
# Do not treat all-zero rows as negative or infer unknown ABO subtype codes.
family_diagnostics=[]
for name,prefix in FAMILIES.items():
 cols=[c for c in raw if c.startswith(prefix+'_')]
 if not cols:raise ValueError('Missing indicator family: '+prefix)
 z=raw[cols].apply(pd.to_numeric,errors='coerce')
 hits=z.eq(1); nhits=hits.sum(axis=1)
 label=hits.idxmax(axis=1).str[len(prefix)+1:]
 label=label.where((nhits==1)&(~z.isna().any(axis=1)),'UNRESOLVED')
 if name=='ABO':
  mapping={'O':'O','A':'A','A1':'A','A2':'A','B':'B','AB':'AB','A1B':'AB','A2B':'AB'}
  label=label.map(lambda s:mapping.get(s,'UNRESOLVED'))
 raw[name]=label.astype('object')
 family_diagnostics.append({'Family':name,'Indicators':len(cols),'Resolved':int((label!='UNRESOLVED').sum()),'Unresolved':int((label=='UNRESOLVED').sum()),'Dictionary_Verified':False})
pd.DataFrame(family_diagnostics).to_csv(DIAG/'01_category_reconstruction_integrity.csv',index=False)

# Missing clinical values are imputed within each training fold. The resulting model
# can still learn data-availability patterns; no prospective availability is assumed.
for c in BASE+CLINICAL:
 if c not in raw:raw[c]=np.nan
for c in BASE+CLINICAL:
 if c!='Gender':raw[c]=pd.to_numeric(raw[c],errors='coerce')
for c in ['Gender','Cause','ABO','Mechanism','OPO']:
 if c in raw:raw[c]=raw[c].astype('object').where(raw[c].notna(),'UNRESOLVED').astype(str)

cohorts={'Authorized_All':raw, 'Authorized_BrainDeath':raw.loc[raw.brain_death==1].copy()}
rows=[]; preds=[]; diagnostics=[]
for cohort_name,df in cohorts.items():
 df=df.reset_index(drop=True); y=df[Y].astype(int).to_numpy()
 cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
 splits=list(cv.split(df,y))
 for set_name,columns in FEATURE_SETS.items():
  nums=[c for c in columns if c not in ['Gender','Cause','ABO','Mechanism']]
  cats=[c for c in columns if c in ['Gender','Cause','ABO','Mechanism']]
  # Fixed, clinically interpretable imputation. Unknown category stays explicit.
  prep=ColumnTransformer([
   ('num',SimpleImputer(strategy='median',keep_empty_features=True),nums),
   ('cat',OneHotEncoder(handle_unknown='ignore'),cats)
  ],sparse_threshold=0.3)
  model=Pipeline([('prep',prep),('rf',RandomForestClassifier(n_estimators=500,min_samples_leaf=2,random_state=SEED,n_jobs=-1))])
  p=np.full(len(df),np.nan)
  for fold,(tr,te) in enumerate(splits,1):
   model.fit(df.iloc[tr][columns],y[tr]);p[te]=model.predict_proba(df.iloc[te][columns])[:,1]
   for idx in te:
    preds.append({'Cohort':cohort_name,'Feature_Set':set_name,'Row_Index':int(idx),'Fold':fold,'Outcome':int(y[idx]),'Probability':float(p[idx]),'OPO':str(df.iloc[idx].get('OPO','')),'Referral_Year':str(df.iloc[idx].get('Referral_Year',''))})
  rows.append({'Cohort':cohort_name,'Feature_Set':set_name,'N':len(df),'Events':int(y.sum()),'ROC_AUC':roc_auc_score(y,p),'Average_Precision':average_precision_score(y,p),'Brier':brier_score_loss(y,p)})
 # Grouped OPO validation for primary cohort, all groups must have both classes globally.
 if cohort_name=='Authorized_All' and 'OPO' in df:
  groups=df.OPO.fillna('UNRESOLVED').astype(str).to_numpy()
  if len(np.unique(groups))>=3:
   for set_name,columns in FEATURE_SETS.items():
    nums=[c for c in columns if c not in ['Gender','Cause','ABO','Mechanism']];cats=[c for c in columns if c in ['Gender','Cause','ABO','Mechanism']]
    prep=ColumnTransformer([('num',SimpleImputer(strategy='median',keep_empty_features=True),nums),('cat',OneHotEncoder(handle_unknown='ignore'),cats)])
    model=Pipeline([('prep',prep),('rf',RandomForestClassifier(n_estimators=500,min_samples_leaf=2,random_state=SEED,n_jobs=-1))])
    p=np.full(len(df),np.nan)
    for fold,(tr,te) in enumerate(GroupKFold(n_splits=min(5,len(np.unique(groups)))).split(df,y,groups),1):
     if len(np.unique(y[tr]))<2:continue
     model.fit(df.iloc[tr][columns],y[tr]);p[te]=model.predict_proba(df.iloc[te][columns])[:,1]
     for idx in te:preds.append({'Cohort':'Authorized_OPO_Grouped','Feature_Set':set_name,'Row_Index':int(idx),'Fold':fold,'Outcome':int(y[idx]),'Probability':float(p[idx]),'OPO':groups[idx],'Referral_Year':str(df.iloc[idx].get('Referral_Year',''))})
    good=np.isfinite(p)
    if good.sum() and len(np.unique(y[good]))==2:rows.append({'Cohort':'Authorized_OPO_Grouped','Feature_Set':set_name,'N':int(good.sum()),'Events':int(y[good].sum()),'ROC_AUC':roc_auc_score(y[good],p[good]),'Average_Precision':average_precision_score(y[good],p[good]),'Brier':brier_score_loss(y[good],p[good])})

result=pd.DataFrame(rows);result.to_csv(TABLES/'01_model_comparison.csv',index=False)
pred=pd.DataFrame(preds);pred.to_csv(DATA/'02_heldout_predictions.csv',index=False)
# Paired bootstrap on fixed out-of-fold predictions. One comparison per expanded feature set.
rng=np.random.default_rng(SEED); boot=[]
for cohort,part in pred.groupby('Cohort'):
 pivot=part.pivot(index='Row_Index',columns='Feature_Set',values='Probability')
 yy=part.drop_duplicates('Row_Index').set_index('Row_Index').loc[pivot.index,'Outcome'].to_numpy(dtype=int)
 if 'A_Clinical' not in pivot:continue
 for name in FEATURE_SETS:
  if name=='A_Clinical' or name not in pivot:continue
  pair=pivot[['A_Clinical',name]].dropna(); ypair=part.drop_duplicates('Row_Index').set_index('Row_Index').loc[pair.index,'Outcome'].to_numpy(dtype=int)
  a=pair['A_Clinical'].to_numpy();b=pair[name].to_numpy(); diffs=[]; apdiff=[]
  for _ in range(2000):
   ix=rng.integers(0,len(ypair),len(ypair));yb=ypair[ix]
   if len(np.unique(yb))<2:continue
   diffs.append(roc_auc_score(yb,b[ix])-roc_auc_score(yb,a[ix]))
   apdiff.append(average_precision_score(yb,b[ix])-average_precision_score(yb,a[ix]))
  boot.append({'Cohort':cohort,'Expanded_Set':name,'N':len(ypair),'Delta_ROC_AUC':roc_auc_score(ypair,b)-roc_auc_score(ypair,a),'ROC_CI_Low':np.quantile(diffs,.025),'ROC_CI_High':np.quantile(diffs,.975),'Delta_AP':average_precision_score(ypair,b)-average_precision_score(ypair,a),'AP_CI_Low':np.quantile(apdiff,.025),'AP_CI_High':np.quantile(apdiff,.975)})
pd.DataFrame(boot).to_csv(TABLES/'03_paired_incremental_bootstrap.csv',index=False)
(DIAG/'README.txt').write_text('Retrospective model comparison only. Category reconstruction is provisional and requires ORCHID dictionary confirmation. Unknown indicator families are UNRESOLVED, not negative. No clinical measurement timing is verified. Random and grouped cross-validation are internal validation. Paired bootstrap conditions on fixed out-of-fold predictions and does not account for model refitting. All comparisons exploratory. Do not infer heart suitability, offer, acceptance or discard.\n',encoding='utf-8')
(DIAG/'run_notes.json').write_text(json.dumps({'Source':str(SOURCE),'Cohorts':{k:len(v) for k,v in cohorts.items()},'Feature_sets':FEATURE_SETS,'Seed':SEED,'Folds':5},indent=2),encoding='utf-8')
print('\nPhase 17 complete:',OUT);print(result.to_string(index=False))
