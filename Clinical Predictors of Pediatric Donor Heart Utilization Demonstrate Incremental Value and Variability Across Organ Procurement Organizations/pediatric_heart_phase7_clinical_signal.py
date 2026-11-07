from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.inspection import permutation_importance

SOURCE=Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
ROOT=Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
OUT=ROOT/'06_Model_Interpretation'/'Clinical_Signal_Analysis'
for sub in ('Data','Tables','Figures','Diagnostics'): (OUT/sub).mkdir(parents=True,exist_ok=True)
TARGET='outcome_heart_Transplanted'; SEED=42; N_SPLITS=5
if not SOURCE.exists(): raise FileNotFoundError(f'Input not found: {SOURCE}')
df=pd.read_csv(SOURCE,low_memory=False)
required=['Age','authorized',TARGET,'Gender','WeightKg','HeightIn','brain_death']
missing=[c for c in required if c not in df.columns]
if missing: raise ValueError(f'Missing required columns: {missing}')
for c in required: df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.loc[df.Age.between(0,17)&df.authorized.eq(1)&df[TARGET].isin([0,1])].copy()
if 'PatientID' in df and df.PatientID.dropna().duplicated().any(): raise ValueError('Duplicate PatientID')
for c,lo,hi in [('WeightKg',.3,250),('HeightIn',8,90)]:
    df.loc[df[c].notna()&~df[c].between(lo,hi),c]=np.nan
base=['Age','Gender','WeightKg','HeightIn']
clinical=['HR','SBP','DBP','PO2','PCO2','PH','O2SAT','FIO2','PEEP','Rate','TV','Hgb','Hct','Ptl','UOP','Output_Urine']
clinical=[c for c in clinical if c in df]
for c in clinical: df[c]=pd.to_numeric(df[c],errors='coerce')
moderate=[c for c in clinical if df[c].notna().mean()>=.40]
if not moderate: raise ValueError('No clinical variables meet 40% coverage threshold')
for c in moderate: df['available__'+c]=df[c].notna().astype(int)
avail=['available__'+c for c in moderate]
sets={'A_Baseline':base,'B_Clinical_Values_No_Indicators':base+moderate,
      'C_Availability_Only':base+avail,'D_Values_Plus_Availability':base+moderate+avail}
try:
    from xgboost import XGBClassifier
    algorithms=['LR','RF','XGB']
except ImportError:
    algorithms=['LR','RF']; warnings.warn('XGBoost unavailable; LR and RF only')
def make_pipe(cols,alg):
    prep=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=False,keep_empty_features=True)),('scale',StandardScaler())]),cols)])
    if alg=='LR': model=LogisticRegression(max_iter=3000,random_state=SEED)
    elif alg=='RF': model=RandomForestClassifier(n_estimators=350,min_samples_leaf=5,n_jobs=-1,random_state=SEED)
    else: model=XGBClassifier(n_estimators=250,max_depth=3,learning_rate=.04,subsample=.85,colsample_bytree=.85,eval_metric='logloss',random_state=SEED,n_jobs=1)
    return Pipeline([('prep',prep),('model',model)])
results=[]; importance=[]; oof={}
def run(frame,cohort):
    y=frame[TARGET].astype(int).to_numpy()
    if min(np.bincount(y,minlength=2))<N_SPLITS:
        warnings.warn(f'Skipping {cohort}: too few events/non-events'); return
    folds=list(StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED).split(frame,y))
    for setname,cols in sets.items():
        for alg in algorithms:
            pred=np.full(len(frame),np.nan); fold_col=np.zeros(len(frame),dtype=int)
            for k,(tr,te) in enumerate(folds,1):
                model=make_pipe(cols,alg); model.fit(frame.iloc[tr][cols],y[tr]); pred[te]=model.predict_proba(frame.iloc[te][cols])[:,1]; fold_col[te]=k
                if alg=='RF' and setname=='D_Values_Plus_Availability':
                    # Held-out permutation, not training-set importance.
                    pi=permutation_importance(model,frame.iloc[te][cols],y[te],scoring='roc_auc',n_repeats=5,random_state=SEED+k,n_jobs=1)
                    for c,m,s in zip(cols,pi.importances_mean,pi.importances_std): importance.append({'Cohort':cohort,'Fold':k,'Feature':c,'Mean_AUC_Decrease':m,'SD':s})
            assert np.isfinite(pred).all()
            results.append({'Cohort':cohort,'Feature_Set':setname,'Algorithm':alg,'N':len(y),'Events':int(y.sum()),'Prevalence':y.mean(),'ROC_AUC':roc_auc_score(y,pred),'Average_Precision':average_precision_score(y,pred),'Brier':brier_score_loss(y,pred),'Log_Loss':log_loss(y,pred,labels=[0,1])})
            oof[(cohort,setname,alg)]=(y,pred)
            pd.DataFrame({'source_row_index':frame.index,'fold':fold_col,'y_true':y,'oof_probability':pred}).to_csv(OUT/'Data'/f'{cohort}_{setname}_{alg}_OOF.csv',index=False)
run(df,'Authorized_All')
run(df.loc[df.brain_death.eq(1)].copy(),'BrainDeath_Documented')
# Selection analysis: complete enough on moderate-coverage features, explicitly not all-variable complete cases.
obs=df[moderate].notna().mean(axis=1)
selection=pd.DataFrame({'Coverage_Threshold':[0,.5,.7,.9,1], 'N':[int((obs>=t).sum()) for t in [0,.5,.7,.9,1]],'Events':[int(df.loc[obs>=t,TARGET].sum()) for t in [0,.5,.7,.9,1]]})
selection['Event_Rate']=selection['Events']/selection['N'].replace(0,np.nan)
selection.to_csv(OUT/'Tables'/'03_completeness_selection_counts.csv',index=False)
selected=df.loc[obs>=.7].copy()
if len(selected)>=100 and selected[TARGET].value_counts().min()>=N_SPLITS: run(selected,'AtLeast70Pct_Moderate_Clinical_Observed')
else: warnings.warn('70% completeness subset too small for 5-fold modeling; counts saved')
pd.DataFrame(results).to_csv(OUT/'Tables'/'01_model_comparison.csv',index=False)
if importance:
    pd.DataFrame(importance).to_csv(OUT/'Tables'/'04_foldwise_heldout_permutation_importance.csv',index=False)
    pd.DataFrame(importance).groupby(['Cohort','Feature'],as_index=False).agg(Mean_AUC_Decrease=('Mean_AUC_Decrease','mean'),SD_Across_Folds=('Mean_AUC_Decrease','std')).sort_values(['Cohort','Mean_AUC_Decrease'],ascending=[True,False]).to_csv(OUT/'Tables'/'05_permutation_importance_summary.csv',index=False)
# Paired bootstrap on fixed OOF predictions, same folds and individuals.
rng=np.random.default_rng(SEED); comparisons=[]
for (cohort,setname,alg),(y,p) in oof.items():
    if setname=='A_Baseline': continue
    y0,p0=oof[(cohort,'A_Baseline',alg)]
    if not np.array_equal(y,y0): raise RuntimeError('OOF labels misaligned')
    delta=[]
    for _ in range(1000):
        ix=rng.integers(0,len(y),len(y))
        if len(np.unique(y[ix]))<2: continue
        delta.append(roc_auc_score(y[ix],p[ix])-roc_auc_score(y[ix],p0[ix]))
    comparisons.append({'Cohort':cohort,'Algorithm':alg,'Feature_Set':setname,'Delta_ROC_AUC':roc_auc_score(y,p)-roc_auc_score(y,p0),'CI_Low':np.percentile(delta,2.5),'CI_High':np.percentile(delta,97.5)})
pd.DataFrame(comparisons).to_csv(OUT/'Tables'/'02_paired_auc_differences.csv',index=False)
pd.DataFrame([{'Feature':c,'Coverage':df[c].notna().mean(),'Transplanted_Coverage':df.loc[df[TARGET].eq(1),c].notna().mean(),'Nontransplanted_Coverage':df.loc[df[TARGET].eq(0),c].notna().mean()} for c in moderate]).to_csv(OUT/'Tables'/'06_moderate_feature_coverage.csv',index=False)
with open(OUT/'Diagnostics'/'phase7_notes.json','w') as f: json.dump({'N':len(df),'events':int(df[TARGET].sum()),'clinical_features':moderate,'feature_sets':sets,'folds':N_SPLITS,'seed':SEED,'warnings':['Retrospective association only; timing unknown','Values-only models use median imputation without explicit indicators, but missingness can still affect model behavior','Completeness-selected subgroup has selection bias and is sensitivity-only','Bootstrap confidence intervals condition on fixed OOF predictions','Permutation importance measures predictive reliance, not causality','No external validation or claim of clinical suitability']},f,indent=2)
print('Phase 7 complete:',OUT)
print(pd.DataFrame(results).to_string(index=False))
print('\nCompleteness subset sizes:\n',selection.to_string(index=False))
