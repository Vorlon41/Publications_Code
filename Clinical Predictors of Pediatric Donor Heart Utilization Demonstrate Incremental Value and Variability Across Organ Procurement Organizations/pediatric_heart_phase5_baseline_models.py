from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, accuracy_score, roc_curve, precision_recall_curve
from sklearn.base import clone

INPUT = Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
ROOT = Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
OUT = ROOT / '05_Machine_Learning'
for part in ('Data','Tables','Figures','Diagnostics'):
    (OUT/part).mkdir(parents=True, exist_ok=True)
TARGET = 'outcome_heart_Transplanted'
SEED = 42
N_SPLITS = 5

if not INPUT.exists():
    raise FileNotFoundError(f'Original donor CSV not found: {INPUT}')
df = pd.read_csv(INPUT, low_memory=False)
required = ['Age','authorized',TARGET,'Gender','WeightKg','HeightIn']
missing = [c for c in required if c not in df]
if missing: raise ValueError(f'Missing required source columns: {missing}')
for c in required + [c for c in ('Referral_Year','OPO','brain_death','PatientID') if c in df]:
    if c != 'OPO': df[c] = pd.to_numeric(df[c],errors='coerce')
# Do not silently treat missing authorization or target as zero.
df = df.loc[(df['Age'] >= 0)&(df['Age'] < 18)&(df['authorized'] == 1)&(df[TARGET].isin([0,1]))].copy()
df[TARGET] = df[TARGET].astype(int)
if 'PatientID' in df and df['PatientID'].notna().any() and df.loc[df['PatientID'].notna(),'PatientID'].duplicated().any():
    raise ValueError('Repeated PatientID values: donor-level CV unsafe; resolve duplicates first.')
if len(df) != 819 or df[TARGET].sum() != 441:
    warnings.warn(f'Cohort differs from Phase 4: n={len(df)}, positives={df[TARGET].sum()}. Investigate before publication.')

# A documented audit gate: never use outcome/process fields in primary baseline.
feature_sets = {'A_Demographics':['Age','Gender','WeightKg','HeightIn']}
if 'Referral_Year' in df:
    feature_sets['B_Plus_Referral_Year'] = feature_sets['A_Demographics']+['Referral_Year']
if 'OPO' in df and 'Referral_Year' in df:
    feature_sets['C_Plus_OPO_Exploratory'] = feature_sets['B_Plus_Referral_Year']+['OPO']

# Prespecified broad plausibility bounds: flagged as missing, not whole-row deletion.
# These are data hygiene bounds, NOT pediatric reference ranges.
bounds={'Age':(0,17),'WeightKg':(0.3,250),'HeightIn':(8,90)}
flag_rows=[]
for col,(lo,hi) in bounds.items():
    bad=df[col].notna() & ~df[col].between(lo,hi)
    flag_rows.append({'Feature':col,'Lower_bound':lo,'Upper_bound':hi,'Flagged_n':int(bad.sum())})
    df.loc[bad,col]=np.nan
pd.DataFrame(flag_rows).to_csv(OUT/'Diagnostics'/'plausibility_flag_counts.csv',index=False)

try:
    from xgboost import XGBClassifier
    xgb_available=True
except ImportError:
    xgb_available=False
    warnings.warn('xgboost unavailable: LR and RF will run; install xgboost for XGB.')

def make_pipe(cols, algorithm):
    numeric=[c for c in cols if c!='OPO']
    categorical=[c for c in cols if c=='OPO']
    transformers=[]
    if numeric:
        transformers.append(('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),numeric))
    if categorical:
        transformers.append(('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),categorical))
    prep=ColumnTransformer(transformers)
    if algorithm=='LR': clf=LogisticRegression(max_iter=3000,random_state=SEED)
    elif algorithm=='RF': clf=RandomForestClassifier(n_estimators=350,min_samples_leaf=5,random_state=SEED,n_jobs=-1)
    else: clf=XGBClassifier(n_estimators=250,max_depth=3,learning_rate=.04,subsample=.85,colsample_bytree=.85,eval_metric='logloss',random_state=SEED,n_jobs=1)
    return Pipeline([('preprocess',prep),('classifier',clf)])

def evaluate(data, label):
    y=data[TARGET].to_numpy(dtype=int)
    if min(np.bincount(y,minlength=2)) < N_SPLITS:
        return [{'Cohort':label,'Status':'INSUFFICIENT_POSITIVES_OR_NEGATIVES'}]
    splits=list(StratifiedKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED).split(np.zeros(len(y)),y))
    rows=[]
    for fs,cols in feature_sets.items():
        for alg in ['LR','RF']+(['XGB'] if xgb_available else []):
            X=data[cols].copy()
            if 'OPO' in X: X['OPO']=X['OPO'].astype('string').fillna('Missing').astype(str)
            oof=np.full(len(y),np.nan)
            for fold,(train,test) in enumerate(splits,1):
                pipe=make_pipe(cols,alg)
                pipe.fit(X.iloc[train],y[train])
                oof[test]=pipe.predict_proba(X.iloc[test])[:,1]
            assert np.isfinite(oof).all()
            metrics={'Cohort':label,'Feature_Set':fs,'Algorithm':alg,'N':len(y),'Positive_N':int(y.sum()),'Prevalence':float(y.mean()),
                     'ROC_AUC':roc_auc_score(y,oof),'Average_Precision':average_precision_score(y,oof),
                     'Brier':brier_score_loss(y,oof),'Log_Loss':log_loss(y,oof,labels=[0,1]),'Status':'OK'}
            rows.append(metrics)
            pd.DataFrame({'source_row_index':data.index,'observed_heart_transplant':y,'oof_probability':oof}).to_csv(OUT/'Data'/f'{label}_{fs}_{alg}_OOF.csv',index=False)
            fpr,tpr,_=roc_curve(y,oof)
            plt.figure(figsize=(6,5));plt.plot(fpr,tpr,label=f'AUC={metrics["ROC_AUC"]:.3f}');plt.plot([0,1],[0,1],'--',color='gray');plt.xlabel('False positive rate');plt.ylabel('True positive rate');plt.title(f'{label}: {fs} / {alg}');plt.legend();plt.tight_layout();plt.savefig(OUT/'Figures'/f'{label}_{fs}_{alg}_ROC.png',dpi=200);plt.close()
    return rows

results=evaluate(df,'Authorized_All')
if 'brain_death' in df:
    bd=df.loc[df['brain_death']==1].copy()
    results+=evaluate(bd,'Authorized_BrainDeath_Documented')
    pd.DataFrame({'brain_death_recorded':df['brain_death'].fillna(-1).astype(str),'heart_transplanted':df[TARGET]}).groupby(['brain_death_recorded','heart_transplanted']).size().reset_index(name='N').to_csv(OUT/'Tables'/'brain_death_status_by_outcome.csv',index=False)
summary=pd.DataFrame(results)
summary.to_csv(OUT/'Tables'/'baseline_model_comparison.csv',index=False)
cohort=pd.DataFrame([{'Cohort':'Authorized_All','N':len(df),'Heart_Transplanted_N':int(df[TARGET].sum()),'Heart_Not_Transplanted_N':int((df[TARGET]==0).sum())}])
cohort.to_csv(OUT/'Tables'/'cohort_accounting.csv',index=False)
with open(OUT/'Diagnostics'/'analysis_notes.json','w') as f:
    json.dump({'input':str(INPUT),'output':str(OUT),'random_seed':SEED,'folds':N_SPLITS,'feature_sets':feature_sets,
               'prediction_claim':'retrospective classification only; feature timing unverified',
               'model_selection':'no hyperparameter tuning or threshold optimization; results are exploratory',
               'brain_death':'sensitivity subset only; not included as a primary predictor',
               'caution':'OPO model is exploratory and random-fold validation does not assess unseen-OPO transportability'},f,indent=2)
print('\nPhase 5 complete. Results:',OUT/'Tables'/'baseline_model_comparison.csv')
print(summary.to_string(index=False))
