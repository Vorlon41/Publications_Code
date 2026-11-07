from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
ROOT = Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
OUT = ROOT / '04_Feature_Engineering'
for sub in ['Data','Tables','Diagnostics']:
    (OUT/sub).mkdir(parents=True, exist_ok=True)
TARGET='outcome_heart_Transplanted'
raw=pd.read_csv(SOURCE,low_memory=False)
required=['Age','authorized',TARGET]
missing=[c for c in required if c not in raw]
if missing: raise ValueError(f'Missing required columns: {missing}')
raw['Age']=pd.to_numeric(raw['Age'],errors='coerce')
raw['authorized']=pd.to_numeric(raw['authorized'],errors='coerce')
raw[TARGET]=pd.to_numeric(raw[TARGET],errors='coerce')
df=raw.loc[raw['Age'].ge(0)&raw['Age'].lt(18)&raw['authorized'].eq(1)&raw[TARGET].isin([0,1])].copy()
print('Authorized pediatric donors:',len(df),'utilized:',int(df[TARGET].sum()))

# Explicit, conservative classifications: never automatically approve all 392 columns.
baseline={'Age','Gender','WeightKg','HeightIn'}
context={'OPO','Referral_Year','brain_death','time_brain_death','authorized','time_authorized','approached','time_approached','time_referred'}
clinical={'HR','SBP','DBP','Temp','PH','PCO2','PO2','HCO3','BE','FIO2','O2SAT','PIP','PEEP','TV','Rate','Creatinine','BUN','Lactate','TroponinI','TroponinT','BNP','CKMB','Hct','Hgb','Ptl','INR','PT','PTT','UOP','Output_Urine'}
rows=[]
for c in raw.columns:
    lc=c.lower()
    if c==TARGET or lc.startswith('outcome_') or c in {'PatientID','HospitalID','Procured_Year','procured','transplanted','time_procured'} or 'transplant' in lc or 'recovered' in lc:
        role='EXCLUDE_outcome_identifier_or_postdecision'
    elif c in context:
        role='CONTEXT_only_not_predictor'
    elif c in baseline:
        role='BASELINE_candidate_verify_timing'
    elif c in clinical:
        role='CLINICAL_conditional_verify_timing_units'
    else:
        role='HOLD_manual_dictionary_review'
    s=df[c]
    rows.append(dict(Variable=c,Role=role,Dtype=str(s.dtype),N_observed=int(s.notna().sum()),N_missing=int(s.isna().sum()),Missing_pct=round(100*s.isna().mean(),2),Unique_observed=int(s.nunique(dropna=True)),Timing_verified=False,Dictionary_verified=False))
inv=pd.DataFrame(rows).sort_values(['Role','Missing_pct','Variable'])
inv.to_csv(OUT/'Tables'/'01_feature_eligibility_inventory.csv',index=False)

# Missingness separately by outcome, no imputation.
cover=[]
for c in raw.columns:
    for outcome in (0,1):
        x=df.loc[df[TARGET].eq(outcome),c]
        cover.append(dict(Variable=c,Heart_transplanted=outcome,N=len(x),Observed=int(x.notna().sum()),Missing_pct=round(100*x.isna().mean(),2)))
pd.DataFrame(cover).to_csv(OUT/'Tables'/'02_coverage_by_outcome.csv',index=False)

# Candidate baseline characteristics, including missing categories; no clinical claims.
base_cols=[c for c in ['Age','Gender','WeightKg','HeightIn','Referral_Year','brain_death','OPO'] if c in df]
summary=[]
for c in base_cols:
    for outcome in (0,1):
        s=df.loc[df[TARGET].eq(outcome),c]
        num=pd.to_numeric(s,errors='coerce')
        summary.append(dict(Variable=c,Outcome=outcome,N=len(s),Observed=int(s.notna().sum()),Median=float(num.median()) if num.notna().any() else np.nan,Q1=float(num.quantile(.25)) if num.notna().any() else np.nan,Q3=float(num.quantile(.75)) if num.notna().any() else np.nan,Unique=int(s.nunique(dropna=True))))
pd.DataFrame(summary).to_csv(OUT/'Tables'/'03_baseline_summary.csv',index=False)

# Clinical measurement coverage by age and outcome.
df['Age_group']=pd.cut(df.Age,bins=[-0.001,0.999,4.999,11.999,17.999],labels=['<1','1-4','5-11','12-17'])
clinical_rows=[]
for c in sorted(clinical.intersection(df.columns)):
    for (age,outcome),g in df.groupby(['Age_group',TARGET],observed=True):
        clinical_rows.append(dict(Variable=c,Age_group=str(age),Outcome=int(outcome),N=len(g),Observed=int(g[c].notna().sum()),Missing_pct=round(100*g[c].isna().mean(),2)))
pd.DataFrame(clinical_rows).to_csv(OUT/'Tables'/'04_clinical_coverage_by_age_outcome.csv',index=False)

# Plausibility screening only: do not automatically overwrite values.
limits={'Age':(0,17),'WeightKg':(0.3,300),'HeightIn':(8,100),'HR':(20,300),'SBP':(20,250),'DBP':(5,180),'O2SAT':(0,100),'FIO2':(0,100),'PH':(6.5,8),'PO2':(0,800)}
flags=[]
for c,(lo,hi) in limits.items():
    if c not in df: continue
    s=pd.to_numeric(df[c],errors='coerce')
    flags.append(dict(Variable=c,Lower_screen=lo,Upper_screen=hi,Numeric_N=int(s.notna().sum()),Outside_screen_N=int(((s<lo)|(s>hi)).sum()),Observed_min=s.min(),Observed_max=s.max(),Note='Screening thresholds only; verify units and source definitions'))
pd.DataFrame(flags).to_csv(OUT/'Diagnostics'/'05_plausibility_flags.csv',index=False)

# Model readiness: only an audit, not a claim of prospective availability.
readiness=pd.DataFrame([{'Check':'Prediction time defined and documented','Status':'PENDING'}, {'Check':'Authorization semantics verified','Status':'PENDING'}, {'Check':'Brain-death/DCD pathway verified','Status':'PENDING'}, {'Check':'Clinical measurement timestamps verified','Status':'PENDING'}, {'Check':'Donor identifiers and repeated records audited','Status':'PENDING'}, {'Check':'No postdecision features in model','Status':'REQUIRED'}, {'Check':'OPO grouping strategy for validation','Status':'PENDING'}, {'Check':'Age and baseline variable unit validation','Status':'PENDING'}])
readiness.to_csv(OUT/'Diagnostics'/'06_model_readiness_checklist.csv',index=False)
with open(OUT/'Diagnostics'/'README.txt','w') as f:
    f.write('Phase 4 is a feature-eligibility audit only. No models, imputation, or patient-level exports.\nAll unverified columns are held from predictive modeling until adjudicated.\n')
print('Phase 4 audit complete. Results:',OUT)
