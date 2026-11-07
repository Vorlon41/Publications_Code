
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact

SOURCE = Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
ROOT = Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
OUT = ROOT / '09_Post_Procurement_Analysis'
TABLES, DIAG, FIGS = [OUT / s for s in ('Tables','Diagnostics','Figures')]
for p in (TABLES, DIAG, FIGS): p.mkdir(parents=True, exist_ok=True)
TARGET = 'outcome_heart_Transplanted'
RECOVERY = ['outcome_heart_Recovered for Research', 'outcome_heart_Recovered for Transplant but not Transplanted']
if not SOURCE.exists():
    raise FileNotFoundError(f'Input file missing: {SOURCE}; edit SOURCE at top of script.')
df = pd.read_csv(SOURCE, low_memory=False)
needed = ['Age','authorized','procured',TARGET]
if any(c not in df for c in needed):
    raise ValueError(f'Missing required columns: {[c for c in needed if c not in df]}')
for c in needed + RECOVERY + ['brain_death','Referral_Year','WeightKg','HeightIn']:
    if c in df: df[c] = pd.to_numeric(df[c],errors='coerce')
ped = df.loc[df.Age.ge(0) & df.Age.lt(18) & df[TARGET].isin([0,1])].copy()
proc = ped.loc[ped.procured.eq(1)].copy()
proc['Age_group'] = pd.cut(proc.Age,[-.001,.999,4.999,11.999,17.999],labels=['<1','1-4','5-11','12-17'])

# Stage integrity and accounting, with explicit unknown values.
rows=[]
for name,sub in [('Pediatric all',ped),('Authorized',ped.loc[ped.authorized.eq(1)]),('Procured',proc),('Procured and authorized',proc.loc[proc.authorized.eq(1)])]:
    rows.append(dict(Cohort=name,N=len(sub),Heart_transplanted=int(sub[TARGET].eq(1).sum()),Heart_not_transplanted=int(sub[TARGET].eq(0).sum()),Brain_death_documented=int(sub.get('brain_death',pd.Series(dtype=float)).eq(1).sum())))
pd.DataFrame(rows).to_csv(TABLES/'01_cohort_accounting.csv',index=False)
stage_cols = [c for c in ['authorized','brain_death','procured',TARGET] if c in ped]
ped.groupby(stage_cols,dropna=False).size().reset_index(name='N').to_csv(DIAG/'01_stage_integrity.csv',index=False)

# Heart-specific outcomes; retain missing as missing, not as zero.
heart_fields=[c for c in proc if c.startswith('outcome_heart_')]
if heart_fields:
    proc.groupby(heart_fields,dropna=False).size().reset_index(name='N').to_csv(TABLES/'02_heart_outcome_combinations.csv',index=False)
    rows=[]
    for c in heart_fields:
        for outcome in [0,1]:
            s=proc.loc[proc[TARGET].eq(outcome),c]
            rows.append(dict(Variable=c,Heart_transplanted=outcome,N=len(s),Yes=int(s.eq(1).sum()),No=int(s.eq(0).sum()),Missing=int(s.isna().sum())))
    pd.DataFrame(rows).to_csv(TABLES/'03_heart_outcome_field_inventory.csv',index=False)

# Utilization by age, year, OPO and donation-stage indicators.
for v in ['Age_group','Age','Referral_Year','OPO','brain_death','authorized']:
    if v not in proc: continue
    tab=proc.groupby([v,TARGET],dropna=False,observed=True).size().unstack(TARGET,fill_value=0).reindex(columns=[0,1],fill_value=0)
    tab.columns=['Not_transplanted','Heart_transplanted']
    tab['Total']=tab.sum(axis=1)
    tab['Heart_utilization_pct']=100*tab['Heart_transplanted']/tab['Total'].replace(0,np.nan)
    tab.reset_index().to_csv(TABLES/f'04_utilization_by_{v}.csv',index=False)

# Raw observed-data summaries, without imputation or clinical significance claims.
clinical=['Age','WeightKg','HeightIn','HR','SBP','DBP','PO2','PCO2','PH','O2SAT','FIO2','PEEP','Rate','TV','Hgb','Hct','Ptl','UOP','Output_Urine','Creatinine','BUN','Lactate','TroponinI','TroponinT','BNP']
continuous=[]
for v in clinical:
    if v not in proc: continue
    a=pd.to_numeric(proc.loc[proc[TARGET].eq(1),v],errors='coerce').dropna()
    b=pd.to_numeric(proc.loc[proc[TARGET].eq(0),v],errors='coerce').dropna()
    pooled=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2) if len(a)>1 and len(b)>1 else np.nan
    p=mannwhitneyu(a,b,alternative='two-sided').pvalue if len(a)>=3 and len(b)>=3 else np.nan
    continuous.append(dict(Variable=v,N_tx=len(a),Median_tx=a.median(),Q1_tx=a.quantile(.25),Q3_tx=a.quantile(.75),N_no_tx=len(b),Median_no_tx=b.median(),Q1_no_tx=b.quantile(.25),Q3_no_tx=b.quantile(.75),SMD=(a.mean()-b.mean())/pooled if pd.notna(pooled) and pooled>0 else np.nan,Exploratory_p=p))
pd.DataFrame(continuous).to_csv(TABLES/'05_clinical_observed_comparison.csv',index=False)

missing=[]
for v in proc.columns:
    if v in ['PatientID','HospitalID']: continue
    for outcome in [0,1]:
        s=proc.loc[proc[TARGET].eq(outcome),v]
        missing.append(dict(Variable=v,Heart_transplanted=outcome,N=len(s),Observed=int(s.notna().sum()),Missing=int(s.isna().sum()),Missing_pct=100*s.isna().mean() if len(s) else np.nan))
pd.DataFrame(missing).to_csv(TABLES/'06_missingness_by_outcome.csv',index=False)

# Report completeness within each age group and outcome, without inferring evaluation was performed.
rows=[]
for v in clinical:
    if v not in proc: continue
    for (age,outcome),sub in proc.groupby(['Age_group',TARGET],observed=True):
        s=pd.to_numeric(sub[v],errors='coerce')
        rows.append(dict(Variable=v,Age_group=str(age),Heart_transplanted=int(outcome),N=len(s),N_observed=int(s.notna().sum()),Observed_pct=100*s.notna().mean()))
pd.DataFrame(rows).to_csv(TABLES/'07_clinical_coverage_by_age.csv',index=False)

# Descriptive completeness based on a fixed list of clinical variables, not an evaluation score.
selected=[v for v in clinical if v in proc and v not in ['Age','WeightKg','HeightIn']]
proc['Clinical_fields_observed']=proc[selected].notna().sum(axis=1) if selected else 0
proc['Clinical_fields_observed_pct']=100*proc['Clinical_fields_observed']/len(selected) if selected else np.nan
proc.groupby(TARGET).agg(N=('Age','size'),Median_fields=('Clinical_fields_observed','median'),Median_pct=('Clinical_fields_observed_pct','median'),Mean_pct=('Clinical_fields_observed_pct','mean')).reset_index().to_csv(TABLES/'08_clinical_field_completeness.csv',index=False)

notes={'source':str(SOURCE),'output':str(OUT),'pediatric_n':len(ped),'procured_n':len(proc),'heart_transplanted_n':int(proc[TARGET].eq(1).sum()),'heart_not_transplanted_n':int(proc[TARGET].eq(0).sum()),'clinical_variables_counted':selected,'limitations':['Procured may refer to any organ; verify ORCHID dictionary.','Heart not transplanted is not synonymous with declined, unsuitable, or discarded.','Missing measurements do not prove clinical tests were not performed.','No prediction models, no imputation, no causal claims.','Source field timing and units remain unverified.']}
(DIAG/'phase10_notes.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
print('Phase 10 complete:',OUT)
print(pd.DataFrame(rows[:0])) if False else None
print('Procured donors:',len(proc),'| Hearts transplanted:',int(proc[TARGET].eq(1).sum()),'| Not transplanted:',int(proc[TARGET].eq(0).sum()))
