from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.stats.multitest import multipletests

SOURCE = Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
ROOT = Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
OUT = ROOT / '09_Post_Procurement_Analysis' / 'Brain_Death_Subgroup' / 'Phase15_Categorical_Audit'
TABLES = OUT / 'Tables'; DIAG = OUT / 'Diagnostics'
for p in (TABLES, DIAG): p.mkdir(parents=True, exist_ok=True)
TARGET='outcome_heart_Transplanted'

def binary_indicator(s):
    z=pd.to_numeric(s,errors='coerce')
    return z.notna().any() and set(z.dropna().unique()).issubset({0,1})

def family_of(c):
    for prefix in ('Cause_of_Death_OPO_','Cause_of_Death_UNOS_','Mechanism_of_Death_','Circumstances_of_Death_','ABO_BloodType_'):
        if c.startswith(prefix): return prefix.rstrip('_')
    if c.startswith('outcome_') or any(t in c.lower() for t in ('procure','transplant','recover','authorized','approached','time_')): return 'PROCESS_OR_OUTCOME'
    return 'Other'

raw=pd.read_csv(SOURCE,low_memory=False)
for c in ('Age',TARGET,'procured','brain_death','authorized'):
    if c not in raw: raise ValueError(f'Missing required field: {c}')
    raw[c]=pd.to_numeric(raw[c],errors='coerce')
cohorts={
    'Procured_BrainDeath':raw[(raw.Age<18)&(raw.procured==1)&(raw.brain_death==1)&raw[TARGET].isin([0,1])].copy(),
    'Authorized_All':raw[(raw.Age<18)&(raw.authorized==1)&raw[TARGET].isin([0,1])].copy()
}
rows=[]; family_rows=[]; checks=[]
for cohort_name,df in cohorts.items():
    n1=int((df[TARGET]==1).sum()); n0=int((df[TARGET]==0).sum())
    print(cohort_name, 'N=',len(df),'Tx=',n1,'NoTx=',n0)
    for col in df.columns:
        fam=family_of(col)
        if fam=='PROCESS_OR_OUTCOME' or col in (TARGET,'PatientID','HospitalID','OPO','Referral_Year','Procured_Year','Age','WeightKg','HeightIn'):continue
        s=df[col]; is_bin=binary_indicator(s)
        if not is_bin and (pd.api.types.is_numeric_dtype(s) or s.nunique(dropna=True)>30):continue
        tx=s[df[TARGET]==1]; nt=s[df[TARGET]==0]
        a=int(tx.notna().sum()); b=int(nt.notna().sum())
        if not n1 or not n0:continue
        coverage=min(a/n1,b/n0)
        if coverage<.70:continue
        if is_bin:
            v1=pd.to_numeric(tx,errors='coerce');v0=pd.to_numeric(nt,errors='coerce')
            pos1=int((v1==1).sum());pos0=int((v0==1).sum())
            p=fisher_exact([[pos1,a-pos1],[pos0,b-pos0]])[1] if a and b else np.nan
            rows.append(dict(Cohort=cohort_name,Family=fam,Variable=col,Type='binary_indicator',N_Tx=n1,N_NoTx=n0,Observed_Tx=a,Observed_NoTx=b,Coverage_Tx=a/n1,Coverage_NoTx=b/n0,Positive_Tx=pos1,Positive_NoTx=pos0,Positive_Pct_Tx=pos1/a if a else np.nan,Positive_Pct_NoTx=pos0/b if b else np.nan,P=p,Pass80Both=coverage>=.80,Interpretation='Indicator 0 may mean not selected/unknown; source encoding unverified'))
        else:
            tab=pd.crosstab(s.fillna('<<MISSING>>').astype(str),df[TARGET]);p=np.nan
            if tab.shape[0]>1 and tab.shape[1]==2:
                try:p=chi2_contingency(tab)[1]
                except ValueError:pass
            rows.append(dict(Cohort=cohort_name,Family=fam,Variable=col,Type='categorical',N_Tx=n1,N_NoTx=n0,Observed_Tx=a,Observed_NoTx=b,Coverage_Tx=a/n1,Coverage_NoTx=b/n0,Positive_Tx=np.nan,Positive_NoTx=np.nan,Positive_Pct_Tx=np.nan,Positive_Pct_NoTx=np.nan,P=p,Pass80Both=coverage>=.80,Interpretation='Inspect levels and missing category before interpretation'))
    for fam in ('Cause_of_Death_OPO','Cause_of_Death_UNOS','Mechanism_of_Death','Circumstances_of_Death','ABO_BloodType'):
        cols=[c for c in df if family_of(c)==fam and binary_indicator(df[c])]
        if not cols:continue
        z=df[cols].apply(pd.to_numeric,errors='coerce')
        count=(z==1).sum(axis=1)
        for status in (0,1):
            k=count[df[TARGET]==status]
            family_rows.append(dict(Cohort=cohort_name,Family=fam,HeartTransplanted=status,N=len(k),Fields=len(cols),ZeroPositive=int((k==0).sum()),ExactlyOnePositive=int((k==1).sum()),MultiplePositive=int((k>1).sum()),AnyMissingField=int(z.loc[k.index].isna().any(axis=1).sum()),Interpretation='Zero positives is not necessarily negative/unknown; multiple positives may be legitimate or encoding artifacts'))
        if fam=='ABO_BloodType':checks.append(dict(Cohort=cohort_name,Family=fam,Warning='ABO one-hot family includes subtype labels; do not infer exclusive ABO type from all columns without dictionary'))
result=pd.DataFrame(rows)
if not result.empty:
    result['FDR_within_cohort_family']=np.nan
    for (_, _),ix in result.groupby(['Cohort','Family']).groups.items():
        valid=result.loc[ix,'P'].notna(); idx=result.loc[ix].index[valid]
        if len(idx):result.loc[idx,'FDR_within_cohort_family']=multipletests(result.loc[idx,'P'],method='fdr_bh')[1]
    result.sort_values(['Cohort','Family','P'],na_position='last').to_csv(TABLES/'01_categorical_variable_screen.csv',index=False)
    result[result.Pass80Both].to_csv(TABLES/'02_ge80_both_groups.csv',index=False)
    result[(result.Pass80Both)&(result.FDR_within_cohort_family<.05)].to_csv(TABLES/'03_exploratory_fdr_signals.csv',index=False)
pd.DataFrame(family_rows).to_csv(TABLES/'04_indicator_family_integrity.csv',index=False)
pd.DataFrame(checks).to_csv(DIAG/'05_dictionary_checks.csv',index=False)
(DIAG/'README.txt').write_text('Exploratory categorical audit. No model fitting. Do not interpret one-hot zero as confirmed absence. Cause-of-death labels are not harmonized. FDR is within cohort and heuristic family, not across all tests. Review original ORCHID data dictionary and measurement timing before clinical claims.\n',encoding='utf-8')
print('Saved to:',OUT)
