from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
SOURCE = Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
OUT = ROOT / '03_Age_Adjustment'
for sub in ('Data','Tables','Figures','Diagnostics'):
    (OUT/sub).mkdir(parents=True,exist_ok=True)
TARGET='outcome_heart_Transplanted'

def numeric(s): return pd.to_numeric(s,errors='coerce')
def smd(a,b):
    a=numeric(a).dropna();b=numeric(b).dropna()
    if len(a)<2 or len(b)<2:return np.nan
    den=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
    return (a.mean()-b.mean())/den if den>0 else np.nan

def summary(df,col,by=TARGET):
    result={'Variable':col}
    for val,label in [(0,'Not_transplanted'),(1,'Transplanted')]:
        s=numeric(df.loc[df[by]==val,col]).dropna()
        result.update({f'{label}_N':len(s),f'{label}_Median':s.median(),f'{label}_Q1':s.quantile(.25),f'{label}_Q3':s.quantile(.75)})
    a=numeric(df.loc[df[by]==1,col]).dropna();b=numeric(df.loc[df[by]==0,col]).dropna()
    result['SMD']=smd(a,b)
    result['P_exploratory']=mannwhitneyu(a,b,alternative='two-sided').pvalue if len(a)>0 and len(b)>0 else np.nan
    return result

if not SOURCE.exists():raise FileNotFoundError(f'Original CSV not found: {SOURCE}')
df=pd.read_csv(SOURCE,low_memory=False)
needed=['Age','authorized',TARGET]
missing=[c for c in needed if c not in df]
if missing:raise ValueError(f'Required columns absent: {missing}')
for c in needed:df[c]=numeric(df[c])
df=df[df['Age'].ge(0)&df['Age'].lt(18)&df[TARGET].isin([0,1])].copy()
raw_n=len(df)
auth=df[df['authorized'].eq(1)].copy()
auth['Age_group']=pd.cut(auth['Age'],bins=[-0.01,0.999,4.999,11.999,17.999],labels=['<1','1-4','5-11','12-17'])

pd.DataFrame([{'All_pediatric_with_valid_age_outcome':raw_n,'Authorized_N':len(auth),'Heart_transplanted_N':int(auth[TARGET].eq(1).sum()),'Not_transplanted_N':int(auth[TARGET].eq(0).sum()),'Age_missing_excluded':int(df['Age'].isna().sum())}]).to_csv(OUT/'Tables'/'01_cohort_counts.csv',index=False)

age_table=pd.crosstab(auth['Age_group'],auth[TARGET]).reindex(columns=[0,1],fill_value=0)
age_table.columns=['Not_transplanted','Transplanted']
age_table['Total']=age_table.sum(axis=1)
age_table['Utilization_pct']=100*age_table['Transplanted']/age_table['Total']
age_table.to_csv(OUT/'Tables'/'02_age_group_counts.csv')

age_exact=pd.crosstab(auth['Age'],auth[TARGET]).reindex(columns=[0,1],fill_value=0)
age_exact.columns=['Not_transplanted','Transplanted']
age_exact['Total']=age_exact.sum(axis=1)
age_exact['Utilization_pct']=100*age_exact['Transplanted']/age_exact['Total']
age_exact.to_csv(OUT/'Tables'/'03_exact_age_counts.csv')

age_balance=pd.DataFrame([summary(auth,'Age')]+[dict(Age_group=str(g),**summary(part,'Age')) for g,part in auth.groupby('Age_group',observed=True)])
age_balance.to_csv(OUT/'Tables'/'04_age_balance.csv',index=False)

clinical=['WeightKg','HeightIn','HR','SBP','DBP','Creatinine','BUN','PO2','PCO2','FIO2','O2SAT','Lactate','TroponinI','TroponinT','BNP','TV','Rate','PIP','PEEP']
clinical=[c for c in clinical if c in auth.columns]
for c in clinical:auth[c]=numeric(auth[c])
rows=[];missing_rows=[]
for grp,sub in [('Overall',auth)]+[(str(g),p) for g,p in auth.groupby('Age_group',observed=True)]:
    for col in ['Age']+clinical:
        rows.append({'Age_group':grp,**summary(sub,col)})
        missing_rows.append({'Age_group':grp,'Variable':col,'Transplanted_N':int(sub[TARGET].eq(1).sum()),'Not_transplanted_N':int(sub[TARGET].eq(0).sum()),'Transplanted_missing_pct':100*sub.loc[sub[TARGET].eq(1),col].isna().mean(),'Not_transplanted_missing_pct':100*sub.loc[sub[TARGET].eq(0),col].isna().mean()})
res=pd.DataFrame(rows)
res['FDR_within_age_group']=np.nan
for g,idx in res.groupby('Age_group').groups.items():
    valid=res.loc[idx,'P_exploratory'].notna()
    if valid.any():res.loc[res.loc[idx].index[valid],'FDR_within_age_group']=multipletests(res.loc[res.loc[idx].index[valid],'P_exploratory'],method='fdr_bh')[1]
res.to_csv(OUT/'Tables'/'05_age_stratified_clinical_comparisons.csv',index=False)
pd.DataFrame(missing_rows).to_csv(OUT/'Tables'/'06_age_stratified_missingness.csv',index=False)

# Separate descriptive diagnostic, not prospective model features.
flags=['brain_death','procured','approached','transplanted']
flag_rows=[]
for c in flags:
    if c not in auth:continue
    for group,sub in [('Overall',auth)]+[(str(g),p) for g,p in auth.groupby('Age_group',observed=True)]:
        for y in (0,1):
            vals=numeric(sub.loc[sub[TARGET].eq(y),c])
            flag_rows.append({'Variable':c,'Age_group':group,'Heart_transplanted':y,'N':len(vals),'Observed_N':int(vals.notna().sum()),'Yes_N':int(vals.eq(1).sum()),'Missing_N':int(vals.isna().sum())})
pd.DataFrame(flag_rows).to_csv(OUT/'Diagnostics'/'07_donation_stage_by_age.csv',index=False)

fig,ax=plt.subplots(figsize=(8,4.5))
xs=np.arange(len(age_table));w=.36
ax.bar(xs-w/2,age_table['Not_transplanted'],w,label='Not transplanted')
ax.bar(xs+w/2,age_table['Transplanted'],w,label='Transplanted')
ax.set_xticks(xs,age_table.index.astype(str));ax.set_xlabel('Age group (years)');ax.set_ylabel('Donors');ax.legend();fig.tight_layout()
fig.savefig(OUT/'Figures'/'age_group_counts.png',dpi=250);plt.close(fig)

with open(OUT/'Diagnostics'/'README.txt','w',encoding='utf8') as f:
    f.write('Phase 3 descriptive audit only. No imputation or ML.\n')
    f.write('SMD = transplanted minus non-transplanted standardized mean difference.\n')
    f.write('P values exploratory; missingness denominators include all records in each outcome group.\n')
    f.write('Age coded in integer years; age zero includes all infants under one year.\n')
    f.write('Clinical variable units and predecision timing are NOT verified.\n')
    f.write('Donation process flags are descriptive only, not eligible predictors.\n')
print('Phase 3 complete:',OUT,'Authorized donors:',len(auth))
