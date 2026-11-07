from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE=Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
ROOT=Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
OUT=ROOT/'09_Post_Procurement_Analysis'/'Brain_Death_Subgroup'
TABLES=OUT/'Tables'; FIGS=OUT/'Figures'; DIAG=OUT/'Diagnostics'
for p in (TABLES,FIGS,DIAG): p.mkdir(parents=True,exist_ok=True)
TARGET='outcome_heart_Transplanted'
CLINICAL=['HR','SBP','DBP','PO2','PCO2','PH','O2SAT','FIO2','PIP','PEEP','Rate','TV','Hgb','Hct','Ptl','UOP','Output_Urine','Creatinine','BUN','Lactate','TroponinI','TroponinT','BNP']
if not SOURCE.exists(): raise FileNotFoundError(f'Edit SOURCE: {SOURCE}')
df=pd.read_csv(SOURCE,low_memory=False)
required=['Age','procured','brain_death',TARGET]
missing=[c for c in required if c not in df]
if missing: raise ValueError(f'Missing required columns: {missing}')
for c in required+['WeightKg','HeightIn','Referral_Year']+CLINICAL:
    if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
cohort=df.loc[df.Age.ge(0)&df.Age.lt(18)&df.procured.eq(1)&df.brain_death.eq(1)&df[TARGET].isin([0,1])].copy()
cohort['Age_group']=pd.cut(cohort.Age,[-.001,.999,4.999,11.999,17.999],labels=['<1','1-4','5-11','12-17'])
cohort['Heart_transplanted']=cohort[TARGET].astype(int)
cohort['Clinical_fields_observed']=cohort[[c for c in CLINICAL if c in cohort]].notna().sum(axis=1)
cohort['Clinical_fields_total']=len([c for c in CLINICAL if c in cohort])
cohort['Clinical_fields_observed_pct']=100*cohort['Clinical_fields_observed']/cohort['Clinical_fields_total'] if len([c for c in CLINICAL if c in cohort]) else np.nan

pd.DataFrame([{'Cohort':'Procured + documented brain death','N':len(cohort),'Transplanted':int(cohort.Heart_transplanted.sum()),'Not_transplanted':int((cohort.Heart_transplanted==0).sum()),'Observed_clinical_fields':len([c for c in CLINICAL if c in cohort])}]).to_csv(TABLES/'01_cohort_accounting.csv',index=False)

for col in ['Age_group','Age','OPO','Referral_Year']:
    if col not in cohort: continue
    t=cohort.groupby([col,'Heart_transplanted'],dropna=False,observed=True).size().unstack('Heart_transplanted',fill_value=0).reindex(columns=[0,1],fill_value=0)
    t.columns=['Not_transplanted','Transplanted']; t['N']=t.sum(axis=1)
    t['Utilization_pct']=100*t['Transplanted']/t['N'].replace(0,np.nan)
    t.reset_index().to_csv(TABLES/f'02_utilization_by_{col}.csv',index=False)

rows=[]
for col in ['Age','WeightKg','HeightIn']+[c for c in CLINICAL if c in cohort]:
    a=cohort.loc[cohort.Heart_transplanted.eq(1),col].dropna()
    b=cohort.loc[cohort.Heart_transplanted.eq(0),col].dropna()
    sd=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2) if min(len(a),len(b))>1 else np.nan
    p=mannwhitneyu(a,b,alternative='two-sided').pvalue if min(len(a),len(b))>=3 else np.nan
    rows.append({'Variable':col,'Tx_N_observed':len(a),'Tx_median':a.median(),'Tx_Q1':a.quantile(.25),'Tx_Q3':a.quantile(.75),'NoTx_N_observed':len(b),'NoTx_median':b.median(),'NoTx_Q1':b.quantile(.25),'NoTx_Q3':b.quantile(.75),'SMD':(a.mean()-b.mean())/sd if pd.notna(sd) and sd>0 else np.nan,'Exploratory_p':p})
summary=pd.DataFrame(rows)
summary['FDR_BH']=np.nan
mask=summary.Exploratory_p.notna()
if mask.any(): summary.loc[mask,'FDR_BH']=multipletests(summary.loc[mask,'Exploratory_p'],method='fdr_bh')[1]
summary.to_csv(TABLES/'03_observed_clinical_comparison.csv',index=False)

rows=[]
for col in ['Age','WeightKg','HeightIn']+[c for c in CLINICAL if c in cohort]:
    for group,sub in cohort.groupby('Heart_transplanted'):
        s=sub[col]
        rows.append({'Variable':col,'Heart_transplanted':group,'N':len(sub),'Observed':int(s.notna().sum()),'Missing':int(s.isna().sum()),'Missing_pct':100*s.isna().mean()})
miss=pd.DataFrame(rows); miss.to_csv(TABLES/'04_missingness_by_outcome.csv',index=False)

rows=[]
for (age,y),sub in cohort.groupby(['Age_group','Heart_transplanted'],observed=True):
    rows.append({'Age_group':str(age),'Heart_transplanted':y,'N':len(sub),'Median_observed_fields':sub.Clinical_fields_observed.median(),'Mean_observed_pct':sub.Clinical_fields_observed_pct.mean(),'Median_observed_pct':sub.Clinical_fields_observed_pct.median()})
pd.DataFrame(rows).to_csv(TABLES/'05_completeness_by_age_outcome.csv',index=False)

# Adjusted association of clinical-field availability with transplantation.
# This is not a causal analysis and is not an estimate of clinical evaluation quality.
model_df=cohort[['Heart_transplanted','Age','Clinical_fields_observed_pct']].dropna().copy()
model_df['Completeness_10pct']=model_df.Clinical_fields_observed_pct/10
if model_df.Heart_transplanted.nunique()==2:
    m=smf.logit('Heart_transplanted ~ bs(Age, df=4) + Completeness_10pct',data=model_df).fit(disp=False)
    conf=m.conf_int()
    pd.DataFrame({'Variable':m.params.index,'OR':np.exp(m.params.values),'CI_low':np.exp(conf[0].values),'CI_high':np.exp(conf[1].values),'p':m.pvalues.values}).to_csv(TABLES/'06_age_adjusted_completeness_association.csv',index=False)
    (DIAG/'age_adjusted_model_summary.txt').write_text(m.summary().as_text(),encoding='utf-8')

# Visualizations
agefile=TABLES/'02_utilization_by_Age_group.csv'
if agefile.exists():
    d=pd.read_csv(agefile)
    fig,ax=plt.subplots(figsize=(8,5)); ax.bar(d.Age_group.astype(str),d.Utilization_pct)
    ax.set(xlabel='Age group (years)',ylabel='Heart utilization (%)',title='Procured donors with documented brain death'); ax.set_ylim(0,100)
    fig.tight_layout();fig.savefig(FIGS/'age_group_utilization.png',dpi=300);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5))
for y,label in [(0,'Not transplanted'),(1,'Transplanted')]:
    s=cohort.loc[cohort.Heart_transplanted.eq(y),'Clinical_fields_observed_pct'].dropna()
    ax.hist(s,bins=np.arange(0,105,10),alpha=.5,label=f'{label} (n={len(s)})')
ax.set(xlabel='Percentage of selected clinical fields recorded',ylabel='Donors',title='Clinical data completeness (not evaluation completion)');ax.legend()
fig.tight_layout();fig.savefig(FIGS/'completeness_distribution.png',dpi=300);plt.close(fig)
notes={'source':str(SOURCE),'cohort_n':len(cohort),'transplanted_n':int(cohort.Heart_transplanted.sum()),'not_transplanted_n':int((cohort.Heart_transplanted==0).sum()),'clinical_fields':[c for c in CLINICAL if c in cohort],'limitations':['Procurement may refer to any organ.','Brain death indicator requires source dictionary verification.','Recorded data completeness does not prove testing or evaluation was or was not performed.','Associations are not causal.','Clinical measurement timing and units unverified.','No inference of donor-heart suitability, offer, decline or discard.']}
(DIAG/'phase11_notes.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
print('Phase 11 complete:',OUT,'| N=',len(cohort),'| Tx=',int(cohort.Heart_transplanted.sum()))
