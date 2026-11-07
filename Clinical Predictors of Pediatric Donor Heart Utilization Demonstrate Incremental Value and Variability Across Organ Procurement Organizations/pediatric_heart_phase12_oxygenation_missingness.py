from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu

SOURCE = Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
ROOT = Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
OUT = ROOT/'09_Post_Procurement_Analysis'/'Brain_Death_Subgroup'/'Phase12_Oxygenation_Missingness'
TABLES, FIGS, DIAG = (OUT/s for s in ('Tables','Figures','Diagnostics'))
for p in (TABLES, FIGS, DIAG): p.mkdir(parents=True, exist_ok=True)
TARGET='outcome_heart_Transplanted'
CLINICAL=['HR','SBP','DBP','PO2','PCO2','PH','O2SAT','FIO2','PIP','PEEP','Rate','TV','Hgb','Hct','Ptl','UOP','Output_Urine','Creatinine','BUN','Lactate','TroponinI','TroponinT','BNP']
if not SOURCE.exists(): raise FileNotFoundError(f'Input not found: {SOURCE}')
df=pd.read_csv(SOURCE,low_memory=False)
required=['Age','procured','brain_death',TARGET]
missing=[c for c in required if c not in df]
if missing: raise ValueError(f'Missing required columns: {missing}')
for c in required+['PO2','FIO2','O2SAT','Referral_Year','WeightKg']:
    if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
df=df[(df['Age']>=0)&(df['Age']<18)&(df['procured']==1)&(df['brain_death']==1)&(df[TARGET].isin([0,1]))].copy()
df['Heart_transplanted']=df[TARGET].astype(int)
if len(df)==0: raise ValueError('No eligible records')
clinical=[c for c in CLINICAL if c in df.columns]
for c in clinical: df[c]=pd.to_numeric(df[c],errors='coerce')
df['Observed_clinical_count']=df[clinical].notna().sum(axis=1)
df['Observed_clinical_pct']=100*df['Observed_clinical_count']/len(clinical)
df['PO2_recorded']=df['PO2'].notna().astype(int) if 'PO2' in df else 0
if 'FIO2' in df:
    fio2=df['FIO2'].copy()
    # Only convert percentages when a coherent encoding can be verified; this is a flagged exploratory ratio.
    fio2_fraction=np.where((fio2>1)&(fio2<=100),fio2/100,np.where((fio2>0)&(fio2<=1),fio2,np.nan))
    df['PF_exploratory']=df['PO2']/fio2_fraction
    df.loc[(df['PF_exploratory']<=0)|(df['PF_exploratory']>1000),'PF_exploratory']=np.nan
else: df['PF_exploratory']=np.nan

# Cohort and availability
pd.DataFrame([{'N':len(df),'Transplanted':int(df['Heart_transplanted'].sum()),'Not_transplanted':int((df['Heart_transplanted']==0).sum()),'Clinical_fields':len(clinical)}]).to_csv(TABLES/'01_cohort.csv',index=False)
coverage=[]
for variable in ['PO2','FIO2','O2SAT','PF_exploratory']:
    if variable not in df: continue
    for outcome,g in df.groupby('Heart_transplanted'):
        x=g[variable].dropna()
        coverage.append({'Variable':variable,'Heart_transplanted':outcome,'N_total':len(g),'N_observed':len(x),'Coverage_pct':100*len(x)/len(g),'Median':x.median(),'Q1':x.quantile(.25),'Q3':x.quantile(.75)})
pd.DataFrame(coverage).to_csv(TABLES/'02_oxygenation_coverage.csv',index=False)

# Regressions are association analyses in selected observed records, not prospective models.
results=[]
def fit_association(label, formula, data, minimum=50):
    needed=list(set([v for v in ['Age','Heart_transplanted'] if v in data.columns]))
    if len(data)<minimum or data['Heart_transplanted'].nunique()<2:
        results.append({'Analysis':label,'N':len(data),'Status':'insufficient observations or outcome variation'}); return
    try:
        model=smf.logit(formula,data=data).fit(disp=False,maxiter=250)
        for term in model.params.index:
            if term=='Intercept': continue
            ci=model.conf_int().loc[term]
            results.append({'Analysis':label,'N':int(model.nobs),'Term':term,'OR':float(np.exp(model.params[term])),'CI_low':float(np.exp(ci.iloc[0])),'CI_high':float(np.exp(ci.iloc[1])),'p_value':float(model.pvalues[term]),'Status':'ok'})
    except Exception as e:
        results.append({'Analysis':label,'N':len(data),'Status':f'not estimable: {type(e).__name__}: {str(e)[:150]}'})

adjust='bs(Age, df=4)'
if 'OPO' in df and df['OPO'].nunique(dropna=True)>1: adjust+=' + C(OPO)'
if 'Referral_Year' in df and df['Referral_Year'].nunique(dropna=True)>1: adjust+=' + Referral_Year'

fit_association('PO2 availability',f'Heart_transplanted ~ PO2_recorded + {adjust}',df)
fit_association('Completeness continuous',f'Heart_transplanted ~ I(Observed_clinical_pct / 10) + {adjust}',df)
for var,scale in [('PO2',50),('O2SAT',5),('PF_exploratory',100)]:
    if var not in df: continue
    sub=df[df[var].notna()].copy()
    if var=='PF_exploratory': sub=sub[(sub[var]>=50)&(sub[var]<=800)]
    sub['Scaled_value']=sub[var]/scale
    fit_association(f'{var} per {scale} units (observed only)',f'Heart_transplanted ~ Scaled_value + {adjust}',sub)
pd.DataFrame(results).to_csv(TABLES/'03_adjusted_associations.csv',index=False)

# Document missingness patterns without calling them clinical evaluations.
for groupvar in ['OPO','Referral_Year','Heart_transplanted']:
    if groupvar not in df: continue
    rows=[]
    for level,g in df.groupby(groupvar,dropna=False):
        rows.append({'Grouping':groupvar,'Level':level,'N':len(g),'Utilization_pct':100*g['Heart_transplanted'].mean(),'Median_clinical_pct':g['Observed_clinical_pct'].median(),'Mean_clinical_pct':g['Observed_clinical_pct'].mean(),'Pct_very_low_coverage_lt10':100*(g['Observed_clinical_pct']<10).mean(),'Pct_high_coverage_ge70':100*(g['Observed_clinical_pct']>=70).mean(),'PO2_coverage_pct':100*g['PO2_recorded'].mean()})
    pd.DataFrame(rows).to_csv(TABLES/f'04_completeness_by_{groupvar}.csv',index=False)

# Cross-tab documentation regime by OPO and year.
df['Completeness_regime']=pd.cut(df['Observed_clinical_pct'],bins=[-0.01,10,40,70,100.01],labels=['<10%','10–<40%','40–<70%','>=70%'],right=False)
for groupvar in ['OPO','Referral_Year']:
    if groupvar in df:
        pd.crosstab(df[groupvar],df['Completeness_regime'],margins=True).to_csv(TABLES/f'05_regime_by_{groupvar}.csv')

fig,ax=plt.subplots(figsize=(9,5))
for outcome,label in [(0,'Not transplanted'),(1,'Transplanted')]:
    ax.hist(df.loc[df['Heart_transplanted']==outcome,'Observed_clinical_pct'],bins=np.arange(0,110,10),alpha=.5,label=label)
ax.set(xlabel='Selected clinical fields recorded (%)',ylabel='Donors',title='Clinical-field completeness by heart utilization')
ax.legend();fig.tight_layout();fig.savefig(FIGS/'clinical_completeness_distribution.png',dpi=250);plt.close(fig)

with open(DIAG/'phase12_notes.json','w',encoding='utf-8') as f:
    json.dump({'N':len(df),'Clinical_fields':clinical,'Adjustments':adjust,'Cautions':['PO2/FiO2 uses heuristic FiO2 conversion; confirm units before interpretation','No measurement timestamps; associations may involve postdecision values','Observed-only regressions are selected subsets','OPO and year may be correlated; estimates can be unstable','No suitability, offer, decline, discard or causal claims','Non-estimable models are recorded rather than silently omitted']},f,indent=2)
print(f'Phase 12 complete: {OUT}')
print(pd.DataFrame(results).query("Term == 'Scaled_value' or Term == 'PO2_recorded' or Term == 'I(Observed_clinical_pct / 10)'").to_string(index=False) if results else 'No models')
