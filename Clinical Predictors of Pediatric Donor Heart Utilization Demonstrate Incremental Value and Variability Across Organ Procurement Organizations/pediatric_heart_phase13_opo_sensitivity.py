from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu

SOURCE=Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
ROOT=Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
OUT=ROOT/'09_Post_Procurement_Analysis'/'Brain_Death_Subgroup'/'Phase13_OPO_Sensitivity'
TABLES=OUT/'Tables'; DIAG=OUT/'Diagnostics'; FIGS=OUT/'Figures'
for d in (TABLES,DIAG,FIGS): d.mkdir(parents=True,exist_ok=True)
if not SOURCE.exists(): raise FileNotFoundError(f'Original CSV not found: {SOURCE}')
df=pd.read_csv(SOURCE,low_memory=False)
needed=['Age','procured','brain_death','outcome_heart_Transplanted','OPO','PO2','FIO2','O2SAT','Referral_Year']
absent=[c for c in needed if c not in df.columns]
if absent: raise ValueError(f'Missing required fields: {absent}')
for c in needed:
    if c!='OPO': df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.loc[df['Age'].between(0,17)&df['procured'].eq(1)&df['brain_death'].eq(1)&df['outcome_heart_Transplanted'].isin([0,1])].copy()
df['Heart_transplanted']=df['outcome_heart_Transplanted'].astype(int)
df['OPO']=df['OPO'].astype(str)
# Keep oxygenation measurements observed; do not impute or treat absent measurements as normal.
# FiO2 ratio is not modeled until source units and paired measurement timing are confirmed.
groups={'OPO1':['OPO1'],'OPO2':['OPO2'],'OPO1_OPO2':['OPO1','OPO2'],'OPO3':['OPO3'],'All_six_OPOs':sorted(df.OPO.unique())}
counts=[]; summaries=[]; regressions=[]
for name,opos in groups.items():
    g=df[df.OPO.isin(opos)].copy()
    for outcome,part in g.groupby('Heart_transplanted'):
        observed=part['PO2'].dropna()
        counts.append({'Group':name,'Outcome':int(outcome),'N':len(part),'PO2_observed':len(observed),'PO2_coverage_pct':100*len(observed)/len(part) if len(part) else np.nan})
        summaries.append({'Group':name,'Outcome':int(outcome),'N_PO2':len(observed),'PO2_median':observed.median(),'PO2_Q1':observed.quantile(.25),'PO2_Q3':observed.quantile(.75)})
    sub=g.dropna(subset=['PO2','Age','Referral_Year']).copy()
    n0=int((sub.Heart_transplanted==0).sum()); n1=int((sub.Heart_transplanted==1).sum())
    row={'Group':name,'N':len(sub),'Events':n1,'Non_events':n0,'PO2_unit':'per 50 recorded units'}
    if n0<10 or n1<10 or sub['PO2'].nunique()<5:
        row['Status']='insufficient outcomes or PO2 variation'; regressions.append(row); continue
    sub['PO2_per50']=sub['PO2']/50
    # Age represented parsimoniously for smaller groups; year centered for stability.
    sub['Year_centered']=sub['Referral_Year']-sub['Referral_Year'].median()
    formula='Heart_transplanted ~ PO2_per50 + Age + Year_centered'
    if name=='OPO1_OPO2': formula+=' + C(OPO)'
    elif name=='All_six_OPOs': formula+=' + C(OPO)'
    try:
        fitted=smf.logit(formula,data=sub).fit(disp=False,maxiter=200)
        ci=fitted.conf_int().loc['PO2_per50']; beta=fitted.params['PO2_per50']
        row.update({'Status':'ok','OR':float(np.exp(beta)),'CI_low':float(np.exp(ci.iloc[0])),'CI_high':float(np.exp(ci.iloc[1])),'p_value':float(fitted.pvalues['PO2_per50']),'Formula':formula})
    except Exception as exc:
        row.update({'Status':f'not estimable: {type(exc).__name__}: {str(exc)[:160]}','Formula':formula})
    regressions.append(row)
pd.DataFrame(counts).to_csv(TABLES/'01_opo_outcome_coverage.csv',index=False)
pd.DataFrame(summaries).to_csv(TABLES/'02_opo_observed_po2_summary.csv',index=False)
pd.DataFrame(regressions).to_csv(TABLES/'03_within_opo_adjusted_po2_associations.csv',index=False)
# Document whether all OPOs have overlapping observed PO2; no extrapolation across zero-coverage OPOs.
coverage=df.groupby('OPO').agg(N=('Heart_transplanted','size'),Events=('Heart_transplanted','sum'),PO2_observed=('PO2','count')).reset_index()
coverage['PO2_coverage_pct']=100*coverage.PO2_observed/coverage.N
coverage.to_csv(TABLES/'04_opo_positivity_diagnostics.csv',index=False)
notes={'cohort_N':len(df),'purpose':'Within-OPO oxygenation association sensitivity','warnings':['Observational associations, not organ suitability or acceptance decisions','PO2 timing and source units unverified','OPO4-OPO6 have no PO2 values in prior audit and cannot contribute to within-OPO PO2 associations','OPO3 small and sparse; adjusted estimates may be unstable','No imputation or model selection; exploratory confidence intervals','FiO2-derived ratios intentionally excluded pending unit/timing confirmation']}
(DIAG/'phase13_notes.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
print('Phase 13 complete:',OUT)
print(pd.DataFrame(regressions).to_string(index=False))
