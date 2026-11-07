from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

SOURCE = Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
ROOT = Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
OUT = ROOT/'09_Post_Procurement_Analysis'/'Brain_Death_Subgroup'/'Phase14_High_Coverage'
TABLES=OUT/'Tables'; DIAG=OUT/'Diagnostics'
for p in (TABLES, DIAG): p.mkdir(parents=True, exist_ok=True)
if not SOURCE.exists(): raise FileNotFoundError(SOURCE)
df=pd.read_csv(SOURCE,low_memory=False)
required=['Age','procured','brain_death','outcome_heart_Transplanted','OPO','authorized']
missing=[x for x in required if x not in df.columns]
if missing: raise ValueError(f'Missing columns: {missing}')
for c in ['Age','procured','brain_death','outcome_heart_Transplanted','authorized']:
    df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.loc[df.Age.between(0,17)&df['outcome_heart_Transplanted'].isin([0,1])].copy()
df['Y']=df['outcome_heart_Transplanted'].astype(int)
cohorts={
 'Procured_BrainDeath':df.loc[df.procured.eq(1)&df.brain_death.eq(1)],
 'Authorized_All':df.loc[df.authorized.eq(1)],
 'Procured_BrainDeath_OPO1_2':df.loc[df.procured.eq(1)&df.brain_death.eq(1)&df.OPO.astype(str).isin(['OPO1','OPO2'])],
 'Procured_BrainDeath_OPO1':df.loc[df.procured.eq(1)&df.brain_death.eq(1)&df.OPO.astype(str).eq('OPO1')],
 'Procured_BrainDeath_OPO2':df.loc[df.procured.eq(1)&df.brain_death.eq(1)&df.OPO.astype(str).eq('OPO2')],
}
# Exclude donor IDs, outcomes, donation-stage variables and downstream events from clinical comparison.
# Unverified fields are INVENTORIED but NOT automatically approved as clinically eligible.
excluded_exact={'Y','PatientID','HospitalID','OPO','authorized','approached','procured','transplanted','brain_death','Referral_Year','Procured_Year','time_asystole','time_brain_death','time_referred','time_approached','time_authorized','time_procured'}
excluded_prefix=('outcome_','time_','t_')
excluded_fragments=('transplant','recovered','procure','authorization','recipient','patientid','hospitalid')
rows=[]; count_rows=[]
for cname,g in cohorts.items():
    n1=int(g.Y.sum()); n0=int((g.Y==0).sum())
    count_rows.append({'Cohort':cname,'N':len(g),'Transplanted':n1,'Not_transplanted':n0})
    if min(n1,n0)==0: continue
    for c in g.columns:
        if c in excluded_exact or c.startswith(excluded_prefix) or any(f in c.lower() for f in excluded_fragments): continue
        a=g.loc[g.Y==1,c]; b=g.loc[g.Y==0,c]
        nobs1=int(a.notna().sum()); nobs0=int(b.notna().sum())
        cov1=nobs1/n1; cov0=nobs0/n0
        row={'Cohort':cname,'Variable':c,'N_Tx':n1,'N_NoTx':n0,'Observed_Tx':nobs1,'Observed_NoTx':nobs0,
             'Coverage_Tx_pct':100*cov1,'Coverage_NoTx_pct':100*cov0,'Min_group_coverage_pct':100*min(cov1,cov0),
             'Pass70_both':min(cov1,cov0)>=.70,'Pass80_both':min(cov1,cov0)>=.80,
             'Missing_difference_pct':100*(cov1-cov0),'Timing_verified':False,'Dictionary_verified':False}
        # A variable with binary values is categorical even when numeric.
        numeric=pd.to_numeric(g[c],errors='coerce')
        raw_nonnull=g[c].dropna()
        numeric_fraction=numeric.notna().sum()/len(raw_nonnull) if len(raw_nonnull) else 0
        distinct=set(numeric.dropna().unique()) if numeric_fraction>.99 else set()
        is_numeric=numeric_fraction>.99 and len(distinct)>2 and not c.lower().endswith('year')
        if is_numeric:
            x=numeric[g.Y==1].dropna(); y=numeric[g.Y==0].dropna()
            row['Type']='continuous'
            for label,s in [('Tx',x),('NoTx',y)]:
                row[f'Median_{label}']=s.median();row[f'Q1_{label}']=s.quantile(.25);row[f'Q3_{label}']=s.quantile(.75)
                row[f'Mean_{label}']=s.mean();row[f'SD_{label}']=s.std()
            if len(x)>2 and len(y)>2:
                try: row['p_value']=mannwhitneyu(x,y,alternative='two-sided').pvalue
                except ValueError: pass
                sd=np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
                row['SMD']=(x.mean()-y.mean())/sd if sd>0 else np.nan
        else:
            row['Type']='categorical'
            aa=a.astype('string').dropna();bb=b.astype('string').dropna()
            levels=sorted(set(aa.unique()).union(set(bb.unique())))
            row['Levels_observed']=len(levels)
            row['Counts_Tx']='; '.join(f'{v}:{int((aa==v).sum())}' for v in levels[:30])
            row['Counts_NoTx']='; '.join(f'{v}:{int((bb==v).sum())}' for v in levels[:30])
            if 1<len(levels)<=30 and len(aa)>0 and len(bb)>0:
                tab=pd.crosstab(pd.concat([aa,bb],ignore_index=True),pd.Series(['Tx']*len(aa)+['NoTx']*len(bb)))
                try:
                    row['p_value']=fisher_exact(tab.values)[1] if tab.shape==(2,2) else chi2_contingency(tab)[1]
                except ValueError: pass
        rows.append(row)
results=pd.DataFrame(rows)
results['FDR_within_cohort']=np.nan
for cname,idx in results.groupby('Cohort').groups.items():
    p=results.loc[idx,'p_value'].astype(float)
    valid=p.notna()
    if valid.any(): results.loc[p.index[valid],'FDR_within_cohort']=multipletests(p[valid],method='fdr_bh')[1]
results.to_csv(TABLES/'01_all_variable_coverage_and_comparisons.csv',index=False)
for threshold in [70,80]:
    sub=results[results[f'Pass{threshold}_both']].sort_values(['Cohort','Type','Variable'])
    sub.to_csv(TABLES/f'02_variables_ge{threshold}_both_groups.csv',index=False)
    print(f'>={threshold}% both groups:',sub.groupby('Cohort').size().to_dict())
pd.DataFrame(count_rows).to_csv(TABLES/'00_cohort_counts.csv',index=False)
results.groupby(['Cohort','Type']).agg(Variables=('Variable','size'),Meet70=('Pass70_both','sum'),Meet80=('Pass80_both','sum')).reset_index().to_csv(TABLES/'03_coverage_summary.csv',index=False)
notes={'source':str(SOURCE),'cohorts':count_rows,'threshold_rule':'coverage must meet threshold in EACH outcome group',
 'limitations':['No imputation or causal inference','Unverified timing and dictionary for every field',
 'Categorical fields with >30 levels are listed but not hypothesis-tested',
 'Unadjusted tests are exploratory; FDR applied across tested variables within each cohort',
 'Clinical feature eligibility requires manual review; high coverage alone does not establish suitability',
 'Some categorical variables may be one-hot encoded indicators; review related columns jointly',
 'Missingness is OPO-dependent; OPO1/OPO2 estimates are not representative of all OPOs']}
(DIAG/'phase14_notes.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
print('Phase 14 complete:',OUT)
