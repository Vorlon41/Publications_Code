from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests

SOURCE=Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
ROOT=Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
OUT=ROOT/'09_Post_Procurement_Analysis'/'Brain_Death_Subgroup'/'Phase16_Reconstructed_Categories'
T=OUT/'Tables'; D=OUT/'Diagnostics'
for p in (T,D):p.mkdir(parents=True,exist_ok=True)
Y='outcome_heart_Transplanted'
raw=pd.read_csv(SOURCE,low_memory=False)
for c in ['Age','procured','brain_death','authorized',Y,'Referral_Year']:
    if c not in raw:raise ValueError('Required column absent: '+c)
    raw[c]=pd.to_numeric(raw[c],errors='coerce')
cohorts={
 'Procured_BrainDeath':raw.loc[(raw.Age<18)&(raw.procured==1)&(raw.brain_death==1)&raw[Y].isin([0,1])].copy(),
 'Authorized_All':raw.loc[(raw.Age<18)&(raw.authorized==1)&raw[Y].isin([0,1])].copy()
}
families=['Cause_of_Death_UNOS','Mechanism_of_Death','Circumstances_of_Death','ABO_BloodType']
all_counts=[]; all_levels=[]; all_models=[]; all_diagnostics=[]
for cohort_name,df in cohorts.items():
    df['HeartTx']=df[Y].astype(int)
    for fam in families:
        cols=[c for c in df.columns if c.startswith(fam+'_')]
        if not cols:continue
        indicators=df[cols].apply(pd.to_numeric,errors='coerce')
        positive=indicators.eq(1)
        npositive=positive.sum(axis=1)
        labels=positive.idxmax(axis=1).str[len(fam)+1:]
        labels=labels.where(npositive==1,'UNRESOLVED')
        # A subtype-to-major-group mapping is provisional, subject to data dictionary confirmation.
        if fam=='ABO_BloodType':
            abo_map={'O':'O','A':'A','A1':'A','A2':'A','B':'B','AB':'AB','A1B':'AB','A2B':'AB'}
            labels=labels.map(lambda v:abo_map.get(v,'UNRESOLVED'))
        df[fam+'_Reconstructed']=labels
        for outcome in [0,1]:
            d=df[df.HeartTx==outcome]
            all_counts.append(dict(Cohort=cohort_name,Family=fam,Outcome=outcome,N=len(d),NoPositive=int((npositive.loc[d.index]==0).sum()),MultiplePositive=int((npositive.loc[d.index]>1).sum()),MissingAnyIndicator=int(indicators.loc[d.index].isna().any(axis=1).sum())))
        # Keep unresolved as a distinct category for descriptive tables; do not call it a negative.
        tab=pd.crosstab(labels,df.HeartTx).reindex(columns=[0,1],fill_value=0)
        p_global=np.nan
        if len(tab)>1:
            try:p_global=chi2_contingency(tab)[1]
            except ValueError:pass
        for level,row in tab.iterrows():
            n0=int(row[0]);n1=int(row[1]);tot0=int((df.HeartTx==0).sum());tot1=int((df.HeartTx==1).sum())
            p=fisher_exact([[n1,tot1-n1],[n0,tot0-n0]])[1] if n1+n0 else np.nan
            all_levels.append(dict(Cohort=cohort_name,Family=fam,Category=level,Tx_N=n1,Tx_Total=tot1,Tx_Pct=n1/tot1*100,NoTx_N=n0,NoTx_Total=tot0,NoTx_Pct=n0/tot0*100,Unadjusted_Fisher_P=p,Global_ChiSquare_P=p_global))
        # Adjusted category-specific contrasts: one category versus other resolved categories.
        # Sparse categories excluded, and no fitting if insufficient outcome variation.
        resolved=df.loc[labels!='UNRESOLVED'].copy()
        for level in sorted(labels.unique()):
            if level=='UNRESOLVED':continue
            work=resolved[['HeartTx','Age','OPO','Referral_Year',fam+'_Reconstructed']].dropna().copy() if 'OPO' in resolved else pd.DataFrame()
            if work.empty:continue
            work['Exposure']=(work[fam+'_Reconstructed']==level).astype(int)
            ct=pd.crosstab(work.Exposure,work.HeartTx).reindex(index=[0,1],columns=[0,1],fill_value=0)
            if min(ct.loc[1].tolist())<5 or min(ct.loc[0].tolist())<5:
                all_models.append(dict(Cohort=cohort_name,Family=fam,Category=level,N=len(work),Status='SKIPPED_sparse_cell'))
                continue
            try:
                # Parsimonious age adjustment; OPO and year as fixed effects may cause separation.
                fit=smf.logit('HeartTx ~ Exposure + Age + C(OPO) + Referral_Year',data=work).fit(disp=False,maxiter=200)
                ci=fit.conf_int().loc['Exposure']
                all_models.append(dict(Cohort=cohort_name,Family=fam,Category=level,N=len(work),Status='OK' if fit.mle_retvals.get('converged') else 'NOT_CONVERGED',OR=float(np.exp(fit.params['Exposure'])),CI_Low=float(np.exp(ci.iloc[0])),CI_High=float(np.exp(ci.iloc[1])),P=float(fit.pvalues['Exposure'])))
            except Exception as exc:
                all_models.append(dict(Cohort=cohort_name,Family=fam,Category=level,N=len(work),Status='FAILED_'+type(exc).__name__))
        all_diagnostics.append(dict(Cohort=cohort_name,Family=fam,Categories=int(labels.nunique()),Unresolved=int((labels=='UNRESOLVED').sum()),DictionaryVerified=False))
    print(cohort_name,'N',len(df),'Tx',int(df.HeartTx.sum()))

pd.DataFrame(all_counts).to_csv(T/'01_family_encoding_integrity.csv',index=False)
levels=pd.DataFrame(all_levels)
levels['FDR_within_cohort_family']=np.nan
for _,indices in levels.groupby(['Cohort','Family']).groups.items():
    ix=levels.loc[indices].index[levels.loc[indices,'Unadjusted_Fisher_P'].notna()]
    if len(ix):levels.loc[ix,'FDR_within_cohort_family']=multipletests(levels.loc[ix,'Unadjusted_Fisher_P'],method='fdr_bh')[1]
levels.to_csv(T/'02_reconstructed_category_counts.csv',index=False)
models=pd.DataFrame(all_models)
if 'P' in models:
    models['FDR_exploratory_within_cohort']=np.nan
    for cohort,ix in models.groupby('Cohort').groups.items():
        valid=models.loc[ix].index[models.loc[ix,'P'].notna()]
        if len(valid):models.loc[valid,'FDR_exploratory_within_cohort']=multipletests(models.loc[valid,'P'],method='fdr_bh')[1]
models.to_csv(T/'03_adjusted_category_associations.csv',index=False)
pd.DataFrame(all_diagnostics).to_csv(D/'04_dictionary_and_positivity_flags.csv',index=False)
(D/'README.txt').write_text('Exploratory retrospective analyses. ABO subtype mapping is provisional. Category reconstruction requires original dictionary verification. UNRESOLVED means zero/multiple positive indicators, not confirmed absence. Adjusted models are category-vs-other-resolved and may fail from sparse cells. P values are exploratory. No heart suitability, offers, acceptance, discard, or causal inference.\n',encoding='utf-8')
print('Saved:',OUT)
