from pathlib import Path
import json
import numpy as np
import pandas as pd

SOURCE = Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
OUT = Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2') / 'Phase2_Cohort'
TARGET = 'outcome_heart_Transplanted'
OUT.mkdir(parents=True, exist_ok=True)
if not SOURCE.exists():
    raise FileNotFoundError(f'Source CSV not found: {SOURCE}. Update SOURCE at top of script.')
df = pd.read_csv(SOURCE, low_memory=False)
required = ['Age', 'authorized', TARGET]
missing = [c for c in required if c not in df]
if missing: raise ValueError(f'Required columns absent: {missing}')
for c in ['Age','authorized','brain_death','procured','approached',TARGET,'Referral_Year','WeightKg']:
    if c in df: df[c] = pd.to_numeric(df[c], errors='coerce')
original_n = len(df)
# Preserve unknown ages and invalid outcomes in accounting, never silently treat them as negative.
ped = df.loc[df['Age'].ge(0) & df['Age'].lt(18) & df[TARGET].isin([0,1])].copy()
auth = ped.loc[ped['authorized'].eq(1)].copy()
proc = ped.loc[ped['procured'].eq(1)].copy() if 'procured' in ped else pd.DataFrame()
for d in [ped, auth, proc]:
    if len(d):
        d['Age_group'] = pd.cut(d['Age'], bins=[-0.001,0.999,4.999,11.999,17.999], labels=['<1','1-4','5-11','12-17'])
account = pd.DataFrame([
    ['Source rows',original_n],
    ['Age missing/invalid',int((~df['Age'].ge(0) | ~df['Age'].lt(18)).sum())],
    ['Pediatric with binary heart outcome',len(ped)],
    ['Pediatric authorized=1',len(auth)],
    ['Pediatric procured=1',len(proc)],
    ['Authorized with heart transplanted',int(auth[TARGET].eq(1).sum())],
    ['Authorized without heart transplanted',int(auth[TARGET].eq(0).sum())],
],columns=['Criterion','N'])
account.to_csv(OUT/'01_cohort_accounting.csv',index=False)

stage = [c for c in ['approached','authorized','brain_death','procured','transplanted',TARGET] if c in ped]
if stage:
    stages = ped.groupby(stage, dropna=False, observed=True).size().reset_index(name='N').sort_values('N',ascending=False)
    stages.to_csv(OUT/'02_joint_donation_pathway.csv',index=False)

for name, cohort in [('all_pediatric',ped),('authorized',auth),('procured',proc)]:
    if cohort.empty: continue
    for groupvar in ['Age_group','Referral_Year','brain_death','procured','OPO']:
        if groupvar not in cohort: continue
        tab = cohort.groupby([groupvar,TARGET],dropna=False,observed=True).size().unstack(TARGET,fill_value=0)
        tab = tab.reindex(columns=[0,1],fill_value=0)
        tab.columns=['Not_transplanted','Heart_transplanted']
        tab['Total']=tab.sum(axis=1)
        tab['Utilization_pct']=np.where(tab['Total']>0,100*tab['Heart_transplanted']/tab['Total'],np.nan)
        tab.reset_index().to_csv(OUT/f'03_{name}_by_{groupvar}.csv',index=False)

# Record availability and observed values separately. No imputation.
excluded = {'PatientID','HospitalID'}
rows=[]
for c in auth.columns:
    if c in excluded or c=='Age_group': continue
    for y in [0,1]:
        sub=auth.loc[auth[TARGET].eq(y),c]
        rows.append({'Variable':c,'Heart_transplanted':y,'N_group':len(sub),
                     'N_observed':int(sub.notna().sum()),'N_missing':int(sub.isna().sum()),
                     'Missing_pct':round(100*sub.isna().mean(),2) if len(sub) else np.nan,
                     'N_unique_observed':int(sub.nunique(dropna=True))})
pd.DataFrame(rows).to_csv(OUT/'04_authorized_missingness_by_outcome.csv',index=False)

# Inventory every source field related to cardiac recovery and other organ outcomes.
heart_fields=[c for c in ped if ('heart' in c.lower() or 'cardiac' in c.lower())]
organ_fields=[c for c in ped if c.startswith('outcome_')]
field_rows=[]
for c in sorted(set(heart_fields+organ_fields)):
    for name,cohort in [('all_pediatric',ped),('authorized',auth),('procured',proc)]:
        if len(cohort)==0: continue
        s=cohort[c]
        numeric=pd.to_numeric(s,errors='coerce')
        field_rows.append({'Variable':c,'Cohort':name,'N':len(cohort),'N_observed':int(s.notna().sum()),
                           'N_equal_1':int(numeric.eq(1).sum()),'N_equal_0':int(numeric.eq(0).sum()),
                           'Observed_levels':str(s.value_counts(dropna=False).head(12).to_dict())})
pd.DataFrame(field_rows).to_csv(OUT/'05_heart_recovery_and_organ_outcome_inventory.csv',index=False)

# Cross-tab heart-recovery indicators against transplant outcome; do not assume field meaning.
for c in heart_fields:
    if c==TARGET:continue
    if auth[c].nunique(dropna=True)>15:continue
    pd.crosstab(auth[c].fillna('MISSING').astype(str),auth[TARGET],dropna=False).to_csv(OUT/f'06_heart_field_{heart_fields.index(c):02d}_cross_tab.csv')

# Raw measurement range and coverage in authorized cohort.
clinical=['Age','WeightKg','HeightIn','HR','SBP','DBP','PO2','PCO2','PH','FIO2','O2SAT','TroponinI','TroponinT','BNP','Lactate','Creatinine','TV','Rate']
summary=[]
for c in clinical:
    if c not in auth:continue
    for y in [0,1]:
        s=pd.to_numeric(auth.loc[auth[TARGET].eq(y),c],errors='coerce')
        summary.append({'Variable':c,'Heart_transplanted':y,'N_observed':int(s.notna().sum()),
                        'Median':s.median(),'Q1':s.quantile(.25),'Q3':s.quantile(.75),
                        'Min':s.min(),'Max':s.max()})
pd.DataFrame(summary).to_csv(OUT/'07_authorized_clinical_coverage_and_ranges.csv',index=False)

# Explicit manual timing review: no automated assumption of predecision availability.
review=[]
for c in df.columns:
    l=c.lower()
    if c in ['PatientID','HospitalID'] or l.startswith('outcome_') or 'transplant' in l or 'procure' in l:
        category='Exclude from predecision prediction (ID/outcome/postdecision)'
    elif c in ['authorized','approached','brain_death','time_authorized','time_approached','time_brain_death','time_referred','time_asystole','Referral_Year','OPO']:
        category='Cohort/stratification or timing review; not automatically a predictor'
    else:
        category='Candidate only; verify availability at prediction time and clinical meaning'
    review.append({'Variable':c,'Preliminary_role':category,'Data_dictionary_verified':False,'Predecision_timing_verified':False})
pd.DataFrame(review).to_csv(OUT/'08_feature_eligibility_review.csv',index=False)

notes={'source_rows':original_n,'pediatric_rows':len(ped),'authorized_rows':len(auth),'procured_rows':len(proc),
       'primary_outcome':TARGET,'primary_candidate_cohort':'authorized == 1',
       'important':'Authorization for any organ is not proof of heart-specific authorization, offer, or suitability.',
       'prediction_time':'NOT YET ESTABLISHED. No ML until measurement timing is verified.',
       'missingness':'Observed data only; no imputation performed.',
       'heart_recovery':'Field names and distributions inventoried; definitions require data dictionary confirmation.'}
(OUT/'00_readme.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
print(account.to_string(index=False))
print(f'\nSaved Phase 2 audit to: {OUT}')
