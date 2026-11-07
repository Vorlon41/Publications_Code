from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT_CSV = Path(r'C:\Users\User\Desktop\HarvardExtensionSchool\Back End Web\Cloud Services\Cloud Computing Project\P_Pulm_Transplant.csv')
OUTPUT_DIR = Path(r'C:\Users\User\Desktop\Ped Heart Transplant\New2')
TARGET = 'outcome_heart_Transplanted'
ID_CANDIDATES = ['PatientID', 'patient_id', 'donor_id']
PROCESS_COLUMNS = ['brain_death','approached','authorized','procured','transplanted']
AGE_BINS = [-np.inf, 1, 5, 12, 17, np.inf]
AGE_LABELS = ['<1 year', '1–4 years', '5–11 years', '12–17 years', '18+ years']


def write_csv(frame, name):
    frame.to_csv(OUTPUT_DIR / name, index=False)


def numeric(series):
    return pd.to_numeric(series, errors='coerce')


def counts_by_group(df, col, group):
    return pd.crosstab(df[col].astype('string').fillna('Missing'),
                       df[group].astype('string').fillna('Missing'), dropna=False).reset_index()


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f'Input CSV not found: {INPUT_CSV}. Update INPUT_CSV to its actual location.')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    if TARGET not in df.columns:
        raise KeyError(f'Required target {TARGET!r} missing. Check the original CSV columns.')
    if 'Age' not in df.columns:
        raise KeyError('Age column missing. Verify source schema before defining pediatric cohort.')
    notes = []
    def note(msg):
        notes.append(msg)
        print(msg)

    note(f'Source: {INPUT_CSV}')
    note(f'Original shape: {df.shape[0]:,} rows, {df.shape[1]:,} columns')
    write_csv(pd.DataFrame({'column': df.columns, 'dtype': df.dtypes.astype(str).values}), '01_source_schema.csv')
    id_col = next((x for x in ID_CANDIDATES if x in df.columns), None)
    if id_col:
        note(f'{id_col}: {df[id_col].nunique(dropna=True):,} distinct nonmissing IDs; {df[id_col].isna().sum():,} missing IDs; {df[id_col].duplicated(keep=False).sum():,} rows with duplicated IDs (including missing)')
        dup = df.loc[df[id_col].notna() & df[id_col].duplicated(keep=False), [id_col]].copy()
        write_csv(dup, '02_duplicate_donor_ids.csv')
    else:
        note('No known donor ID column found; cannot confirm unique donor-level records.')
    df['__age'] = numeric(df['Age'])
    df['__target'] = numeric(df[TARGET])
    eligible = df['__age'].ge(0) & df['__age'].lt(18) & df['__target'].isin([0, 1])
    note(f'Age missing/non-numeric: {df["__age"].isna().sum():,}; negative ages: {df["__age"].lt(0).sum():,}; nonbinary/missing target: {(~df["__target"].isin([0,1])).sum():,}')
    note(f'Pediatric donors with valid binary outcome: {eligible.sum():,}')
    ped = df.loc[eligible].copy()
    ped['Heart_Utilized'] = ped['__target'].astype(int)
    ped['Age_Group'] = pd.cut(ped['__age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    # Correct final pediatric age band: bins [12,18) by using separate explicit edges.
    ped['Age_Group'] = pd.cut(ped['__age'], bins=[0,1,5,12,18], labels=AGE_LABELS[:4], right=False)
    write_csv(pd.DataFrame([{'Source_rows':len(df),'Pediatric_valid_outcome_rows':len(ped),
                             'Heart_utilized':int(ped['Heart_Utilized'].sum()),
                             'Heart_not_utilized':int((ped['Heart_Utilized']==0).sum()),
                             'Missing_age':int(df['__age'].isna().sum()),
                             'Negative_age':int(df['__age'].lt(0).sum()),
                             'Invalid_or_missing_outcome':int((~df['__target'].isin([0,1])).sum()),
                             'Unique_IDs':df[id_col].nunique(dropna=True) if id_col else np.nan}]), '03_cohort_accounting.csv')
    age_summary = ped.groupby(['Age_Group','Heart_Utilized'], observed=False).agg(
        N=('__age','size'), Age_median=('__age','median'), Age_Q1=('__age',lambda x:x.quantile(.25)),
        Age_Q3=('__age',lambda x:x.quantile(.75))).reset_index()
    write_csv(age_summary,'04_age_strata.csv')
    age_fig, ax = plt.subplots(figsize=(9,5))
    for outcome, name in [(0,'Not utilized'),(1,'Utilized')]:
        vals = ped.loc[ped['Heart_Utilized']==outcome,'__age'].dropna()
        if len(vals): ax.hist(vals,bins=np.arange(0,19),alpha=.5,label=f'{name} (n={len(vals)})')
    ax.set(xlabel='Age in years (as recorded)',ylabel='Donors',title='Pediatric donor age distribution')
    ax.legend(); age_fig.tight_layout(); age_fig.savefig(OUTPUT_DIR/'04_age_distribution.png',dpi=180);plt.close(age_fig)

    # Preserve original categories and explicitly show missingness rather than filling nulls with 0.
    for col in PROCESS_COLUMNS + ['Referral_Year','Procured_Year','OPO','Gender']:
        if col in ped.columns:
            write_csv(counts_by_group(ped,col,'Heart_Utilized'),f'05_group_counts_{col}.csv')
    for col in ['Referral_Year','Procured_Year']:
        if col in ped.columns:
            vals=numeric(ped[col]);note(f'{col}: observed numeric min={vals.min()}, max={vals.max()}, missing={vals.isna().sum():,}')
    # Nested stage counts are descriptive intersections, not verified clinical eligibility.
    stage_rows=[]
    for name, mask in [('All pediatric records',pd.Series(True,index=ped.index)),
                       ('Heart utilized',ped['Heart_Utilized'].eq(1)),
                       ('Heart not utilized',ped['Heart_Utilized'].eq(0))]:
        stage_rows.append({'Group':name,'N':int(mask.sum())})
    for col in PROCESS_COLUMNS:
        if col in ped.columns:
            s=numeric(ped[col]);
            for value in [0,1]:
                stage_rows.append({'Group':f'{col} == {value}','N':int(s.eq(value).sum())})
            stage_rows.append({'Group':f'{col} missing/unknown','N':int(s.isna().sum())})
            stage_rows.append({'Group':f'{col} == 1 AND heart utilized','N':int((s.eq(1)&ped['Heart_Utilized'].eq(1)).sum())})
            stage_rows.append({'Group':f'{col} == 1 AND heart not utilized','N':int((s.eq(1)&ped['Heart_Utilized'].eq(0)).sum())})
    write_csv(pd.DataFrame(stage_rows),'06_donation_stage_counts.csv')

    # Missingness and data-availability profiles, no imputation.
    exclude=['__age','__target','Heart_Utilized','Age_Group']
    columns=[c for c in ped.columns if c not in exclude]
    missing_rows=[]
    for c in columns:
        for group,part in [('All',ped),('Heart_not_utilized',ped[ped.Heart_Utilized==0]),
                           ('Heart_utilized',ped[ped.Heart_Utilized==1])]:
            n=len(part);nonmissing=int(part[c].notna().sum())
            missing_rows.append({'Variable':c,'Group':group,'N':n,'Observed_N':nonmissing,
                                 'Missing_N':n-nonmissing,'Missing_pct':100*(n-nonmissing)/n if n else np.nan,
                                 'Distinct_nonmissing':part[c].nunique(dropna=True)})
    write_csv(pd.DataFrame(missing_rows),'07_missingness_by_outcome.csv')
    age_missing=[]
    for (age,outcome),part in ped.groupby(['Age_Group','Heart_Utilized'],observed=True):
        for c in columns:
            age_missing.append({'Age_Group':str(age),'Heart_Utilized':outcome,'Variable':c,
                                'N':len(part),'Missing_pct':100*part[c].isna().mean()})
    write_csv(pd.DataFrame(age_missing),'08_missingness_by_age_and_outcome.csv')
    for stage in ['authorized','procured','brain_death']:
        if stage in ped.columns:
            tmp=ped.copy();tmp['Stage_Value']=tmp[stage].astype('string').fillna('Missing')
            rows=[]
            for stage_val,part in tmp.groupby('Stage_Value'):
                for c in columns:
                    rows.append({'Stage':stage,'Stage_Value':stage_val,'Variable':c,'N':len(part),
                                 'Missing_pct':100*part[c].isna().mean()})
            write_csv(pd.DataFrame(rows),f'09_missingness_by_{stage}.csv')
    if 'Referral_Year' in ped.columns:
        tmp=ped.copy();tmp['Year']=tmp['Referral_Year'].astype('string').fillna('Missing')
        write_csv(tmp.groupby('Year',dropna=False).agg(N=('Heart_Utilized','size'),
                     Hearts_utilized=('Heart_Utilized','sum'),Utilization_rate=('Heart_Utilized','mean')).reset_index(),
                  '10_referral_year_utilization.csv')
        yr_rows=[]
        for year,part in tmp.groupby('Year'):
            for c in columns:
                yr_rows.append({'Year':year,'Variable':c,'N':len(part),'Missing_pct':100*part[c].isna().mean()})
        write_csv(pd.DataFrame(yr_rows),'10_missingness_by_referral_year.csv')

    # Numerical summaries are descriptive only; clinical units and time semantics require source documentation.
    num_rows=[]
    for c in columns:
        s=numeric(ped[c]); n=int(s.notna().sum()); original_n=int(ped[c].notna().sum())
        if n and n >= .8*max(original_n,1):
            num_rows.append({'Variable':c,'Numeric_N':n,'Min':s.min(),'Q01':s.quantile(.01),
                             'Median':s.median(),'Q99':s.quantile(.99),'Max':s.max(),
                             'Negative_N':int(s.lt(0).sum()),'Zero_N':int(s.eq(0).sum())})
    write_csv(pd.DataFrame(num_rows),'11_numeric_range_screen.csv')
    leakage_terms=['outcome_','transplant','procure','authorization','authorized','approach','asystole',
                   'brain_death','time_','referral_year','patientid','hospitalid']
    audit=[]
    for c in columns:
        hits=[term for term in leakage_terms if term in c.lower()]
        audit.append({'Variable':c,'Potential_postdecision_or_identifier_flag':int(bool(hits)),
                      'Matched_terms':'; '.join(hits), 'Requires_timing_and_definition_review':True})
    write_csv(pd.DataFrame(audit),'12_feature_timing_review.csv')
    # Check if apparently binary process variables are actually binary, without assuming unknown=0.
    binary_rows=[]
    for c in PROCESS_COLUMNS:
        if c in ped.columns:
            values=numeric(ped[c]);unexpected=values.notna()&~values.isin([0,1])
            binary_rows.append({'Variable':c,'Missing_N':int(values.isna().sum()),
                                'Unexpected_nonbinary_N':int(unexpected.sum()),
                                'Observed_values_sample':str(values.dropna().unique()[:20].tolist())})
    write_csv(pd.DataFrame(binary_rows),'13_process_code_validation.csv')
    note('This audit does not establish whether hearts were offered, declined, clinically suitable, or eligible for transplantation.')
    note('Age units, all measurement units, timestamp semantics, and ORCHID variable definitions must be verified from the data dictionary.')
    note('Missing process fields are UNKNOWN, not negative. Other-organ transplant does not prove heart offer or acceptability.')
    note('Potential leakage flags are screening prompts, not definitive classifications.')
    (OUTPUT_DIR/'00_README_and_audit_notes.txt').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    (OUTPUT_DIR/'00_run_config.json').write_text(json.dumps({'input_csv':str(INPUT_CSV),'output_dir':str(OUTPUT_DIR),
      'target':TARGET,'phase':'data audit only','imputation':False,'models_trained':False},indent=2),encoding='utf-8')
    print(f'Finished. Results: {OUTPUT_DIR}')

if __name__=='__main__':
    main()
