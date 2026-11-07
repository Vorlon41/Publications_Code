
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
OUT = ROOT / 'analysis_step42'

def read(name, folder):
    path = ROOT / folder / (name + '.csv')
    if not path.exists():
        raise FileNotFoundError(f'Required source missing: {path}')
    return pd.read_csv(path)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    early = read('era5_attrex_validation_summary', 'analysis_step11b')
    early['segment_id'] = early['segment'].str.extract(r'(\d+)$').astype(int)
    a = early.rename(columns={'matched_rows':'n_matched','total_rows':'n_total','coverage_pct':'coverage_pct'})
    cols=['segment_id','n_total','n_matched','coverage_pct','vector_rmse_mps','u_bias_mps','v_bias_mps']
    rows=[a[cols]]
    s15=read('step15a_validation_summary','analysis_step15a')
    rows.append(s15.rename(columns={'observations_total':'n_total','observations_matched':'n_matched','match_percent':'coverage_pct'})[cols])
    for step,folder in [('step24','analysis_step24'),('step32','analysis_step32')]:
        d=read(step+'_wind_metrics',folder)
        d=d.rename(columns={'segment':'segment_id','total_rows':'n_total','n':'n_matched','bias_u_mps':'u_bias_mps','bias_v_mps':'v_bias_mps'})
        d['coverage_pct']=100*d['valid_fraction']
        rows.append(d[cols])
    validation=pd.concat(rows,ignore_index=True).sort_values('segment_id')
    if validation.segment_id.duplicated().any():
        raise ValueError('Duplicate segment validation rows; check source selection')
    validation.to_csv(OUT/'table1_era5_wind_validation.csv',index=False)

    nominal=read('step38_paired_comparison','analysis_step38')
    nominal=nominal.loc[np.isclose(nominal.spacing_km,0.25)].copy()
    nominal.to_csv(OUT/'table2_nominal_routing.csv',index=False)
    fixed=read('step39_sensitivity_summary','analysis_step39')
    fixed.to_csv(OUT/'table3_fixed_route_uncertainty.csv',index=False)
    insample=read('step40_summary','analysis_step40')
    insample.to_csv(OUT/'table4_in_sample_reoptimization.csv',index=False)
    separate=read('step41_summary','analysis_step41')
    separate.to_csv(OUT/'table5_separate_field_robustness.csv',index=False)
    numerical=read('step35_summary','analysis_step35')
    numerical.to_csv(OUT/'supplement_numerical_convergence.csv',index=False)

    print('\nSTEP 42: FINAL TABLES')
    for name,df in [('Wind validation',validation),('Nominal routing',nominal),('Fixed-route uncertainty',fixed),('In-sample reoptimization',insample),('Separate-field robustness',separate)]:
        print('\n'+name+'\n'+df.to_string(index=False))
    print('\nSaved to:',OUT)
    print('IMPORTANT: These are descriptive outputs; trials are not independent flights.')
    print('IMPORTANT: Verify source file provenance and time coverage in Step 43.')
    print('IMPORTANT: No operational flight-time or fuel savings inferred.')

if __name__=='__main__':
    main()
