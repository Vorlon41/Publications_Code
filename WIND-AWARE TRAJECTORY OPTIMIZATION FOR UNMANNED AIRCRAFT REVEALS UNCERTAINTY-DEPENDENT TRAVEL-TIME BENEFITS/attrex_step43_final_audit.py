
from pathlib import Path
import hashlib, json, sys, platform
import numpy as np
import pandas as pd

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
T42 = ROOT / 'analysis_step42'
T38 = ROOT / 'analysis_step38'
OUT = ROOT / 'analysis_step43'
TOL_SECONDS = 0.01


def check(rows, name, passed, detail, category='automated'):
    rows.append(dict(check=name, status='PASS' if passed else 'FAIL', category=category, detail=str(detail)))


def great_circle_km(a, b):
    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)
    dlat = lat2-lat1
    dlon = (lon2-lon1+np.pi)%(2*np.pi)-np.pi
    h = np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0088 * 2*np.arcsin(np.sqrt(np.clip(h,0,1)))


def route_length_km(points):
    return sum(great_circle_km(points[i],points[i+1]) for i in range(len(points)-1))


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    files = {
        'wind': 'table1_era5_wind_validation.csv',
        'nominal': 'table2_nominal_routing.csv',
        'fixed': 'table3_fixed_route_uncertainty.csv',
        'insample': 'table4_in_sample_reoptimization.csv',
        'separate': 'table5_separate_field_robustness.csv',
        'convergence': 'supplement_numerical_convergence.csv'
    }
    tables = {}
    rows = []
    manifest = []
    for name, filename in files.items():
        p = T42 / filename
        if not p.exists():
            check(rows, f'input_{name}', False, f'Missing {p}')
            continue
        tables[name] = pd.read_csv(p)
        manifest.append(dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        check(rows, f'input_{name}', True, f'{len(tables[name])} rows')
    if len(tables) != len(files):
        pd.DataFrame(rows).to_csv(OUT/'step43_audit_checks.csv',index=False)
        raise FileNotFoundError('Missing Step 42 tables. Run Step 42 and rerun this audit.')

    w=tables['wind']
    check(rows,'wind_coverage', bool(((w.n_matched<=w.n_total)&(w.n_matched>=0)&(w.coverage_pct.between(0,100))).all()), 'Coverage counts and percentages within bounds')
    pct=100*w.n_matched/w.n_total
    check(rows,'wind_coverage_arithmetic', bool(np.allclose(pct,w.coverage_pct,atol=0.05)), f'Max percentage-point discrepancy {np.max(np.abs(pct-w.coverage_pct)):.5g}')
    check(rows,'wind_rmse_finite', bool(np.isfinite(w.vector_rmse_mps).all() and (w.vector_rmse_mps>=0).all()), 'Vector RMSE finite and nonnegative')

    n=tables['nominal']
    calc=n.distance_time_s-n.wind_aware_time_s
    check(rows,'nominal_advantage_arithmetic',bool(np.allclose(calc,n.advantage_s,atol=1e-5)),f'Max discrepancy {np.max(np.abs(calc-n.advantage_s)):.7f} s')
    calc_pct=100*calc/n.distance_time_s
    check(rows,'nominal_percent_arithmetic',bool(np.allclose(calc_pct,n.saving_pct,atol=1e-5)),f'Max discrepancy {np.max(np.abs(calc_pct-n.saving_pct)):.7f} percentage points')
    check(rows,'nominal_nonnegative_distance',bool(((n.distance_km>0)&(n.wind_aware_km>0)).all()),'Positive route lengths')
    check(rows,'nominal_fifo_sample',bool((n.fifo_violations==0).all()),'Sampled violations zero; NOT a global FIFO proof')

    f=tables['fixed']
    check(rows,'fixed_percentiles',bool(((f.p025_advantage_s<=f.median_advantage_s)&(f.median_advantage_s<=f.p975_advantage_s)).all()),'Percentile ordering')
    check(rows,'fixed_trial_counts',bool(((f.n_valid<=f.n_total)&(f.n_valid>0)).all()),'Valid trials do not exceed total')
    check(rows,'fixed_fraction',bool(f.fraction_wind_aware_faster.between(0,1).all()),'Fractions in [0,1]')

    for key in ['insample','separate']:
        d=tables[key]
        check(rows,f'{key}_trial_counts',bool(((d.n_valid<=d.n_total)&(d.n_valid>=0)).all()),'Valid trial counts')
        frac_cols=[c for c in d if c.startswith('fraction_')]
        check(rows,f'{key}_fractions',bool(all(d[c].between(0,1).all() for c in frac_cols)),f'Fractions: {frac_cols}')

    c=tables['convergence']
    if {'scenario','spacing_km','max_equivalent_difference_s'}.issubset(c):
        finest=c.loc[c.groupby('scenario').spacing_km.idxmin()]
        check(rows,'equivalent_route_tolerance',bool((finest.max_equivalent_difference_s.abs()<TOL_SECONDS).all()),f'Finest-spacing max {finest.max_equivalent_difference_s.abs().max():.8f} s')
    else:
        check(rows,'equivalent_route_tolerance',False,'Missing required convergence columns')

    # Inspect original route geometry; display-only simplified routes are NOT used for travel time.
    for width in (0,50,100):
        lengths={}; ends={}
        for mode in ('distance','wind_aware'):
            p=T38/f'width{width}_{mode}_route.csv'
            if not p.exists():
                check(rows,f'route_{width}_{mode}',False,f'Missing {p}')
                continue
            xy=pd.read_csv(p)[['latitude','longitude']].to_numpy(float)
            valid=len(xy)>=2 and np.isfinite(xy).all() and np.all(np.abs(xy[:,0])<=90) and np.all(np.abs(xy[:,1])<=360)
            check(rows,f'route_{width}_{mode}',bool(valid),f'{len(xy)} vertices')
            if valid:
                lengths[mode]=route_length_km(xy)
                ends[mode]=(xy[0],xy[-1])
            manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        if len(ends)==2:
            ok=np.allclose(ends['distance'][0],ends['wind_aware'][0],atol=1e-8) and np.allclose(ends['distance'][1],ends['wind_aware'][1],atol=1e-8)
            check(rows,f'route_{width}_common_endpoints',bool(ok),'Same endpoints for both objectives')
        if len(lengths)==2:
            row=n.loc[n.width_km==width]
            if len(row):
                diffs=[abs(lengths[m]-float(row.iloc[0][('distance_km' if m=='distance' else 'wind_aware_km')])) for m in lengths]
                check(rows,f'route_{width}_length_consistency',bool(max(diffs)<0.1),f'Max difference {max(diffs):.5f} km; spherical geodesic audit tolerance 0.1 km')

    # Explicit unresolved issues; these are not silently counted as passes.
    manual=[
      ('global_fifo','Global FIFO certification or independent non-FIFO-safe search comparison not provided'),
      ('search_optimality','Optimality of time-dependent label-setting not independently certified'),
      ('wind_conventions','Confirm ATTREX MMS U/V signs, units, and pressure definition against documentation'),
      ('meteorology_provenance','Confirm ERA5 product, retrieval settings, source and time conventions'),
      ('aircraft_feasibility','No turn radius, fuel, airspace, or mission constraints certified'),
      ('uncertainty_calibration','Off-track spatial/temporal covariance remains assumed, not empirically calibrated'),
      ('external_validation','Separate simulated perturbations around same ERA5 field are not independent observed weather')
    ]
    for name,detail in manual:
        rows.append(dict(check=name,status='UNRESOLVED',category='manual',detail=detail))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'step43_audit_checks.csv',index=False)
    pd.DataFrame(manifest).to_csv(OUT/'step43_input_manifest.csv',index=False)
    metadata={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'audit_tolerance_seconds':TOL_SECONDS}
    (OUT/'step43_environment.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('\nSTEP 43 AUDIT RESULTS\n',result.to_string(index=False))
    print('\nCOUNTS:',result.status.value_counts().to_dict())
    print('Saved:',OUT)
    print('IMPORTANT: Automated passes do not resolve manual methodological limitations.')
    print('Do not freeze analysis until all FAIL items are corrected and UNRESOLVED items are addressed or explicitly scoped as limitations.')

if __name__=='__main__':
    main()
