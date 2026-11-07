
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
OUT = ROOT / 'analysis_step15b'
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261025)
BOOT = 2000
BLOCK_SECONDS = 300
# These are fixed-route sensitivity demonstrations, NOT estimates of optimized route savings.
TAS_BY_SEGMENT = {'1925':178.21, '1986':176.58, '1450':167.02}


def load_matches():
    found = {}
    for segment in TAS_BY_SEGMENT:
        candidates = list(ROOT.glob(f'analysis_step*/**/*segment_{segment}*era5*matched*.csv'))
        candidates += list(ROOT.glob(f'analysis_step*/**/*segment_{segment}*matched*.csv'))
        # prioritize latest non-Step15B source
        candidates = [p for p in candidates if 'analysis_step15b' not in str(p)]
        candidates = sorted(set(candidates), key=lambda p: (('era5_matched' in p.name), p.stat().st_mtime), reverse=True)
        for p in candidates:
            d = pd.read_csv(p)
            if {'attrex_u_mps','attrex_v_mps','era5_u_mps','era5_v_mps'}.issubset(d.columns):
                found[segment] = (p,d)
                break
    return found


def bootstrap_blocks(block_means, n_reps=BOOT):
    """Resample non-overlapping time blocks; preserve within-block autocorrelation."""
    vals = np.asarray(block_means, dtype=float)
    idx = RNG.integers(0,len(vals),size=(n_reps,len(vals)))
    return vals[idx].mean(axis=1)


def analyze(segment, path, df):
    d = df.copy()
    d['datetime_utc'] = pd.to_datetime(d['datetime_utc'], utc=True, errors='coerce')
    d = d.dropna(subset=['datetime_utc','attrex_u_mps','attrex_v_mps','era5_u_mps','era5_v_mps']).sort_values('datetime_utc')
    d['du'] = d['era5_u_mps'] - d['attrex_u_mps']
    d['dv'] = d['era5_v_mps'] - d['attrex_v_mps']
    d['err2'] = d['du']**2 + d['dv']**2
    d['elapsed_s'] = (d['datetime_utc'] - d['datetime_utc'].iloc[0]).dt.total_seconds()
    d['block'] = (d['elapsed_s']//BLOCK_SECONDS).astype(int)
    blocks = d.groupby('block',sort=True).agg(du=('du','mean'),dv=('dv','mean'),err2=('err2','mean'),n=('du','size'))
    # Only full 5-minute blocks for block bootstrap; short remainder is excluded from CIs.
    full = blocks[blocks['n']>=0.8*BLOCK_SECONDS]
    if len(full)<4:
        print(f'WARNING {segment}: fewer than 4 full blocks; confidence intervals unreliable.')
        full = blocks
    if len(full)<2:
        raise ValueError(f'{segment}: not enough blocks')
    # Bias CI: equal weighting of complete blocks; RMSE point estimate uses all rows.
    bu = bootstrap_blocks(full['du'].to_numpy())
    bv = bootstrap_blocks(full['dv'].to_numpy())
    b2 = bootstrap_blocks(full['err2'].to_numpy())
    def ci(a): return (float(np.quantile(a,.025)),float(np.quantile(a,.975)))
    tas = TAS_BY_SEGMENT[segment]
    # Error projection along the observed horizontal ground-track tangent.
    # This estimates local along-track wind error, not a counterfactual off-track field.
    lat=np.deg2rad(d['latitude'].to_numpy()); lon=np.unwrap(np.deg2rad(d['longitude'].to_numpy()))
    east=np.gradient(lon)*np.cos(lat)*6371000
    north=np.gradient(lat)*6371000
    norm=np.hypot(east,north)
    ok=norm>0.1
    ex=np.divide(east,norm,out=np.full_like(east,np.nan),where=ok)
    ny=np.divide(north,norm,out=np.full_like(north,np.nan),where=ok)
    along=d['du'].to_numpy()*ex+d['dv'].to_numpy()*ny
    d['along_error_mps']=along
    # Illustrative effect on a 500 km fixed straight route at constant TAS.
    # First-order perturbation delta_t ~ -L * delta_w / TAS^2.
    L=500_000.0
    along_blocks=d.groupby('block')['along_error_mps'].agg(['mean','count'])
    along_full=along_blocks[along_blocks['count']>=0.8*BLOCK_SECONDS]['mean'].dropna()
    if len(along_full)<2: along_full=along_blocks['mean'].dropna()
    along_boot=bootstrap_blocks(along_full.to_numpy())
    time_error_s=-L*along_boot/(tas*tas)
    row={
        'segment':segment,'file':str(path),'matched_rows':len(d),
        'full_5min_blocks':len(full),'u_bias_mps':d['du'].mean(),
        'v_bias_mps':d['dv'].mean(),'vector_rmse_mps':np.sqrt(d['err2'].mean()),
        'u_bias_ci_low_mps':ci(bu)[0],'u_bias_ci_high_mps':ci(bu)[1],
        'v_bias_ci_low_mps':ci(bv)[0],'v_bias_ci_high_mps':ci(bv)[1],
        'vector_rmse_ci_low_mps':ci(np.sqrt(b2))[0],
        'vector_rmse_ci_high_mps':ci(np.sqrt(b2))[1],
        'alongtrack_bias_mps':np.nanmean(along),
        'alongtrack_bias_ci_low_mps':ci(along_boot)[0],
        'alongtrack_bias_ci_high_mps':ci(along_boot)[1],
        'illustrative_500km_time_error_s':-L*np.nanmean(along)/(tas*tas),
        'illustrative_500km_time_error_ci_low_s':ci(time_error_s)[0],
        'illustrative_500km_time_error_ci_high_s':ci(time_error_s)[1],
    }
    d.to_csv(OUT/f'segment_{segment}_wind_error_diagnostics.csv',index=False)
    fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True,constrained_layout=True)
    axs[0].plot(d['elapsed_s']/60,d['du'],label='ERA5 − ATTREX U',lw=.8)
    axs[0].plot(d['elapsed_s']/60,d['dv'],label='ERA5 − ATTREX V',lw=.8)
    axs[0].axhline(0,color='k',lw=.7);axs[0].legend();axs[0].set_ylabel('Component error (m/s)')
    axs[1].plot(d['elapsed_s']/60,d['along_error_mps'],lw=.8)
    axs[1].axhline(0,color='k',lw=.7);axs[1].set_ylabel('Along-track error (m/s)');axs[1].set_xlabel('Elapsed minutes')
    fig.suptitle(f'Step 15B | segment {segment} | 5-min blocked uncertainty')
    fig.savefig(OUT/f'segment_{segment}_wind_error.png',dpi=170)
    plt.close(fig)
    return row


def main():
    files=load_matches()
    if not files:
        raise FileNotFoundError('No matched ERA5–ATTREX CSVs found in analysis_step* folders.')
    rows=[]
    for segment,(path,df) in files.items():
        print('Analyzing:',segment,path)
        try: rows.append(analyze(segment,path,df))
        except Exception as exc: print('ERROR:',segment,exc)
    if not rows: raise RuntimeError('No segment analyzed')
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'step15b_uncertainty_summary.csv',index=False)
    print('\n',summary[['segment','matched_rows','full_5min_blocks','vector_rmse_mps','alongtrack_bias_mps','illustrative_500km_time_error_s']].to_string(index=False))
    print('\nSaved:',OUT)
    print('CAUTION: blocked-bootstrap intervals are descriptive, conditional on these flight segments.')
    print('CAUTION: illustrative fixed-route sensitivity is not optimized-route uncertainty or a savings estimate.')
    missing=set(TAS_BY_SEGMENT)-set(files)
    if missing: print('Missing matched CSVs (not analyzed):',', '.join(sorted(missing)))

if __name__=='__main__': main()
