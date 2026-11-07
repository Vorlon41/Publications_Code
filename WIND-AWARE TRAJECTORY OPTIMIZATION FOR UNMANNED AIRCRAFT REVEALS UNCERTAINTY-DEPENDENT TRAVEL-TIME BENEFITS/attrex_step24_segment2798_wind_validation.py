from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
OBS = ROOT / 'analysis_step23' / '20140311_segment_2798_observations.csv'
GRIB = ROOT / 'ERA5_ATTREX_20140311_20140312_expanded.grib'
OUT = ROOT / 'analysis_step24'
CHUNK = 250
BLOCK_SECONDS = 300
BOOTSTRAPS = 1000
SEED = 2401


def interp_wind(ds, obs):
    import xarray as xr
    times = pd.to_datetime(obs['datetime_utc'], utc=True)
    lats = obs['G_LAT'].to_numpy(float)
    lons = obs['G_LONG'].to_numpy(float)
    pressures = obs['P'].to_numpy(float)
    glon = np.asarray(ds.longitude.values, float)
    if glon.min() >= 0 and glon.max() > 180:
        lons = np.mod(lons, 360)
    else:
        lons = (lons + 180) % 360 - 180
    time_grid = pd.to_datetime(ds.time.values, utc=True)
    domain_ok = (np.isfinite(lats) & np.isfinite(lons) & np.isfinite(pressures)
                 & (lats >= float(ds.latitude.min())) & (lats <= float(ds.latitude.max()))
                 & (lons >= glon.min()) & (lons <= glon.max())
                 & (pressures >= float(ds.isobaricInhPa.min()))
                 & (pressures <= float(ds.isobaricInhPa.max()))
                 & (times >= time_grid.min()).to_numpy()
                 & (times <= time_grid.max()).to_numpy())
    u = np.full(len(obs), np.nan)
    v = np.full(len(obs), np.nan)
    indices = np.flatnonzero(domain_ok)
    for offset in range(0, len(indices), CHUNK):
        ix = indices[offset:offset + CHUNK]
        q = ds[['u', 'v']].interp(
            time=xr.DataArray(times.iloc[ix].to_numpy(dtype='datetime64[ns]'), dims='point'),
            latitude=xr.DataArray(lats[ix], dims='point'),
            longitude=xr.DataArray(lons[ix], dims='point'),
            isobaricInhPa=xr.DataArray(pressures[ix], dims='point'),
            method='linear',
        )
        u[ix] = np.asarray(q.u.values).reshape(-1)
        v[ix] = np.asarray(q.v.values).reshape(-1)
    return u, v, domain_ok


def summary_metrics(df):
    du = df['du'].to_numpy(float)
    dv = df['dv'].to_numpy(float)
    e = np.hypot(du, dv)
    obs_u, obs_v = df.U.to_numpy(float), df.V.to_numpy(float)
    era_u, era_v = df.era5_u.to_numpy(float), df.era5_v.to_numpy(float)
    # Vector RMSE = sqrt(mean(du^2 + dv^2)), not component-wise average.
    return dict(n=len(df), bias_u_mps=float(np.mean(du)), bias_v_mps=float(np.mean(dv)),
                rmse_u_mps=float(np.sqrt(np.mean(du**2))),
                rmse_v_mps=float(np.sqrt(np.mean(dv**2))),
                vector_rmse_mps=float(np.sqrt(np.mean(e**2))),
                vector_mae_mps=float(np.mean(e)),
                corr_u=float(np.corrcoef(obs_u, era_u)[0, 1]) if np.std(obs_u)*np.std(era_u) > 0 else np.nan,
                corr_v=float(np.corrcoef(obs_v, era_v)[0, 1]) if np.std(obs_v)*np.std(era_v) > 0 else np.nan)


def bootstrap(df):
    """Resample contiguous 300-s time blocks; intervals are descriptive, not independent trials."""
    d = df.copy()
    seconds = (pd.to_datetime(d.datetime_utc, utc=True) - pd.to_datetime(d.datetime_utc.iloc[0], utc=True)).dt.total_seconds()
    d['block'] = np.floor(seconds / BLOCK_SECONDS).astype(int)
    groups = [g for _, g in d.groupby('block')]
    rng = np.random.default_rng(SEED)
    stats = []
    for _ in range(BOOTSTRAPS):
        chosen = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in chosen], ignore_index=True)
        m = summary_metrics(sample)
        stats.append([m['bias_u_mps'], m['bias_v_mps'], m['vector_rmse_mps']])
    arr = np.asarray(stats)
    rows = []
    for j, key in enumerate(['bias_u_mps', 'bias_v_mps', 'vector_rmse_mps']):
        rows.append(dict(metric=key, estimate=summary_metrics(df)[key],
                         percentile_2_5=float(np.percentile(arr[:, j], 2.5)),
                         percentile_97_5=float(np.percentile(arr[:, j], 97.5)),
                         n_blocks=len(groups), block_seconds=BLOCK_SECONDS,
                         bootstrap_replicates=BOOTSTRAPS))
    return pd.DataFrame(rows)


def figures(df):
    t = (pd.to_datetime(df.datetime_utc, utc=True) - pd.to_datetime(df.datetime_utc.iloc[0], utc=True)).dt.total_seconds().to_numpy() / 60
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, name, era, label in [(axes[0], 'U', 'era5_u', 'Eastward U (m/s)'),
                                  (axes[1], 'V', 'era5_v', 'Northward V (m/s)')]:
        ax.plot(t, df[name], label='ATTREX MMS', lw=1)
        ax.plot(t, df[era], label='ERA5 collocated', lw=1, alpha=.85)
        ax.set_ylabel(label); ax.grid(alpha=.25); ax.legend(loc='best')
    axes[2].plot(t, df.vector_error_mps, lw=1, color='tab:purple')
    axes[2].set_ylabel('Vector error magnitude (m/s)')
    axes[2].set_xlabel('Minutes since segment start (UTC)'); axes[2].grid(alpha=.25)
    fig.suptitle('Step 24 | Segment 2798 | ATTREX vs ERA5 winds')
    fig.tight_layout(); fig.savefig(OUT / 'step24_wind_timeseries.png', dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, o, e, label in [(axes[0], 'U', 'era5_u', 'U (m/s)'), (axes[1], 'V', 'era5_v', 'V (m/s)')]:
        ax.scatter(df[o], df[e], s=5, alpha=.25)
        lo = min(df[o].min(), df[e].min()); hi = max(df[o].max(), df[e].max())
        ax.plot([lo, hi], [lo, hi], '--', color='black', lw=1)
        ax.set(xlabel='ATTREX ' + label, ylabel='ERA5 ' + label)
        ax.grid(alpha=.2)
    fig.suptitle('Step 24 | Collocated wind component agreement')
    fig.tight_layout(); fig.savefig(OUT / 'step24_wind_scatter.png', dpi=180); plt.close(fig)


def main():
    import xarray as xr
    if not OBS.exists():
        raise FileNotFoundError(f'Missing observations: {OBS}. Run Step 23 first.')
    if not GRIB.exists():
        raise FileNotFoundError(f'Missing GRIB: {GRIB}')
    OUT.mkdir(parents=True, exist_ok=True)
    obs = pd.read_csv(OBS)
    required = {'datetime_utc', 'P', 'G_LAT', 'G_LONG', 'U', 'V'}
    if not required.issubset(obs):
        raise ValueError(f'Observation file missing: {sorted(required - set(obs))}')
    for col in ['P', 'G_LAT', 'G_LONG', 'U', 'V']:
        obs[col] = pd.to_numeric(obs[col], errors='coerce')
    obs['datetime_utc'] = pd.to_datetime(obs.datetime_utc, utc=True)
    obs = obs.sort_values('datetime_utc').reset_index(drop=True)
    ds = xr.open_dataset(GRIB, engine='cfgrib', backend_kwargs={'indexpath': ''})
    try:
        print('ERA5:', GRIB)
        print('ERA5 time:', pd.to_datetime(ds.time.values, utc=True).min(), 'to', pd.to_datetime(ds.time.values, utc=True).max())
        if not {'u', 'v'}.issubset(ds.data_vars):
            raise ValueError('ERA5 GRIB must contain u and v')
        eu, ev, ok = interp_wind(ds, obs)
    finally:
        ds.close()
    obs['era5_u'] = eu; obs['era5_v'] = ev
    obs['within_era5_domain'] = ok
    obs['du'] = obs.era5_u - obs.U
    obs['dv'] = obs.era5_v - obs.V
    obs['vector_error_mps'] = np.hypot(obs.du, obs.dv)
    obs.to_csv(OUT / 'step24_full_collocation.csv', index=False)
    valid = obs.dropna(subset=['U', 'V', 'era5_u', 'era5_v']).copy()
    print(f'Observations: {len(obs)}; within domain: {ok.sum()}; valid comparisons: {len(valid)}')
    if len(valid) < 100:
        raise ValueError('Insufficient valid paired winds; review GRIB coverage and input columns')
    metrics = summary_metrics(valid)
    pd.DataFrame([dict(segment=2798, **metrics, total_rows=len(obs),
                       valid_fraction=len(valid)/len(obs))]).to_csv(OUT / 'step24_wind_metrics.csv', index=False)
    ci = bootstrap(valid)
    ci.to_csv(OUT / 'step24_block_bootstrap.csv', index=False)
    figures(valid)
    print('\nWIND VALIDATION (ERA5 minus aircraft; units m/s):')
    for key, value in metrics.items():
        print(f'  {key}: {value:.5f}' if isinstance(value, float) else f'  {key}: {value}')
    print('\nBLOCK BOOTSTRAP (descriptive 2.5–97.5 percentiles):')
    print(ci.to_string(index=False))
    print('\nSaved:', OUT)
    print('CAUTION: ATTREX MMS U/V convention must be verified against dataset documentation.')
    print('CAUTION: ERA5 agreement is only evaluated along the observed trajectory.')
    print('CAUTION: autocorrelation and systematic model bias limit bootstrap interpretation.')
    print('CAUTION: no routing benefit, fuel savings, or aircraft feasibility inferred.')


if __name__ == '__main__':
    main()
