
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
GRIB = ROOT / '6ae42b7a70fdd5fdc0662512e3830732.grib'
OBS = ROOT / 'analysis_step9' / '20150305_segment_1925_observations.csv'
OUT = ROOT / 'analysis_step15a'
OUT.mkdir(parents=True, exist_ok=True)

print('Loading observations:', OBS)
df = pd.read_csv(OBS)
print('Observation columns:', list(df.columns))
print('Loading ERA5:', GRIB)
ds = xr.open_dataset(GRIB, engine='cfgrib', backend_kwargs={'indexpath': ''}).sortby('isobaricInhPa')

# ICARTT TIME_UTC is seconds since midnight UTC, not a Unix timestamp.
# The 2015-03-05 file date is the reference even if seconds exceed 86400.
def col(options):
    lookup = {str(c).lower(): c for c in df.columns}
    for name in options:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise KeyError(f'Expected one of {options}; available: {list(df.columns)}')

time_col = col(['TIME_UTC', 'time_utc'])
lat_col = col(['G_LAT', 'latitude', 'lat'])
lon_col = col(['G_LONG', 'longitude', 'lon'])
alt_col = col(['G_ALT', 'altitude_m', 'altitude'])
u_col = col(['U', 'u'])
v_col = col(['V', 'v'])

work = pd.DataFrame({
    'time_utc_s': pd.to_numeric(df[time_col], errors='coerce'),
    'latitude': pd.to_numeric(df[lat_col], errors='coerce'),
    'longitude': pd.to_numeric(df[lon_col], errors='coerce'),
    'altitude_m': pd.to_numeric(df[alt_col], errors='coerce'),
    'attrex_u_mps': pd.to_numeric(df[u_col], errors='coerce'),
    'attrex_v_mps': pd.to_numeric(df[v_col], errors='coerce'),
})
work['datetime_utc'] = pd.Timestamp('2015-03-05') + pd.to_timedelta(work['time_utc_s'], unit='s')
work = work.replace([np.inf, -np.inf], np.nan)

# First interpolate ERA5 in time/latitude/longitude at all available pressure levels.
# xarray broadcasts vectorized 'sample' queries and retains the pressure dimension.
valid = work[['datetime_utc','latitude','longitude','altitude_m','attrex_u_mps','attrex_v_mps']].notna().all(axis=1)
latmin, latmax = float(ds.latitude.min()), float(ds.latitude.max())
lonmin, lonmax = float(ds.longitude.min()), float(ds.longitude.max())
tmin, tmax = pd.Timestamp(ds.time.values.min()), pd.Timestamp(ds.time.values.max())
valid &= work.latitude.between(latmin, latmax) & work.longitude.between(lonmin, lonmax)
valid &= work.datetime_utc.between(tmin, tmax)
print(f'Within ERA5 time/spatial bounds: {valid.sum()}/{len(work)}')
subset = work.loc[valid].copy()
if subset.empty:
    raise RuntimeError('No matching observations; check file dates and spatial coverage.')
query = dict(
    time=xr.DataArray(subset.datetime_utc.to_numpy(dtype='datetime64[ns]'), dims='sample'),
    latitude=xr.DataArray(subset.latitude.to_numpy(), dims='sample'),
    longitude=xr.DataArray(subset.longitude.to_numpy(), dims='sample'),
)
interp = ds[['u','v','z']].interp(query, method='linear')
# geopotential (m^2/s^2) -> geopotential height (m), approximately.
height = np.asarray((interp.z / 9.80665).transpose('sample','isobaricInhPa'))
ui = np.asarray(interp.u.transpose('sample','isobaricInhPa'))
vi = np.asarray(interp.v.transpose('sample','isobaricInhPa'))
target_alt = subset.altitude_m.to_numpy()

def interpolate_height(h, values, target):
    ok = np.isfinite(h) & np.isfinite(values)
    if ok.sum() < 2:
        return np.nan
    hh = h[ok]; vv = values[ok]
    order = np.argsort(hh)
    hh, vv = hh[order], vv[order]
    if target < hh[0] or target > hh[-1]:
        return np.nan  # No extrapolation beyond pressure-level coverage
    return np.interp(target, hh, vv)

subset['era5_u_mps'] = [interpolate_height(height[i], ui[i], target_alt[i]) for i in range(len(subset))]
subset['era5_v_mps'] = [interpolate_height(height[i], vi[i], target_alt[i]) for i in range(len(subset))]
subset = subset.dropna(subset=['era5_u_mps','era5_v_mps'])
subset['u_error_mps'] = subset.era5_u_mps - subset.attrex_u_mps
subset['v_error_mps'] = subset.era5_v_mps - subset.attrex_v_mps
subset['vector_error_mps'] = np.hypot(subset.u_error_mps, subset.v_error_mps)
subset['attrex_wind_mps'] = np.hypot(subset.attrex_u_mps, subset.attrex_v_mps)
subset['era5_wind_mps'] = np.hypot(subset.era5_u_mps, subset.era5_v_mps)

def safe_corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a,b)[0,1])

summary = {
    'segment_id': 1925,
    'observations_total': len(work),
    'observations_matched': len(subset),
    'match_percent': 100 * len(subset) / len(work),
    'vector_rmse_mps': np.sqrt(np.mean(subset.vector_error_mps ** 2)) if len(subset) else np.nan,
    'vector_mae_mps': subset.vector_error_mps.mean(),
    'u_bias_mps': subset.u_error_mps.mean(),
    'v_bias_mps': subset.v_error_mps.mean(),
    'u_rmse_mps': np.sqrt(np.mean(subset.u_error_mps ** 2)) if len(subset) else np.nan,
    'v_rmse_mps': np.sqrt(np.mean(subset.v_error_mps ** 2)) if len(subset) else np.nan,
    'u_correlation': safe_corr(subset.attrex_u_mps, subset.era5_u_mps),
    'v_correlation': safe_corr(subset.attrex_v_mps, subset.era5_v_mps),
    'observed_median_wind_mps': subset.attrex_wind_mps.median(),
    'era5_median_wind_mps': subset.era5_wind_mps.median(),
}
subset.to_csv(OUT / '20150305_segment_1925_era5_matched.csv', index=False)
pd.DataFrame([summary]).to_csv(OUT / 'step15a_validation_summary.csv', index=False)

fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for ax, observed, model, title in [
    (axes[0], 'attrex_u_mps','era5_u_mps','Eastward wind (U)'),
    (axes[1], 'attrex_v_mps','era5_v_mps','Northward wind (V)'),
]:
    ax.plot(subset.datetime_utc, subset[observed], label='ATTREX', lw=1)
    ax.plot(subset.datetime_utc, subset[model], label='ERA5', lw=1.4)
    ax.set_ylabel('Wind (m/s)'); ax.set_title(title); ax.legend(); ax.grid(alpha=.2)
axes[2].plot(subset.datetime_utc, subset.vector_error_mps, lw=1)
axes[2].set_ylabel('Vector error (m/s)'); axes[2].set_xlabel('UTC'); axes[2].grid(alpha=.2)
fig.suptitle(f"Step 15A | 2015-03-05 segment 1925 | vector RMSE {summary['vector_rmse_mps']:.2f} m/s")
fig.tight_layout()
fig.savefig(OUT / 'step15a_era5_validation.png', dpi=180)
plt.show()
print(pd.DataFrame([summary]).to_string(index=False))
print('Saved:', OUT)
print('CAUTION: geopotential height approximation; ERA5 hourly resolution; no off-track observations.')
