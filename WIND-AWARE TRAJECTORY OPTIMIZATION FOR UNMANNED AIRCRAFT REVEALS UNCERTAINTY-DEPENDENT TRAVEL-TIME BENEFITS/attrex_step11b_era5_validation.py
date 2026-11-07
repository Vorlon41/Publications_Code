
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825")
GRIB = ROOT / "4218a346c038961a43934d722972a5ad.grib"
OBS_DIR = ROOT / "analysis_step9"
OUT = ROOT / "analysis_step11b"
FILES = ["20140311_segment_1986_observations.csv", "20140311_segment_1450_observations.csv"]
G0 = 9.80665


def load_era5(path):
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    for variable in ("u", "v", "z"):
        if variable not in ds:
            raise ValueError(f"ERA5 missing {variable}")
    for dim in ("time", "isobaricInhPa", "latitude", "longitude"):
        if dim not in ds.dims:
            raise ValueError(f"ERA5 missing dimension {dim}")
    ds = ds.sortby("time").sortby("isobaricInhPa").sortby("latitude").sortby("longitude")
    times = ds.time.values.astype("datetime64[ns]").astype("int64") / 1e9
    pressure = np.log(ds.isobaricInhPa.values.astype(float))
    lat = ds.latitude.values.astype(float)
    lon = ds.longitude.values.astype(float)
    axes = (times, pressure, lat, lon)
    fields = {}
    for key in ("u", "v", "z"):
        fields[key] = RegularGridInterpolator(axes, ds[key].values, method="linear", bounds_error=False, fill_value=np.nan)
    return ds, axes, fields


def process_one(path, axes, fields):
    df = pd.read_csv(path)
    needed = ["TIME_UTC", "P", "U", "V", "G_LAT", "G_LONG", "G_ALT"]
    for name in needed:
        if name not in df:
            raise ValueError(f"{path.name}: missing {name}")
    # ICARTT TIME_UTC is elapsed seconds from UTC midnight of file date.
    # If values exceed 86400, pandas correctly rolls them into the next date.
    day = pd.Timestamp(path.name[:8], tz="UTC")
    t = day + pd.to_timedelta(df.TIME_UTC, unit="s")
    t_seconds = t.astype("int64").to_numpy() / 1e9
    longitude = np.mod(pd.to_numeric(df.G_LONG, errors="coerce").to_numpy(), 360.0)
    lat = pd.to_numeric(df.G_LAT, errors="coerce").to_numpy()
    p = pd.to_numeric(df.P, errors="coerce").to_numpy()
    queries = np.column_stack((t_seconds, np.log(np.where(p > 0, p, np.nan)), lat, longitude))
    finite = np.all(np.isfinite(queries), axis=1)
    inside = finite.copy()
    for k, axis in enumerate(axes):
        inside &= (queries[:, k] >= axis[0]) & (queries[:, k] <= axis[-1])
    df["timestamp_utc"] = t.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    df["inside_era5_domain"] = inside
    for var in ("u", "v", "z"):
        vals = np.full(len(df), np.nan)
        if inside.any():
            vals[inside] = fields[var](queries[inside])
        df[f"era5_{var}"] = vals
    df["era5_geopotential_height_m"] = df.era5_z / G0
    df["height_difference_gps_minus_era5_m"] = df.G_ALT - df.era5_geopotential_height_m
    df["u_error_mps"] = df.era5_u - df.U
    df["v_error_mps"] = df.era5_v - df.V
    df["vector_error_mps"] = np.hypot(df.u_error_mps, df.v_error_mps)
    observed_speed = np.hypot(df.U, df.V)
    modeled_speed = np.hypot(df.era5_u, df.era5_v)
    df["speed_error_mps"] = modeled_speed - observed_speed
    obs_direction = np.arctan2(df.V, df.U)
    mod_direction = np.arctan2(df.era5_v, df.era5_u)
    df["direction_abs_error_deg"] = np.abs(np.rad2deg(np.arctan2(np.sin(mod_direction-obs_direction), np.cos(mod_direction-obs_direction))))
    # Direction is ill-defined at very low wind speeds; exclude winds < 2 m/s.
    df.loc[(observed_speed < 2) | (modeled_speed < 2), "direction_abs_error_deg"] = np.nan
    valid = np.isfinite(df.vector_error_mps.to_numpy())
    def mean(series):
        a = np.asarray(series, float)
        return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
    def rmse(series):
        a = np.asarray(series, float)
        return float(np.sqrt(np.nanmean(a*a))) if np.isfinite(a).any() else np.nan
    summary = dict(segment=path.name.replace("_observations.csv", ""), total_rows=len(df), matched_rows=int(valid.sum()), coverage_pct=round(100*valid.mean(),2),
                   first_utc=t.min().isoformat(), last_utc=t.max().isoformat(), pressure_min_hpa=float(np.nanmin(p)), pressure_max_hpa=float(np.nanmax(p)),
                   u_bias_mps=mean(df.u_error_mps), v_bias_mps=mean(df.v_error_mps), u_rmse_mps=rmse(df.u_error_mps), v_rmse_mps=rmse(df.v_error_mps),
                   vector_rmse_mps=rmse(df.vector_error_mps), speed_mae_mps=mean(np.abs(df.speed_error_mps)),
                   direction_mae_deg=mean(df.direction_abs_error_deg), gps_minus_geopotential_height_median_m=float(np.nanmedian(df.height_difference_gps_minus_era5_m)) if valid.any() else np.nan)
    return df, summary


def figure(df, summary, output):
    t = pd.to_datetime(df.timestamp_utc, utc=True)
    fig, axs = plt.subplots(3,1,figsize=(12,9),sharex=True)
    axs[0].plot(t, df.U, label="ATTREX U", lw=.8)
    axs[0].plot(t, df.era5_u, label="ERA5 U", lw=1.2)
    axs[0].set_ylabel("Eastward wind (m/s)"); axs[0].legend(ncol=2)
    axs[1].plot(t, df.V, label="ATTREX V", lw=.8)
    axs[1].plot(t, df.era5_v, label="ERA5 V", lw=1.2)
    axs[1].set_ylabel("Northward wind (m/s)"); axs[1].legend(ncol=2)
    axs[2].plot(t, df.vector_error_mps, lw=.9)
    axs[2].set_ylabel("Vector error (m/s)"); axs[2].set_xlabel("UTC")
    fig.suptitle(f"{summary['segment']} | matched {summary['matched_rows']}/{summary['total_rows']} | vector RMSE {summary['vector_rmse_mps']:.2f} m/s")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(output,dpi=170,bbox_inches="tight"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading ERA5:", GRIB)
    ds, axes, fields = load_era5(GRIB)
    print("ERA5 time coverage:",str(ds.time.values[0]),"to",str(ds.time.values[-1]))
    print("ERA5 pressure levels:",ds.isobaricInhPa.values)
    summaries=[]
    for name in FILES:
        df, summary = process_one(OBS_DIR/name, axes, fields)
        summaries.append(summary)
        stem = name.replace("_observations.csv", "")
        df.to_csv(OUT/f"{stem}_era5_collocated.csv",index=False)
        figure(df,summary,OUT/f"{stem}_era5_comparison.png")
        print(f"{stem}: {summary['matched_rows']}/{summary['total_rows']} matched ({summary['coverage_pct']:.1f}%), vector RMSE {summary['vector_rmse_mps']:.2f} m/s")
        if summary['coverage_pct'] < 95:
            print("  WARNING: low matching coverage; inspect pressure and geographic limits before interpretation.")
    pd.DataFrame(summaries).to_csv(OUT/"era5_attrex_validation_summary.csv",index=False)
    print("Saved outputs:",OUT)
    print("NOTE: hourly reanalysis vs 1-Hz measurements; errors include unresolved spatial/temporal variability.")

if __name__ == "__main__":
    main()
