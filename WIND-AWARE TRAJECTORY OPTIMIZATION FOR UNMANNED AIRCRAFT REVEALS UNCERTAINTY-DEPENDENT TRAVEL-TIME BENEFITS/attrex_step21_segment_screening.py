
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
OUT = ROOT / 'analysis_step21'
SCREEN = ROOT / 'analysis_step4' / 'segment_screening.csv'
# Set False only after separately resolving the 2014-03-13/14 provenance concerns.
EXCLUDE_UNRESOLVED_MAR13_14 = True
MIN_DURATION_MIN = 20.0
MIN_DIRECT_KM = 200.0
TOP_N = 15


def locate_screen():
    if SCREEN.exists():
        return SCREEN
    found = sorted(ROOT.rglob('segment_screening.csv'))
    if not found:
        raise FileNotFoundError('Cannot find segment_screening.csv beneath '+str(ROOT))
    if len(found)>1:
        print('Multiple screening files found; using:', found[0])
    return found[0]


def date_from_name(s):
    match = re.search(r'(20\d{6})', str(s))
    return match.group(1) if match else None


def robust_unit_interval(s):
    """Rank by empirical percentiles, avoiding invented physical thresholds."""
    return pd.to_numeric(s, errors='coerce').rank(pct=True).fillna(0.0)


def find_obs(date, segment):
    if not date:
        return None
    f = ROOT / 'analysis_step9' / f'{date}_segment_{segment}_observations.csv'
    return f if f.exists() else None


def wind_diagnostics(obs_path):
    """Descriptive ATTREX along-track variability only, not ERA5 off-track."""
    result = {'obs_available': False, 'obs_valid_wind_rows': 0,
              'obs_wind_p90_mps': np.nan, 'obs_wind_speed_iqr_mps': np.nan,
              'obs_crosswind_abs_p90_mps': np.nan,
              'obs_alongwind_sd_mps': np.nan,
              'obs_wind_first_last_vector_change_mps': np.nan}
    if obs_path is None:
        return result
    try:
        df = pd.read_csv(obs_path, usecols=lambda x: x in
                         {'U','V','G_LAT','G_LONG','TIME_UTC'})
        for col in ('U','V','G_LAT','G_LONG'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.replace([np.inf,-np.inf],np.nan).dropna(subset=['U','V','G_LAT','G_LONG'])
        if len(df)<10:
            return result
        u,v = df.U.to_numpy(),df.V.to_numpy()
        speed = np.hypot(u,v)
        lat = np.deg2rad(df.G_LAT.to_numpy())
        lon = np.unwrap(np.deg2rad(df.G_LONG.to_numpy()))
        north = np.gradient(lat)
        east = np.gradient(lon)*np.cos(lat)
        mag = np.hypot(east,north)
        good = mag>1e-10
        along = np.full(len(df),np.nan)
        cross = np.full(len(df),np.nan)
        along[good]=(u[good]*east[good]+v[good]*north[good])/mag[good]
        cross[good]=(-u[good]*north[good]+v[good]*east[good])/mag[good]
        k = max(1,len(df)//10)
        first = np.array([np.median(u[:k]),np.median(v[:k])])
        last = np.array([np.median(u[-k:]),np.median(v[-k:])])
        result.update(obs_available=True,obs_valid_wind_rows=len(df),
                      obs_wind_p90_mps=float(np.percentile(speed,90)),
                      obs_wind_speed_iqr_mps=float(np.percentile(speed,75)-np.percentile(speed,25)),
                      obs_crosswind_abs_p90_mps=float(np.nanpercentile(np.abs(cross),90)),
                      obs_alongwind_sd_mps=float(np.nanstd(along)),
                      obs_wind_first_last_vector_change_mps=float(np.linalg.norm(last-first)))
    except (ValueError,KeyError,OSError) as e:
        print('Observation diagnostic unavailable for',obs_path.name,':',e)
    return result


def era5_inventory():
    """Inspect GRIB coordinate extents, not per-segment coverage or wind skill."""
    rows=[]
    gribs=sorted(ROOT.glob('*.grib'))+sorted(ROOT.glob('*.grb'))
    if not gribs:
        return pd.DataFrame(columns=['grib','status','time_min','time_max',
                                     'lat_min','lat_max','lon_min','lon_max'])
    try:
        import xarray as xr
    except ImportError:
        return pd.DataFrame([{'grib':p.name,'status':'xarray unavailable'} for p in gribs])
    for path in gribs:
        item={'grib':path.name,'status':'unreadable'}
        try:
            ds=xr.open_dataset(path,engine='cfgrib',backend_kwargs={'indexpath':''})
            for name in ('latitude','longitude'):
                if name in ds.coords:
                    a=np.asarray(ds[name].values,float)
                    item['lat_min' if name=='latitude' else 'lon_min']=float(np.nanmin(a))
                    item['lat_max' if name=='latitude' else 'lon_max']=float(np.nanmax(a))
            if 'time' in ds.coords:
                t=pd.to_datetime(ds.time.values,utc=True)
                item['time_min']=str(t.min());item['time_max']=str(t.max())
            item['status']='opened'
            ds.close()
        except Exception as e:
            item['status']=f'{type(e).__name__}: {str(e)[:100]}'
        rows.append(item)
    return pd.DataFrame(rows)


def main():
    if not ROOT.exists():
        raise FileNotFoundError('ATTREX project directory does not exist: '+str(ROOT))
    OUT.mkdir(parents=True,exist_ok=True)
    src=locate_screen()
    df=pd.read_csv(src)
    required=['file','segment_id','duration_min','direct_distance_km',
              'route_ratio','median_wind_mps','wind_speed_sd_mps']
    missing=[c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Missing screening columns: '+str(missing))
    df['flight_date']=df['file'].map(date_from_name)
    df['segment_id']=pd.to_numeric(df.segment_id,errors='coerce').astype('Int64')
    df['unresolved_provenance']=df.flight_date.isin(['20140313','20140314'])
    df['meets_basic_length']=(pd.to_numeric(df.duration_min,errors='coerce')>=MIN_DURATION_MIN)&(pd.to_numeric(df.direct_distance_km,errors='coerce')>=MIN_DIRECT_KM)
    df['eligible_initial']=df['eligible_initial'].astype(str).str.lower().isin(['true','1','yes']) if 'eligible_initial' in df else True
    df['screen_eligible']=df.meets_basic_length & df.eligible_initial
    if EXCLUDE_UNRESOLVED_MAR13_14:
        df['screen_eligible'] &= ~df.unresolved_provenance
    # Wind variability is only observed on the flown track; use as a heuristic.
    weights={'direct_distance_km':.22,'duration_min':.10,'median_wind_mps':.12,
             'wind_speed_sd_mps':.25,'p95_wind_mps':.16}
    df['screen_score']=0.0
    for key,weight in weights.items():
        if key in df:
            df['screen_score']+=weight*robust_unit_interval(df[key])
    # Route ratio indicates path geometry, not free route optimization opportunity.
    df['screen_score']+=.15*robust_unit_interval(df['route_ratio'])
    df['screen_score']=df['screen_score'].round(4)
    df['observation_csv']=''
    diagnostics=[]
    for _,row in df.iterrows():
        if pd.isna(row.segment_id):
            diagnostics.append(wind_diagnostics(None));continue
        p=find_obs(row.flight_date,int(row.segment_id))
        diagnostics.append(wind_diagnostics(p))
        if p is not None:
            df.at[row.name,'observation_csv']=str(p)
    df=pd.concat([df.reset_index(drop=True),pd.DataFrame(diagnostics)],axis=1)
    df=df.sort_values(['screen_eligible','screen_score'],ascending=[False,False]).reset_index(drop=True)
    df['screen_rank']=np.where(df.screen_eligible,df.screen_eligible.cumsum(),np.nan)
    df.to_csv(OUT/'step21_all_segments_screened.csv',index=False)
    shortlist=df[df.screen_eligible].head(TOP_N).copy()
    shortlist.to_csv(OUT/'step21_candidate_shortlist.csv',index=False)
    era=era5_inventory()
    era.to_csv(OUT/'step21_era5_file_inventory.csv',index=False)
    # Metadata-only coverage warnings. Longitude/time/pressure need verification.
    if len(shortlist):
        fig,ax=plt.subplots(figsize=(10,max(4,0.42*len(shortlist)+1)))
        labels=[f'{d} / {s}' for d,s in zip(shortlist.flight_date,shortlist.segment_id)]
        ax.barh(labels[::-1],shortlist.screen_score.to_numpy()[::-1])
        ax.set(xlabel='Heuristic screening score (relative; not modeled savings)',
               title='ATTREX Step 21 | additional flight candidates')
        ax.grid(axis='x',alpha=.2)
        fig.tight_layout()
        fig.savefig(OUT/'step21_candidate_ranking.png',dpi=190)
        plt.close(fig)
    cols=['flight_date','segment_id','duration_min','direct_distance_km',
          'median_wind_mps','wind_speed_sd_mps','screen_score',
          'obs_available','unresolved_provenance']
    print('\nScreened:',len(df),'segments; eligible:',int(df.screen_eligible.sum()))
    print('\nTOP CANDIDATES (NOT ROUTING RESULTS):\n',shortlist[cols].to_string(index=False))
    print('\nERA5 files inspected:',len(era))
    if len(era):
        print(era[['grib','status','time_min','time_max']].to_string(index=False)
              if {'time_min','time_max'}.issubset(era.columns) else era[['grib','status']].to_string(index=False))
    print('\nSaved:',OUT)
    print('CAUTION: observational winds are along-track only; no off-track gradient or savings inferred.')
    print('CAUTION: GRIB inventory does not establish spatial-temporal coverage of any candidate.')
    print('CAUTION: 2014-03-13/14 excluded pending independent provenance resolution.' if EXCLUDE_UNRESOLVED_MAR13_14 else 'WARNING: provenance-excluded dates enabled.')
    print('NEXT: check full ERA5 coverage and science/airspace constraints before optimization.')

if __name__=='__main__':
    main()
