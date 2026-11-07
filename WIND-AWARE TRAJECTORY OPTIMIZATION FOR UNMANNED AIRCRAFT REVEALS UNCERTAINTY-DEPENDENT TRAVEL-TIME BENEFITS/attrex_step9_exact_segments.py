
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
SCREEN = ROOT / 'analysis_step4' / 'segment_screening.csv'
CLEAN = ROOT / 'analysis_step2'
OUT = ROOT / 'analysis_step9'
OUT.mkdir(exist_ok=True)
TARGETS = [('20150305',1925),('20140311',1986),('20140311',1450)]

def haversine_km(lat1, lon1, lat2, lon2):
    r=6371.0088
    p1,p2=np.radians(lat1),np.radians(lat2)
    dp=p2-p1
    dl=np.radians((lon2-lon1+180)%360-180)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*r*np.arcsin(np.sqrt(np.clip(a,0,1)))

screen=pd.read_csv(SCREEN)
records=[]
for date, segid in TARGETS:
    rows=screen[screen['file'].str.contains(date,regex=False) & (screen.segment_id==segid)]
    if len(rows)!=1:
        raise ValueError(f'Expected one screening row for {date}/{segid}; found {len(rows)}')
    info=rows.iloc[0]
    path=CLEAN/info['file']
    df=pd.read_csv(path)
    required=['TIME_UTC','G_LAT','G_LONG','G_ALT','TAS','U','V']
    missing=set(required)-set(df.columns)
    if missing: raise ValueError(f'{path.name} missing {sorted(missing)}')
    df=df.loc[df.TIME_UTC.between(info.start_s-0.002,info.end_s+0.002)].copy()
    df=df.sort_values('TIME_UTC').drop_duplicates('TIME_UTC')
    for c in required: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.replace([np.inf,-np.inf],np.nan)
    df=df.dropna(subset=required)
    df=df[df.G_LAT.between(-90,90) & df.G_LONG.between(-180.01,180.01) & (df.TAS>0)]
    if len(df)<100: raise ValueError(f'Too few valid samples for {date}/{segid}')
    df['wind_speed_mps']=np.hypot(df.U,df.V)
    df['elapsed_min']=(df.TIME_UTC-df.TIME_UTC.iloc[0])/60
    df['lon_unwrapped']=np.degrees(np.unwrap(np.radians(df.G_LONG.to_numpy())))
    lat=df.G_LAT.to_numpy(); lon=df.G_LONG.to_numpy(); t=df.TIME_UTC.to_numpy()
    steps=haversine_km(lat[:-1],lon[:-1],lat[1:],lon[1:])
    dt=np.diff(t)
    valid=(dt>0)&(dt<=2.0)
    observed_km=float(steps[valid].sum())
    direct_km=float(haversine_km(lat[0],lon[0],lat[-1],lon[-1]))
    name=f'{date}_segment_{segid}'
    df.to_csv(OUT/f'{name}_observations.csv',index=False)
    fig,axs=plt.subplots(2,1,figsize=(11,8),layout='constrained')
    scatter=axs[0].scatter(df.lon_unwrapped,df.G_LAT,c=df.wind_speed_mps,s=3,cmap='viridis',rasterized=True)
    axs[0].scatter([df.lon_unwrapped.iloc[0]],[df.G_LAT.iloc[0]],marker='o',s=70,c='lime',edgecolor='black',label='Start')
    axs[0].scatter([df.lon_unwrapped.iloc[-1]],[df.G_LAT.iloc[-1]],marker='X',s=70,c='red',label='End')
    axs[0].set(xlabel='Unwrapped longitude (degrees)',ylabel='Latitude (degrees)',title=f'{date} segment {segid} — observed track')
    axs[0].legend()
    fig.colorbar(scatter,ax=axs[0],label='Observed horizontal wind speed (m/s)')
    axs[1].plot(df.elapsed_min,df.wind_speed_mps,label='Wind speed (m/s)')
    axs[1].plot(df.elapsed_min,df.TAS,label='True airspeed (m/s)',alpha=.7)
    axs[1].set(xlabel='Elapsed minutes',ylabel='Speed (m/s)',title='Observed along-track conditions')
    axs[1].legend()
    fig.savefig(OUT/f'{name}_diagnostics.png',dpi=160)
    plt.close(fig)
    records.append(dict(date=date,segment_id=segid,samples=len(df),start_s=float(df.TIME_UTC.iloc[0]),end_s=float(df.TIME_UTC.iloc[-1]),duration_min=float(df.elapsed_min.iloc[-1]),observed_distance_km=observed_km,direct_distance_km=direct_km,route_ratio=observed_km/direct_km,median_wind_mps=float(df.wind_speed_mps.median()),p95_wind_mps=float(df.wind_speed_mps.quantile(.95)),median_tas_mps=float(df.TAS.median()),max_time_gap_s=float(np.max(dt)),source_file=path.name))
    print(f'{name}: {len(df):,} rows, wind median {records[-1]["median_wind_mps"]:.2f} m/s, direct {direct_km:.1f} km')
summary=pd.DataFrame(records)
summary.to_csv(OUT/'exact_segment_summary.csv',index=False)
print('\nSaved outputs to:',OUT)
print(summary[['date','segment_id','samples','duration_min','direct_distance_km','median_wind_mps']].to_string(index=False))
