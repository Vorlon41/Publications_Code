
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
ERA5 = ROOT / 'ERA5_ATTREX_20140311_20140312_expanded.grib'
OBS = ROOT / 'analysis_step23' / '20140311_segment_2798_observations.csv'
ROUTES = ROOT / 'analysis_step26'
OUT = ROOT / 'analysis_step36'
SPACINGS = (5., 2., 1., 0.5, 0.25)
R = 6371008.8

def unit(latlon):
    lat, lon = np.deg2rad(latlon)
    return np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])

def point(u):
    u = u / np.linalg.norm(u)
    return np.rad2deg([np.arctan2(u[2], np.hypot(u[0],u[1])), np.arctan2(u[1],u[0])])

def arc(a,b):
    return float(R*np.arccos(np.clip(unit(a)@unit(b),-1.,1.)))

def interpolate(a,b,f):
    u,v=unit(a),unit(b)
    theta=np.arccos(np.clip(u@v,-1.,1.))
    if theta<1e-12: return np.asarray(a,float).copy()
    return point((np.sin((1-f)*theta)*u+np.sin(f*theta)*v)/np.sin(theta))

def heading(a,b):
    """Unit east/north tangent to great-circle at a."""
    p=unit(a); q=unit(b)
    n=np.cross(p,q)
    tangent=np.cross(n,p)
    tangent/=np.linalg.norm(tangent)
    lat,lon=np.deg2rad(a)
    east=np.array([-np.sin(lon),np.cos(lon),0.])
    north=np.array([-np.sin(lat)*np.cos(lon),-np.sin(lat)*np.sin(lon),np.cos(lat)])
    return np.array([tangent@east,tangent@north])

class Wind:
    def __init__(self,ds,pressure):
        ds=ds[['u','v']].sortby('time').sortby('latitude').sortby('longitude').sortby('isobaricInhPa').load()
        if not (float(ds.isobaricInhPa.min())<=pressure<=float(ds.isobaricInhPa.max())):
            raise ValueError('Pressure outside ERA5 levels')
        layer=ds.interp(isobaricInhPa=pressure)
        self.times=ds.time.values.astype('datetime64[ns]').astype('int64')/1e9
        self.lat=np.asarray(ds.latitude.values,float);self.lon=np.asarray(ds.longitude.values,float)
        grid=(self.times,self.lat,self.lon)
        self.u=RegularGridInterpolator(grid,layer.u.values,bounds_error=False,fill_value=np.nan)
        self.v=RegularGridInterpolator(grid,layer.v.values,bounds_error=False,fill_value=np.nan)
    def at(self,t,p):
        q=np.array([[t,p[0],p[1]]]);return np.array([float(self.u(q)[0]),float(self.v(q)[0])])

def evaluate(route,spacing_km,departure,tas,wind=None):
    """Preserve all route vertices; great-circle sublegs; midpoint time iteration."""
    route=np.asarray(route,float)
    if route.ndim!=2 or route.shape[1]!=2 or len(route)<2 or not np.isfinite(route).all():
        raise ValueError('Invalid route')
    elapsed=0.; distance=0.; samples=0
    for a,b in zip(route[:-1],route[1:]):
        length=arc(a,b)
        if length<1e-5: continue
        n=max(1,int(np.ceil(length/(1000*spacing_km))))
        for j in range(n):
            p=interpolate(a,b,j/n);q=interpolate(a,b,(j+1)/n)
            mid=interpolate(a,b,(j+0.5)/n)
            segment_length=length/n
            h=heading(p,q)
            dt=segment_length/tas
            for _ in range(30):
                w=np.zeros(2) if wind is None else wind.at(departure+elapsed+dt/2,mid)
                if not np.isfinite(w).all():return np.inf,distance/1000,samples
                along=float(w@h); cross=float(-w[0]*h[1]+w[1]*h[0])
                if abs(cross)>=tas:return np.inf,distance/1000,samples
                gs=np.sqrt(tas**2-cross**2)+along
                if gs<=0:return np.inf,distance/1000,samples
                new_dt=segment_length/gs
                if abs(new_dt-dt)<1e-10:
                    dt=new_dt;break
                dt=new_dt
            elapsed+=dt;distance+=segment_length;samples+=1
    return elapsed,distance/1000,samples

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not ERA5.is_file() or not OBS.is_file():
        raise FileNotFoundError(f'Check ERA5={ERA5.exists()}, observations={OBS.exists()}')
    obs=pd.read_csv(OBS).sort_values('TIME_UTC').dropna(subset=['TIME_UTC','G_LAT','G_LONG','P','TAS'])
    a=obs[['G_LAT','G_LONG']].iloc[0].to_numpy(float)
    b=obs[['G_LAT','G_LONG']].iloc[-1].to_numpy(float)
    departure=pd.Timestamp('2014-03-11',tz='UTC').timestamp()+float(obs.TIME_UTC.iloc[0])
    pressure=float(obs.P.median());tas=float(obs.TAS.median())
    with xr.open_dataset(ERA5,engine='cfgrib',backend_kwargs={'indexpath':''}) as ds:
        wind=Wind(ds,pressure)
    direct=np.vstack([a,b]);split4=np.vstack([interpolate(a,b,f) for f in np.linspace(0,1,5)])
    split16=np.vstack([interpolate(a,b,f) for f in np.linspace(0,1,17)])
    routes={'direct_2pt':direct,'direct_5pt':split4,'direct_17pt':split16}
    for width in (0,50,100):
        for name in ('distance','wind_aware'):
            path=ROUTES/f'step26_width_{width}km_{name}_route.csv'
            if path.is_file():
                routes[f'width{width}_{name}']=pd.read_csv(path)[['latitude','longitude']].to_numpy(float)
            else:
                print('Missing optional Step 26 route:',path)
    records=[]
    for scenario in ('zero','era5'):
        for spacing in SPACINGS:
            for name,xy in routes.items():
                seconds,km,n=evaluate(xy,spacing,departure,tas,None if scenario=='zero' else wind)
                records.append(dict(scenario=scenario,spacing_km=spacing,route=name,
                                    travel_time_s=seconds,distance_km=km,sublegs=n))
    details=pd.DataFrame(records)
    details.to_csv(OUT/'step36_details.csv',index=False)
    summaries=[]
    for (scenario,spacing),g in details.groupby(['scenario','spacing_km']):
        m=g.set_index('route')
        equivalent=max(abs(m.loc['direct_2pt','travel_time_s']-m.loc[r,'travel_time_s'])
                       for r in ('direct_5pt','direct_17pt'))
        analytic=arc(a,b)/tas if scenario=='zero' else np.nan
        row=dict(scenario=scenario,spacing_km=spacing,direct_time_s=m.loc['direct_2pt','travel_time_s'],
                 equivalent_max_difference_s=equivalent,
                 zero_wind_analytic_error_s=(m.loc['direct_2pt','travel_time_s']-analytic) if scenario=='zero' else np.nan)
        for width in (0,50,100):
            d=f'width{width}_distance';w=f'width{width}_wind_aware'
            if d in m.index and w in m.index:
                row[f'width{width}_advantage_s']=m.loc[d,'travel_time_s']-m.loc[w,'travel_time_s']
                row[f'width{width}_distance_difference_m']=1000*(m.loc[w,'distance_km']-m.loc[d,'distance_km'])
        summaries.append(row)
    summary=pd.DataFrame(summaries).sort_values(['scenario','spacing_km'])
    summary.to_csv(OUT/'step36_summary.csv',index=False)
    print('\nSTEP 36 SUMMARY\n',summary.to_string(index=False))
    fine=summary[summary.spacing_km==0.25]
    assert fine.equivalent_max_difference_s.max()<0.01,'Equivalent great-circle routes disagree'
    assert summary[summary.scenario=='zero'].zero_wind_analytic_error_s.abs().max()<1e-6,'Zero wind analytic failure'
    print('\nPASS: canonical great-circle subdivision and zero-wind analytical checks.')
    print('NOTE: Step 26 route comparison is REEVALUATION ONLY; no A* reoptimization.')
    print('NOTE: 0.25-km convergence is a numerical check, not an accuracy guarantee.')
    print('NOTE: FIFO, off-track wind validity, turn dynamics and mission constraints remain unverified.')
    print('Saved:',OUT)

if __name__=='__main__':main()
