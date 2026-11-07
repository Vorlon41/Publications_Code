
from pathlib import Path
import heapq
import math
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
GRIB = ROOT / '4218a346c038961a43934d722972a5ad.grib'
OBS = ROOT / 'analysis_step9' / '20140311_segment_1986_observations.csv'
OUT = ROOT / 'analysis_step12'
OUT.mkdir(parents=True, exist_ok=True)
GRID_DEG = 0.10  # approximately 9–11 km cells; diagonal movement allowed
MARGIN_DEG = 1.0  # planning corridor margin around observed endpoints
EARTH_RADIUS_KM = 6371.0088


def haversine_km(a, b):
    la1, lo1 = np.radians(a)
    la2, lo2 = np.radians(b)
    dlat, dlon = la2-la1, lo2-lo1
    h = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
    return 2*EARTH_RADIUS_KM*np.arcsin(np.minimum(1, np.sqrt(h)))


def east_north_m(a, b):
    lat1, lon1 = a
    lat2, lon2 = b
    dn = (lat2-lat1)*np.pi/180*EARTH_RADIUS_KM*1000
    de = (lon2-lon1)*np.pi/180*EARTH_RADIUS_KM*1000*np.cos(np.radians((lat1+lat2)/2))
    return de, dn


def main():
    if not GRIB.exists():
        raise FileNotFoundError(f'ERA5 GRIB not found: {GRIB}')
    if not OBS.exists():
        raise FileNotFoundError(f'Step 9 observations not found: {OBS}')
    obs = pd.read_csv(OBS).sort_values('TIME_UTC')
    obs = obs.dropna(subset=['G_LAT','G_LONG','P','TAS','TIME_UTC'])
    start = (float(obs.G_LAT.iloc[0]), float(obs.G_LONG.iloc[0]))
    end = (float(obs.G_LAT.iloc[-1]), float(obs.G_LONG.iloc[-1]))
    pressure = float(obs.P.median())
    tas = float(obs.TAS.median())
    depart = pd.Timestamp('2014-03-11', tz='UTC') + pd.to_timedelta(float(obs.TIME_UTC.iloc[0]), unit='s')
    print(f'Start: {start}; End: {end}')
    print(f'Departure UTC: {depart}; fixed pressure: {pressure:.1f} hPa; fixed TAS: {tas:.1f} m/s')

    ds = xr.open_dataset(GRIB, engine='cfgrib', backend_kwargs={'indexpath':''})
    ds = ds[['u','v']].sortby('isobaricInhPa').sortby('latitude')
    times = pd.DatetimeIndex(ds.time.values).tz_localize('UTC')
    epoch = times.asi8 / 1e9
    levels = ds.isobaricInhPa.values.astype(float)
    latitudes = ds.latitude.values.astype(float)
    longitudes = ds.longitude.values.astype(float)
    if not levels[0] <= pressure <= levels[-1]:
        raise ValueError(f'Pressure {pressure:.1f} hPa outside ERA5 levels {levels}')
    ufun = RegularGridInterpolator((epoch,levels,latitudes,longitudes), ds.u.values,
                                    bounds_error=False, fill_value=np.nan)
    vfun = RegularGridInterpolator((epoch,levels,latitudes,longitudes), ds.v.values,
                                    bounds_error=False, fill_value=np.nan)

    lo_lat = max(latitudes.min(), min(start[0],end[0])-MARGIN_DEG)
    hi_lat = min(latitudes.max(), max(start[0],end[0])+MARGIN_DEG)
    lo_lon = max(longitudes.min(), min(start[1],end[1])-MARGIN_DEG)
    hi_lon = min(longitudes.max(), max(start[1],end[1])+MARGIN_DEG)
    lats = np.arange(np.floor(lo_lat/GRID_DEG)*GRID_DEG,
                     np.ceil(hi_lat/GRID_DEG)*GRID_DEG+GRID_DEG/2,GRID_DEG)
    lons = np.arange(np.floor(lo_lon/GRID_DEG)*GRID_DEG,
                     np.ceil(hi_lon/GRID_DEG)*GRID_DEG+GRID_DEG/2,GRID_DEG)
    lats = lats[(lats>=latitudes.min()) & (lats<=latitudes.max())]
    lons = lons[(lons>=longitudes.min()) & (lons<=longitudes.max())]
    nr, nc = len(lats), len(lons)
    nodes = [(float(lat),float(lon)) for lat in lats for lon in lons]
    start_id, end_id = len(nodes), len(nodes)+1
    nodes.extend([start,end])
    nearest = lambda p: (int(np.argmin(abs(lats-p[0]))), int(np.argmin(abs(lons-p[1]))))
    def index(i,j): return i*nc+j
    def neighbors(k):
        if k >= nr*nc:
            p = nodes[k]
            i,j = nearest(p)
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    ii,jj=i+di,j+dj
                    if 0<=ii<nr and 0<=jj<nc:
                        yield index(ii,jj)
            if k == start_id:
                yield end_id  # direct edge is always a candidate
        else:
            i,j=divmod(k,nc)
            for di in (-1,0,1):
                for dj in (-1,0,1):
                    if di==0 and dj==0: continue
                    ii,jj=i+di,j+dj
                    if 0<=ii<nr and 0<=jj<nc: yield index(ii,jj)
            for special in (start_id,end_id):
                si,sj=nearest(nodes[special])
                if abs(i-si)<=1 and abs(j-sj)<=1: yield special

    cache = {}
    def edge(a,b,elapsed_seconds,wind):
        pa,pb=nodes[a],nodes[b]
        east,north=east_north_m(pa,pb)
        distance=np.hypot(east,north)
        if distance < 0.001: return 0.0, 0.0
        if not wind: return distance, distance/tas
        # Midpoint in space and midpoint estimated time; iterate twice.
        travel=distance/tas
        for _ in range(2):
            t=depart.timestamp()+elapsed_seconds+travel/2
            midpoint=((pa[0]+pb[0])/2,(pa[1]+pb[1])/2)
            pt=np.array([[t,pressure,midpoint[0],midpoint[1]]])
            u=float(ufun(pt)[0]); v=float(vfun(pt)[0])
            if not (np.isfinite(u) and np.isfinite(v)):
                return distance, np.inf
            # Wind-corrected ground speed maintaining the desired ground track.
            along=(u*east+v*north)/distance
            cross=(-u*north+v*east)/distance
            if abs(cross)>=tas: return distance,np.inf
            ground=np.sqrt(tas*tas-cross*cross)+along
            if ground<=0: return distance,np.inf
            travel=distance/ground
        return distance,travel

    def astar(wind):
        # An admissible optimistic heuristic for wind-aware time search.
        maxwind=float(np.nanmax(np.hypot(ds.u.values,ds.v.values)))
        def heuristic(k):
            d=haversine_km(nodes[k],end)*1000
            return d/(tas+maxwind) if wind else d
        best={start_id:0.0}; arrival={start_id:0.0}; parent={}
        queue=[(heuristic(start_id),0.0,start_id)]
        visited=0
        while queue:
            _,cost,a=heapq.heappop(queue)
            if cost>best.get(a,np.inf)+1e-7: continue
            visited+=1
            if a==end_id:
                path=[a]
                while path[-1]!=start_id: path.append(parent[path[-1]])
                path.reverse()
                return path,visited
            for b in neighbors(a):
                d,t=edge(a,b,arrival[a],wind)
                if not np.isfinite(t): continue
                new=cost+(t if wind else d)
                if new+1e-7<best.get(b,np.inf):
                    best[b]=new
                    arrival[b]=arrival[a]+t
                    parent[b]=a
                    heapq.heappush(queue,(new+heuristic(b),new,b))
        raise RuntimeError('No feasible route in selected planning grid')

    def evaluate(path):
        dist=0.0; sec=0.0
        for a,b in zip(path[:-1],path[1:]):
            d,t=edge(a,b,sec,True)
            if not np.isfinite(t): raise RuntimeError('A route cannot be flown under ERA5 wind')
            dist+=d;sec+=t
        return dist/1000,sec/60

    distance_path, visited_d=astar(False)
    wind_path, visited_w=astar(True)
    d0,t0=evaluate(distance_path)
    d1,t1=evaluate(wind_path)
    saving=100*(t0-t1)/t0
    print(f'Conventional A*: {d0:.2f} km, {t0:.2f} min; explored {visited_d} nodes')
    print(f'Wind-aware A*:   {d1:.2f} km, {t1:.2f} min; explored {visited_w} nodes')
    print(f'Modeled time saving: {saving:.2f}%')
    print('NOTE: 2D constant-pressure, constant-TAS model; no turn-radius, airspace, fuel, or mission constraints.')
    print('NOTE: Wind-aware route evaluated with departure-time-dependent ERA5 winds; this is a preliminary comparison.')

    for label,path in [('distance_astar',distance_path),('wind_aware_astar',wind_path)]:
        pd.DataFrame([{'latitude':nodes[k][0],'longitude':nodes[k][1]} for k in path]).to_csv(OUT/f'{label}_route.csv',index=False)
    pd.DataFrame([{'case':'20140311_segment_1986','fixed_pressure_hpa':pressure,'fixed_tas_mps':tas,
                   'baseline_distance_km':d0,'baseline_time_min':t0,
                   'wind_aware_distance_km':d1,'wind_aware_time_min':t1,
                   'modeled_time_saving_pct':saving,'baseline_nodes':len(distance_path),
                   'wind_aware_nodes':len(wind_path)}]).to_csv(OUT/'step12_route_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(11,6))
    ax.plot(obs.G_LONG,obs.G_LAT,color='0.65',lw=1.5,label='Observed ATTREX track')
    for path,color,label in [(distance_path,'black','Distance-minimizing A*'),
                             (wind_path,'tab:red','Wind-aware A*')]:
        points=np.array([nodes[k] for k in path])
        ax.plot(points[:,1],points[:,0],color=color,lw=2,label=label)
    ax.scatter([start[1],end[1]],[start[0],end[0]],c=['green','red'],s=60,zorder=5)
    ax.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',
           title=f'Step 12: preliminary routing | modeled saving {saving:.2f}%')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(OUT/'step12_route_comparison.png',dpi=180)
    plt.show()
    ds.close()
    print(f'Saved outputs: {OUT}')

if __name__=='__main__':
    main()
