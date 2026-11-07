
from pathlib import Path
import heapq
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(r'C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825')
GRIB = ROOT / '4218a346c038961a43934d722972a5ad.grib'
OUT = ROOT / 'analysis_step18'
SEGMENTS = (1450, 1986)
GRID_DEGREES = (0.25, 0.125) # sensitivity requires actual reoptimization
CONNECT_K = 8
MAX_LEG_KM = 25.0 # subleg integration of wind for all edges, including direct endpoint edge
EARTH = 6371008.8

def edge_geometry(a,b):
    la1,lo1=np.deg2rad(a); la2,lo2=np.deg2rad(b)
    dl=lo2-lo1
    h=np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin(dl/2)**2
    d=2*EARTH*np.arcsin(np.sqrt(np.clip(h,0,1)))
    east=np.sin(dl)*np.cos(la2)
    north=np.cos(la1)*np.sin(la2)-np.sin(la1)*np.cos(la2)*np.cos(dl)
    az=np.arctan2(east,north)
    return float(d),np.array([np.sin(az),np.cos(az)])

class Wind:
    def __init__(self,ds,pressure):
        ds=ds.sortby('time').sortby('latitude').sortby('longitude').sortby('isobaricInhPa')
        p=ds[['u','v']].interp(isobaricInhPa=pressure)
        self.times=ds.time.values.astype('datetime64[ns]').astype('int64')/1e9
        self.lat=np.asarray(ds.latitude.values,float)
        self.lon=np.asarray(ds.longitude.values,float)
        grid=(self.times,self.lat,self.lon)
        self.u=RegularGridInterpolator(grid,np.asarray(p.u.values,float),bounds_error=False,fill_value=np.nan)
        self.v=RegularGridInterpolator(grid,np.asarray(p.v.values,float),bounds_error=False,fill_value=np.nan)
    def at(self,t,lat,lon):
        q=np.array([[t,lat,lon]])
        return np.array([float(self.u(q)[0]),float(self.v(q)[0])])

def travel(a,b,departure,wind,tas):
    dist,_=edge_geometry(a,b)
    if dist<0.01:return 0.
    n=max(1,int(np.ceil(dist/(MAX_LEG_KM*1000))))
    total=0.
    for j in range(n):
        p=a+(b-a)*j/n; q=a+(b-a)*(j+1)/n
        length,heading=edge_geometry(p,q)
        mid=(p+q)/2
        dt=length/tas
        for _ in range(3):
            w=wind.at(departure+total+dt/2,*mid)
            if not np.isfinite(w).all():return np.inf
            along=float(w@heading)
            cross=float(-w[0]*heading[1]+w[1]*heading[0])
            if abs(cross)>=tas:return np.inf
            groundspeed=np.sqrt(tas*tas-cross*cross)+along
            if groundspeed<=1:return np.inf
            dt=length/groundspeed
        total+=dt
    return total

def load_case(ds,seg,grid_deg):
    f=ROOT/'analysis_step9'/f'20140311_segment_{seg}_observations.csv'
    df=pd.read_csv(f).sort_values('TIME_UTC').dropna(subset=['TIME_UTC','G_LAT','G_LONG','P','TAS'])
    a=df.iloc[0]; b=df.iloc[-1]
    start=np.array([float(a.G_LAT),float(a.G_LONG)])
    goal=np.array([float(b.G_LAT),float(b.G_LONG)])
    depart=pd.Timestamp('2014-03-11',tz='UTC').timestamp()+float(a.TIME_UTC)
    pressure=float(df.P.median()); tas=float(df.TAS.median())
    latlo=max(float(ds.latitude.min()),min(start[0],goal[0])-0.5)
    lathi=min(float(ds.latitude.max()),max(start[0],goal[0])+0.5)
    lonlo=max(float(ds.longitude.min()),min(start[1],goal[1])-2.)
    lonhi=min(float(ds.longitude.max()),max(start[1],goal[1])+2.)
    lats=np.arange(np.ceil(latlo/grid_deg)*grid_deg,lathi+1e-8,grid_deg)
    lons=np.arange(np.ceil(lonlo/grid_deg)*grid_deg,lonhi+1e-8,grid_deg)
    if not len(lats) or not len(lons):raise ValueError('Empty routing grid')
    coords=np.array([(y,x) for y in lats for x in lons]+[tuple(start),tuple(goal)])
    ny,nx=len(lats),len(lons); n=ny*nx; s=n; g=n+1
    adj=[[] for _ in range(n+2)]
    for y in range(ny):
        for x in range(nx):
            i=y*nx+x
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    if dy==dx==0:continue
                    yy,xx=y+dy,x+dx
                    if 0<=yy<ny and 0<=xx<nx:adj[i].append(yy*nx+xx)
    # Connect both exact endpoints to K nearest grid nodes; include all 4
    # cell corners where present. Edges are directed for wind-dependent cost.
    for terminal in (s,g):
        distances=np.array([edge_geometry(coords[terminal],c)[0] for c in coords[:n]])
        neighbors=np.argsort(distances)[:min(CONNECT_K,n)]
        for j in neighbors:
            adj[terminal].append(int(j));adj[j].append(terminal)
    # Direct geodesic is a candidate, with wind integrated spatially along it.
    adj[s].append(g);adj[g].append(s)
    return coords,adj,s,g,depart,pressure,tas,df

def solve(coords,adj,s,g,depart,wind,tas,mode):
    best=np.full(len(coords),np.inf);best[s]=0.
    prev=np.full(len(coords),-1,dtype=int)
    pq=[(0.,s)]; expanded=0
    # For distance mode, labels are meters. For time mode, seconds.
    while pq:
        val,i=heapq.heappop(pq)
        if val>best[i]+1e-7:continue
        expanded+=1
        if i==g:break
        for j in adj[i]:
            w=edge_geometry(coords[i],coords[j])[0] if mode=='distance' else travel(coords[i],coords[j],depart+val,wind,tas)
            if not np.isfinite(w):continue
            new=val+w
            if new<best[j]-1e-7:
                best[j]=new;prev[j]=i;heapq.heappush(pq,(new,j))
    if not np.isfinite(best[g]):raise RuntimeError('No feasible path')
    route=[g]
    while route[-1]!=s:
        route.append(int(prev[route[-1]]))
        if route[-1]<0:raise RuntimeError('Broken predecessor chain')
    return route[::-1],expanded

def evaluate(coords,route,depart,wind,tas):
    elapsed=0.; distance=0.
    for i,j in zip(route[:-1],route[1:]):
        distance+=edge_geometry(coords[i],coords[j])[0]
        elapsed+=travel(coords[i],coords[j],depart+elapsed,wind,tas)
    return elapsed/60,distance/1000

def turns(xy):
    heads=[edge_geometry(a,b)[1] for a,b in zip(xy[:-1],xy[1:])]
    if len(heads)<2:return 0.,0
    angles=np.rad2deg(np.arccos(np.clip(np.sum(np.asarray(heads[:-1])*np.asarray(heads[1:]),axis=1),-1,1)))
    return float(angles.max()),int(np.sum(angles>30))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not GRIB.exists():raise FileNotFoundError(f'ERA5 file missing: {GRIB}')
    ds=xr.open_dataset(GRIB,engine='cfgrib',backend_kwargs={'indexpath':''})
    rows=[]
    for seg in SEGMENTS:
        for resolution in GRID_DEGREES:
            coords,adj,s,g,depart,pressure,tas,obs=load_case(ds,seg,resolution)
            wind=Wind(ds,pressure)
            if not(wind.times[0]<=depart<=wind.times[-1]):raise ValueError('Departure outside ERA5 time coverage')
            result={}
            for mode in ('distance','time'):
                route,expanded=solve(coords,adj,s,g,depart,wind,tas,mode)
                minutes,km=evaluate(coords,route,depart,wind,tas)
                xy=coords[route]
                maximum,sharp=turns(xy)
                name='distance' if mode=='distance' else 'wind_aware'
                result[name]=(minutes,km,xy)
                pd.DataFrame(xy,columns=['latitude','longitude']).to_csv(OUT/f'segment_{seg}_grid_{resolution:g}_{name}_route.csv',index=False)
                rows.append(dict(segment=seg,grid_deg=resolution,route=name,modeled_time_min=minutes,
                                 distance_km=km,waypoints=len(route),expanded=expanded,
                                 max_turn_deg=maximum,turns_over_30=sharp,
                                 exact_start_error_m=edge_geometry(xy[0],coords[s])[0],
                                 exact_goal_error_m=edge_geometry(xy[-1],coords[g])[0],
                                 direct_geodesic_selected=len(route)==2))
            delta=(result['distance'][0]-result['wind_aware'][0])*60
            print(f'Segment {seg} grid {resolution:g}: distance {result["distance"][0]:.4f} min; '
                  f'wind-aware {result["wind_aware"][0]:.4f} min; advantage {delta:.3f} s')
            fig,ax=plt.subplots(figsize=(9,6))
            ax.plot(obs.G_LONG,obs.G_LAT,color='0.75',lw=1,label='Observed science track')
            for name,color in [('distance','black'),('wind_aware','red')]:
                xy=result[name][2];ax.plot(xy[:,1],xy[:,0],lw=2,color=color,label=name)
            ax.scatter([coords[s,1],coords[g,1]],[coords[s,0],coords[g,0]],c=['green','blue'],s=50,zorder=5)
            ax.set(xlabel='Longitude (degrees E)',ylabel='Latitude (degrees N)',
                   title=f'Step 18 | segment {seg} | grid {resolution:g} | advantage {delta:.2f} s')
            ax.grid(alpha=.25);ax.legend();fig.tight_layout()
            fig.savefig(OUT/f'segment_{seg}_grid_{resolution:g}_routes.png',dpi=180);plt.close(fig)
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'step18_exact_endpoint_routes.csv',index=False)
    paired=summary.pivot(index=['segment','grid_deg'],columns='route',values='modeled_time_min').reset_index()
    paired['wind_aware_advantage_seconds']=60*(paired['distance']-paired['wind_aware'])
    paired['saving_pct']=100*(paired['distance']-paired['wind_aware'])/paired['distance']
    paired.to_csv(OUT/'step18_paired_comparison.csv',index=False)
    print('\n',paired.to_string(index=False),'\nSaved:',OUT)
    print('CAUTIONS: exact endpoints but no aircraft turn dynamics, fuel, mission constraints, or certified FIFO.')
    print('Direct great-circle edge is permitted; both algorithms use identical candidate edges.')
    print('This is a fixed-pressure, fixed-TAS, time-dependent ERA5 model, not measured savings.')
if __name__=='__main__':main()
