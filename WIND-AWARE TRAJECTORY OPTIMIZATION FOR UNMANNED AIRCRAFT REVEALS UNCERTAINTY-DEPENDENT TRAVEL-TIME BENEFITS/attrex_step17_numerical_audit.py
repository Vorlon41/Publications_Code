
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\chris\OneDrive\Desktop\Publication Projects\Eathdata\ATTREX\ATTREX-Aircraft_navigational_and_meteorological_Measurements_1-20261025_192825")
GRIB = ROOT / "4218a346c038961a43934d722972a5ad.grib"
OUT = ROOT / "analysis_step17"
OUT.mkdir(exist_ok=True, parents=True)
EARTH = 6371008.8
CONFIG = {
    1450: dict(depart="2014-03-11T17:18:14.200Z", tas=167.02, pressure=106.07,
               start=(12.8599,145.5738), end=(32.97302,145.65088)),
    1986: dict(depart="2014-03-11T21:20:15.200Z", tas=176.58, pressure=99.09,
               start=(33.06891,146.88303), end=(33.15909,152.17472)),
}
# Integration uses straight geodesic approximations between adjacent grid waypoints.
# At the short segment lengths here, local bearing approximation is adequate for sensitivity.
def leg_info(a,b):
    lat=np.deg2rad((a[0]+b[0])/2)
    north=EARTH*np.deg2rad(b[0]-a[0])
    east=EARTH*np.cos(lat)*np.deg2rad(b[1]-a[1])
    d=np.hypot(east,north)
    return d,np.array([east,north])/max(d,1e-12)

def angles(coords):
    dirs=[]
    for a,b in zip(coords[:-1],coords[1:]):
        d,v=leg_info(a,b)
        if d>1: dirs.append(v)
    if len(dirs)<2: return np.array([])
    return np.rad2deg(np.arccos(np.clip(np.sum(np.array(dirs[:-1])*np.array(dirs[1:]),axis=1),-1,1)))

def load_route(seg, kind):
    path=ROOT/"analysis_step14"/f"step14_segment_{seg}_{kind}_route.csv"
    a=pd.read_csv(path)[["latitude","longitude"]].to_numpy(float)
    if len(a)<2 or not np.isfinite(a).all(): raise ValueError(f"Invalid route: {path}")
    return a

def evaluate(ds, coords, cfg, subdivisions=1, time_dependent=True):
    """Integrate each original leg using N sublegs and two midpoint time iterations."""
    elapsed=0.; total=0.
    departure=pd.Timestamp(cfg["depart"]).tz_convert("UTC").tz_localize(None)
    for a,b in zip(coords[:-1],coords[1:]):
        for j in range(subdivisions):
            p=a+(b-a)*j/subdivisions
            q=a+(b-a)*(j+1)/subdivisions
            d,direction=leg_info(p,q)
            if d<1: continue
            mid=(p+q)/2
            dt=d/cfg["tas"]
            for _ in range(3):
                sample_s=(elapsed+dt/2) if time_dependent else 0.
                when=np.datetime64((departure+pd.Timedelta(seconds=sample_s)).to_datetime64(),"ns")
                v=ds[["u","v"]].interp(time=when,isobaricInhPa=cfg["pressure"],
                    latitude=float(mid[0]),longitude=float(mid[1]),method="linear")
                wind=np.array([float(v.u.values),float(v.v.values)])
                if not np.isfinite(wind).all():
                    return np.nan,np.nan
                speed=cfg["tas"]+float(wind@direction)
                if speed<=5: return np.nan,np.nan
                dt=d/speed
            elapsed+=dt; total+=d
    return elapsed/60,total/1000

def endpoints_corrected(coords,cfg):
    """Replace only first and last snapped waypoints with observed coordinates."""
    c=coords.copy()
    c[0]=cfg["start"];c[-1]=cfg["end"]
    return c

def main():
    ds=xr.open_dataset(GRIB,engine="cfgrib",backend_kwargs={"indexpath":""})
    ds=ds.sortby("isobaricInhPa").sortby("latitude").sortby("longitude")
    results=[]; turns=[]
    for seg,cfg in CONFIG.items():
        routes={k:load_route(seg,k) for k in ("distance","wind_aware")}
        for kind,original in routes.items():
            for correction in (False,True):
                coords=endpoints_corrected(original,cfg) if correction else original
                a=angles(coords)
                turns.append(dict(segment=seg,route=kind,actual_endpoints=correction,
                    n_waypoints=len(coords),max_turn_deg=float(a.max()) if len(a) else np.nan,
                    turns_over_30=int(np.sum(a>30)),turns_over_60=int(np.sum(a>60)),
                    turns_over_90=int(np.sum(a>90)),
                    start_snap_km=leg_info(original[0],np.array(cfg["start"]))[0]/1000,
                    end_snap_km=leg_info(original[-1],np.array(cfg["end"]))[0]/1000))
                for n in (1,2,4,8):
                    for timed in (False,True):
                        mins,km=evaluate(ds,coords,cfg,n,timed)
                        results.append(dict(segment=seg,route=kind,
                            actual_endpoints=correction,sublegs_per_grid_edge=n,
                            time_dependent=timed,time_min=mins,distance_km=km))
                        print(seg,kind,"actual endpoints",correction,
                              "sublegs",n,"dynamic",timed,"time_min",round(mins,5))
        print("Completed segment",seg)
    df=pd.DataFrame(results)
    df.to_csv(OUT/"numerical_audit.csv",index=False)
    pd.DataFrame(turns).to_csv(OUT/"turn_geometry_audit.csv",index=False)
    comparison=df.pivot_table(index=["segment","actual_endpoints","sublegs_per_grid_edge","time_dependent"],
                              columns="route",values="time_min").reset_index()
    comparison["delta_seconds"]=(comparison["distance"]-comparison["wind_aware"])*60
    comparison["saving_pct"]=100*(comparison["distance"]-comparison["wind_aware"])/comparison["distance"]
    comparison.to_csv(OUT/"paired_numerical_comparison.csv",index=False)
    for seg in CONFIG:
        subset=comparison[(comparison.segment==seg)&(comparison.time_dependent)]
        fig,ax=plt.subplots(figsize=(8,5))
        for correction,g in subset.groupby("actual_endpoints"):
            g=g.sort_values("sublegs_per_grid_edge")
            ax.plot(g.sublegs_per_grid_edge,g.delta_seconds,"o-",label="Observed endpoints" if correction else "Snapped endpoints")
        ax.axhline(0,color="black",lw=1,ls="--")
        ax.set(xlabel="Sublegs per grid edge",ylabel="Modeled wind-aware advantage (seconds)",
               title=f"Step 17 | segment {seg} | temporal and spatial integration")
        ax.legend();fig.tight_layout()
        fig.savefig(OUT/f"segment_{seg}_integration_sensitivity.png",dpi=180)
        plt.close(fig)
    print("\nDYNAMIC-WIND ROUTE COMPARISONS\n")
    print(comparison[comparison.time_dependent].to_string(index=False))
    print("\nTURN GEOMETRY\n")
    print(pd.DataFrame(turns).to_string(index=False))
    print("\nSaved outputs:",OUT)
    print("CAUTION: this is a FIXED-ROUTE evaluator audit, not a grid-resolution or A* reoptimization study.")
    print("CAUTION: replacing snapped endpoints may change first/last-leg geometry and creates no turn-radius constraints.")
    print("CAUTION: route waypoints may contain sharp corners; no aircraft dynamics, fuel, or mission constraints.")
if __name__=="__main__":
    main()
