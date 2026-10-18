
# COMPLETE SELF-CONTAINED SYNTHETIC FIGURE GENERATOR

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks, hilbert, stft, butter, sosfiltfilt

SEED = 20261016
rng = np.random.default_rng(SEED)
# On Windows, export directly to the requested book-chapter directory.
# Elsewhere, use a local folder so the script remains portable.
import os
OUT = (Path(r'C:\Users\User\Desktop\Book Chapter\Signal Decompostion Chapter')
       if os.name == 'nt' else Path('chapter7_figures'))
OUT.mkdir(parents=True, exist_ok=True)
print('Figure output directory:', OUT.resolve())
MM = 1 / 25.4
W, HMAX = 130*MM, 184*MM
plt.rcParams.update({'figure.dpi':120,'savefig.dpi':300,'font.size':7.1,
    'axes.titlesize':7.5,'axes.labelsize':7.0,'xtick.labelsize':6.3,
    'ytick.labelsize':6.3,'legend.fontsize':6.1,'axes.spines.top':False,
    'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.20,
    'pdf.fonttype':42,'ps.fonttype':42})
C = dict(sig='#17365D', est='#B22222', green='#237A57', purple='#7651A8')

def db(x):
    x=np.asarray(x); return 10*np.log10(np.maximum(x,1e-14))
def ma(x,y,mask=None):
    ok=np.isfinite(x)&np.isfinite(y)
    if mask is not None: ok &= mask
    return np.mean(np.abs(np.asarray(x)[ok]-np.asarray(y)[ok])) if ok.any() else np.nan
def corr(x,y,mask=None):
    ok=np.isfinite(x)&np.isfinite(y)
    if mask is not None: ok &= mask
    if ok.sum()<3 or np.std(np.asarray(x)[ok])<1e-12 or np.std(np.asarray(y)[ok])<1e-12:return np.nan
    return np.corrcoef(np.asarray(x)[ok],np.asarray(y)[ok])[0,1]
def norm(x):
    x=np.asarray(x); return (x-np.mean(x))/(np.std(x)+1e-12)
def lowpass(x,fs,hz,order=3):
    return sosfiltfilt(butter(order,hz,fs=fs,btype='low',output='sos'),x)
def bandpass(x,fs,lo,hi,order=3):
    return sosfiltfilt(butter(order,[lo,hi],fs=fs,btype='band',output='sos'),x)
def spec(x,fs,nperseg,noverlap=None):
    nperseg=min(int(nperseg),len(x)); noverlap=int(.85*nperseg) if noverlap is None else noverlap
    f,t,z=stft(x,fs=fs,nperseg=nperseg,noverlap=noverlap,boundary=None,padded=False)
    return f,t,np.abs(z)**2

def cwt_morlet(x,fs,freqs,w=6,decim=1):
    
    xd=np.asarray(x)[::decim]; fsd=fs/decim
    td=np.arange(len(xd))/fsd
    out=np.empty((len(freqs),len(xd)),dtype=complex)
    for i,f in enumerate(freqs):
        sigma=w/(2*np.pi*f)*fsd
        radius=min(int(np.ceil(4*sigma)),max(3,len(xd)//2))
        k=np.arange(-radius,radius+1)
        wave=np.exp(1j*2*np.pi*f*k/fsd)*np.exp(-.5*(k/sigma)**2)
        wave/=np.sqrt(np.sum(np.abs(wave)**2))
        out[i]=signal.fftconvolve(xd,np.conj(wave[::-1]),mode='same')
    return td,out

def vmd(x,K=2,alpha=1800,tau=0,tol=1e-6,max_iter=350):
    
    x=np.asarray(x,float)
    n=len(x)
    if n%2: x=np.pad(x,(0,1)); n+=1
    # Mirror extension mitigates artificial periodic boundaries.
    ext=np.r_[x[:n//2][::-1],x,x[-n//2:][::-1]]
    N=len(ext); f=np.fft.fftshift(np.fft.fftfreq(N))
    X=np.fft.fftshift(np.fft.fft(ext)); X[f<0]=0
    U=np.zeros((K,N),complex); omega=np.linspace(.015,.15,K)
    dual=np.zeros(N,complex)
    for it in range(max_iter):
        old=U.copy()
        for k in range(K):
            residual=X-np.sum(U,axis=0)+U[k]-dual/2
            U[k]=residual/(1+2*alpha*(f-omega[k])**2)
            pos=f>0
            denom=np.sum(np.abs(U[k,pos])**2)
            if denom>1e-14: omega[k]=np.sum(f[pos]*np.abs(U[k,pos])**2)/denom
        dual+=tau*(np.sum(U,axis=0)-X)
        if np.linalg.norm(U-old)/(np.linalg.norm(old)+1e-12)<tol:break
    modes=[]
    for k in range(K):
        S=np.zeros(N,complex); S[f>=0]=U[k,f>=0]
        # Real inverse: double strictly positive frequencies, retain DC.
        full=np.fft.ifftshift(S)
        mode=2*np.real(np.fft.ifft(full))
        modes.append(mode[n//2:n//2+n])
    return np.array(modes)[:,:len(x)],omega,it+1

def fig_axes(n):
    height={3:6.40,4:7.05,5:7.20}[n]
    fig,ax=plt.subplots(n,1,figsize=(W,height),sharex=False)
    fig.subplots_adjust(left=.17,right=.81,bottom=.085,top=.965,hspace=.62)
    return fig,np.atleast_1d(ax)
def title(ax,letter,s):ax.set_title(f'({letter}) {s}',loc='left',fontweight='semibold')
def cb(fig,ax,im,txt):
    pos=ax.get_position()
    ca=fig.add_axes([.835,pos.y0,.022,pos.height])
    c=fig.colorbar(im,cax=ca);c.set_label(txt,fontsize=6.3);c.ax.tick_params(labelsize=5.9,length=2)
def legend(ax,**kw):ax.legend(framealpha=.88,handlelength=1.5,borderpad=.25,**kw)
def xlim(ax,a,b):
    for z in ax:z.set_xlim(a,b)
def transition(ax,a,b):
    for z in ax:
        z.axvspan(a,b,color='orange',alpha=.12,zorder=0)
        z.axvline(a,color=C['est'],lw=.7,ls='--')
def shade(ax,edge,end):
    for z in ax:
        z.axvspan(0,edge,color='gray',alpha=.11)
        z.axvspan(end-edge,end,color='gray',alpha=.11)
def save(fig,stem):
    w,h=fig.get_size_inches()
    assert w<=W+1e-9 and h<=HMAX+1e-9
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'{stem}.{ext}',dpi=300,facecolor='white',bbox_inches=None)
    plt.show();plt.close(fig)
    print(f'{stem}: {w*25.4:.1f} x {h*25.4:.1f} mm')

# FIGURE 7.1 — ECG: 75 to 125 bpm, 10–11 s transition

fs=250; t=np.arange(0,20,1/fs)
hr_truth=np.where(t<10,75,np.where(t<11,75+50*(t-10),125))
phase=np.cumsum(hr_truth/60/fs)
beat_t=np.interp(np.arange(1,int(phase[-1])+1),phase,t)
ecg=.012*rng.normal(size=len(t))+.025*np.sin(2*np.pi*.28*t)
for bt in beat_t:
    d=t-bt
    ecg+=.11*np.exp(-.5*((d+.18)/.045)**2)-.12*np.exp(-.5*((d+.018)/.012)**2)
    ecg+=1.0*np.exp(-.5*(d/.009)**2)-.22*np.exp(-.5*((d-.024)/.015)**2)
    ecg+=.23*np.exp(-.5*((d-.26)/.075)**2)
peaks,_=find_peaks(ecg,height=.50,distance=int(fs*.35),prominence=.40)
pt=t[peaks]; hr_times=pt[1:]; hr_ecg=60/np.diff(pt)
f_ecg=np.linspace(2,40,65); tc,Wc=cwt_morlet(ecg,fs,f_ecg,decim=5)
fig,ax=fig_axes(3)
ax[0].plot(t,ecg,color=C['sig'],lw=.48);ax[0].set_ylabel('ECG (a.u.)');title(ax[0],'A','Synthetic ECG waveform')
ax[1].plot(t,ecg,color='.70',lw=.40);ax[1].scatter(pt,ecg[peaks],s=6,color=C['est'],label='R peaks',zorder=3)
ax[1].set_ylabel('ECG (a.u.)');title(ax[1],'B','R peaks and heart rate');legend(ax[1],loc='upper left')
a2=ax[1].twinx();a2.step(hr_times,hr_ecg,where='post',color=C['green'],lw=1.0);a2.set_ylabel('Heart rate (bpm)',color=C['green']);a2.tick_params(axis='y',labelcolor=C['green']);a2.grid(False)
im=ax[2].pcolormesh(tc,f_ecg,db(abs(Wc)**2),shading='auto',cmap='viridis',rasterized=True)
ax[2].set_ylabel('Frequency (Hz)');ax[2].set_xlabel('Time (s)');title(ax[2],'C','ECG CWT power');cb(fig,ax[2],im,'Power (relative dB)')
transition(ax,10,11);xlim(ax,0,20);save(fig,'Figure_7_1_ECG')


# FIGURE 7.2 — EEG: alpha to delta, 10–15 s transition

fs=128;t=np.arange(0,30,1/fs)
a=np.clip((t-10)/5,0,1);noise=.15*rng.normal(size=len(t))
eeg=(1-a)*1.15*np.sin(2*np.pi*10*t)+a*1.45*np.sin(2*np.pi*2*t)+.23*np.sin(2*np.pi*6*t)+noise
f,ts,p=spec(eeg,fs,512)
bands={'Delta (0.5–4 Hz)':(.5,4),'Theta (4–8 Hz)':(4,8),'Alpha (8–13 Hz)':(8,13),'Beta (13–30 Hz)':(13,30)}
bp={name:np.trapz(p[(f>=lo)&(f<hi)],f[(f>=lo)&(f<hi)],axis=0) for name,(lo,hi) in bands.items()}
total=np.sum(list(bp.values()),axis=0);rel={k:v/np.maximum(total,1e-14) for k,v in bp.items()}
fig,ax=fig_axes(3)
ax[0].plot(t,eeg,color=C['sig'],lw=.4);ax[0].set_ylabel('EEG (a.u.)');title(ax[0],'A','Synthetic alpha-to-delta transition')
im=ax[1].pcolormesh(ts,f,db(p),shading='auto',cmap='magma',rasterized=True);ax[1].set_ylim(0,35);ax[1].set_ylabel('Frequency (Hz)');title(ax[1],'B','STFT power spectrum');cb(fig,ax[1],im,'Power (relative dB)')
for name,val in rel.items():ax[2].plot(ts,val,lw=.95,label=name)
ax[2].set_ylim(0,1.04);ax[2].set_ylabel('Selected-band fraction');ax[2].set_xlabel('Time (s)');title(ax[2],'C','Relative power within selected EEG bands');legend(ax[2],loc='upper left',ncol=2)
transition(ax,10,15);xlim(ax,0,30);save(fig,'Figure_7_2_EEG')


# FIGURE 7.3 — EMG: shift from higher to lower spectral content
# Force is independently simulated (not inferred from EMG).

fs=1000;t=np.arange(0,20,1/fs)
noise=rng.normal(size=len(t));high=bandpass(noise,fs,95,210);low=bandpass(rng.normal(size=len(t)),fs,25,85)
a=np.clip(t/20,0,1)
emg=(1-.65*a)*high/(np.std(high)+1e-12)+(.25+.8*a)*low/(np.std(low)+1e-12)
emg*=.55+.08*np.sin(2*np.pi*.25*t)
force=55-1.3*t+1.1*np.sin(2*np.pi*.13*t)+.7*rng.normal(size=len(t))
f,ts,p=spec(emg,fs,1024)
use=(f>=10)&(f<=250);ff=f[use];pp=p[use]
cum=np.cumsum(pp,axis=0);target=cum[-1]/2
idx=np.argmax(cum>=target[None,:],axis=0);median=ff[idx]
median_s=gaussian_filter1d(median,sigma=max(1,1/np.median(np.diff(ts))))
fig,ax=fig_axes(4)
ax[0].plot(t,emg,color=C['sig'],lw=.25);ax[0].set_ylabel('EMG (a.u.)');title(ax[0],'A','Synthetic surface EMG')
im=ax[1].pcolormesh(ts,f,db(p),shading='auto',cmap='plasma',rasterized=True);ax[1].set_ylim(0,250);ax[1].set_ylabel('Frequency (Hz)');title(ax[1],'B','EMG spectral power');cb(fig,ax[1],im,'Power (relative dB)')
ax[2].plot(ts,median,color='.68',lw=.6,label='Raw');ax[2].plot(ts,median_s,color=C['est'],lw=1.2,label='Smoothed');ax[2].set_ylabel('Median freq. (Hz)');title(ax[2],'C','Median-frequency trajectory');legend(ax[2],loc='upper right')
ax[3].plot(t,force,color=C['green'],lw=.55);ax[3].set_ylabel('Force (a.u.)');ax[3].set_xlabel('Time (s)');title(ax[3],'D','Independently simulated force');xlim(ax,0,20);save(fig,'Figure_7_3_EMG')


# FIGURE 7.4 — RESPIRATION: time-varying 12 to 21 breaths/min

fs=20;duration=80;t=np.arange(0,duration,1/fs)
rr_truth=12+9*np.clip((t-25)/35,0,1)
phase=np.cumsum(rr_truth/60/fs)
resp=np.sin(2*np.pi*phase)+.12*rng.normal(size=len(t))
fresp=np.linspace(.12,.48,70);tc,Wr=cwt_morlet(resp,fs,fresp)
power=np.abs(Wr)**2
ridge=fresp[np.argmax(power,axis=0)];ridge_s=gaussian_filter1d(ridge,3)
rr_est=60*ridge_s
edge=9;valid=(t>=edge)&(t<=duration-edge)
err=ma(rr_truth,rr_est,valid)
fig,ax=fig_axes(3)
ax[0].plot(t,resp,color=C['sig'],lw=.6);ax[0].set_ylabel('Resp. (a.u.)');title(ax[0],'A','Synthetic respiratory waveform')
im=ax[1].pcolormesh(tc,fresp,power,shading='auto',cmap='cividis',rasterized=True)
ax[1].plot(tc,ridge_s,color=C['est'],lw=1.0,label='CWT ridge');ax[1].set_ylabel('Frequency (Hz)');title(ax[1],'B','CWT power and frequency ridge');legend(ax[1],loc='upper right');cb(fig,ax[1],im,'Power (a.u.)')
ax[2].plot(t,rr_truth,'k--',lw=1,label='Known reference');ax[2].plot(t,np.where(valid,rr_est,np.nan),color=C['est'],lw=1.1,label='CWT estimate')
ax[2].set_ylabel('Breaths/min');ax[2].set_xlabel('Time (s)');title(ax[2],'C','Respiratory-rate tracking');legend(ax[2],loc='upper right')
shade(ax,edge,duration);xlim(ax,0,duration);save(fig,'Figure_7_4_Respiration')
print(f'Respiratory rate: interior MAE = {err:.2f} breaths/min (edge exclusion {edge:.0f} s)')


# FIGURE 7.5 — PPG: two-mode numerical VMD and Hilbert envelope

fs=50;duration=40;t=np.arange(0,duration,1/fs)
resp_ref=np.sin(2*np.pi*.24*t)
cardiac=np.sin(2*np.pi*1.35*t)+.25*np.sin(2*np.pi*2.7*t)
ppg=(1+.28*resp_ref)*cardiac+.16*resp_ref+.07*rng.normal(size=len(t))
modes,omega,niter=vmd(ppg,K=2,alpha=1800,max_iter=350)
# Choose mode by center-frequency proximity to known synthetic cardiac carrier.
card_idx=int(np.argmin(abs(omega*fs-1.35)))
card_mode=modes[card_idx];resp_mode=modes[1-card_idx]
env=np.abs(hilbert(card_mode))
env_lp=lowpass(env,fs,.50)
resp_est=norm(env_lp);resp_norm=norm(resp_ref)
edge=5;valid=(t>=edge)&(t<=duration-edge);r=corr(resp_norm,resp_est,valid)
fig,ax=fig_axes(5)
for j,(y,lab,desc,color) in enumerate([
    (ppg,'PPG (a.u.)','Synthetic amplitude-modulated PPG',C['sig']),
    (card_mode,'Mode (a.u.)','VMD cardiac-frequency mode',C['est']),
    (resp_mode,'Mode (a.u.)','Other VMD mode (not necessarily respiratory)',C['green']),
    (env,'Envelope (a.u.)','Cardiac-mode Hilbert envelope',C['purple'])]):
    ax[j].plot(t,y,color=color,lw=.6);ax[j].set_ylabel(lab);title(ax[j],'ABCDE'[j],desc)
ax[4].plot(t,resp_norm,'k--',lw=1,label='Known modulation');ax[4].plot(t,np.where(valid,resp_est,np.nan),color=C['green'],lw=1,label='Envelope proxy')
ax[4].set_ylabel('Normalized value');ax[4].set_xlabel('Time (s)');title(ax[4],'E','Respiratory modulation comparison');legend(ax[4],loc='upper right')
shade(ax,edge,duration);xlim(ax,0,duration);save(fig,'Figure_7_5_PPG')
print(f'PPG VMD: iterations={niter}, center frequencies={np.sort(omega*fs).round(3)} Hz, interior r={r:.3f}')
print(f'VMD reconstruction relative RMS error: {np.linalg.norm(ppg-modes.sum(axis=0))/np.linalg.norm(ppg):.3f} (algorithm diagnostic)')


# FIGURE 7.6 — ABP: respiratory pulse-pressure modulation

fs=100;duration=65;t=np.arange(0,duration,1/fs)
resp_hz=.25;resp_abp=np.sin(2*np.pi*resp_hz*t)
heart_hz=1.3
beats=np.arange(.5,duration-.5,1/heart_hz)
abp=np.full(len(t),75.,dtype=float)
for bt in beats:
    pp=39*(1+.16*np.sin(2*np.pi*resp_hz*bt))
    d=t-bt
    # A causal pulse with rapid rise, slower decay, and a dicrotic feature.
    u=np.maximum(d,0)
    pulse=(np.exp(-u/.22)-np.exp(-u/.027))*(d>=0)
    pulse/=.66
    abp+=pp*pulse+2.0*np.exp(-.5*((d-.28)/.028)**2)
abp+=.45*rng.normal(size=len(t))
peaks,_=find_peaks(abp,distance=int(fs*.58),prominence=15)
# Beatwise PP: max-min in each cycle, avoiding a cross-beat minimum.
pp_times=[];pulse_pressures=[]
for i in range(len(peaks)-1):
    lo,hi=peaks[i],peaks[i+1]
    if hi-lo<3:continue
    pp_times.append(t[lo]);pulse_pressures.append(abp[lo]-np.min(abp[lo:hi]))
pp_times=np.asarray(pp_times);pulse_pressures=np.asarray(pulse_pressures)
# Interpolate beatwise PP onto uniform grid for a conventional CWT.
tpp=np.arange(pp_times[0],pp_times[-1],.20)
pp_uniform=np.interp(tpp,pp_times,pulse_pressures)
fpp=np.linspace(.10,.50,65);tc,Wpp=cwt_morlet(pp_uniform,5,fpp)
# Approximate Morlet cone of influence (COI), using the Gaussian
# e-folding width. This masks boundary-sensitive power, not a formal
# statistical significance region.
pp_coi = np.zeros(Wpp.shape, dtype=bool)
for i, frequency in enumerate(fpp):
    edge_seconds = np.sqrt(2)*6/(2*np.pi*frequency)
    pp_coi[i] = ((tc >= tc[0] + edge_seconds) &
                 (tc <= tc[-1] - edge_seconds))
pp_power = np.ma.masked_where(~pp_coi, np.abs(Wpp)**2)
pp_cmap = plt.get_cmap('viridis').copy()
pp_cmap.set_bad('lightgray')
fig,ax=fig_axes(4)
ax[0].plot(t,abp,color=C['sig'],lw=.45);ax[0].scatter(t[peaks],abp[peaks],s=3,color=C['est'],label='Systolic peaks');ax[0].set_ylabel('ABP (mmHg)');title(ax[0],'A','Synthetic arterial pressure');legend(ax[0],loc='upper right')
ax[1].plot(pp_times,pulse_pressures,'o-',color=C['est'],ms=1.7,lw=.6);ax[1].set_ylabel('PP (mmHg)');title(ax[1],'B','Beatwise pulse pressure')
im=ax[2].pcolormesh(tc,fpp,pp_power,shading='auto',cmap=pp_cmap,rasterized=True);ax[2].axhline(resp_hz,color='white',ls='--',lw=1)
ax[2].set_ylabel('Frequency (Hz)');title(ax[2],'C','Pulse-pressure CWT (COI masked)');cb(fig,ax[2],im,'Power (a.u.)')
ax[3].plot(t,resp_abp,color='black',lw=.7);ax[3].set_ylabel('Resp. (a.u.)');ax[3].set_xlabel('Time (s)');title(ax[3],'D','Known respiratory reference');xlim(ax,0,duration);save(fig,'Figure_7_6_Arterial_Pressure')


# FIGURE 7.7 — ABP–CBFV: locally smoothed wavelet coherence


fs=10;duration=120;t=np.arange(0,duration,1/fs)
resp=np.sin(2*np.pi*.25*t)
noise1=lowpass(rng.normal(size=len(t)),fs,.75)
noise2=lowpass(rng.normal(size=len(t)),fs,.75)
abp_cbfv=92+6*resp+1.8*noise1
# Transition from shared oscillation to an independent oscillation.
a=np.clip((t-55)/15,0,1)
cbfv=42+4.5*((1-a)*np.sin(2*np.pi*.25*t-.35)+a*np.sin(2*np.pi*.32*t+1.1))+1.5*noise2
freq=np.linspace(.08,.7,65)
tc,Wa=cwt_morlet(abp_cbfv-np.mean(abp_cbfv),fs,freq)
_,Wb=cwt_morlet(cbfv-np.mean(cbfv),fs,freq)
# Scale-adaptive smoothing: time bandwidth varies with Morlet scale.
Saa=np.zeros_like(Wa.real);Sbb=np.zeros_like(Wa.real);Sab=np.zeros_like(Wa)
for i,f in enumerate(freq):
    sigma_t=max(2.0,0.65*fs/f)
    saa=abs(Wa[i])**2; sbb=abs(Wb[i])**2; sab=Wa[i]*np.conj(Wb[i])
    Saa[i]=gaussian_filter1d(saa,sigma_t)
    Sbb[i]=gaussian_filter1d(sbb,sigma_t)
    Sab[i]=gaussian_filter1d(sab.real,sigma_t)+1j*gaussian_filter1d(sab.imag,sigma_t)
# Smooth across neighboring scales as well.
Saa=gaussian_filter(Saa,(1.2,0));Sbb=gaussian_filter(Sbb,(1.2,0))
Sab=gaussian_filter(Sab.real,(1.2,0))+1j*gaussian_filter(Sab.imag,(1.2,0))
coh=np.clip(abs(Sab)**2/np.maximum(Saa*Sbb,1e-16),0,1)
# Conservative COI: approximately sqrt(2)*Morlet Gaussian width.
coi=np.zeros_like(coh,dtype=bool)
for i,f in enumerate(freq):
    edge=np.sqrt(2)*6/(2*np.pi*f)
    coi[i]=(tc>=edge)&(tc<=duration-edge)
masked=np.ma.array(coh,mask=~coi)
idx=np.argmin(abs(freq-.25));trajectory=np.where(coi[idx],coh[idx],np.nan)
# Do not smooth across invalid edge regions.
valid=np.isfinite(trajectory);smooth=np.full_like(trajectory,np.nan)
if valid.any():smooth[valid]=gaussian_filter1d(trajectory[valid],sigma=8)
before=(tc>=15)&(tc<=45)&valid;after=(tc>=80)&(tc<=105)&valid
mean_before=np.mean(trajectory[before]) if before.any() else np.nan
mean_after=np.mean(trajectory[after]) if after.any() else np.nan
fig,ax=fig_axes(5)
ax[0].plot(t,abp_cbfv,color=C['sig'],lw=.6);ax[0].set_ylabel('ABP (mmHg)');title(ax[0],'A','Synthetic arterial pressure')
ax[1].plot(t,cbfv,color=C['est'],lw=.6);ax[1].set_ylabel('CBFV (a.u.)');title(ax[1],'B','Cerebral flow-velocity proxy')
im=ax[2].pcolormesh(tc,freq,db(abs(Wa)**2),shading='auto',cmap='viridis',rasterized=True);ax[2].set_ylabel('Frequency (Hz)');title(ax[2],'C','ABP CWT power');cb(fig,ax[2],im,'Power (relative dB)')
cmap=plt.get_cmap('magma').copy();cmap.set_bad('lightgray')
im=ax[3].pcolormesh(tc,freq,masked,shading='auto',cmap=cmap,vmin=0,vmax=1,rasterized=True);ax[3].axhline(.25,color='cyan',lw=.8,ls='--');ax[3].set_ylabel('Frequency (Hz)');title(ax[3],'D','Wavelet coherence (COI masked)');cb(fig,ax[3],im,'Squared coherence')
ax[4].plot(tc,trajectory,color='.7',lw=.55,label='Estimate');ax[4].plot(tc,smooth,color=C['est'],lw=1.05,label='Display smoothing')
if np.isfinite(mean_before):ax[4].axhline(mean_before,color=C['green'],ls=':',lw=.9,label=f'Before {mean_before:.2f}')
if np.isfinite(mean_after):ax[4].axhline(mean_after,color=C['purple'],ls=':',lw=.9,label=f'After {mean_after:.2f}')
ax[4].set_ylim(0,1.05);ax[4].set_ylabel('Coherence');ax[4].set_xlabel('Time (s)');title(ax[4],'E',f'Coherence at {freq[idx]:.2f} Hz');legend(ax[4],loc='upper right',ncol=2)
transition(ax,55,70);xlim(ax,0,duration);save(fig,'Figure_7_7_Cerebral_Hemodynamics')
print(f'Coherence at {freq[idx]:.3f} Hz: before={mean_before:.3f}, after={mean_after:.3f}; valid n={before.sum()}/{after.sum()} (not independent)')

# FIGURE 7.8 — DOPPLER: narrowband complex IQ and ridge estimate

fs=4000;duration=3;t=np.arange(0,duration,1/fs)
vel_true=26+13*np.maximum(np.sin(2*np.pi*1.15*t),0)**2
# Positive-frequency analytic Doppler IQ; velocity-frequency calibration
# is chosen synthetically as 35 Hz per cm/s.
scale=35
phase=2*np.pi*np.cumsum(vel_true*scale)/fs
noise=(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))/np.sqrt(2)
doppler=np.exp(1j*phase)+.17*noise
f,ts,z=stft(doppler,fs=fs,nperseg=512,noverlap=464,return_onesided=False,boundary=None,padded=False)
pos=f>=0;f=f[pos];power=abs(z[pos])**2
vel_axis=f/scale
region=(vel_axis>=10)&(vel_axis<=55)
indices=np.argmax(power[region],axis=0)
vel_est=vel_axis[region][indices]
vel_est=gaussian_filter1d(vel_est,1.2)
ref=np.interp(ts,t,vel_true)
edge=.08;valid=(ts>=edge)&(ts<=duration-edge)
err=ma(ref,vel_est,valid)
segment=(t>=.5)&(t<=.53)
fig,ax=fig_axes(3)
ax[0].plot((t[segment]-.5)*1000,np.real(doppler[segment]),color=C['sig'],lw=.6)
ax[0].set_ylabel('Signal (a.u.)');ax[0].set_xlabel('Segment time (ms)');title(ax[0],'A','Short synthetic Doppler waveform')
im=ax[1].pcolormesh(ts,vel_axis,db(power),shading='auto',cmap='inferno',rasterized=True)
ax[1].set_ylim(0,55);ax[1].set_ylabel('Velocity (cm/s)');title(ax[1],'B','Spectral Doppler representation');cb(fig,ax[1],im,'Power (relative dB)')
ax[2].plot(t,vel_true,'k--',lw=1,label='Known reference');ax[2].plot(ts,np.where(valid,vel_est,np.nan),color=C['est'],lw=1.0,label='Spectral ridge')
ax[2].set_ylabel('Velocity (cm/s)');ax[2].set_xlabel('Time (s)');title(ax[2],'C','Spectral-ridge velocity estimate');legend(ax[2],loc='lower left')
ax[1].set_xlim(0,duration);ax[2].set_xlim(0,duration);save(fig,'Figure_7_8_Doppler')
print(f'Doppler interior MAE = {err:.3f} cm/s over {valid.sum()} overlapping windows')


# EXPORT VERIFICATION

from matplotlib.backends.backend_pdf import PdfPages
try:
    from pypdf import PdfReader
    for file in sorted(OUT.glob('Figure_7_*.pdf')):
        page=PdfReader(str(file)).pages[0]
        width=float(page.mediabox.width)*25.4/72
        height=float(page.mediabox.height)*25.4/72
        assert width<=130.01 and height<=184.01,(file,width,height)
        print(f'Verified PDF: {file.name} — {width:.2f} x {height:.2f} mm')
except ImportError:
    print('Install pypdf to verify PDF MediaBox dimensions.')
print('\nDONE: Eight synthetic figures exported as PNG and PDF.')
print('Scientific caveat: illustrative synthetic models; no clinical validation or significance testing.')
