#!/usr/bin/env python3
"""
Cond2 timelapse (one-way communication).
Two synthetic cells share AHL.
  - Cell A: pCon–LuxI (+ optional burden, unused here)
  - Cell B: LuxR + pLux–GFP
Defaults: N=500, T_end=6000, steady window 2500–3000.
Outputs:
  - raw CSV: cond2_timelapse_raw.csv (GA & GB timelines)
  - plot PNG: cond2_timelapse_plot.png (GB focus)
  - summary CSV: cond2_summary.csv (on GFP_total = GA + GB)
"""
import os, math, argparse
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Params2:
    RNAP_total:int=50; Ribo_total:int=200
    n_pCon_luxI_A:int=10; n_pLux_gfp_B:int=10
    LuxR0_B:int=100
    kon_act:float=1e-3; koff_act:float=1e-3
    kon_RL:float=1e-3; koff_RL:float=1e-2
    kon_RNAP:float=5e-4; kcat_tx_active:float=0.08; kcat_tx_basal:float=0.008
    kon_ribo:float=1e-3; kcat_tl:float=0.5
    gamma_m:float=0.01; gamma_p:float=5e-4; kmat:float=0.01
    k_I:float=0.02; gamma_AHL:float=0.001
    gamma_G:float=0.01
    T_end:float=6000.0

def gillespie_cond2(P:Params2, seed:int=None):
    rng = np.random.default_rng(seed)
    AHL = 0
    A = {'RNAP_free':P.RNAP_total,'Ribo_free':P.Ribo_total,
         'pCon_luxI_free':P.n_pCon_luxI_A,'pCon_luxI_RNAP':0,
         'm_luxI':0,'ribo_luxI':0,'LuxI':0,'Gi':0,'G':0}
    B = {'RNAP_free':P.RNAP_total,'Ribo_free':P.Ribo_total,
         'LuxR':P.LuxR0_B,'RL':0,
         'pLux_gfp_free':P.n_pLux_gfp_B,'pLux_gfp_act':0,
         'pLux_gfp_RNAP_inact':0,'pLux_gfp_RNAP_act':0,
         'm_gfp':0,'ribo_gfp':0,'Gi':0,'G':0}
    t=0.0; it=0; max_it=int(2e6)
    times=[0.0]; GA=[0]; GB=[0]
    while t<P.T_end and it<max_it:
        it+=1; props=[]; rxns=[]
        # A: LuxI production
        if A['RNAP_free']>0 and A['pCon_luxI_free']>0: props.append(P.kon_RNAP*A['RNAP_free']*A['pCon_luxI_free']); rxns.append(('A','bind_pCon_luxI'))
        if A['pCon_luxI_RNAP']>0: props.append(P.kcat_tx_active*A['pCon_luxI_RNAP']); rxns.append(('A','tx_pCon_luxI'))
        if A['Ribo_free']>0 and A['m_luxI']>0: props.append(P.kon_ribo*A['Ribo_free']*A['m_luxI']); rxns.append(('A','bind_ribo_luxI'))
        if A['ribo_luxI']>0: props.append(P.kcat_tl*A['ribo_luxI']); rxns.append(('A','tl_luxI'))
        if A['m_luxI']>0: props.append(P.gamma_m*A['m_luxI']); rxns.append(('A','deg_m_luxI'))
        if A['LuxI']>0: props.append(P.gamma_p*A['LuxI']); rxns.append(('A','deg_LuxI'))
        if A['LuxI']>0: props.append(P.k_I*A['LuxI']); rxns.append(('S','syn_AHL'))
        # B: LuxR + pLux–GFP
        if B['LuxR']>0 and AHL>0: props.append(P.kon_RL*B['LuxR']*AHL); rxns.append(('B','bind_RL'))
        if B['RL']>0: props.append(P.koff_RL*B['RL']); rxns.append(('B','unbind_RL'))
        if B['RL']>0 and B['pLux_gfp_free']>0: props.append(P.kon_act*B['RL']*B['pLux_gfp_free']); rxns.append(('B','act_pLux_gfp'))
        if B['pLux_gfp_act']>0: props.append(P.koff_act*B['pLux_gfp_act']); rxns.append(('B','deact_pLux_gfp'))
        if B['RNAP_free']>0:
            if B['pLux_gfp_free']>0: props.append(P.kon_RNAP*B['RNAP_free']*B['pLux_gfp_free']); rxns.append(('B','bind_pLux_gfp_inact'))
            if B['pLux_gfp_act']>0: props.append(P.kon_RNAP*B['RNAP_free']*B['pLux_gfp_act']); rxns.append(('B','bind_pLux_gfp_act'))
        if B['pLux_gfp_RNAP_inact']>0: props.append(P.kcat_tx_basal*B['pLux_gfp_RNAP_inact']); rxns.append(('B','tx_pLux_gfp_inact'))
        if B['pLux_gfp_RNAP_act']>0: props.append(P.kcat_tx_active*B['pLux_gfp_RNAP_act']); rxns.append(('B','tx_pLux_gfp_act'))
        if B['Ribo_free']>0 and B['m_gfp']>0: props.append(P.kon_ribo*B['Ribo_free']*B['m_gfp']); rxns.append(('B','bind_ribo_gfp'))
        if B['ribo_gfp']>0: props.append(P.kcat_tl*B['ribo_gfp']); rxns.append(('B','tl_gfp'))
        if B['m_gfp']>0: props.append(P.gamma_m*B['m_gfp']); rxns.append(('B','deg_m_gfp'))
        if B['Gi']>0: props.append(P.kmat*B['Gi']); rxns.append(('B','mat_G'))
        if B['G']>0: props.append(P.gamma_G*B['G']); rxns.append(('B','deg_G'))
        # Shared AHL decay
        if AHL>0: props.append(P.gamma_AHL*AHL); rxns.append(('S','deg_AHL'))
        # Fire
        a0=sum(props)
        if a0<=0: break
        tau=-math.log(rng.random())/a0; t+=tau
        r2=rng.random()*a0; cum=0.0; idx=-1
        for i,p in enumerate(props):
            cum+=p
            if r2<=cum: idx=i; break
        who,rx = rxns[idx]
        if who=='A':
            if rx=='bind_pCon_luxI': A['RNAP_free']-=1; A['pCon_luxI_free']-=1; A['pCon_luxI_RNAP']+=1
            elif rx=='tx_pCon_luxI': A['pCon_luxI_RNAP']-=1; A['RNAP_free']+=1; A['pCon_luxI_free']+=1; A['m_luxI']+=1
            elif rx=='bind_ribo_luxI': A['Ribo_free']-=1; A['m_luxI']-=1; A['ribo_luxI']+=1
            elif rx=='tl_luxI': A['ribo_luxI']-=1; A['Ribo_free']+=1; A['LuxI']+=1
            elif rx=='deg_m_luxI': A['m_luxI']-=1
            elif rx=='deg_LuxI': A['LuxI']-=1
        elif who=='B':
            if rx=='bind_RL': B['LuxR']-=1; AHL-=1; B['RL']+=1
            elif rx=='unbind_RL': B['RL']-=1; B['LuxR']+=1; AHL+=1
            elif rx=='act_pLux_gfp': B['RL']-=1; B['pLux_gfp_free']-=1; B['pLux_gfp_act']+=1
            elif rx=='deact_pLux_gfp': B['pLux_gfp_act']-=1; B['pLux_gfp_free']+=1; B['RL']+=1
            elif rx=='bind_pLux_gfp_inact': B['RNAP_free']-=1; B['pLux_gfp_free']-=1; B['pLux_gfp_RNAP_inact']+=1
            elif rx=='bind_pLux_gfp_act': B['RNAP_free']-=1; B['pLux_gfp_act']-=1; B['pLux_gfp_RNAP_act']+=1
            elif rx=='tx_pLux_gfp_inact': B['pLux_gfp_RNAP_inact']-=1; B['RNAP_free']+=1; B['pLux_gfp_free']+=1; B['m_gfp']+=1
            elif rx=='tx_pLux_gfp_act': B['pLux_gfp_RNAP_act']-=1; B['RNAP_free']+=1; B['pLux_gfp_act']+=1; B['m_gfp']+=1
            elif rx=='bind_ribo_gfp': B['Ribo_free']-=1; B['m_gfp']-=1; B['ribo_gfp']+=1
            elif rx=='tl_gfp': B['ribo_gfp']-=1; B['Ribo_free']+=1; B['Gi']+=1
            elif rx=='deg_m_gfp': B['m_gfp']-=1
            elif rx=='mat_G': B['Gi']-=1; B['G']+=1
            elif rx=='deg_G': B['G']-=1
        else:
            if rx=='syn_AHL': AHL+=1
            elif rx=='deg_AHL': AHL-=1
        times.append(t); GA.append(A['G']); GB.append(B['G'])
    return np.array(times), np.array(GA), np.array(GB)

def run(N=500, T_end=6000, window=(2500,3000), outdir="cond2_timelapse_out", seed=222):
    os.makedirs(outdir, exist_ok=True)
    P = Params2(T_end=T_end)
    t_grid = np.arange(int(T_end)+1)
    GA_all = np.zeros((N, len(t_grid))); GB_all = np.zeros((N, len(t_grid)))
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 2**31-1, size=N, dtype=np.int64)
    for i in range(N):
        ti,GA,GB = gillespie_cond2(P, seed=int(seeds[i]))
        idx = np.searchsorted(ti, t_grid, side='right')-1; idx[idx<0]=0
        GA_all[i,:]=GA[idx]; GB_all[i,:]=GB[idx]
    # save raw
    cols = [f"GA_t{{t}}" for t in t_grid] + [f"GB_t{{t}}" for t in t_grid]
    raw = np.hstack([GA_all, GB_all])
    pd.DataFrame(raw, columns=cols).assign(replicate_id=lambda d: np.arange(N)).to_csv(os.path.join(outdir,"cond2_timelapse_raw.csv"), index=False)
    # summary on GFP_total
    w0,w1=window
    A_means = GA_all[:, w0:w1+1].mean(axis=1)
    B_means = GB_all[:, w0:w1+1].mean(axis=1)
    TOT = A_means + B_means
    mean=float(TOT.mean()); sd=float(TOT.std(ddof=0)); cv=float(sd/mean) if mean>0 else float("nan"); cv2=float(cv*cv) if mean>0 else float("nan")
    pd.DataFrame([{"mean":mean,"sd":sd,"CV":cv,"CV2":cv2,"N":N,"T_end":T_end,"window_start":w0,"window_end":w1}]).to_csv(os.path.join(outdir,"cond2_summary.csv"), index=False)
    # plot GB
    meanB=GB_all.mean(axis=0); p10=np.percentile(GB_all,10,axis=0); p90=np.percentile(GB_all,90,axis=0)
    plt.figure(figsize=(10,5))
    for i in range(min(N,100)): plt.plot(t_grid, GB_all[i,:], alpha=0.08, linewidth=0.5)
    plt.plot(t_grid, meanB, linewidth=2.0, label="Mean (B)")
    plt.fill_between(t_grid, p10, p90, alpha=0.25, label="10–90% (B)")
    plt.axvspan(w0, w1, alpha=0.1, label="Steady window")
    plt.xlabel("Time"); plt.ylabel("GFP (cell B)"); plt.title("Cond2 timelapse")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"cond2_timelapse_plot.png"), dpi=150); plt.close()

def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=500)
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--w0", type=int, default=2500)
    ap.add_argument("--w1", type=int, default=3000)
    ap.add_argument("--out", type=str, default="cond2_timelapse_out")
    return ap.parse_args()

if __name__=="__main__":
    args = parse_args()
    run(N=args.N, T_end=args.T, window=(args.w0,args.w1), outdir=args.out)
