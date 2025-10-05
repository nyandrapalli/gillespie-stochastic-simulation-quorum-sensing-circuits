#!/usr/bin/env python3
"""
Cond1 timelapse (single synthetic cell with LuxI + LuxR + pLux–GFP).
Runs N replicates (default 500) to T_end (default 6000).
Outputs:
  - raw CSV: cond1_timelapse_raw.csv
  - plot PNG: cond1_timelapse_plot.png
  - summary CSV: cond1_summary.csv  (Mean, SD, CV, CV² over steady window)
"""
import os, math, argparse
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Params:
    RNAP_total:int=50; Ribo_total:int=200
    n_pLux_gfp:int=10; n_pCon_luxI:int=10
    LuxR0:int=100
    kon_act:float=1e-3; koff_act:float=1e-3
    kon_RL:float=1e-3; koff_RL:float=1e-2
    kon_RNAP:float=5e-4; kcat_tx_active:float=0.08; kcat_tx_basal:float=0.008
    kon_ribo:float=1e-3; kcat_tl:float=0.5
    gamma_m:float=0.01; gamma_p:float=5e-4; kmat:float=0.01
    k_I:float=0.02; gamma_AHL:float=0.001
    gamma_G:float=0.01
    T_end:float=6000.0
    load_promoters:int=0

def gillespie_cond1(P:Params, seed:int=None):
    rng = np.random.default_rng(seed)
    AHL=0
    c={'RNAP_free':P.RNAP_total,'Ribo_free':P.Ribo_total,
       'LuxR':P.LuxR0,'RL':0,
       'pLux_gfp_free':P.n_pLux_gfp,'pLux_gfp_act':0,
       'pLux_gfp_RNAP_inact':0,'pLux_gfp_RNAP_act':0,
       'pCon_luxI_free':P.n_pCon_luxI,'pCon_luxI_RNAP':0,
       'm_gfp':0,'m_luxI':0,'ribo_gfp':0,'ribo_luxI':0,'LuxI':0,'Gi':0,'G':0,
       'pLoad_free':P.load_promoters,'pLoad_RNAP':0,'m_load':0,'ribo_load':0,'prot_load':0}
    t=0.0; times=[0.0]; G=[0]; it=0; max_it=int(2e6)
    while t<P.T_end and it<max_it:
        it+=1; props=[]; rxns=[]
        # RL formation
        if c['LuxR']>0 and AHL>0: props.append(P.kon_RL*c['LuxR']*AHL); rxns.append(('B','bind_RL'))
        if c['RL']>0: props.append(P.koff_RL*c['RL']); rxns.append(('B','unbind_RL'))
        # Activation
        if c['RL']>0 and c['pLux_gfp_free']>0: props.append(P.kon_act*c['RL']*c['pLux_gfp_free']); rxns.append(('B','act'))
        if c['pLux_gfp_act']>0: props.append(P.koff_act*c['pLux_gfp_act']); rxns.append(('B','deact'))
        # RNAP binding
        if c['RNAP_free']>0:
            if c['pLux_gfp_free']>0: props.append(P.kon_RNAP*c['RNAP_free']*c['pLux_gfp_free']); rxns.append(('B','bind_inact'))
            if c['pLux_gfp_act']>0: props.append(P.kon_RNAP*c['RNAP_free']*c['pLux_gfp_act']); rxns.append(('B','bind_act'))
            if c['pCon_luxI_free']>0: props.append(P.kon_RNAP*c['RNAP_free']*c['pCon_luxI_free']); rxns.append(('A','bind_I'))
            if c['pLoad_free']>0: props.append(P.kon_RNAP*c['RNAP_free']*c['pLoad_free']); rxns.append(('L','bind_load'))
        # Transcription
        if c['pLux_gfp_RNAP_inact']>0: props.append(P.kcat_tx_basal*c['pLux_gfp_RNAP_inact']); rxns.append(('B','tx_inact'))
        if c['pLux_gfp_RNAP_act']>0: props.append(P.kcat_tx_active*c['pLux_gfp_RNAP_act']); rxns.append(('B','tx_act'))
        if c['pCon_luxI_RNAP']>0: props.append(P.kcat_tx_active*c['pCon_luxI_RNAP']); rxns.append(('A','tx_I'))
        if c['pLoad_RNAP']>0: props.append(P.kcat_tx_active*c['pLoad_RNAP']); rxns.append(('L','tx_load'))
        # Translation
        if c['Ribo_free']>0 and c['m_gfp']>0: props.append(P.kon_ribo*c['Ribo_free']*c['m_gfp']); rxns.append(('B','bind_ribo_gfp'))
        if c['Ribo_free']>0 and c['m_luxI']>0: props.append(P.kon_ribo*c['Ribo_free']*c['m_luxI']); rxns.append(('A','bind_ribo_I'))
        if c['Ribo_free']>0 and c['m_load']>0: props.append(P.kon_ribo*c['Ribo_free']*c['m_load']); rxns.append(('L','bind_ribo_load'))
        if c['ribo_gfp']>0: props.append(P.kcat_tl*c['ribo_gfp']); rxns.append(('B','tl_gfp'))
        if c['ribo_luxI']>0: props.append(P.kcat_tl*c['ribo_luxI']); rxns.append(('A','tl_I'))
        if c['ribo_load']>0: props.append(P.kcat_tl*c['ribo_load']); rxns.append(('L','tl_load'))
        # Decay/maturation
        if c['m_gfp']>0: props.append(P.gamma_m*c['m_gfp']); rxns.append(('B','deg_m_gfp'))
        if c['m_luxI']>0: props.append(P.gamma_m*c['m_luxI']); rxns.append(('A','deg_m_I'))
        if c['m_load']>0: props.append(P.gamma_m*c['m_load']); rxns.append(('L','deg_m_load'))
        if c['LuxI']>0: props.append(P.gamma_p*c['LuxI']); rxns.append(('A','deg_I'))
        if c['prot_load']>0: props.append(P.gamma_p*c['prot_load']); rxns.append(('L','deg_prot_load'))
        if c['Gi']>0: props.append(P.kmat*c['Gi']); rxns.append(('B','mat'))
        if c['G']>0: props.append(P.gamma_G*c['G']); rxns.append(('B','deg_G'))
        # Signal
        if c['LuxI']>0: props.append(P.k_I*c['LuxI']); rxns.append(('S','syn_AHL'))
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
        # updates
        if who=='B' and rx=='bind_RL': c['LuxR']-=1; AHL-=1; c['RL']+=1
        elif who=='B' and rx=='unbind_RL': c['RL']-=1; c['LuxR']+=1; AHL+=1
        elif who=='B' and rx=='act': c['RL']-=1; c['pLux_gfp_free']-=1; c['pLux_gfp_act']+=1
        elif who=='B' and rx=='deact': c['pLux_gfp_act']-=1; c['pLux_gfp_free']+=1; c['RL']+=1
        elif who=='B' and rx=='bind_inact': c['RNAP_free']-=1; c['pLux_gfp_free']-=1; c['pLux_gfp_RNAP_inact']+=1
        elif who=='B' and rx=='bind_act': c['RNAP_free']-=1; c['pLux_gfp_act']-=1; c['pLux_gfp_RNAP_act']+=1
        elif who=='A' and rx=='bind_I': c['RNAP_free']-=1; c['pCon_luxI_free']-=1; c['pCon_luxI_RNAP']+=1
        elif who=='L' and rx=='bind_load': c['RNAP_free']-=1; c['pLoad_free']-=1; c['pLoad_RNAP']+=1
        elif who=='B' and rx=='tx_inact': c['pLux_gfp_RNAP_inact']-=1; c['RNAP_free']+=1; c['pLux_gfp_free']+=1; c['m_gfp']+=1
        elif who=='B' and rx=='tx_act': c['pLux_gfp_RNAP_act']-=1; c['RNAP_free']+=1; c['pLux_gfp_act']+=1; c['m_gfp']+=1
        elif who=='A' and rx=='tx_I': c['pCon_luxI_RNAP']-=1; c['RNAP_free']+=1; c['pCon_luxI_free']+=1; c['m_luxI']+=1
        elif who=='L' and rx=='tx_load': c['pLoad_RNAP']-=1; c['RNAP_free']+=1; c['pLoad_free']+=1; c['m_load']+=1
        elif who=='B' and rx=='bind_ribo_gfp': c['Ribo_free']-=1; c['m_gfp']-=1; c['ribo_gfp']+=1
        elif who=='A' and rx=='bind_ribo_I': c['Ribo_free']-=1; c['m_luxI']-=1; c['ribo_luxI']+=1
        elif who=='L' and rx=='bind_ribo_load': c['Ribo_free']-=1; c['m_load']-=1; c['ribo_load']+=1
        elif who=='B' and rx=='tl_gfp': c['ribo_gfp']-=1; c['Ribo_free']+=1; c['Gi']+=1
        elif who=='A' and rx=='tl_I': c['ribo_luxI']-=1; c['Ribo_free']+=1; c['LuxI']+=1
        elif who=='L' and rx=='tl_load': c['ribo_load']-=1; c['Ribo_free']+=1; c['prot_load']+=1
        elif who=='B' and rx=='deg_m_gfp': c['m_gfp']-=1
        elif who=='A' and rx=='deg_m_I': c['m_luxI']-=1
        elif who=='L' and rx=='deg_m_load': c['m_load']-=1
        elif who=='A' and rx=='deg_I': c['LuxI']-=1
        elif who=='L' and rx=='deg_prot_load': c['prot_load']-=1
        elif who=='B' and rx=='mat': c['Gi']-=1; c['G']+=1
        elif who=='B' and rx=='deg_G': c['G']-=1
        elif who=='S' and rx=='syn_AHL': AHL+=1
        elif who=='S' and rx=='deg_AHL': AHL-=1
        times.append(t); G.append(c['G'])
    return np.array(times), np.array(G)

def run(N=500, T_end=6000, window=(2500,3000), outdir="cond1_timelapse_out", seed=111):
    os.makedirs(outdir, exist_ok=True)
    P = Params(T_end=T_end)
    t_grid = np.arange(int(T_end)+1)
    curves = np.zeros((N, len(t_grid)))
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 2**31-1, size=N, dtype=np.int64)
    for i in range(N):
        ti,Gi = gillespie_cond1(P, seed=int(seeds[i]))
        idx = np.searchsorted(ti, t_grid, side='right')-1; idx[idx<0]=0
        curves[i,:]=Gi[idx]
    # save raw
    raw_csv = os.path.join(outdir,"cond1_timelapse_raw.csv")
    pd.DataFrame(curves, columns=[f"t{{t}}" for t in t_grid]).assign(unit_id=lambda d: np.arange(N)).to_csv(raw_csv, index=False)
    # metrics
    w0,w1=window
    block = curves[:, w0:w1+1].mean(axis=1)
    mean=float(block.mean()); sd=float(block.std(ddof=0)); cv=float(sd/mean) if mean>0 else float("nan"); cv2=float(cv*cv) if mean>0 else float("nan")
    pd.DataFrame([{"mean":mean,"sd":sd,"CV":cv,"CV2":cv2,"N":N,"T_end":T_end,"window_start":w0,"window_end":w1}]).to_csv(os.path.join(outdir,"cond1_summary.csv"), index=False)
    # plot
    mean_curve = curves.mean(axis=0); p10=np.percentile(curves,10,axis=0); p90=np.percentile(curves,90,axis=0)
    plt.figure(figsize=(10,5))
    for i in range(min(N,100)): plt.plot(t_grid, curves[i,:], alpha=0.08, linewidth=0.5)
    plt.plot(t_grid, mean_curve, linewidth=2.0, label="Mean")
    plt.fill_between(t_grid, p10, p90, alpha=0.25, label="10–90%")
    plt.axvspan(w0, w1, alpha=0.1, label="Steady window")
    plt.xlabel("Time"); plt.ylabel("GFP"); plt.title("Cond1 timelapse")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"cond1_timelapse_plot.png"), dpi=150); plt.close()

def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=500)
    ap.add_argument("--T", type=int, default=6000)
    ap.add_argument("--w0", type=int, default=2500)
    ap.add_argument("--w1", type=int, default=3000)
    ap.add_argument("--out", type=str, default="cond1_timelapse_out")
    return ap.parse_args()

if __name__=="__main__":
    args = parse_args()
    run(N=args.N, T_end=args.T, window=(args.w0,args.w1), outdir=args.out)
