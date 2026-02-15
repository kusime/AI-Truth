"""
Section 32: 维度相变 - 从渐变到崩塌
Dimensional Phase Transition: From Gradient to Collapse

Theory:
1. Phase I: SNR Scaling (Energy Law) -> SNR ~ 1/d
2. Phase II: BBP Threshold (The Singularity) -> Critical Gamma_c = 4.0
3. Phase III: Death of Direction (Information Limit) -> Overlap <v,u>^2 -> 0

Implementation:
- Full GPU Acceleration (PyTorch) for high-dimensional SVD.
- Unified execution flow generating all visualizations.
"""

import os
import time

import numpy as np
import plotly.graph_objects as go
import torch
from sklearn.datasets import fetch_openml, load_digits
from sklearn.metrics import accuracy_score

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on Device: {device}")

# Create Output Dirs (Unified)
os.makedirs('output/sec_32', exist_ok=True)
os.makedirs('output/sec_34', exist_ok=True) # Keep for compatibility or just sec_32? Doc uses both.

def run_phase_i_snr_scaling():
    print(f"\n{'='*50}\nPHASE I: ENERGY SCALING (SNR ~ 1/d)\n{'='*50}")
    
    # 1. Load Data
    try:
        data = fetch_openml('mnist_784', version=1, return_X_y=False, as_frame=False, cache=True)
        X_raw = data.data
        y = data.target
        X_raw = X_raw[y == '0'][:1000] # Digit 0, 1000 samples
        d0 = 784
    except:
        print("Fetch MNIST failed, using Digits fallback")
        X_raw, y = load_digits(return_X_y=True)
        X_raw = X_raw[y == 0]
        d0 = 64
    
    # Preprocess
    X_torch = torch.tensor(X_raw, dtype=torch.float32, device=device)
    X_torch = X_torch - X_torch.mean(dim=0, keepdim=True)
    std = X_torch.std() + 1e-8
    X_torch = X_torch / std
    
    E_signal = torch.mean(torch.sum(X_torch**2, dim=1)).item()
    n_samples = X_torch.shape[0]
    results = []
    
    # Scale dimensions
    dims = np.logspace(np.log10(d0), np.log10(10000), 20).astype(int)
    noise_sigma = 1.0
    
    for d_total in dims:
        if d_total <= d0: continue
        d_noise = d_total - d0
        
        # Generate Noise
        Noise = torch.randn(n_samples, d_noise, device=device) * noise_sigma
        X_full = torch.cat([X_torch, Noise], dim=1)
        
        # SVD (Eigenvalues)
        try:
             _, S, _ = torch.linalg.svd(X_full / np.sqrt(n_samples), full_matrices=False)
             lambda_1 = (S[0]**2).item()
        except:
             lambda_1 = 0
        
        # SNR
        E_noise = torch.mean(torch.sum(Noise**2, dim=1)).item()
        snr = E_signal / E_noise
        
        # Theoretical Bound
        gamma = d_total / n_samples
        mp_bound = (noise_sigma**2) * (1 + np.sqrt(gamma))**2
        
        results.append({'d': d_total, 'snr': snr, 'lambda_1': lambda_1, 'mp_bound': mp_bound})
        if d_total > 1000 and d_total < 2000:
            print(f"Dim {d_total} | SNR {snr:.4f} | Lam1 {lambda_1:.2f} vs MP {mp_bound:.2f}")

    # Viz
    dims_v = [r['d'] for r in results]
    snrs = [r['snr'] for r in results]
    lam1 = [r['lambda_1'] for r in results]
    mp = [r['mp_bound'] for r in results]
    
    # Combined Viz 1
    fig_snr = go.Figure()
    fig_snr.add_trace(go.Scatter(x=dims_v, y=snrs, mode='markers', name='Exp SNR', marker=dict(color='#00bdff')))
    k = snrs[0] * dims_v[0]
    fig_snr.add_trace(go.Scatter(x=dims_v, y=[k/d for d in dims_v], mode='lines', name='1/d Theory', line=dict(dash='dash', color='gray')))
    fig_snr.update_layout(title="Phase I: Energy Scaling", xaxis_type="log", yaxis_type="log", template='plotly_dark')
    
    with open('output/sec_32/spectral_viz.html', 'w') as f:
        f.write(fig_snr.to_html(full_html=False, include_plotlyjs='cdn'))
    print("Phase I Viz Saved: output/sec_32/spectral_viz.html")


def run_phase_ii_iii_bbp_rigorous():
    print(f"\n{'='*50}\nPHASE II & III: BBP SINGULARITY & INFORMATION LIMIT\n{'='*50}")
    
    n_samples = 1000
    alpha = 2.0
    sigma = 1.0
    
    # Fine Gamma Scan
    gammas = np.linspace(2.0, 6.0, 41)
    ds = (gammas * n_samples).astype(int)
    
    results = []
    
    for gamma, d in zip(gammas, ds):
        u = torch.randn(d, device=device)
        u = u / torch.norm(u)
        
        z = torch.randn(n_samples, device=device) * np.sqrt(alpha)
        S = torch.outer(z, u)
        N_mat = torch.randn(n_samples, d, device=device) * sigma
        X = S + N_mat
        
        # SVD
        try:
             _, S_vals, Vh = torch.linalg.svd(X / np.sqrt(n_samples), full_matrices=False)
             v1 = Vh[0] # Top direction
        except:
             v1 = torch.zeros(d, device=device)

        # Overlap (Phase III)
        overlap = (torch.dot(v1, u).item())**2
        
        # Accuracy (Algorithms)
        X_cpu = X.cpu().numpy()
        y_cpu = (z.cpu().numpy() > 0).astype(int)
        
        # Eigen Classifier
        scores = X_cpu @ v1.cpu().numpy()
        pred = (scores > 0).astype(int)
        acc_eigen = accuracy_score(y_cpu, pred)
        if acc_eigen < 0.5: acc_eigen = 1 - acc_eigen
        
        # Oracle Classifier
        scores_oracle = X_cpu @ u.cpu().numpy()
        pred_oracle = (scores_oracle > 0).astype(int)
        if np.corrcoef(scores_oracle, z.cpu().numpy())[0,1] < 0: pred_oracle = 1 - pred_oracle
        acc_oracle = accuracy_score(y_cpu, pred_oracle)
        
        results.append({
            'gamma': gamma, 
            'overlap': overlap,
            'acc_eigen': acc_eigen,
            'acc_oracle': acc_oracle
        })
        
        if len(results) % 10 == 0:
            print(f"Gamma {gamma:.2f} | Overlap {overlap:.4f} | Acc {acc_eigen:.2f}")

    # Viz
    gammas_v = [r['gamma'] for r in results]
    
    # Combined Viz 2 (Rigorous)
    fig_rigorous = go.Figure()
    
    # Overlap
    fig_rigorous.add_trace(go.Scatter(x=gammas_v, y=[r['overlap'] for r in results], name='Direction Overlap', line=dict(color='#00ff88', width=3)))
    
    # Accuracies
    fig_rigorous.add_trace(go.Scatter(x=gammas_v, y=[r['acc_eigen'] for r in results], name='Eigenvector Acc', line=dict(color='#ff0055', width=2)))
    fig_rigorous.add_trace(go.Scatter(x=gammas_v, y=[r['acc_oracle'] for r in results], name='Oracle Acc', line=dict(color='#00bdff', dash='dot')))
    
    fig_rigorous.add_vline(x=4.0, line_dash="dash", line_color="yellow", annotation_text="Singularity (Gamma=4.0)")
    fig_rigorous.update_layout(title="Phase II & III: The Collapse", xaxis_title="Gamma (d/n)", template='plotly_dark')
    
    # Save to the path expected by docs (sec_34 folder but unified concept)
    with open('output/sec_34/bbp_rigorous_viz.html', 'w') as f:
        f.write(fig_rigorous.to_html(full_html=False, include_plotlyjs='cdn'))
    print("Phase II/III Viz Saved: output/sec_34/bbp_rigorous_viz.html")

if __name__ == "__main__":
    run_phase_i_snr_scaling()
    run_phase_ii_iii_bbp_rigorous()
