
import os
import sys
import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add src to path to import GravityEngine
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.sec_36_gravity_engine import GravityEngine, get_data

# --- Configuration ---
OUTPUT_DIR = "output/sec_37"
STEPS = 100
DT = 0.1 # Time step (epsilon)
NOISE_SCALE = 0.5 # Temperature

# ==========================================
# PART 1: The Physics of Reverse Entropy
# ==========================================

def langevin_dynamics(engine, target_digit, steps=STEPS, dt=DT, noise_scale=NOISE_SCALE):
    """
    Simulates the 'Growth' of a digit from pure noise.
    Equation: dx = - grad(E)*dt + sqrt(2*dt)*noise
    """
    well = engine.wells[target_digit]
    mu = well['mu']      # (784,)
    sigma2 = well['sigma2'] # (784,)
    
    # Start from pure chaos (White Noise) aka Standard Gaussian
    # In our space, 'Standard' means mean=0.5, var=1 ? Or match data stats?
    # Let's start from standard normal N(0,1) scaled to pixel range
    current_x = np.random.normal(0.5, 0.5, size=(784,))
    
    history = [current_x.copy()]
    
    print(f"Diffusing digit {target_digit}...")
    
    for t in range(steps):
        # 1. Calculate Gradient of Potential E
        # E = 0.5 * (x - mu)^2 / sigma2
        # grad(E) = (x - mu) / sigma2
        grad_E = (current_x - mu) / sigma2
        
        # 2. Langevin Update
        # deterministic drift + thermal fluctuation
        drift = -grad_E * dt
        diffusion = np.sqrt(2 * dt) * np.random.normal(0, noise_scale, size=(784,))
        
        current_x = current_x + drift + diffusion
        
        # Clip to valid pixel range (optional, but physics respects boundaries/walls)
        # In pure physics, potential would go to infinity at boundaries.
        # Here we just clamp.
        current_x = np.clip(current_x, 0, 1)
        
        history.append(current_x.copy())
        
    return history

# ==========================================
# PART 2: Visualization
# ==========================================

def visualize_combined_diffusion(histories, targets):
    """
    Creates a combined animation of the diffusion process for multiple digits.
    """
    n_digits = len(targets)
    steps = len(histories[0])
    
    # Subsample frames
    frames = list(range(0, steps, max(1, steps//20)))
    if (steps-1) not in frames: frames.append(steps-1)
    
    # Grid: Rows = Digits, Cols = Time Steps
    fig = make_subplots(
        rows=n_digits, cols=len(frames),
        subplot_titles=[f"t={t}" if i==0 else "" for i in range(n_digits) for t in frames],
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
        row_titles=[f"Digit {d}" for d in targets]
    )
    
    for row_idx, (digit, history) in enumerate(zip(targets, histories)):
        for col_idx, t in enumerate(frames):
            img = history[t].reshape(28, 28)
            
            # Heatmap
            fig.add_trace(go.Heatmap(
                z=img, colorscale='Greys', showscale=False,
                zmin=0, zmax=1
            ), row=row_idx+1, col=col_idx+1)
            
            # Clean axes
            fig.update_xaxes(visible=False, row=row_idx+1, col=col_idx+1)
            fig.update_yaxes(visible=False, scaleanchor=f"x{col_idx+1}", autorange='reversed', row=row_idx+1, col=col_idx+1)

    fig.update_layout(
        title="<b>Macroscopic Diffusion: Generating from Chaos</b><br>Langevin Dynamics acting on Gravitational Potential Field (Pure NumPy)",
        height=200 * n_digits,
        margin=dict(l=50, r=10, t=80, b=10),
        template="plotly_dark"
    )
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "macroscopic_diffusion.html")
    fig.write_html(out_path)
    print(f"Saved combined diffusion viz to {out_path}")

def run_diffusion_experiment():
    print("Initialising Gravity Engine...")
    train_x, train_y, _, _ = get_data() 
    
    engine = GravityEngine()
    engine.fit(train_x, train_y)
    
    # Generate 3 digits
    targets = [8, 3, 0,9,4]
    histories = []
    
    for digit in targets:
        # Longer steps for better viz
        history = langevin_dynamics(engine, digit, steps=60, dt=0.15, noise_scale=0.2)
        histories.append(history)
        
    visualize_combined_diffusion(histories, targets)

if __name__ == "__main__":
    run_diffusion_experiment()
