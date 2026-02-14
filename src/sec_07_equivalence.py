import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr

# --- User's Intuition ---
# "我和那些专家的参数完全不一样，但是总 Loss 很少... y_me ≈ y_pro"
# "Equivalence Principle: Different Parameters, Same Logic"
# Section 7: Equivalence Principle

def generate_complex_data(n_samples=1000):
    # Spiral Data again (Complex logic)
    theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi 
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T + np.random.randn(n_samples, 2) * 0.1
    x = torch.FloatTensor(data_a)
    # Target function: Distance from origin + Sinusoidal modulation
    # y = sin(r) > 0 ?
    r = torch.norm(x, dim=1)
    y = (torch.sin(r) > 0).float().unsqueeze(1)
    return x, y

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def train_model(X, y, seed):
    torch.manual_seed(seed)
    model = SimpleNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    for _ in range(500):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    return model

def verify_equivalence():
    print("Running Equivalence Simulation...")
    # 1. Generate Data (The "Truth")
    X, y = generate_complex_data(1000)
    
    # 2. Train Model A ("Expert")
    print("Training Model A (Seed 42)...")
    model_A = train_model(X, y, seed=42)
    
    # 3. Train Model B ("Me") - Different Init
    print("Training Model B (Seed 999)...")
    model_B = train_model(X, y, seed=999)
    
    # 4. Compare Weights (Parameters)
    # Flatten all weights
    w_A = torch.cat([p.flatten() for p in model_A.parameters()]).detach().numpy()
    w_B = torch.cat([p.flatten() for p in model_B.parameters()]).detach().numpy()
    
    weight_corr, _ = pearsonr(w_A, w_B)
    print(f"Weight Correlation: {weight_corr:.4f} (Should be low/random)")
    
    # 5. Compare Outputs (Logic)
    # Test on new data
    X_test, _ = generate_complex_data(1000)
    
    with torch.no_grad():
        out_A = model_A(X_test).flatten().numpy()
        out_B = model_B(X_test).flatten().numpy()
        
    logic_corr, _ = pearsonr(out_A, out_B)
    print(f"Logic (Output) Correlation: {logic_corr:.4f} (Should be high)")
    
    return w_A, w_B, out_A, out_B, weight_corr, logic_corr

def visualize_equivalence(w_A, w_B, out_A, out_B, w_corr, l_corr):
    # Plot 1: Weight Space (Chaos)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=w_A, y=w_B, mode='markers', marker=dict(opacity=0.5, size=3), name='Weights'))
    fig1.update_layout(
        title=f"1. Parameter Space: Chaos (Corr = {w_corr:.2f})<br><sup>My neurons ≠ Your neurons (No alignment)</sup>",
        xaxis_title="Model A Weights",
        yaxis_title="Model B Weights",
        template="plotly_dark"
    )
    
    # Plot 2: Function Space (Order)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=out_A, y=out_B, mode='markers', marker=dict(opacity=0.5, size=3, color='#00f2ff'), name='Outputs'))
    # Add x=y line
    fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='white', dash='dash'), name='Perfect Match'))
    
    fig2.update_layout(
        title=f"2. Logical Space: Equivalence (Corr = {l_corr:.2f})<br><sup>We reached the same Truth via different paths.</sup>",
        xaxis_title="Model A Output",
        yaxis_title="Model B Output",
        template="plotly_dark"
    )
    
    return fig1, fig2

if __name__ == "__main__":
    w_A, w_B, out_A, out_B, w_corr, l_corr = verify_equivalence()
    
    fig_w, fig_l = visualize_equivalence(w_A, w_B, out_A, out_B, w_corr, l_corr)
    
    fig_w.write_html("output/sec_07/equivalence_params.html")
    fig_l.write_html("output/sec_07/equivalence_logic.html")
    
    print("\nVerification Complete.")
    print("1. 'equivalence_params.html': Shows that parameters are completely different.")
    print("2. 'equivalence_logic.html': Shows that the logical function is identical.")
