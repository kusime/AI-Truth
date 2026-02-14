import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F

# --- User's Intuition ---
# "在高维空间里面...我的直觉描述和真实的论文在高维语义空间里发生了 QK 重叠"
# "我的 Query 撞击了真理的 Key... 产生与共鸣"
# Section 6: Resonance & QK Overlap

def verify_resonance(n_keys=1000, max_dim=512):
    dims = np.arange(16, max_dim + 1, 32)
    
    avg_probs = [] # Probability assigned to the correct key
    entropies = [] # Sharpness of the distribution (Lower = Sharper)
    
    print(f"Running Resonance Simulation: {n_keys} Keys, Dim 16 -> {max_dim}...")
    
    for d in dims:
        # 1. Generate Knowledge Base (Keys)
        # Random high-dim vectors (Normalized)
        K = torch.randn(n_keys, d)
        K = F.normalize(K, p=2, dim=1)
        
        # 2. Generate a Query
        # Fix: Normalize noise so it has constant angular deviation (e.g. 0.5 radian), regardless of dim.
        true_key = K[0]
        noise = torch.randn(d)
        noise = F.normalize(noise, dim=0) * 0.5 # Constant magnitude noise (Angle is constant)
        Q = true_key + noise
        Q = F.normalize(Q, p=2, dim=0)
        
        # 3. Calculate Attention Scores (Discovery: "Blessing of Dimensionality" Scaling)
        # We multiply by sqrt(d). Why?
        # <Q, True> is approx constant (e.g. 0.8).
        # <Q, False> is noise with std 1/sqrt(d).
        # The Signal-to-Noise Ratio (Z-score) is 0.8 / (1/sqrt(d)) = 0.8 * sqrt(d).
        # As D increases, the "Gap" between Truth and Falsehood becomes infinite in terms of standard deviations.
        # This causes Softmax to lock on to index 0 with probability 1.0.
        scores = torch.matmul(K, Q) * np.sqrt(d)
        
        # 4. Apply Rescaling Temperature (Beta)
        # In modern transformers, we might use temperature.
        # But even with standard 1/sqrt(d), high dim makes dot products distribute differently.
        # Let's observe the raw Softmax.
        attn_weights = F.softmax(scores, dim=0)
        
        # 5. Measure "Resonance"
        # How much probability mass is on the TRUE Key[0]?
        prob_correct = attn_weights[0].item()
        avg_probs.append(prob_correct)
        
        # Entropy: -Sum(p * log(p))
        entropy = -(attn_weights * torch.log(attn_weights + 1e-9)).sum().item()
        entropies.append(entropy)
        
        if d % 50 <= 32:
            print(f"Dim {d}: Prob(Correct) = {prob_correct:.4f}, Entropy = {entropy:.4f}")
            
    return dims, avg_probs, entropies

def visualize_resonance(dims, probs, entropies):
    # Plot 1: Resonance Probability
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=dims, y=probs, mode='lines+markers', name='P(Correct Key)', line=dict(color='#f1c40f', width=3)))
    
    fig1.update_layout(
        title="1. Resonance: Probability of Matching the 'Truth' Key",
        xaxis_title="Dimension",
        yaxis_title="Probability (Softmax Score)",
        template="plotly_dark",
        yaxis=dict(range=[0, 1.1])
    )
    
    # Plot 2: Entropy (Sharpening)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=dims, y=entropies, mode='lines+markers', name='Attention Entropy', line=dict(color='#00f2ff', width=3)))
    
    fig2.update_layout(
        title="2. Focus: Attention Distribution Entropy (Lower = Sharper)",
        xaxis_title="Dimension",
        yaxis_title="Entropy (Uncertainty)",
        template="plotly_dark"
    )
    
    return fig1, fig2

if __name__ == "__main__":
    dims, probs, entropies = verify_resonance(max_dim=512)
    
    fig_res, fig_ent = visualize_resonance(dims, probs, entropies)
    
    fig_res.write_html("output/sec_06/resonance_probability.html")
    fig_ent.write_html("output/sec_06/resonance_entropy.html")
    
    print("\nVerification Complete.")
    print("1. 'resonance_probability.html': Shows how high dimensions allow the Query to 'lock on' to the Truth.")
    print("2. 'resonance_entropy.html': Shows how uncertainty collapses (Resonance) as dimension increases.")
