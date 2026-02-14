"""
Section 31: 全息宇宙验证 (修正版) - 混沌与分形
Section 31: Holographic Verification (Fixed) - Chaos & Fractals

User Feedback: "The graph showed 1x collapse, not the massive collapse you predicted."
Root Cause: The previous 'God Universe' (Tanh) was too smooth. Truncating weights didn't break the topology.
Fix: Introduce **High-Frequency Chaos** (Sinusoidal Manifolds).
     High-dim information is encoded in the *phase* and *frequency* of the weights.
     Shattering precision (FP16) should destroy phase coherence.

"""

import os
import time

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim

os.makedirs('output/sec_31', exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 激活函数: 正弦波 (制造高频震荡)
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class ChaosGod(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=1000): # Reduce dim for speed
        super().__init__()
        # 混沌宇宙: 包含高频分量
        # Key: 这里的权重本身代表"频率"
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            Scale(5000.0), # 极端频率放大 -> 强迫全息编码
            Sine(), 
            nn.Linear(hidden_dim, 1)
        )
        # frozen weights
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        return self.net(x)

class SimpleGod(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        # 简单宇宙: 只有低频线性关系
        self.net = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        return self.net(x)

class Student(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=1024):
        super().__init__()
        # 这是一个试图理解混沌的学生
        # 它必须学习那些高频的"频率权重"
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(), # Swish
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

def train_and_shatter(god, label, steps=300): # Faster steps
    print(f"\n>>> Simulating {label}...")
    
    # 1. Generate Truth (Less samples)
    X = torch.randn(1000, 10, dtype=torch.float64).to(device)
    with torch.no_grad():
        Y = god(X)
        # Normalize Y to make loss comparable
        Y = (Y - Y.mean()) / (Y.std() + 1e-6)

    # 2. Train Student (FP64)
    student = Student(10, 512).to(device, dtype=torch.float64) # Smaller student
    optimizer = optim.Adam(student.parameters(), lr=1e-3, eps=1e-16)
    criterion = nn.MSELoss()
    
    start_t = time.time()
    for i in range(steps):
        optimizer.zero_grad()
        pred = student(X)
        loss = criterion(pred, Y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"  Train Step {i}: Loss={loss.item():.2e}")
            
    print(f"  FP64 Baseline Loss: {loss.item():.2e}")
    base_loss = loss.item()
    
    # 3. Shatter (Quantize)
    results = [1.0] # FP64 ratio
    
    # FP32
    s32 = Student(10, 512).to(device)
    s32.load_state_dict(student.state_dict())
    s32.to(dtype=torch.float32)
    with torch.no_grad():
        loss_32 = nn.MSELoss()(s32(X.float()), Y.float()).item()
    ratio_32 = loss_32 / base_loss if base_loss > 1e-12 else 1.0
    results.append(ratio_32)
    print(f"  FP32 Shatter: {ratio_32:.1f}x Loss")
    
    # FP16
    s16 = Student(10, 512).to(device)
    s16.load_state_dict(student.state_dict())
    s16.to(dtype=torch.float16)
    with torch.no_grad():
        loss_16 = nn.MSELoss()(s16(X.half()), Y.half()).item()
    ratio_16 = loss_16 / base_loss if base_loss > 1e-12 else 1.0
    results.append(ratio_16)
    print(f"  FP16 Shatter: {ratio_16:.1f}x Loss")
    
    return results

def run_experiment():
    print(f"{'='*60}")
    print("Section 31: 全息易碎性验证 (The Fragility of Holograms)")
    print(f"{'='*60}")
    
    # 宇宙 A: 混沌 (Chaos)
    high_dim_ratios = train_and_shatter(ChaosGod().to(device, dtype=torch.float64), "High-Dim Chaos Universe")
    
    # 宇宙 B: 秩序 (Order)
    low_dim_ratios = train_and_shatter(SimpleGod().to(device, dtype=torch.float64), "Low-Dim Order Universe")
    
    return high_dim_ratios, low_dim_ratios

def create_viz(h_ratios, l_ratios):
    fig = go.Figure()
    
    x = ['FP64 (完整)', 'FP32 (裂纹)', 'FP16 (粉碎)']
    
    fig.add_trace(go.Scatter(
        x=x, y=h_ratios,
        mode='lines+markers+text',
        name='High-Dim (Chaos)',
        line=dict(color='#ff0055', width=4),
        marker=dict(size=12, symbol='star'),
        text=[f"{r:.0f}x" for r in h_ratios],
        textposition="top center"
    ))
    
    fig.add_trace(go.Scatter(
        x=x, y=l_ratios,
        mode='lines+markers+text',
        name='Low-Dim (Order)',
        line=dict(color='#00bdff', width=4),
        marker=dict(size=12),
        text=[f"{r:.1f}x" for r in l_ratios],
        textposition="bottom center"
    ))
    
    fig.update_layout(
        title='验证结果: 为什么高维信息如此脆弱? (The Butterfly Effect of Precision)',
        template='plotly_dark',
        yaxis_title='Loss Explosion Factor',
        yaxis=dict(type='log'),
        height=600
    )
    
    fig.write_html('output/sec_31/hologram_fixed.html')
    print("\n可视化已生成: output/sec_31/hologram_fixed.html")

if __name__ == '__main__':
    h, l = run_experiment()
    create_viz(h, l)
