"""
Section 30: 最终验证 - 精度与容量的三阶段壁垒
Section 30: Final Verdict - The Three-Stage Limits (Precision, Capacity, Hardware)

本代码汇总了之前的实验发现 (Section 30, 31, Stress Test)，只保留最终结论性验证。
It consolidates previous findings into a single, definitive experiment.

实验目标 / Goal:
清晰展示 AI 在逼近真理时的三道墙 (Three Walls):
1. 精度墙 (The Precision Wall): Float16 的物理底噪 (Loss ~ 1e-4).
2. 容量墙 (The Capacity Wall): Float32/64 受限于模型拟合能力 (Loss ~ 1e-5).
3. 硬件墙 (The Hardware Wall): Float64 在消费级显卡上的算力惩罚 (Time Explosion).

"""

import gc
import os
import time

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import train_test_split

# 配置
OUTPUT_DIR = 'output/sec_30'
os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_data(n_samples=10000):
    # 足够的数据量以排除过拟合干扰
    X, t = make_swiss_roll(n_samples=n_samples, noise=0.0)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X.astype(np.float64), t

class UniversalEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=2000):
        super().__init__()
        # 标准的大容量架构
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def run_trial(name, dtype, hidden_dim, epochs=200, eps=1e-8):
    print(f"  [{name}] Running on {device} (H={hidden_dim})...", end="")
    torch.cuda.empty_cache()
    
    # Data Prep
    X, _ = generate_data(n_samples=10000)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    X_train_t = torch.tensor(X_train, dtype=dtype).to(device)
    X_test_t = torch.tensor(X_test, dtype=dtype).to(device)
    
    model = UniversalEncoder(hidden_dim=hidden_dim).to(dtype=dtype).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=eps)
    criterion = nn.MSELoss()
    
    start_time = time.time()
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        _, output = model(X_train_t)
        loss = criterion(output, X_train_t)
        loss.backward()
        optimizer.step()
    
    # Eval
    model.eval()
    with torch.no_grad():
        _, recovered = model(X_test_t)
        final_loss = criterion(recovered, X_test_t).item()
        
    duration = time.time() - start_time
    print(f" -> Loss: {final_loss:.2e} | Time: {duration:.2f}s")
    
    return final_loss, duration

def final_experiment():
    print(f"{'='*60}")
    print("Section 30: 最终验证 (The Final Verdict)")
    print(f"{'='*60}")
    
    # 我们选取两个代表性的容量点
    configs = [
        {"dim": 2000, "desc": "Standard Capacity"},
        {"dim": 10000, "desc": "High Capacity"} 
    ]
    
    metrics = {
        'Float16': {'loss': [], 'time': []},
        'Float32': {'loss': [], 'time': []},
        'Float64': {'loss': [], 'time': []}
    }
    
    dims = []
    
    for cfg in configs:
        h_dim = cfg['dim']
        dims.append(h_dim)
        print(f"\n>>> Scene: {cfg['desc']} (Neurons={h_dim})")
        
        # 1. FP16 (The Precision Wall)
        # 注意: FP16 经常溢出, 我们容忍它
        try:
            l, t = run_trial("Float16", torch.float16, h_dim, eps=1e-5)
            metrics['Float16']['loss'].append(l)
            metrics['Float16']['time'].append(t)
        except Exception as e:
            print(f" -> Failed: {e}")
            metrics['Float16']['loss'].append(None)
            
        # 2. FP32 (The Silicon Chosen One)
        l, t = run_trial("Float32", torch.float32, h_dim, eps=1e-8)
        metrics['Float32']['loss'].append(l)
        metrics['Float32']['time'].append(t)
        
        # 3. FP64 (The Theoretical King)
        l, t = run_trial("Float64", torch.float64, h_dim, eps=1e-16)
        metrics['Float64']['loss'].append(l)
        metrics['Float64']['time'].append(t)

    return dims, metrics

def create_final_viz(dims, metrics):
    fig = go.Figure()
    
    x_vals = [str(d) for d in dims]
    
    # Bar Chart for Loss (Log Scale)
    fig.add_trace(go.Bar(
        x=x_vals, y=metrics['Float16']['loss'],
        name='Float16 (精度受限)',
        marker_color='#ff0055', # Red
        text=[f"{l:.2e}" if l else "NaN" for l in metrics['Float16']['loss']],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        x=x_vals, y=metrics['Float32']['loss'],
        name='Float32 (容量受限)',
        marker_color='#00bdff', # Cyan
        text=[f"{l:.2e}" for l in metrics['Float32']['loss']],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        x=x_vals, y=metrics['Float64']['loss'],
        name='Float64 (算力受限)',
        marker_color='#00ff88', # Green
        text=[f"{l:.2e}" for l in metrics['Float64']['loss']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title='Section 30: 精度与容量的最终判决 (Loss Comparison)',
        template='plotly_dark',
        xaxis_title='模型容量 (Hidden Dimension)',
        yaxis_title='重构误差 (Log Scale)',
        yaxis=dict(type='log'),
        barmode='group'
    )
    
    fig.write_html(f'{OUTPUT_DIR}/final_verdict.html')
    print(f"\n可视化已生成: {OUTPUT_DIR}/final_verdict.html")

if __name__ == '__main__':
    d, m = final_experiment()
    create_final_viz(d, m)
