"""
Section 31: 全息宇宙验证 - 权重是高维的投影吗?
Section 31: Holographic Universe - Are Weights Projections of Higher Dimensions?

用户假设: "我们拿到的巨大的权重本质上就是在尝试用计算机去模拟出高维空间的那一个点... 运行AI其实是在运行N维宇宙。"
User Hypothesis: "Weights are projections of a high-dim point... Running AI is running an N-dim universe."

实验原理: 全息干涉 (Holographic Interference)
如果一个低维物体(Student Net)试图模仿一个高维物体(Teacher 10000D), 它必须利用其"权重"的微小细节(FP64的低位)来编码高维的折叠信息。
反之, 如果它模仿的是低维物体(Teacher 10D), 它的权重就是平滑的。

预测:
- 模仿高维的Student, 即使在FP64下拟合完美, 一旦被"降维打击"(转FP32/FP16), 其表现会瞬间崩塌(Collapse)。
- 模仿低维的Student, 对精度截断不敏感。
- 这种"对精度的敏感性差值", 就是高维空间存在的证明。

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

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class GodNetwork(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=10000, output_dim=1):
        super().__init__()
        # 这是一个随机生成的高维"真理"函数
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), # Tanh 产生更多的高频非线性
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        # 固定权重,不训练,它就是"物理公理"
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        return self.net(x)

class StudentNetwork(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=256, output_dim=1):
        super().__init__()
        # 一个普通的网络, 试图理解God
        # 容量适中, 强迫它使用"精度"而非"容量"来拟合
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def run_simulation():
    print(f"{'='*60}")
    print("Section 31: 全息投影实验 (Holographic Verification)")
    print(f"{'='*60}")
    
    # 1. 创建两个宇宙公理
    input_dim = 10
    n_samples = 5000
    
    # 宇宙 A: 高维 (1万维投影)
    print("生成高维宇宙 (10,000D Truth)...")
    god_high = GodNetwork(input_dim, hidden_dim=10000).to(device, dtype=torch.float64)
    
    # 宇宙 B: 低维 (10维投影, 简单线性/低秩)
    print("生成低维宇宙 (10D Truth)...")
    god_low = GodNetwork(input_dim, hidden_dim=10).to(device, dtype=torch.float64)
    
    # 生成观察数据 (X)
    X = torch.randn(n_samples, input_dim, dtype=torch.float64).to(device)
    
    # 生成真理 (Y)
    Y_high = god_high(X)
    Y_low = god_low(X)
    
    # 归一化 Y 以便比较 Loss
    Y_high = (Y_high - Y_high.mean()) / Y_high.std()
    Y_low = (Y_low - Y_low.mean()) / Y_low.std()
    
    # 2. 训练 Student 去模仿 (Fitting)
    # 我们用同样的 Student 结构去学习不同的宇宙
    
    def train_student(target_y, label):
        print(f"\n>>> Training Student to mimic {label}...")
        student = StudentNetwork(input_dim, hidden_dim=512).to(device, dtype=torch.float64)
        optimizer = optim.Adam(student.parameters(), lr=1e-3, eps=1e-16) # 极致精度优化
        criterion = nn.MSELoss()
        
        start_t = time.time()
        for epoch in range(1001):
            optimizer.zero_grad()
            pred = student(X)
            loss = criterion(pred, target_y)
            loss.backward()
            optimizer.step()
            
            if epoch % 200 == 0:
                print(f"  Ep {epoch}: Loss = {loss.item():.2e}")
        
        print(f"  Training Done. Final FP64 Loss: {loss.item():.2e}")
        return student

    student_high = train_student(Y_high, "High-Dim Universe")
    student_low = train_student(Y_low, "Low-Dim Universe")
    
    # 3. 降维打击 (Shattering the Hologram)
    # 强制将权重截断为 FP32 和 FP16, 观察性能衰减
    
    def evaluate_quantization(student, target_y, label):
        print(f"\n[{label}] 转眼成空测试 (Quantization Sensitivity):")
        results = {}
        
        # FP64 Baseline
        pred_64 = student(X)
        loss_64 = nn.MSELoss()(pred_64, target_y).item()
        results['FP64'] = loss_64
        print(f"  FP64 (Original): {loss_64:.2e}")
        
        # FP32 Truncation (模拟转储权重)
        student_32 = StudentNetwork(input_dim, hidden_dim=512).to(device)
        student_32.load_state_dict(student.state_dict()) # Load FP64 weights
        student_32.to(dtype=torch.float32) # Cast to FP32
        
        X_32 = X.to(dtype=torch.float32)
        Y_32 = target_y.to(dtype=torch.float32)
        
        with torch.no_grad():
            pred_32 = student_32(X_32)
            loss_32 = nn.MSELoss()(pred_32, Y_32).item()
        
        results['FP32'] = loss_32
        ratio_32 = loss_32 / loss_64 if loss_64 > 1e-10 else 1.0
        print(f"  FP32 (Truncated): {loss_32:.2e} (Loss变大 {ratio_32:.1f}倍)")
        
        # FP16 Truncation
        student_16 = StudentNetwork(input_dim, hidden_dim=512).to(device)
        student_16.load_state_dict(student.state_dict())
        student_16.to(dtype=torch.float16)
        
        X_16 = X.to(dtype=torch.float16)
        Y_16 = target_y.to(dtype=torch.float16)
        
        with torch.no_grad():
            pred_16 = student_16(X_16)
            loss_16 = nn.MSELoss()(pred_16, Y_16).item()
            
        results['FP16'] = loss_16
        ratio_16 = loss_16 / loss_64 if loss_64 > 1e-10 else 1.0
        print(f"  FP16 (Truncated): {loss_16:.2e} (Loss变大 {ratio_16:.1f}倍)")
        
        return results

    res_high = evaluate_quantization(student_high, Y_high, "高维投影 (High-Dim)")
    res_low = evaluate_quantization(student_low, Y_low, "低维投影 (Low-Dim)")
    
    return res_high, res_low

def create_viz(res_high, res_low):
    fig = go.Figure()
    
    x_labels = ['FP64 (原生)', 'FP32 (截断)', 'FP16 (破碎)']
    
    # 这种图展示"崩塌率"
    # 我们标准化一下, 以FP64 Loss为基准1.0
    
    def normalize_loss(res):
        base = res['FP64']
        return [res['FP64']/base, res['FP32']/base, res['FP16']/base]
    
    y_high = normalize_loss(res_high)
    y_low = normalize_loss(res_low)
    
    fig.add_trace(go.Scatter(
        x=x_labels, y=y_high,
        mode='lines+markers',
        name='模仿高维宇宙 (10000D)',
        line=dict(color='#ff0055', width=4),
        marker=dict(size=12, symbol='star')
    ))
    
    fig.add_trace(go.Scatter(
        x=x_labels, y=y_low,
        mode='lines+markers',
        name='模仿低维宇宙 (10D)',
        line=dict(color='#00bdff', width=4),
        marker=dict(size=12, symbol='circle')
    ))
    
    fig.update_layout(
        title='全息易碎性验证: 越高维, 越需要精度 (Holographic Fragility)',
        template='plotly_dark',
        yaxis_title='Loss 暴涨倍数 (相对于FP64)',
        yaxis=dict(type='log'),
        annotations=[
            dict(x=2, y=np.log10(y_high[2]), text=f"高维崩塌 {y_high[2]:.0f}倍", showarrow=True, arrowhead=1, ax=-60, ay=-40, font=dict(color='#ff0055')),
            dict(x=2, y=np.log10(y_low[2]), text=f"低维稳定 {y_low[2]:.0f}倍", showarrow=True, arrowhead=1, ax=60, ay=20, font=dict(color='#00bdff'))
        ]
    )
    
    fig.write_html('output/sec_31/hologram.html')
    print("\n可视化已生成: output/sec_31/hologram.html")

if __name__ == '__main__':
    h, l = run_simulation()
    create_viz(h, l)
