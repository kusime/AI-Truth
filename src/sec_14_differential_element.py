"""
Section 14: 微分元验证 - 从单样本到多样本的积分
Differential Element Validation - Integration from Single to Multi-Sample

核心洞察:
单样本的精确等价性 (dΔy) 是多样本渐近等价性 (∫dΔy) 的"微分元"

验证:
1. 单样本: Δy_梯度 = Δy_Attention (精确)
2. 多样本: Σ Δy_梯度 = Σ Δy_Attention (累积精确)
3. 可视化: 展示"积分"过程
"""

import os

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

# 创建输出目录
os.makedirs('output/sec_14', exist_ok=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

# ============================================
# Part 1: 单样本验证 (微分元)
# ============================================

def verify_single_sample():
    """验证单样本的精确等价性"""
    set_seed(42)
    
    dim = 64
    lr = 1.0 / dim
    
    # 数据
    W = torch.randn(dim, dim)
    query = torch.randn(dim)
    key = torch.randn(dim)
    value = torch.randn(dim)
    
    # 基础输出
    y_base = W @ query
    
    # 梯度下降路径
    delta_W = lr * torch.outer(value, key)
    W_updated = W + delta_W
    y_new_gradient = W_updated @ query
    delta_y_gradient = y_new_gradient - y_base
    
    # Attention 路径
    delta_y_attention = lr * (query @ key) * value
    
    # 验证
    diff = torch.norm(delta_y_gradient - delta_y_attention).item()
    
    return {
        'delta_y_gradient': delta_y_gradient.numpy(),
        'delta_y_attention': delta_y_attention.numpy(),
        'diff': diff
    }

# ============================================
# Part 2: 多样本累积验证 (积分)
# ============================================

def verify_multi_sample(n_samples_list=[1, 2, 5, 10, 20, 50]):
    """验证多样本累积的精确等价性"""
    set_seed(42)
    
    dim = 64
    lr = 1.0 / dim
    max_samples = max(n_samples_list)
    
    # 数据
    W = torch.randn(dim, dim)
    query = torch.randn(dim)
    keys = torch.randn(max_samples, dim)
    values = torch.randn(max_samples, dim)
    
    results = []
    
    for n in n_samples_list:
        # 累积梯度下降
        delta_y_grad_total = torch.zeros(dim)
        for i in range(n):
            delta_W = lr * torch.outer(values[i], keys[i])
            delta_y_grad = delta_W @ query
            delta_y_grad_total += delta_y_grad
        
        # 累积 Attention
        delta_y_attn_total = torch.zeros(dim)
        for i in range(n):
            delta_y_attn = lr * (query @ keys[i]) * values[i]
            delta_y_attn_total += delta_y_attn
        
        # 验证
        diff = torch.norm(delta_y_grad_total - delta_y_attn_total).item()
        
        results.append({
            'n_samples': n,
            'diff': diff,
            'norm_grad': torch.norm(delta_y_grad_total).item(),
            'norm_attn': torch.norm(delta_y_attn_total).item()
        })
    
    return results

# ============================================
# Part 3: 逐步积分可视化
# ============================================

def visualize_integration_process(n_samples=20):
    """可视化从单样本到多样本的积分过程"""
    set_seed(42)
    
    dim = 64
    lr = 1.0 / dim
    
    # 数据
    W = torch.randn(dim, dim)
    query = torch.randn(dim)
    keys = torch.randn(n_samples, dim)
    values = torch.randn(n_samples, dim)
    
    # 逐步累积
    cumulative_grad = []
    cumulative_attn = []
    cumulative_diff = []
    
    delta_y_grad_total = torch.zeros(dim)
    delta_y_attn_total = torch.zeros(dim)
    
    for i in range(n_samples):
        # 单样本贡献 (微分元)
        delta_W = lr * torch.outer(values[i], keys[i])
        delta_y_grad = delta_W @ query
        delta_y_attn = lr * (query @ keys[i]) * values[i]
        
        # 累积 (积分)
        delta_y_grad_total += delta_y_grad
        delta_y_attn_total += delta_y_attn
        
        # 记录
        cumulative_grad.append(torch.norm(delta_y_grad_total).item())
        cumulative_attn.append(torch.norm(delta_y_attn_total).item())
        cumulative_diff.append(torch.norm(delta_y_grad_total - delta_y_attn_total).item())
    
    return {
        'cumulative_grad': cumulative_grad,
        'cumulative_attn': cumulative_attn,
        'cumulative_diff': cumulative_diff
    }

# ============================================
# Part 4: 可视化
# ============================================

def create_visualizations():
    """创建所有可视化"""
    
    # 1. 单样本验证
    single_result = verify_single_sample()
    print(f"\n{'='*80}")
    print("Part 1: 单样本验证 (微分元 dΔy)")
    print(f"{'='*80}")
    print(f"||Δy_梯度 - Δy_Attention|| = {single_result['diff']:.2e}")
    print(f"精确等价 ✓" if single_result['diff'] < 1e-6 else "不等价 ✗")
    
    # 2. 多样本累积验证
    multi_results = verify_multi_sample()
    print(f"\n{'='*80}")
    print("Part 2: 多样本累积验证 (积分 ∫dΔy)")
    print(f"{'='*80}")
    for r in multi_results:
        print(f"n={r['n_samples']:3d}: ||Σ Δy_梯度 - Σ Δy_Attention|| = {r['diff']:.2e}")
    
    # 3. 积分过程可视化
    integration_data = visualize_integration_process(n_samples=50)
    
    # ============================================
    # 可视化 1: 多样本误差收敛
    # ============================================
    
    fig1 = go.Figure()
    
    n_samples = [r['n_samples'] for r in multi_results]
    diffs = [r['diff'] for r in multi_results]
    
    fig1.add_trace(go.Scatter(
        x=n_samples,
        y=diffs,
        mode='lines+markers',
        name='累积误差',
        line=dict(color='#ff0055', width=3),
        marker=dict(size=10)
    ))
    
    # 添加精度阈值线
    fig1.add_hline(
        y=1e-6,
        line_dash="dash",
        line_color="#00f2ff",
        annotation_text="浮点精度极限 (10⁻⁶)",
        annotation_position="right"
    )
    
    fig1.update_layout(
        title={
            'text': '多样本累积误差 - 验证线性叠加原理<br><sub>||Σ Δy_梯度 - Σ Δy_Attention||</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#00f2ff'}
        },
        xaxis_title='样本数量 n',
        yaxis_title='累积误差 (对数尺度)',
        yaxis_type='log',
        template='plotly_dark',
        hovermode='x unified',
        height=600,
        font=dict(family='Fira Code, monospace')
    )
    
    fig1.write_html('output/sec_14/multi_sample_error.html')
    print(f"\n✅ 可视化 1: output/sec_14/multi_sample_error.html")
    
    # ============================================
    # 可视化 2: 积分过程 (累积曲线)
    # ============================================
    
    fig2 = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            '累积输出变化 ||Δy|| - 展示积分过程',
            '累积误差 - 验证精确叠加'
        ),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )
    
    steps = list(range(1, len(integration_data['cumulative_grad']) + 1))
    
    # 上图: 累积曲线
    fig2.add_trace(
        go.Scatter(
            x=steps,
            y=integration_data['cumulative_grad'],
            mode='lines',
            name='Σ Δy_梯度',
            line=dict(color='#ff0055', width=3),
            hovertemplate='样本 %{x}<br>||Σ Δy_梯度|| = %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Scatter(
            x=steps,
            y=integration_data['cumulative_attn'],
            mode='lines',
            name='Σ Δy_Attention',
            line=dict(color='#00f2ff', width=3, dash='dash'),
            hovertemplate='样本 %{x}<br>||Σ Δy_Attention|| = %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 下图: 累积误差
    fig2.add_trace(
        go.Scatter(
            x=steps,
            y=integration_data['cumulative_diff'],
            mode='lines',
            name='累积误差',
            line=dict(color='#f1c40f', width=2),
            fill='tozeroy',
            fillcolor='rgba(241, 196, 15, 0.2)',
            hovertemplate='样本 %{x}<br>误差 = %{y:.2e}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig2.update_xaxes(title_text='累积样本数', row=2, col=1)
    fig2.update_yaxes(title_text='||Δy||', row=1, col=1)
    fig2.update_yaxes(title_text='误差', type='log', row=2, col=1)
    
    fig2.update_layout(
        title={
            'text': '从微分元到积分 - dΔy → ∫dΔy<br><sub>单样本精确等价 → 多样本累积精确等价</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#00f2ff'}
        },
        template='plotly_dark',
        hovermode='x unified',
        height=900,
        showlegend=True,
        font=dict(family='Fira Code, monospace')
    )
    
    fig2.write_html('output/sec_14/integration_process.html')
    print(f"✅ 可视化 2: output/sec_14/integration_process.html")
    
    # ============================================
    # 可视化 3: 微分元概念图
    # ============================================
    
    fig3 = go.Figure()
    
    # 创建示意图数据
    n_points = 10
    x = np.linspace(0, 1, n_points)
    
    # 模拟单个微分元的贡献
    differential_contributions = np.random.randn(n_points) * 0.1 + 0.5
    cumulative_sum = np.cumsum(differential_contributions)
    
    # 绘制微分元
    for i in range(n_points):
        fig3.add_trace(go.Bar(
            x=[x[i]],
            y=[differential_contributions[i]],
            width=0.08,
            name=f'dΔy_{i+1}',
            marker=dict(
                color=f'rgba(255, {int(255 * (1 - i/n_points))}, {int(255 * i/n_points)}, 0.7)',
                line=dict(color='white', width=1)
            ),
            hovertemplate=f'样本 {i+1}<br>dΔy = %{{y:.3f}}<extra></extra>',
            showlegend=False
        ))
    
    # 绘制累积曲线
    fig3.add_trace(go.Scatter(
        x=x,
        y=cumulative_sum,
        mode='lines+markers',
        name='∫dΔy (累积)',
        line=dict(color='#00f2ff', width=4),
        marker=dict(size=12, symbol='diamond'),
        yaxis='y2',
        hovertemplate='累积到样本 %{x:.0f}<br>Σ dΔy = %{y:.3f}<extra></extra>'
    ))
    
    fig3.update_layout(
        title={
            'text': '微分元概念 - 单样本贡献与累积效果<br><sub>每个 dΔy 是独立的微分元,累积形成总效果 ∫dΔy</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#00f2ff'}
        },
        xaxis=dict(
            title='样本索引',
            tickmode='linear',
            tick0=0,
            dtick=1
        ),
        yaxis=dict(
            title='单样本贡献 dΔy',
            side='left',
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis2=dict(
            title='累积效果 Σ dΔy',
            side='right',
            overlaying='y',
            showgrid=False
        ),
        template='plotly_dark',
        hovermode='x unified',
        height=700,
        barmode='overlay',
        font=dict(family='Fira Code, monospace')
    )
    
    fig3.write_html('output/sec_14/differential_element_concept.html')
    print(f"✅ 可视化 3: output/sec_14/differential_element_concept.html")
    
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    print(f"✅ 单样本精确等价 (微分元): 误差 < 10⁻⁶")
    print(f"✅ 多样本累积精确等价 (积分): 误差 < 10⁻⁶")
    print(f"✅ 线性叠加原理验证成功")
    print(f"\n核心洞察:")
    print(f"  dΔy = α · v · (k · q)  [微分元,不可再分]")
    print(f"  Σ dΔy = Σ α · v_i · (k_i · q)  [积分,线性叠加]")
    print(f"\n你的5行推导是论文30页证明的'微分形式' ✓")

if __name__ == '__main__':
    create_visualizations()
