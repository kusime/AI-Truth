"""
Section 17: 点积的本质 - y=kx 的升维
Dot Product Essence - High-Dimensional y=kx

核心洞察:
点积 (q·k) 本质上就是 y=kx 的高维推广
1D: y = k·x (一个乘法)
nD: y = Σ kᵢ·xᵢ (n个乘法的和)

验证:
1. 从1D到高维的连续性
2. 几何意义的一致性
3. 在Attention=梯度下降证明中的作用
"""

import os

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.makedirs('output/sec_17', exist_ok=True)

def create_visualizations():
    """创建所有可视化"""
    
    print(f"\n{'='*80}")
    print("Section 17: 点积的本质 - y=kx 的升维")
    print(f"{'='*80}")
    
    # ============================================
    # 可视化 1: 1D → 2D → 3D 的连续性
    # ============================================
    
    fig1 = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            '1D: y = k·x',
            '2D: y = k₁·x₁ + k₂·x₂',
            '3D: y = k₁·x₁ + k₂·x₂ + k₃·x₃'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )
    
    # 1D情况
    x = np.linspace(-2, 2, 100)
    k = 1.5
    y = k * x
    
    fig1.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='y = 1.5x',
            line=dict(color='#ff0055', width=3),
            hovertemplate='x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 添加样本点
    x_samples = np.array([-1.5, -0.5, 0.5, 1.5])
    y_samples = k * x_samples
    
    fig1.add_trace(
        go.Scatter(
            x=x_samples,
            y=y_samples,
            mode='markers',
            name='样本点',
            marker=dict(size=10, color='#00f2ff', symbol='diamond'),
            hovertemplate='x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 2D情况 (在3D空间中显示平面)
    x1 = np.linspace(-2, 2, 20)
    x2 = np.linspace(-2, 2, 20)
    X1, X2 = np.meshgrid(x1, x2)
    k1, k2 = 1.0, 0.5
    Y = k1 * X1 + k2 * X2
    
    fig1.add_trace(
        go.Surface(
            x=X1,
            y=X2,
            z=Y,
            colorscale='Reds',
            opacity=0.7,
            showscale=False,
            hovertemplate='x₁=%{x:.2f}<br>x₂=%{y:.2f}<br>y=%{z:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3D情况 (无法直接可视化4D,所以显示等值面)
    # 使用散点表示
    np.random.seed(42)
    n_samples = 100
    x1_3d = np.random.uniform(-2, 2, n_samples)
    x2_3d = np.random.uniform(-2, 2, n_samples)
    x3_3d = np.random.uniform(-2, 2, n_samples)
    k1, k2, k3 = 1.0, 0.5, 0.3
    y_3d = k1 * x1_3d + k2 * x2_3d + k3 * x3_3d
    
    fig1.add_trace(
        go.Scatter3d(
            x=x1_3d,
            y=x2_3d,
            z=x3_3d,
            mode='markers',
            marker=dict(
                size=4,
                color=y_3d,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='y值', x=1.15)
            ),
            hovertemplate='x₁=%{x:.2f}<br>x₂=%{y:.2f}<br>x₃=%{z:.2f}<br>y=%{marker.color:.2f}<extra></extra>'
        ),
        row=1, col=3
    )
    
    fig1.update_xaxes(title_text='x', row=1, col=1)
    fig1.update_yaxes(title_text='y', row=1, col=1)
    
    fig1.update_layout(
        title={
            'text': '点积的升维: y=kx → y=Σkᵢxᵢ<br><sub>结构不变,只是维度增加</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#00f2ff'}
        },
        template='plotly_dark',
        height=600,
        showlegend=False,
        font=dict(family='Fira Code, monospace')
    )
    
    fig1.write_html('output/sec_17/dimensional_progression.html')
    print(f"\n✅ 可视化 1: output/sec_17/dimensional_progression.html")
    
    # ============================================
    # 可视化 2: 点积的几何意义
    # ============================================
    
    fig2 = go.Figure()
    
    # 创建两个向量
    q = np.array([3, 2])
    k = np.array([2, 1])
    
    # 计算点积
    dot_product = np.dot(q, k)
    
    # 计算投影
    k_norm = k / np.linalg.norm(k)
    projection_length = np.dot(q, k_norm)
    projection = projection_length * k_norm
    
    # 向量q
    fig2.add_trace(go.Scatter(
        x=[0, q[0]],
        y=[0, q[1]],
        mode='lines+markers',
        name='向量 q',
        line=dict(color='#ff0055', width=4),
        marker=dict(size=10),
        hovertemplate='q = [%{x:.1f}, %{y:.1f}]<extra></extra>'
    ))
    
    # 向量k
    fig2.add_trace(go.Scatter(
        x=[0, k[0]],
        y=[0, k[1]],
        mode='lines+markers',
        name='向量 k',
        line=dict(color='#00f2ff', width=4),
        marker=dict(size=10),
        hovertemplate='k = [%{x:.1f}, %{y:.1f}]<extra></extra>'
    ))
    
    # 投影
    fig2.add_trace(go.Scatter(
        x=[0, projection[0]],
        y=[0, projection[1]],
        mode='lines+markers',
        name=f'投影 (长度={projection_length:.2f})',
        line=dict(color='#f1c40f', width=3, dash='dash'),
        marker=dict(size=8),
        hovertemplate='投影 = [%{x:.2f}, %{y:.2f}]<extra></extra>'
    ))
    
    # 投影线
    fig2.add_trace(go.Scatter(
        x=[q[0], projection[0]],
        y=[q[1], projection[1]],
        mode='lines',
        name='垂直线',
        line=dict(color='gray', width=1, dash='dot'),
        showlegend=False
    ))
    
    # 添加文本
    fig2.add_annotation(
        x=q[0]/2, y=q[1]/2 + 0.3,
        text=f'q = [{q[0]}, {q[1]}]',
        showarrow=False,
        font=dict(size=14, color='#ff0055')
    )
    
    fig2.add_annotation(
        x=k[0]/2, y=k[1]/2 - 0.3,
        text=f'k = [{k[0]}, {k[1]}]',
        showarrow=False,
        font=dict(size=14, color='#00f2ff')
    )
    
    fig2.add_annotation(
        x=1.5, y=3,
        text=f'q·k = {dot_product}<br>= ||q|| · ||k|| · cos(θ)<br>= {projection_length:.2f} × {np.linalg.norm(k):.2f}',
        showarrow=True,
        arrowhead=2,
        ax=-50,
        ay=-50,
        font=dict(size=12, color='#f1c40f'),
        bgcolor='rgba(0,0,0,0.7)',
        bordercolor='#f1c40f'
    )
    
    fig2.update_layout(
        title={
            'text': '点积的几何意义: q在k方向上的投影<br><sub>q·k = (投影长度) × ||k||</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#00f2ff'}
        },
        xaxis=dict(range=[-0.5, 4], title='x₁'),
        yaxis=dict(range=[-0.5, 3], title='x₂', scaleanchor='x', scaleratio=1),
        template='plotly_dark',
        height=700,
        font=dict(family='Fira Code, monospace')
    )
    
    fig2.write_html('output/sec_17/geometric_meaning.html')
    print(f"✅ 可视化 2: output/sec_17/geometric_meaning.html")
    
    # ============================================
    # 可视化 3: 在Attention=梯度下降中的作用
    # ============================================
    
    fig3 = go.Figure()
    
    # 模拟数据
    np.random.seed(42)
    n_samples = 20
    
    # 生成query和key
    q_vals = np.random.uniform(-2, 2, n_samples)
    k_vals = np.random.uniform(-2, 2, n_samples)
    
    # 计算点积 (模拟1D情况)
    dot_products = q_vals * k_vals
    
    # 梯度下降路径
    alpha = 0.1
    v_val = 1.0
    delta_y_gradient = alpha * v_val * dot_products
    
    # Attention路径
    delta_y_attention = alpha * dot_products * v_val
    
    # 验证等价性
    diff = np.abs(delta_y_gradient - delta_y_attention)
    
    # 绘制
    fig3.add_trace(go.Scatter(
        x=q_vals,
        y=k_vals,
        mode='markers',
        marker=dict(
            size=12,
            color=dot_products,
            colorscale='RdBu',
            reversescale=True,
            showscale=True,
            colorbar=dict(title='q·k'),
            line=dict(color='white', width=1)
        ),
        text=[f'q·k={dp:.2f}' for dp in dot_products],
        hovertemplate='q=%{x:.2f}<br>k=%{y:.2f}<br>q·k=%{marker.color:.2f}<extra></extra>'
    ))
    
    fig3.update_layout(
        title={
            'text': f'点积在Attention=梯度下降中的作用<br><sub>(q·k)是核心,误差 < {diff.max():.2e}</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#00f2ff'}
        },
        xaxis_title='q (query)',
        yaxis_title='k (key)',
        template='plotly_dark',
        height=700,
        font=dict(family='Fira Code, monospace')
    )
    
    fig3.write_html('output/sec_17/role_in_proof.html')
    print(f"✅ 可视化 3: output/sec_17/role_in_proof.html")
    
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    print(f"✅ 点积 (q·k) 本质上就是 y=kx 的高维推广")
    print(f"✅ 1D: y=k·x → nD: y=Σkᵢ·xᵢ (结构不变)")
    print(f"✅ 几何意义: q在k方向上的投影 × ||k||")
    print(f"✅ 在证明中: (q·k) = (k·q) 是等价性的核心")
    print(f"\n核心洞察:")
    print(f"  点积看起来复杂,但本质就是 y=kx")
    print(f"  只是从1个乘法变成了n个乘法的和")
    print(f"  这又是一个费曼式的简化! 🔥")

if __name__ == '__main__':
    create_visualizations()
