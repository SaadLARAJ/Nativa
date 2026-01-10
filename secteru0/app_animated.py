"""
Secteur 0 : Distributed Constellation Control
==============================================
Version with smooth Plotly animation (no page reload)

Run with: streamlit run app_animated.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List
import copy

# ==============================================================================
# Satellite Agent Class
# ==============================================================================

@dataclass
class SatelliteAgent:
    """Satellite with Active Inference + Kuramoto dynamics."""
    id: int
    radius: float
    theta: float
    phi: float = 0.0
    radial_velocity: float = 0.0
    angular_velocity: float = 0.0
    target_radius: float = 1.0
    target_angular_spacing: float = 0.0
    omega: float = 0.1
    precision_radius: float = 2.0
    total_delta_v: float = 0.0
    
    def compute_free_energy(self) -> float:
        error = self.radius - self.target_radius
        return 0.5 * self.precision_radius * error ** 2
    
    def update_kuramoto(self, neighbor_data: List[tuple], K: float, dt: float):
        """Directional Kuramoto with Phase Bias."""
        if not neighbor_data:
            self.theta += self.omega * dt
            return
        
        coupling = 0.0
        for neighbor_theta, direction in neighbor_data:
            expected_diff = direction * self.target_angular_spacing
            actual_diff = neighbor_theta - self.theta
            phase_error = actual_diff - expected_diff
            coupling += np.sin(phase_error)
        
        coupling /= len(neighbor_data)
        self.angular_velocity = self.omega + K * coupling
        self.theta += self.angular_velocity * dt
        self.theta = self.theta % (2 * np.pi)
    
    def update_radius(self, dt: float, damping: float = 0.8):
        action = -self.precision_radius * (self.radius - self.target_radius)
        self.total_delta_v += abs(action * dt)
        self.radial_velocity += action * dt
        self.radial_velocity *= damping
        self.radius += self.radial_velocity * dt
        self.radius = max(0.5, self.radius)
    
    def get_3d_position(self) -> tuple:
        x = self.radius * np.cos(self.theta) * np.cos(self.phi)
        y = self.radius * np.sin(self.theta) * np.cos(self.phi)
        z = self.radius * np.sin(self.phi)
        return x, y, z


# ==============================================================================
# Constellation Class
# ==============================================================================

class SatelliteConstellation:
    """LEO constellation with ring topology."""
    
    def __init__(self, n_satellites: int, target_radius: float, noise: float = 0.0):
        self.n_satellites = n_satellites
        self.target_radius = target_radius
        self.noise = noise
        self.target_spacing = 2 * np.pi / n_satellites
        
        self.satellites: List[SatelliteAgent] = []
        for i in range(n_satellites):
            # Create chaotic initial state for visible energy release
            sat = SatelliteAgent(
                id=i,
                radius=target_radius * np.random.uniform(0.4, 1.8),  # Much more dispersed!
                theta=np.random.uniform(0, 2 * np.pi),
                phi=np.random.uniform(-0.1, 0.1),
                omega=0.05 + np.random.uniform(-0.02, 0.02),
                target_radius=target_radius
            )
            sat.target_angular_spacing = self.target_spacing
            self.satellites.append(sat)
        
        self.time = 0.0
    
    def compute_order_parameter(self) -> float:
        if self.n_satellites < 2:
            return 1.0
        thetas = sorted([s.theta for s in self.satellites])
        diffs = []
        for i in range(len(thetas)):
            diff = thetas[(i+1) % len(thetas)] - thetas[i]
            if diff < 0:
                diff += 2 * np.pi
            diffs.append(diff)
        std = np.std(diffs)
        return 1 - min(std / np.pi, 1.0)
    
    def step(self, K: float, dt: float = 0.1):
        if self.noise > 0:
            for sat in self.satellites:
                sat.theta += np.random.normal(0, self.noise * dt)
                sat.radius += np.random.normal(0, self.noise * 0.1 * dt)
        
        for sat in self.satellites:
            left_id = (sat.id - 1) % self.n_satellites
            right_id = (sat.id + 1) % self.n_satellites
            neighbor_data = [
                (self.satellites[left_id].theta, -1),
                (self.satellites[right_id].theta, +1)
            ]
            sat.update_kuramoto(neighbor_data, K, dt)
            sat.update_radius(dt)
        
        self.time += dt
    
    def get_positions(self):
        """Get all satellite positions as arrays."""
        positions = [s.get_3d_position() for s in self.satellites]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        return xs, ys, zs


# ==============================================================================
# Create Animation with Plotly Frames
# ==============================================================================

def create_animated_figure(n_satellites: int, target_radius: float, 
                          coupling_K: float, noise: float, n_frames: int = 100):
    """
    Pre-computes all simulation steps and creates a Plotly animation.
    This gives smooth millisecond-level animation without page reloads.
    """
    
    # Initialize constellation with random seed based on parameters
    # This ensures different initial configs when clicking "Run"
    import time
    seed = int(time.time() * 1000) % 100000
    np.random.seed(seed)
    constellation = SatelliteConstellation(n_satellites, target_radius, noise)
    
    # Pre-compute all frames
    frame_data = []
    sync_history = []
    fe_history = []  # Free Energy history
    
    for step in range(n_frames):
        xs, ys, zs = constellation.get_positions()
        sync = constellation.compute_order_parameter()
        total_fe = sum(s.compute_free_energy() for s in constellation.satellites)
        
        frame_data.append({
            'xs': xs.copy() if isinstance(xs, list) else list(xs),
            'ys': ys.copy() if isinstance(ys, list) else list(ys),
            'zs': zs.copy() if isinstance(zs, list) else list(zs),
            'sync': sync,
            'fe': total_fe,
            'time': constellation.time
        })
        sync_history.append(sync)
        fe_history.append(total_fe)
        
        constellation.step(coupling_K, dt=0.05)
    
    # Create Earth sphere with procedural continents
    resolution = 40
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    earth_x = 0.4 * np.outer(np.cos(u), np.sin(v))
    earth_y = 0.4 * np.outer(np.sin(u), np.sin(v))
    earth_z = 0.4 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Procedural Earth coloring (continents/oceans)
    earth_color = np.zeros((resolution, resolution))
    np.random.seed(42)
    for i in range(resolution):
        for j in range(resolution):
            lat = (j / resolution - 0.5) * np.pi
            lon = (i / resolution) * 2 * np.pi
            noise = (np.sin(3*lon) * np.cos(2*lat) + 
                    np.sin(5*lon + 1) * np.cos(4*lat + 2) * 0.5 +
                    np.sin(7*lon + 2) * np.cos(3*lat + 1) * 0.3)
            if abs(lat) > 1.3:
                earth_color[i,j] = 1.0  # Polar
            elif noise > 0.3:
                earth_color[i,j] = 0.4 + abs(lat) * 0.3  # Land
            else:
                earth_color[i,j] = 0.1 + noise * 0.1  # Ocean
    
    earth_colorscale = [
        [0.0, '#0a1628'], [0.15, '#1a4a7a'], [0.3, '#2d6a4f'],
        [0.5, '#40916c'], [0.7, '#b7b7a4'], [0.85, '#e9ecef'], [1.0, '#ffffff']
    ]
    
    # Create figure with initial frame
    fig = go.Figure()
    
    # Earth with realistic coloring
    fig.add_trace(go.Surface(
        x=earth_x, y=earth_y, z=earth_z,
        surfacecolor=earth_color,
        colorscale=earth_colorscale,
        showscale=False,
        opacity=1.0,
        name='Earth',
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.9)
    ))
    
    # Orbit ring
    theta_orbit = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter3d(
        x=target_radius * np.cos(theta_orbit),
        y=target_radius * np.sin(theta_orbit),
        z=np.zeros(100),
        mode='lines',
        line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'),
        name='Orbit'
    ))
    
    # Satellites (initial position)
    fig.add_trace(go.Scatter3d(
        x=frame_data[0]['xs'],
        y=frame_data[0]['ys'],
        z=frame_data[0]['zs'],
        mode='markers',
        marker=dict(size=8, color='#00FFFF', symbol='diamond'),
        name='Satellites'
    ))
    
    # Create animation frames with annotations
    frames = []
    max_fe = max(fd['fe'] for fd in frame_data) if frame_data else 1
    
    for i, fd in enumerate(frame_data):
        # Build frame with Earth surface using the same colorscale
        frame = go.Frame(
            data=[
                go.Surface(
                    x=earth_x, y=earth_y, z=earth_z,
                    surfacecolor=earth_color,
                    colorscale=earth_colorscale,
                    showscale=False, opacity=1.0
                ),
                go.Scatter3d(
                    x=target_radius * np.cos(theta_orbit),
                    y=target_radius * np.sin(theta_orbit),
                    z=np.zeros(100),
                    mode='lines',
                    line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash')
                ),
                go.Scatter3d(
                    x=fd['xs'], y=fd['ys'], z=fd['zs'],
                    mode='markers',
                    marker=dict(size=8, color='#00FFFF', symbol='diamond')
                )
            ],
            name=str(i),
            traces=[0, 1, 2],
            layout=go.Layout(
                annotations=[
                    dict(
                        text=f"Sync: {fd['sync']:.1%}",
                        x=0.98, y=0.30, xref='paper', yref='paper',
                        xanchor='right', yanchor='top',
                        font=dict(size=16, color='#00FF88'),
                        showarrow=False, bgcolor='rgba(0,0,0,0.7)'
                    ),
                    dict(
                        text=f"Free Energy: {fd['fe']:.2f}",
                        x=0.98, y=0.22, xref='paper', yref='paper',
                        xanchor='right', yanchor='top',
                        font=dict(size=16, color='#FF6B6B'),
                        showarrow=False, bgcolor='rgba(0,0,0,0.7)'
                    ),
                    dict(
                        text=f"Time: {fd['time']:.1f}s",
                        x=0.98, y=0.14, xref='paper', yref='paper',
                        xanchor='right', yanchor='top',
                        font=dict(size=14, color='#AAAAAA'),
                        showarrow=False, bgcolor='rgba(0,0,0,0.7)'
                    )
                ]
            )
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                y=1,
                x=0.1,
                xanchor='left',
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': 50, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 0}
                        }]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'prefix': 'Frame: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 0},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0,
            'steps': [
                {'args': [[str(i)], {'frame': {'duration': 0, 'redraw': True},
                                     'mode': 'immediate',
                                     'transition': {'duration': 0}}],
                 'label': str(i),
                 'method': 'animate'}
                for i in range(n_frames)
            ]
        }],
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, 
                      showticklabels=False, title='', range=[-2.5, 2.5]),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False,
                      showticklabels=False, title='', range=[-2.5, 2.5]),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False,
                      showticklabels=False, title='', range=[-2.5, 2.5]),
            bgcolor='#020408',
            camera=dict(eye=dict(x=1.8, y=1.2, z=0.6))
        ),
        paper_bgcolor='#020408',
        font=dict(color='white'),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=600
    )
    
    return fig, sync_history, fe_history


# ==============================================================================
# Streamlit App
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Secteur 0 | Constellation Control",
        page_icon="S0",
        layout="wide"
    )
    
    st.markdown("""
    <style>
    .stApp { background-color: #020408; }
    .stSidebar { background-color: #0a0a14; }
    h1 { color: #00CCFF !important; }
    h2, h3 { color: #88AACC !important; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Secteur 0 : Distributed Constellation Control")
    st.markdown("*Validating Active Inference & Kuramoto Dynamics for LEO Swarms*")
    
    # Sidebar controls
    st.sidebar.header("Configuration")
    
    n_satellites = st.sidebar.slider("Satellites (N)", 3, 12, 8)
    coupling_K = st.sidebar.slider("Coupling (K)", 0.5, 5.0, 3.5, 0.1)
    target_radius = st.sidebar.slider("Orbit Radius", 1.0, 2.0, 1.5, 0.1)
    noise_level = st.sidebar.slider("Noise", 0.0, 0.1, 0.02, 0.01)
    n_frames = st.sidebar.slider("Frames", 50, 200, 100, 10)
    
    # Physics-based warning system
    k_minimum_recommended = 0.3 * n_satellites
    
    if coupling_K < k_minimum_recommended:
        st.sidebar.warning(
            f"**Physics Alert**\n\n"
            f"With N={n_satellites} satellites in ring topology, "
            f"propagation latency requires K > {k_minimum_recommended:.1f} for stable synchronization.\n\n"
            f"Current K={coupling_K:.1f} may result in slow or incomplete convergence."
        )
    
    # Metastable states warning for high N
    if n_satellites >= 8:
        st.sidebar.info(
            "**Metastable States (Physics Note)**\n\n"
            "With N ≥ 8 satellites in ring topology, the system may settle into "
            "local energy minima where satellites cluster in pairs.\n\n"
            "This is a real physical phenomenon: each agent only sees its 2 neighbors, "
            "creating multiple stable configurations.\n\n"
            "*Solution: Increase K or Noise to escape these local minima.*"
        )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Simulation Controls")
    st.sidebar.markdown("""
    1. Click **Play** to start animation
    2. Click **Pause** to stop
    3. Drag the **slider** to scrub through frames
    4. Each frame = 1 simulation step
    """)
    
    # Run simulation button
    if st.sidebar.button("Run Constellation Dynamics", type="primary"):
        st.session_state.regenerate = True
    
    # Generate or use cached animation
    cache_key = f"{n_satellites}_{coupling_K}_{target_radius}_{noise_level}_{n_frames}"
    
    if 'animation_cache' not in st.session_state or \
       st.session_state.get('cache_key') != cache_key or \
       st.session_state.get('regenerate', False):
        
        with st.spinner("Generating animation frames..."):
            fig, sync_history, fe_history = create_animated_figure(
                n_satellites, target_radius, coupling_K, noise_level, n_frames
            )
            st.session_state.animation_cache = fig
            st.session_state.sync_history = sync_history
            st.session_state.fe_history = fe_history
            st.session_state.cache_key = cache_key
            st.session_state.regenerate = False
    
    # Main layout
    col_main, col_metrics = st.columns([2, 1])
    
    with col_main:
        # Display animation
        st.plotly_chart(st.session_state.animation_cache, use_container_width=True)
    
    with col_metrics:
        st.subheader("Telemetry")
        
        # Metrics
        st.metric("Initial Sync", f"{st.session_state.sync_history[0]:.1%}")
        st.metric("Final Sync", f"{st.session_state.sync_history[-1]:.1%}")
        st.metric("Initial Free Energy", f"{st.session_state.fe_history[0]:.2f}")
        st.metric("Final Free Energy", f"{st.session_state.fe_history[-1]:.4f}")
        
        # Sync over time graph
        import plotly.express as px
        time_axis = [i * 0.05 for i in range(len(st.session_state.sync_history))]
        
        fig_sync = go.Figure()
        fig_sync.add_trace(go.Scatter(
            x=time_axis, y=st.session_state.sync_history,
            mode='lines', name='Synchronization',
            line=dict(color='#00FF88', width=2)
        ))
        fig_sync.update_layout(
            plot_bgcolor='#0a0a14', paper_bgcolor='#0a0a14',
            font=dict(color='white', size=10),
            xaxis=dict(title='Time (s)', gridcolor='#1a1a2a'),
            yaxis=dict(title='Sync', gridcolor='#1a1a2a', range=[0, 1.1]),
            margin=dict(l=40, r=10, t=30, b=30),
            height=150, title="Synchronization"
        )
        st.plotly_chart(fig_sync, use_container_width=True)
        
        # Free Energy over time graph
        fe_normalized = [f / max(st.session_state.fe_history) if max(st.session_state.fe_history) > 0 else 0 
                        for f in st.session_state.fe_history]
        
        fig_fe = go.Figure()
        fig_fe.add_trace(go.Scatter(
            x=time_axis, y=fe_normalized,
            mode='lines', name='Free Energy',
            line=dict(color='#FF6B6B', width=2)
        ))
        fig_fe.update_layout(
            plot_bgcolor='#0a0a14', paper_bgcolor='#0a0a14',
            font=dict(color='white', size=10),
            xaxis=dict(title='Time (s)', gridcolor='#1a1a2a'),
            yaxis=dict(title='F (norm)', gridcolor='#1a1a2a', range=[0, 1.1]),
            margin=dict(l=40, r=10, t=30, b=30),
            height=150, title="Free Energy"
        )
        st.plotly_chart(fig_fe, use_container_width=True)
    
    # Governing Equations Section
    st.markdown("---")
    
    with st.expander("Governing Equations"):
        col_eq1, col_eq2 = st.columns(2)
        
        with col_eq1:
            st.markdown("""
            **Radial Control (Active Inference)**
            
            Agents minimize Variational Free Energy by reducing 
            sensory prediction error on altitude:
            
            $$F ≈ \\frac{1}{2} \\pi (r - \\mu_{target})^2$$
            
            The agent acts to suppress the discrepancy between 
            observed and expected radius.
            """)
        
        with col_eq2:
            st.markdown("""
            **Angular Coordination (Directional Kuramoto)**
            
            Anisotropic phase coupling with directional bias:
            
            $$\\dot{\\theta}_i = \\omega_i + K \\sum_{j \\in \\mathcal{N}_i} \\sin(\\theta_j - \\theta_i - d_j \\Delta\\phi)$$
            
            where $d_j = \\pm 1$ indicates leading/trailing neighbor.
            
            *This symmetry breaking prevents metastable clustering.*
            """)
    
    st.markdown("---")
    st.markdown("""
    ### Instructions
    
    1. **Play**: Click to start smooth animation
    2. **Slider**: Drag to scrub frame-by-frame
    3. **Rotate**: Click and drag on the 3D view
    
    Each frame = one simulation step.
    """)


if __name__ == "__main__":
    main()
