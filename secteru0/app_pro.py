"""
Secteur 0 : Distributed Constellation Control
==============================================
Validating Active Inference & Kuramoto Dynamics for LEO Swarms

Run with: streamlit run app_pro.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List
from PIL import Image
import requests
from io import BytesIO

# ==============================================================================
# Satellite Agent Class
# ==============================================================================

@dataclass
class SatelliteAgent:
    """
    Satellite agent implementing:
    - Active Inference for radial control (altitude maintenance)
    - Kuramoto dynamics for angular coordination (collision avoidance)
    """
    id: int
    
    # Spherical coordinates
    radius: float
    theta: float
    phi: float = 0.0
    
    # State derivatives
    radial_velocity: float = 0.0
    angular_velocity: float = 0.0
    
    # Reference states
    target_radius: float = 1.0
    target_angular_spacing: float = 0.0
    
    # Intrinsic frequency
    omega: float = 0.1
    
    # Precision (inverse variance)
    precision_radius: float = 2.0
    
    # Accumulated delta-v for fuel cost estimation
    total_delta_v: float = 0.0
    
    def compute_free_energy(self) -> float:
        """
        Variational Free Energy approximation.
        F ≈ (1/2) π (r - μ_target)²
        
        Represents sensory prediction error on altitude.
        """
        prediction_error = self.radius - self.target_radius
        return 0.5 * self.precision_radius * prediction_error ** 2
    
    def compute_radial_action(self) -> float:
        """
        Active Inference action selection.
        Agent acts to suppress prediction error.
        """
        prediction_error = self.radius - self.target_radius
        action = -self.precision_radius * prediction_error
        return action
    
    def update_kuramoto(self, neighbor_data: List[tuple], K: float, dt: float):
        """
        Directional Kuramoto with Phase Bias (Anisotropic Coupling).
        
        Unlike standard Kuramoto (isotropic), this implements directional
        coupling to enforce ring spacing constraints and prevent metastable
        clustering states.
        
        For ring topology:
        - Leading neighbor (i+1): expected at θ + Δφ
        - Trailing neighbor (i-1): expected at θ - Δφ
        
        This symmetry breaking ensures equidistant spacing.
        """
        if not neighbor_data:
            self.theta += self.omega * dt
            return
        
        coupling = 0.0
        for neighbor_theta, direction in neighbor_data:
            # direction: -1 for left neighbor, +1 for right neighbor
            expected_diff = direction * self.target_angular_spacing
            actual_diff = neighbor_theta - self.theta
            phase_error = actual_diff - expected_diff
            coupling += np.sin(phase_error)
        
        coupling /= len(neighbor_data)
        
        self.angular_velocity = self.omega + K * coupling
        self.theta += self.angular_velocity * dt
        self.theta = self.theta % (2 * np.pi)
    
    def update_radius(self, dt: float, damping: float = 0.8):
        """Update radial state via Active Inference."""
        action = self.compute_radial_action()
        
        # Track delta-v for fuel estimation
        self.total_delta_v += abs(action * dt)
        
        self.radial_velocity += action * dt
        self.radial_velocity *= damping
        self.radius += self.radial_velocity * dt
        self.radius = max(0.5, self.radius)
    
    def get_3d_position(self) -> tuple:
        """Spherical to Cartesian transformation."""
        x = self.radius * np.cos(self.theta) * np.cos(self.phi)
        y = self.radius * np.sin(self.theta) * np.cos(self.phi)
        z = self.radius * np.sin(self.phi)
        return x, y, z


# ==============================================================================
# Constellation Class
# ==============================================================================

class SatelliteConstellation:
    """
    LEO constellation with ring topology.
    Agents communicate only with adjacent neighbors.
    """
    
    def __init__(self, n_satellites: int, target_radius: float, noise: float = 0.0):
        self.n_satellites = n_satellites
        self.target_radius = target_radius
        self.noise = noise
        self.target_spacing = 2 * np.pi / n_satellites
        
        self.satellites: List[SatelliteAgent] = []
        for i in range(n_satellites):
            sat = SatelliteAgent(
                id=i,
                radius=target_radius * np.random.uniform(0.6, 1.4),
                theta=np.random.uniform(0, 2 * np.pi),
                phi=np.random.uniform(-0.05, 0.05),
                omega=0.05 + np.random.uniform(-0.01, 0.01),
                target_radius=target_radius
            )
            sat.target_angular_spacing = self.target_spacing
            self.satellites.append(sat)
        
        # Ring topology: each agent connected to left/right neighbors
        self.neighbors = {i: [(i-1) % n_satellites, (i+1) % n_satellites] 
                         for i in range(n_satellites)}
        
        self.history_sync = []
        self.history_F = []
        self.history_delta_v = []
        self.time = 0.0
    
    def compute_order_parameter(self) -> float:
        """
        Angular spacing uniformity metric.
        r = 1 when perfectly equidistant, r → 0 for clustering.
        """
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
        r = 1 - min(std / np.pi, 1.0)
        return r
    
    def compute_total_free_energy(self) -> float:
        """Aggregate free energy across constellation."""
        return sum(s.compute_free_energy() for s in self.satellites)
    
    def compute_total_delta_v(self) -> float:
        """Estimated cumulative delta-v (fuel proxy)."""
        return sum(s.total_delta_v for s in self.satellites)
    
    def step(self, K: float, dt: float = 0.1):
        """Single simulation step."""
        
        # Environmental perturbations (solar radiation, etc.)
        if self.noise > 0:
            for sat in self.satellites:
                sat.theta += np.random.normal(0, self.noise * dt)
                sat.radius += np.random.normal(0, self.noise * 0.1 * dt)
        
        # Update each satellite
        for sat in self.satellites:
            # Get neighbor data with direction (left=-1, right=+1)
            left_id = (sat.id - 1) % self.n_satellites
            right_id = (sat.id + 1) % self.n_satellites
            neighbor_data = [
                (self.satellites[left_id].theta, -1),   # Left neighbor
                (self.satellites[right_id].theta, +1)   # Right neighbor
            ]
            
            sat.update_kuramoto(neighbor_data, K, dt)
            sat.update_radius(dt)
        
        self.history_sync.append(self.compute_order_parameter())
        self.history_F.append(self.compute_total_free_energy())
        self.history_delta_v.append(self.compute_total_delta_v())
        self.time += dt


# ==============================================================================
# Earth Texture Generation
# ==============================================================================

def create_realistic_earth(radius=0.4, resolution=60):
    """
    Create a realistic Earth sphere with surface features.
    Uses procedural generation for continents/oceans.
    """
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Create procedural Earth colors
    # Deep blue oceans, green/brown continents
    colors = np.zeros((resolution, resolution, 3))
    
    np.random.seed(42)  # Reproducible "continents"
    
    for i in range(resolution):
        for j in range(resolution):
            # Base: ocean blue
            lat = (j / resolution - 0.5) * np.pi
            lon = (i / resolution) * 2 * np.pi
            
            # Simple continent generation using noise
            noise = (np.sin(3*lon) * np.cos(2*lat) + 
                    np.sin(5*lon + 1) * np.cos(4*lat + 2) * 0.5 +
                    np.sin(7*lon + 2) * np.cos(3*lat + 1) * 0.3)
            
            if noise > 0.3:  # Land
                # Green to brown based on latitude
                if abs(lat) > 1.2:  # Polar
                    colors[i,j] = [0.95, 0.95, 0.98]  # White/ice
                elif abs(lat) > 0.8:  # Temperate
                    colors[i,j] = [0.2, 0.5, 0.2]  # Green
                else:  # Tropical/desert
                    if noise > 0.6:
                        colors[i,j] = [0.8, 0.7, 0.4]  # Desert
                    else:
                        colors[i,j] = [0.15, 0.4, 0.15]  # Jungle
            else:  # Ocean
                depth_factor = 0.5 + noise * 0.5
                colors[i,j] = [0.05, 0.2 * depth_factor, 0.5 + 0.3 * depth_factor]
    
    # Convert to plotly colorscale format
    # Flatten and create custom colorscale
    return x, y, z, colors


def create_earth_surface(radius=0.4, resolution=50):
    """Create Earth sphere with custom coloring."""
    
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    # Create surface values for color mapping
    surfacecolor = np.zeros((resolution, resolution))
    
    np.random.seed(42)
    
    for i in range(resolution):
        for j in range(resolution):
            lat = (j / resolution - 0.5) * np.pi
            lon = (i / resolution) * 2 * np.pi
            
            # Continent noise
            noise = (np.sin(3*lon) * np.cos(2*lat) + 
                    np.sin(5*lon + 1) * np.cos(4*lat + 2) * 0.5 +
                    np.sin(7*lon + 2) * np.cos(3*lat + 1) * 0.3)
            
            # Polar regions
            if abs(lat) > 1.3:
                surfacecolor[i,j] = 1.0  # White
            elif noise > 0.3:  # Land
                surfacecolor[i,j] = 0.4 + abs(lat) * 0.3  # Green to brown
            else:  # Ocean
                surfacecolor[i,j] = 0.1 + noise * 0.1  # Blue
    
    return x, y, z, surfacecolor


# ==============================================================================
# 3D Visualization
# ==============================================================================

def create_3d_space_plot(constellation: SatelliteConstellation, orbit_radius: float):
    """Professional space visualization."""
    
    fig = go.Figure()
    
    # Earth with realistic coloring
    earth_x, earth_y, earth_z, earth_color = create_earth_surface(radius=0.4)
    
    # Custom colorscale: blue ocean → green land → white polar
    earth_colorscale = [
        [0.0, '#0a1628'],   # Deep ocean
        [0.15, '#1a4a7a'],  # Ocean
        [0.3, '#2d6a4f'],   # Forest
        [0.5, '#40916c'],   # Light green
        [0.7, '#b7b7a4'],   # Desert/brown
        [0.85, '#e9ecef'],  # Snow
        [1.0, '#ffffff']    # Ice
    ]
    
    fig.add_trace(go.Surface(
        x=earth_x, y=earth_y, z=earth_z,
        surfacecolor=earth_color,
        colorscale=earth_colorscale,
        showscale=False,
        opacity=1.0,
        name='Earth',
        hoverinfo='name',
        lighting=dict(
            ambient=0.4,
            diffuse=0.8,
            specular=0.2,
            roughness=0.9
        )
    ))
    
    # Atmosphere glow (slightly larger transparent sphere)
    atm_u = np.linspace(0, 2 * np.pi, 30)
    atm_v = np.linspace(0, np.pi, 30)
    atm_x = 0.42 * np.outer(np.cos(atm_u), np.sin(atm_v))
    atm_y = 0.42 * np.outer(np.sin(atm_u), np.sin(atm_v))
    atm_z = 0.42 * np.outer(np.ones(np.size(atm_u)), np.cos(atm_v))
    
    fig.add_trace(go.Surface(
        x=atm_x, y=atm_y, z=atm_z,
        colorscale=[[0, 'rgba(100,180,255,0.1)'], [1, 'rgba(100,180,255,0.1)']],
        showscale=False,
        opacity=0.3,
        hoverinfo='skip'
    ))
    
    # Target orbit ring
    theta_orbit = np.linspace(0, 2*np.pi, 100)
    x_orbit = orbit_radius * np.cos(theta_orbit)
    y_orbit = orbit_radius * np.sin(theta_orbit)
    z_orbit = np.zeros_like(theta_orbit)
    
    fig.add_trace(go.Scatter3d(
        x=x_orbit, y=y_orbit, z=z_orbit,
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.2)', width=1, dash='dash'),
        name='Reference Orbit',
        hoverinfo='skip'
    ))
    
    # Satellites
    positions = [s.get_3d_position() for s in constellation.satellites]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(
            size=6,
            color='#00FFFF',
            symbol='diamond',
            line=dict(color='white', width=0.5)
        ),
        name='Satellites',
        hovertemplate='SAT-%{customdata[0]}<br>Alt: %{customdata[1]:.3f}<br>Phase: %{customdata[2]:.1f}°<extra></extra>',
        customdata=[[s.id, s.radius, np.degrees(s.theta)] for s in constellation.satellites]
    ))
    
    # Communication links
    for sat in constellation.satellites:
        for neighbor_id in constellation.neighbors[sat.id]:
            if neighbor_id > sat.id:
                neighbor = constellation.satellites[neighbor_id]
                p1 = sat.get_3d_position()
                p2 = neighbor.get_3d_position()
                fig.add_trace(go.Scatter3d(
                    x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                    mode='lines',
                    line=dict(color='rgba(0, 255, 255, 0.15)', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Layout
    axis_range = orbit_radius * 1.5
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, 
                      showticklabels=False, title='', range=[-axis_range, axis_range]),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False,
                      showticklabels=False, title='', range=[-axis_range, axis_range]),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False,
                      showticklabels=False, title='', range=[-axis_range, axis_range]),
            bgcolor='#020408',
            camera=dict(eye=dict(x=1.8, y=1.2, z=0.6))
        ),
        paper_bgcolor='#020408',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.6)', 
                   bordercolor='rgba(255,255,255,0.2)', borderwidth=1),
        margin=dict(l=0, r=0, t=0, b=0),
        height=550
    )
    
    return fig


def create_metrics_plot(constellation: SatelliteConstellation):
    """Metrics visualization."""
    
    if len(constellation.history_sync) < 2:
        return go.Figure()
    
    time_axis = np.arange(len(constellation.history_sync)) * 0.1
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_axis, y=constellation.history_sync,
        mode='lines', name='Spacing Uniformity',
        line=dict(color='#00FF88', width=2)
    ))
    
    F_normalized = np.array(constellation.history_F)
    if F_normalized.max() > 0:
        F_normalized = F_normalized / F_normalized.max()
    
    fig.add_trace(go.Scatter(
        x=time_axis, y=F_normalized,
        mode='lines', name='Free Energy (norm.)',
        line=dict(color='#FF6B6B', width=2)
    ))
    
    fig.add_hline(y=0.95, line_dash='dash', line_color='rgba(255,255,255,0.3)',
                  annotation_text='95%', annotation_font_color='gray')
    
    fig.update_layout(
        plot_bgcolor='#0a0a12',
        paper_bgcolor='#0a0a12',
        font=dict(color='white', size=11),
        legend=dict(x=0.02, y=0.98, font_size=10),
        xaxis=dict(title='Time (s)', gridcolor='#1a1a2a', title_font_size=11),
        yaxis=dict(title='', gridcolor='#1a1a2a', range=[0, 1.1]),
        margin=dict(l=40, r=10, t=10, b=40),
        height=180
    )
    
    return fig


# ==============================================================================
# Streamlit App
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Secteur 0 | Constellation Control",
        page_icon="🛰️",
        layout="wide"
    )
    
    st.markdown("""
    <style>
    .stApp { background-color: #020408; }
    .stSidebar { background-color: #0a0a14; }
    h1 { color: #00CCFF !important; font-weight: 500; }
    h2, h3 { color: #88AACC !important; font-weight: 400; }
    .stMetric { 
        background-color: #0a0a14; 
        padding: 12px; 
        border-radius: 4px;
        border: 1px solid #1a2a3a;
    }
    .stMetric label { color: #6688AA !important; font-size: 12px; }
    .stMetric [data-testid="stMetricValue"] { color: #00FFCC !important; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("Secteur 0 : Distributed Constellation Control")
    st.markdown("*Validating Active Inference & Kuramoto Dynamics for LEO Swarms*")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    n_satellites = st.sidebar.slider("Number of Satellites", 3, 16, 8)
    coupling_K = st.sidebar.slider("Coupling Strength (K)", 0.0, 5.0, 3.5, 0.1)
    target_radius = st.sidebar.slider("Orbital Altitude", 1.0, 2.5, 1.5, 0.1)
    noise_level = st.sidebar.slider("Perturbation Level", 0.0, 0.1, 0.02, 0.01)
    
    # Physics-based warning system
    k_minimum_recommended = 0.3 * n_satellites  # Empirical rule for ring topology
    
    if coupling_K < k_minimum_recommended and coupling_K > 0:
        st.sidebar.warning(
            f"⚠️ **Physics Alert**\n\n"
            f"With N={n_satellites} satellites in ring topology, "
            f"propagation latency requires K > {k_minimum_recommended:.1f} for stable synchronization.\n\n"
            f"Current K={coupling_K:.1f} may result in slow or incomplete convergence."
        )
    
    st.sidebar.markdown("---")
    
    # Ablation study section
    st.sidebar.markdown("### 🧪 Control Loop Ablation")
    st.sidebar.markdown(
        "Set **K → 0** to decouple agents. This demonstrates that "
        "synchronization is an emergent property of the Kuramoto dynamics, "
        "not a hard-coded behavior."
    )
    
    # State management
    if 'constellation' not in st.session_state or st.sidebar.button("↻ Reset Constellation"):
        st.session_state.constellation = SatelliteConstellation(
            n_satellites=n_satellites,
            target_radius=target_radius,
            noise=noise_level
        )
    
    constellation = st.session_state.constellation
    constellation.noise = noise_level
    for sat in constellation.satellites:
        sat.target_radius = target_radius
    
    # Run controls
    col_run, col_step = st.sidebar.columns(2)
    with col_run:
        if st.button("▶ Run 100 steps"):
            for _ in range(100):
                constellation.step(coupling_K)
    with col_step:
        if st.button("⏵ Step"):
            constellation.step(coupling_K)
    
    # Main layout
    col1, col2 = st.columns([2.2, 1])
    
    with col1:
        st.subheader("Orbital View")
        fig_3d = create_3d_space_plot(constellation, target_radius)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with col2:
        st.subheader("Telemetry")
        
        current_sync = constellation.compute_order_parameter()
        current_F = constellation.compute_total_free_energy()
        delta_v = constellation.compute_total_delta_v()
        
        st.metric("Spacing Uniformity", f"{current_sync:.1%}")
        st.metric("Total Free Energy", f"{current_F:.3f}")
        st.metric("Simulation Time", f"{constellation.time:.1f} s")
        st.metric("Est. ΔV Cost", f"{delta_v:.4f} units")
        
        if len(constellation.history_sync) > 1:
            fig_metrics = create_metrics_plot(constellation)
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Real-time stability indicator
        if current_sync < 0.7 and constellation.time > 50:
            st.error(
                "📉 **Low Convergence Detected**\n\n"
                "System not stabilizing. Consider:\n"
                "- Increasing Coupling Strength (K)\n"
                "- Reducing Number of Satellites\n"
                "- Lowering Perturbation Level"
            )
        elif current_sync > 0.95:
            st.success("✓ Constellation synchronized")
    
    # Governing equations
    st.markdown("---")
    
    with st.expander("📐 Governing Equations"):
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
            
            Anisotropic phase coupling with directional bias 
            for collision avoidance and coverage:
            
            $$\\dot{\\theta}_i = \\omega_i + K \\sum_{j \\in \\mathcal{N}_i} \\sin(\\theta_j - \\theta_i - d_j \\Delta\\phi)$$
            
            where $d_j = \\pm 1$ indicates leading/trailing neighbor.
            
            *This symmetry breaking prevents metastable clustering 
            and ensures stable ring formation.*
            """)


if __name__ == "__main__":
    main()
