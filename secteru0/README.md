# Secteur 0 : Distributed Constellation Control

🛰️ Interactive simulation validating **Active Inference** & **Kuramoto Dynamics** for LEO satellite swarms.

![Secteur 0 Demo](https://img.shields.io/badge/Status-Prototype-yellow)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Live Demo

[**Launch Simulation →**](https://secteur0.streamlit.app)

## 📖 Overview

Secteur 0 demonstrates a **decentralized control architecture** where autonomous agents (satellites) coordinate without a central server:

- **Active Inference**: Agents minimize Variational Free Energy to maintain orbital altitude
- **Directional Kuramoto**: Anisotropic phase coupling for even angular spacing
- **Ring Topology**: Each agent communicates only with immediate neighbors

## 🧮 Key Innovation: Directional Phase Bias

Unlike standard Kuramoto (isotropic), we implement **anisotropic coupling** with directional bias:

```
θ̇ᵢ = ωᵢ + K Σⱼ sin(θⱼ - θᵢ - dⱼΔφ)
```

where `dⱼ = ±1` indicates leading/trailing neighbor. This **symmetry breaking** prevents metastable clustering states.

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/secteur0.git
cd secteur0
pip install -r requirements.txt
streamlit run app_pro.py
```

## 🧪 Control Loop Ablation

Set **K → 0** to decouple agents and observe that synchronization is an emergent property of the dynamics, not hard-coded behavior.

## 📐 Governing Equations

### Radial Control (Active Inference)

$$F ≈ \frac{1}{2} \pi (r - \mu_{target})^2$$

### Angular Coordination (Directional Kuramoto)

$$\dot{\theta}_i = \omega_i + K \sum_{j \in \mathcal{N}_i} \sin(\theta_j - \theta_i - d_j \Delta\phi)$$

## 📄 License

MIT License - Saad LARAJ

## 🔗 References

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence
- Olfati-Saber, R. (2007). Consensus and Cooperation in Networked Multi-Agent Systems
