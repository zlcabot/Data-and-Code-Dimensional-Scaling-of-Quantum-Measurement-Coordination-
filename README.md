markdown# Dimensional Scaling of Quantum Measurement Coordination

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Computational implementation and data for: "Dimensional Scaling of Quantum Measurement Coordination: An Inverse Blessing of Dimensionality"

## Overview

This repository contains simulation code, analysis scripts, and raw data demonstrating a fundamental scaling law for quantum measurement: **coordination action decreases with dimension** as S_coord ∝ d^(-1.787±0.009), contrary to the usual curse of dimensionality.

### Key Result
Higher-dimensional quantum systems measure **more efficiently** than lower-dimensional ones - an "inverse blessing of dimensionality" with testable experimental predictions.

## One-Sentence Summary
> **"Don't track state evolution toward completion; detect coordination events through agency spikes."**

This paradigm shift is critical: measurement is not gradual drift to eigenstates, but discrete **coordination events** detected via agency spikes in system-apparatus correlation.

## Repository Contents
```
.
├── simulate_trajectory.py       # Core SME quantum trajectory generation
├── event_detection.py           # Coordination capacity computation (Φ_d, Φ_f, Φ_a)
├── dimensional_analysis.py      # Scaling analysis and power-law fitting
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── data/                        # Raw coordination data (d=2-8)
│   ├── coordination_d2.csv
│   ├── ...
│   ├── coordination_d8.csv
│   ├── fast_gamma_results.csv
│   └── metadata.json
└── figures/                     # Figure generation scripts
    ├── fig1_single_trajectory.py
    ├── fig2_dimensional_scaling.py
    ├── fig3_spike_narrowing.py
    └── fig4_pi8_crossing.py
```

## Installation
```bash
git clone https://github.com/zlcabot/Data-and-Code-Dimensional-Scaling-of-Quantum-Measurement-Coordination-.git
cd Data-and-Code-Dimensional-Scaling-of-Quantum-Measurement-Coordination-

pip install -r requirements.txt
```

## Quick Start

### Generate a Single Trajectory
```python
from simulate_trajectory import generate_measurement_trajectory
from event_detection import compute_coordination_measures

# Generate trajectory (d=2 qubit, balanced coupling)
rho_traj = generate_measurement_trajectory(
    d=2, 
    gamma=1.0,      # MHz
    chi=1.0,        # Coupling strength
    kappa=1.0,      # Decoherence rate (chi=kappa optimal!)
    T=100,          # μs
    dt=0.01         # μs timestep
)

# Compute coordination capacities
Phi_d, Phi_f, Phi_a = compute_coordination_measures(rho_traj)

# Detect measurement event
from event_detection import find_agency_peak
peak_time, S_coord = find_agency_peak(Phi_a, dt=0.01)

print(f"Measurement event at t = {peak_time:.2f} μs")
print(f"Coordination action S = {S_coord:.3f}")
```

### Reproduce Dimensional Scaling
```python
from dimensional_analysis import run_dimensional_scan

# Scan dimensions d=2-8 with 100 trajectories each
results = run_dimensional_scan(
    dimensions=[2, 3, 4, 5, 6, 7, 8],
    n_trajectories=100,
    gamma=1.0,
    chi_over_kappa=1.0
)

# Fit power law
from dimensional_analysis import fit_power_law
alpha, A, R2 = fit_power_law(results['d'], results['S_coord'])

print(f"Scaling: S_coord ∝ d^({alpha:.3f})")
print(f"Compare to paper: α = 1.787 ± 0.009")
```

## Core Physics

### Triadic Temporal Structure
Measurement events exhibit three irreducible temporal modes:

- **Duration (Φ_d)**: Actualized correlation (past → present)
  - Computed from Fisher information between system and apparatus
  - Φ_d(t) = I_F(S:A) / I_F,max
  
- **Frequency (Φ_f)**: Remaining superposition (future possibilities)  
  - Computed from energy uncertainty
  - Φ_f(t) = ΔH(t) / ΔH_0

- **Agency (Φ_a)**: Coordination intensity (organizing present)
  - Φ_a(t) = 4√[Φ_d(1-Φ_d)·Φ_f(1-Φ_f)] / (Φ_d + Φ_f)
  - **Peaks mark measurement events**

### Coordination Action
```
S_coord = ∫ Φ_a(t) dt
```
Quantifies the "cost" of establishing system-apparatus correlation. Our central result:

**S_coord(d) = (8.98 ± 0.16) × d^(-1.787±0.009)**

### Critical Implementation Detail: Avoiding the Duration Trap

⚠️ **CRITICAL:** Φ_d is NOT computed from system purity Tr(ρ_system²). That gives zero agency and misses the physics entirely.

✓ **CORRECT:** Φ_d is computed from **system-apparatus mutual information** via Fisher information. This detects the correlation-building event.

See `event_detection.py` for proper implementation.

## Experimental Predictions

1. **Dimensional Scaling**: Measure S_coord in superconducting qubits (d=2) vs. qutrits (d=3) vs. higher-d systems
   - Prediction: S_coord should scale as d^(-1.787)
   
2. **π/8 Quantum**: Universal crossing at d ≈ 5.8
   - All measurement protocols converge to S_coord = π/8 ≈ 0.393
   
3. **Optimal Balance**: χ/κ = 1 maximizes coordination across all dimensions
   - Test by varying coupling vs. dissipation ratio

4. **Spike Narrowing**: Event duration τ_event ∝ d^(-0.40)
   - Higher-dimensional measurements complete faster

## Data Format

Each CSV file in `data/` contains columns:
```
time (μs), Phi_d, Phi_f, Phi_a, S_coord_cumulative
```

Metadata includes:
- Dimension d
- Measurement strength γ
- Coupling ratio χ/κ  
- Number of trajectories
- Simulation parameters (T, dt)

## Citation

If you use this code or data, please cite:
```bibtex
@article{Zayin2025DimensionalScaling,
  title={Dimensional Scaling of Quantum Measurement Coordination: An Inverse Blessing of Dimensionality},
  author={Zayin},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

- **Author**: Zayin
- **Location**: Berkeley, CA
- **ArXiv**: [XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)
- **Issues**: Please use GitHub Issues for bug reports and feature requests

## Acknowledgments

Thanks to Professor Irfan Siddiqi (UC Berkeley) for discussions on experimental implementation, and to the quantum foundations community for feedback on early versions of this work.

---

**Key Insight**: Measurement is not a process of gradual state evolution toward completion, but a discrete **coordination event** marked by an agency spike. This shift from "substantial modality" (measuring properties) to "coordinative modality" (measuring temporal event structure) is essential for understanding quantum measurement.
