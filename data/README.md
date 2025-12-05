# Data Directory

This directory contains simulation data for dimensional scaling analysis.

## File Structure

```
data/
├── metadata.json              # Complete dataset description
├── coordination_d2.csv        # Qubit (d=2) data
├── coordination_d3.csv        # Qutrit (d=3) data
├── coordination_d4.csv        # d=4 system data
├── coordination_d5.csv        # d=5 system data
├── coordination_d6.csv        # d=6 system data
├── coordination_d7.csv        # d=7 system data
├── coordination_d8.csv        # d=8 system data
├── fast_gamma_results.csv    # Protocol sensitivity data
└── README.md                  # This file
```

## Data Format

Each `coordination_dX.csv` file contains averaged trajectory data:

### Columns

- **time** (μs): Time points sampled at 0.1 μs intervals
- **Phi_d**: Duration coordination (0-1)
  - Actualized system-apparatus correlation
  - Computed from Fisher information
- **Phi_f**: Frequency coordination (0-1)
  - Remaining superposition potential
  - Computed from energy uncertainty
- **Phi_a**: Coordination capacity (0-1)
  - Agency spike marking measurement event
  - Product measure: 4√[Φ_d(1-Φ_d)·Φ_f(1-Φ_f)] / (Φ_d + Φ_f)
- **S_coord_cumulative**: Running integral of Phi_a
  - Coordination action up to time t
  - Final value = total S_coord for trajectory

### Example

```csv
time,Phi_d,Phi_f,Phi_a,S_coord_cumulative
0.0,0.001,0.999,0.063,0.000
0.1,0.012,0.988,0.195,0.013
0.2,0.034,0.966,0.342,0.040
...
```

## Generation

Data was generated using:

```python
from simulate_trajectory import generate_measurement_trajectory
from event_detection import compute_coordination_measures

# Generate trajectory
rho_traj = generate_measurement_trajectory(
    d=2,           # Dimension
    gamma=1.0,     # Measurement strength (MHz)
    chi=1.0,       # Coupling
    kappa=1.0,     # Dissipation
    T=100,         # Time span (μs)
    dt=0.1         # Time step (μs)
)

# Compute coordination measures
Phi_d, Phi_f, Phi_a = compute_coordination_measures(rho_traj)
```

## Reproduction

To regenerate data:

1. Install dependencies: `pip install -r requirements.txt`
2. Run simulation scripts: `python simulate_trajectory.py --dimension 2`
3. Or use example notebooks in `examples/` directory

## Key Results

From this data, the dimensional scaling law was extracted:

**S_coord(d) = (8.98 ± 0.16) × d^(-1.787±0.009)**

- **R² = 0.9987** (excellent fit)
- **π/8 crossing at d ≈ 5.8**
- **Bootstrap validated** (10,000 iterations)

## fast_gamma_results.csv

Protocol sensitivity analysis varying measurement strength γ:

### Columns

- **dimension**: Hilbert space dimension (2-8)
- **gamma**: Measurement strength (0.5-2.0 MHz)
- **S_coord**: Total coordination action
- **peak_time**: Time of maximum Phi_a
- **peak_amplitude**: Maximum value of Phi_a

Shows protocol independence emerges at high dimensions (d ≥ 6).

## Metadata

See `metadata.json` for complete parameter specifications, computational details, and result summary.

## Citation

If using this data, please cite:

```
Cabot, Z. (2024). Dimensional Scaling of Quantum Measurement Coordination: 
An Inverse Blessing of Dimensionality. arXiv:XXXX.XXXXX
```

## License

MIT License - see LICENSE file in repository root.
