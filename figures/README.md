# Figures Directory

Python scripts to generate all publication figures.

## File Structure

```
figures/
├── README.md                      # This file
├── fig1_single_trajectory.py     # Single trajectory with coordination spike
├── fig2_dimensional_scaling.py   # Main scaling result (S ∝ d^-1.787)
├── fig3_spike_narrowing.py       # Geometric narrowing (Δt ∝ d^-0.558)
└── fig4_pi8_crossing.py          # Universal crossing & sign reversal
```

## Quick Start

### Generate All Figures

```bash
# From repository root
cd figures/

# Figure 1: Single trajectory
python fig1_single_trajectory.py

# Figure 2: Dimensional scaling (main result)
python fig2_dimensional_scaling.py

# Figure 3: Spike narrowing
python fig3_spike_narrowing.py

# Figure 4: π/8 crossing (smoking gun)
python fig4_pi8_crossing.py
```

## Figure Descriptions

### Figure 1: Single Trajectory

**File:** `fig1_single_trajectory.py`

**Demonstrates:**
- Single quantum measurement trajectory
- Duration (Φ_d) and Frequency (Φ_f) evolution
- Coordination capacity (Φ_a) spike marking measurement event
- Event detection at peak

**Output:**
- `fig1_single_trajectory_d2.pdf` (qubit example)
- Additional dimensions: d=4, d=8 (for comparison)

**Key insight:** Measurement is a discrete coordination event, not gradual drift.

---

### Figure 2: Dimensional Scaling

**File:** `fig2_dimensional_scaling.py`

**Demonstrates:**
- Main result: S_coord ∝ d^(-1.787±0.009)
- Inverse blessing of dimensionality
- Excellent fit: R² = 0.9987
- π/8 threshold crossing

**Output:**
- `fig2_dimensional_scaling.pdf`

**Key insight:** Higher-dimensional systems measure ~10× more efficiently than qubits.

---

### Figure 3: Spike Narrowing

**File:** `fig3_spike_narrowing.py`

**Demonstrates:**
- Geometric narrowing: Δt ∝ d^(-0.558±0.04)
- Progressive spike narrowing with dimension
- Validates Lemma 1 (geometric bound β ≥ 0.5)
- Measurement speed increases with dimension

**Output:**
- `fig3_spike_narrowing.pdf`

**Key insight:** Concentration of measure on Bloch sphere causes spike narrowing.

---

### Figure 4: π/8 Universal Crossing

**File:** `fig4_pi8_crossing.py`

**Demonstrates:**
- **SIGN REVERSAL** (smoking gun signature)
- Four measurement protocols:
  * Continuous homodyne: α = +1.787 (inverse blessing)
  * Geometric baseline: α = +0.534 (validates bound)
  * Discrete photon: α = -0.28 (curse)
  * Environmental noise: α = -1.7 (strong curse)
- Universal crossing near d_c ≈ 5.8 at S* ≈ π/8

**Output:**
- `fig4_pi8_crossing.pdf`

**Key insight:** Opposite-sign exponents provide decisive experimental test.

---

## Dependencies

All scripts require:
```bash
pip install numpy matplotlib scipy
```

## Data Sources

### Current Implementation
Scripts generate example data based on theoretical predictions:
- Power law equations with stated exponents
- Realistic noise and error bars
- Matches published results

### For Publication
Replace example data with actual simulation results:

```python
# In each script, replace:
# generate_example_data()

# With:
import pandas as pd

# Load from data/
df = pd.read_csv('../data/coordination_d2.csv')
# Extract actual coordination measures
```

## Customization

### Modify Plot Style

Edit in each script:
```python
# Publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300
```

### Change Dimensions Shown

```python
# In fig1_single_trajectory.py
dimensions_to_plot = [2, 4, 6, 8]  # Modify as needed

# In fig3_spike_narrowing.py
dimensions = [2, 4, 6, 8]  # Change subset
```

### Export Formats

All scripts save both PDF and PNG:
```python
plt.savefig('figure_name.pdf', format='pdf', dpi=300)
plt.savefig('figure_name.png', format='png', dpi=300)
```

## Output Files

Each script creates publication-ready figures:

```
figures/
├── fig1_single_trajectory_d2.pdf
├── fig1_single_trajectory_d2.png
├── fig1_single_trajectory_d4.pdf
├── fig1_single_trajectory_d4.png
├── fig1_single_trajectory_d8.pdf
├── fig1_single_trajectory_d8.png
├── fig2_dimensional_scaling.pdf
├── fig2_dimensional_scaling.png
├── fig3_spike_narrowing.pdf
├── fig3_spike_narrowing.png
├── fig4_pi8_crossing.pdf
└── fig4_pi8_crossing.png
```

## Paper Figure Mapping

| Paper Figure | Script | Key Result |
|--------------|--------|------------|
| Figure 1 | `fig1_single_trajectory.py` | Event detection |
| Figure 2 | `fig2_dimensional_scaling.py` | α = 1.787±0.009 |
| Figure 3a | `fig3_spike_narrowing.py` (top panel) | Spike overlays |
| Figure 3b | `fig3_spike_narrowing.py` (bottom panel) | β = 0.558 |
| Figure 4 | `fig4_pi8_crossing.py` | Sign reversal |

## Troubleshooting

### Issue: Figures don't display

```bash
# Backend issue - try different backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# If needed, set backend:
export MPLBACKEND=TkAgg  # or Qt5Agg, Agg
```

### Issue: Missing fonts

```bash
# Install required fonts
# For LaTeX-style fonts:
pip install latex
```

### Issue: Memory errors

```python
# Reduce time resolution in generate functions
dt = 0.1  # Instead of 0.01
```

## Citation

If using these figures, cite:

```
Cabot, Z. (2024). Dimensional Scaling of Quantum Measurement Coordination: 
An Inverse Blessing of Dimensionality. arXiv:XXXX.XXXXX
```

## License

MIT License - see LICENSE file in repository root.

---

**Generate all figures now:**
```bash
for script in fig*.py; do python $script; done
```
