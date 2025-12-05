# Examples Directory

Jupyter notebooks demonstrating key analyses and reproducing paper results.

## File Structure

```
examples/
├── README.md                     # This file
├── 01_quick_start.ipynb         # Basic trajectory analysis
├── 02_dimensional_scaling.ipynb # Main scaling result
└── 03_reproduce_figures.ipynb   # Generate all figures
```

---

## Quick Start

### Launch Jupyter

```bash
# From repository root
cd examples/
jupyter notebook
```

Then open any notebook in your browser.

### Run All Notebooks

```bash
# Convert and execute all notebooks
jupyter nbconvert --to notebook --execute 01_quick_start.ipynb
jupyter nbconvert --to notebook --execute 02_dimensional_scaling.ipynb
jupyter nbconvert --to notebook --execute 03_reproduce_figures.ipynb
```

---

## Notebook Descriptions

### 01_quick_start.ipynb

**Goal:** Get started with trajectory analysis in <5 minutes.

**What it does:**
- Generates a single measurement trajectory
- Computes coordination measures (Φ_d, Φ_f, Φ_a)
- Detects measurement event (agency spike)
- Computes coordination action
- Visualizes full trajectory

**Time:** ~5 minutes  
**Prerequisites:** Basic Python knowledge  
**Output:** Interactive plots, summary statistics

**Key learning:**
> "Measurement is a discrete coordination event, not gradual drift."

---

### 02_dimensional_scaling.ipynb

**Goal:** Reproduce the main result S_coord ∝ d^(-1.787±0.009).

**What it does:**
- Loads coordination data for d=2-8
- Fits power law model
- Computes goodness of fit (R²)
- Performs bootstrap validation
- Generates dimensional scaling figure
- Calculates efficiency gain

**Time:** ~10 minutes  
**Prerequisites:** Understanding of power laws, basic statistics  
**Output:** Publication-quality figure, fit parameters

**Key result:**
> Eight-dimensional systems measure ~10× more efficiently than qubits.

---

### 03_reproduce_figures.ipynb

**Goal:** Generate all paper figures in one notebook.

**What it does:**
- Imports figure generation scripts
- Creates Figure 1: Single trajectory
- Creates Figure 2: Dimensional scaling
- Creates Figure 3: Spike narrowing
- Creates Figure 4: π/8 crossing (sign reversal)
- Saves all figures as PDF

**Time:** ~5 minutes  
**Prerequisites:** None (fully automated)  
**Output:** 4 publication-ready PDF figures

**Key feature:**
> One-click generation of all paper figures.

---

## Dependencies

All notebooks require:

```bash
pip install numpy matplotlib scipy jupyter
```

Optional for enhanced features:
```bash
pip install pandas seaborn qutip
```

---

## Usage Patterns

### Pattern 1: Quick Exploration

```bash
# Just want to see results quickly?
jupyter notebook 03_reproduce_figures.ipynb
# Run all cells → Get all figures
```

### Pattern 2: Learning the Framework

```bash
# Want to understand the physics?
jupyter notebook 01_quick_start.ipynb
# Step through cells, read explanations
```

### Pattern 3: Reproducing Analysis

```bash
# Want to verify scaling law?
jupyter notebook 02_dimensional_scaling.ipynb
# Modify data, re-fit, compare results
```

### Pattern 4: Custom Analysis

```bash
# Want to extend the work?
# Copy any notebook, modify code, add cells
# All notebooks are templates for exploration
```

---

## Data Sources

### Current Implementation

Notebooks use **example data** based on theoretical predictions. This is perfect for:
- Learning the framework
- Testing analysis pipeline
- Generating template figures

### For Publication

Replace example data with actual simulations:

```python
# In any notebook, replace synthetic data generation with:
import pandas as pd

# Load real data
df = pd.read_csv('../data/coordination_d2.csv')
time = df['time'].values
Phi_d = df['Phi_d'].values
Phi_f = df['Phi_f'].values
Phi_a = df['Phi_a'].values
S_coord = df['S_coord_cumulative'].iloc[-1]
```

---

## Customization

### Change Dimensions Analyzed

```python
# In 02_dimensional_scaling.ipynb
dimensions = [2, 3, 4, 5, 6, 7, 8]  # Default
dimensions = [2, 4, 8]              # Subset
dimensions = range(2, 11)           # Extended
```

### Adjust Figure Appearance

```python
# In any notebook
plt.rcParams['figure.figsize'] = (12, 8)  # Larger figures
plt.rcParams['font.size'] = 12            # Bigger text
plt.rcParams['figure.dpi'] = 150          # Higher resolution
```

### Bootstrap Iterations

```python
# In 02_dimensional_scaling.ipynb
n_bootstrap = 1000    # Quick (notebook default)
n_bootstrap = 10000   # Paper quality
```

---

## Outputs

### Generated Files

Running all notebooks creates:

```
examples/
├── (original notebooks)
└── (executed notebooks with outputs)

figures/
├── fig1_single_trajectory_d2.pdf
├── fig2_dimensional_scaling.pdf
├── fig3_spike_narrowing.pdf
└── fig4_pi8_crossing.pdf
```

### Interactive Outputs

- All plots displayed inline
- Summary statistics printed
- Validation checks shown
- Key results highlighted

---

## Troubleshooting

### Issue: Jupyter not found

```bash
pip install jupyter
# or
conda install jupyter
```

### Issue: Kernel crashes

```bash
# Reduce memory usage
# In notebooks, use smaller datasets:
dt = 0.1  # Instead of dt = 0.01
T = 10    # Instead of T = 100
```

### Issue: Figures don't save

```bash
# Check write permissions
ls -la ../figures/
# Should show writable directory

# Or change save location in notebooks
save_path = './my_figures/'
```

### Issue: Import errors

```python
# In notebook cell
import sys
print(sys.path)
# Verify ../figures is accessible

# Add path if needed
sys.path.append('../figures')
```

---

## Learning Path

### Beginner

1. Start with 01_quick_start.ipynb
2. Read all markdown cells carefully
3. Run cells one at a time
4. Experiment by changing parameters

### Intermediate

1. Complete 01_quick_start.ipynb
2. Move to 02_dimensional_scaling.ipynb
3. Modify fitting parameters
4. Try different bootstrap settings
5. Generate custom figures

### Advanced

1. Complete all notebooks
2. Replace synthetic data with real simulations
3. Add new analysis cells
4. Extend to additional protocols
5. Publish derivative work

---

## Best Practices

### Before Running

1. ✓ Check data files exist (`../data/`)
2. ✓ Verify dependencies installed
3. ✓ Read notebook introduction
4. ✓ Understand expected outputs

### While Running

1. ✓ Execute cells in order (top to bottom)
2. ✓ Read output after each cell
3. ✓ Check for errors or warnings
4. ✓ Save notebook after modifications

### After Running

1. ✓ Verify figures generated correctly
2. ✓ Check results match expected values
3. ✓ Save any modified notebooks
4. ✓ Document custom analyses

---

## Citation

If using these notebooks, cite:

```
Cabot, Z. (2024). Dimensional Scaling of Quantum Measurement Coordination: 
An Inverse Blessing of Dimensionality. arXiv:XXXX.XXXXX
```

---

## Support

### Issues?

- Check this README first
- Review notebook markdown cells
- Consult main repository README
- Open GitHub issue if problem persists

### Questions?

- Read paper for theoretical background
- Check code comments for implementation details
- Review figure scripts for visualization logic

---

## License

MIT License - see LICENSE file in repository root.

---

**Get started now:**
```bash
cd examples/
jupyter notebook 01_quick_start.ipynb
```

**Happy analyzing!** 🚀
