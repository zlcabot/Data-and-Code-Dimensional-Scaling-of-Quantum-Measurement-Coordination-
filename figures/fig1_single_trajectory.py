"""
Figure 1: Single Trajectory with Coordination Spike

Demonstrates a single quantum measurement trajectory showing:
- Coordination capacity Phi_a(t) with characteristic spike
- Duration (Phi_d) and Frequency (Phi_f) evolution
- Measurement event detection

Usage:
    python fig1_single_trajectory.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 300


def generate_example_trajectory(d=2, T=20, dt=0.01):
    """
    Generate example coordination measures for visualization.
    
    In actual use, replace with:
        from simulate_trajectory import generate_measurement_trajectory
        from event_detection import compute_coordination_measures
    
    Parameters:
    -----------
    d : int
        Hilbert space dimension
    T : float
        Total time (microseconds)
    dt : float
        Time step
        
    Returns:
    --------
    time, Phi_d, Phi_f, Phi_a : arrays
    """
    time = np.arange(0, T, dt)
    
    # Duration: sigmoid rise
    t_peak = 8.0 / np.sqrt(d)
    width = 2.0 * d**(-0.558)
    Phi_d = 1 / (1 + np.exp(-(time - t_peak) / width))
    
    # Frequency: complementary
    Phi_f = 1 - Phi_d
    
    # Add noise
    np.random.seed(42)
    Phi_d += np.random.normal(0, 0.01, len(time))
    Phi_f += np.random.normal(0, 0.01, len(time))
    Phi_d = np.clip(Phi_d, 0, 1)
    Phi_f = np.clip(Phi_f, 0, 1)
    
    # Coordination capacity (agency spike)
    numerator = 4 * np.sqrt(Phi_d * (1 - Phi_d) * Phi_f * (1 - Phi_f))
    denominator = Phi_d + Phi_f + 1e-10
    Phi_a = numerator / denominator
    
    return time, Phi_d, Phi_f, Phi_a


def create_figure(d=2, save=True):
    """
    Create Figure 1: Single trajectory with coordination spike.
    
    Parameters:
    -----------
    d : int
        Dimension to visualize
    save : bool
        Whether to save figure to file
    """
    
    # Generate trajectory
    time, Phi_d, Phi_f, Phi_a = generate_example_trajectory(d=d)
    
    # Find peak
    peak_idx = np.argmax(Phi_a)
    peak_time = time[peak_idx]
    peak_value = Phi_a[peak_idx]
    
    # Compute coordination action
    S_coord = np.trapz(Phi_a, time)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[1, 1.2], hspace=0.3)
    
    # === TOP PANEL: Duration and Frequency ===
    ax1 = fig.add_subplot(gs[0])
    
    ax1.plot(time, Phi_d, '-', color='#2E86AB', linewidth=2.5, 
             label=r'Duration $\Phi_d$ (actualized correlation)')
    ax1.plot(time, Phi_f, '-', color='#A23B72', linewidth=2.5,
             label=r'Frequency $\Phi_f$ (remaining superposition)')
    
    # Mark the crossing point
    crossing_idx = np.argmin(np.abs(Phi_d - Phi_f))
    ax1.plot(time[crossing_idx], Phi_d[crossing_idx], 'o', 
             color='#E63946', markersize=10, zorder=5,
             label='Crossing point')
    
    ax1.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_ylabel('Coordination Measures', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, time[-1])
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='center right', frameon=True, fontsize=10)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.set_title(f'Quantum Measurement Trajectory (d={d})', 
                  fontsize=13, fontweight='bold', pad=10)
    
    # === BOTTOM PANEL: Coordination Capacity (Agency Spike) ===
    ax2 = fig.add_subplot(gs[1])
    
    ax2.fill_between(time, 0, Phi_a, alpha=0.3, color='#F77F00')
    ax2.plot(time, Phi_a, '-', color='#F77F00', linewidth=2.5,
             label=r'Coordination capacity $\Phi_c$ (agency spike)')
    
    # Mark the peak
    ax2.plot(peak_time, peak_value, 'o', color='#E63946', 
             markersize=12, zorder=5, label='Measurement event')
    ax2.axvline(peak_time, color='#E63946', linestyle='--', 
                linewidth=1.5, alpha=0.7)
    
    # Add annotation
    ax2.annotate(f'Peak at t = {peak_time:.2f} μs\n$\\Phi_c$ = {peak_value:.3f}',
                xy=(peak_time, peak_value),
                xytext=(peak_time + 3, peak_value - 0.1),
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='#E63946', lw=2))
    
    ax2.set_xlabel('Time (μs)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Coordination Capacity', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, time[-1])
    ax2.set_ylim(-0.05, max(Phi_a) * 1.1)
    ax2.legend(loc='upper right', frameon=True, fontsize=10)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add coordination action as text
    ax2.text(0.02, 0.95, 
             f'Coordination action: $S_{{coord}}$ = {S_coord:.3f}',
             transform=ax2.transAxes,
             fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'fig1_single_trajectory_d{d}.pdf', 
                    format='pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'fig1_single_trajectory_d{d}.png', 
                    format='png', bbox_inches='tight', dpi=300)
        print(f"✓ Saved: fig1_single_trajectory_d{d}.pdf/.png")
    
    return fig


def main():
    """Generate Figure 1 for multiple dimensions."""
    
    print("Generating Figure 1: Single Trajectory Examples")
    print("=" * 50)
    
    # Generate for d=2 (main figure)
    print("\nCreating figure for d=2 (qubit)...")
    fig = create_figure(d=2, save=True)
    
    # Optionally create for other dimensions
    print("\nCreating comparison figures...")
    for d in [4, 8]:
        print(f"  d={d}...")
        create_figure(d=d, save=True)
    
    print("\n" + "=" * 50)
    print("✓ Figure 1 generation complete!")
    print("\nKey features demonstrated:")
    print("  • Duration (Phi_d) rises as correlation builds")
    print("  • Frequency (Phi_f) falls as superposition decays")
    print("  • Coordination capacity (Phi_a) peaks at measurement event")
    print("  • Peak narrows with increasing dimension")
    
    plt.show()


if __name__ == '__main__':
    main()
