"""
Generate sample CSV data templates for all dimensions.

This creates realistic-looking sample data based on the theoretical
expectations. Replace with actual simulation data.

Usage:
    python generate_sample_data.py
"""

import numpy as np
import pandas as pd

def generate_sample_trajectory(d, T=20.0, dt=0.1):
    """
    Generate sample coordination measures for dimension d.
    
    This is a TEMPLATE based on theoretical expectations.
    Replace with actual QuTiP simulation data.
    
    Parameters:
    -----------
    d : int
        Hilbert space dimension
    T : float
        Total time (microseconds)
    dt : float
        Time step (microseconds)
        
    Returns:
    --------
    df : pandas.DataFrame
        Columns: time, Phi_d, Phi_f, Phi_a, S_coord_cumulative
    """
    
    # Time array
    time = np.arange(0, T, dt)
    n_points = len(time)
    
    # Peak time scales with dimension (faster for higher d)
    # Empirically: peak around t ~ 5/sqrt(d)
    t_peak = 5.0 / np.sqrt(d)
    
    # Width scales as d^(-0.558) from paper
    width = 2.0 * d**(-0.558)
    
    # Duration coordination: sigmoid rise to equilibrium
    # Faster rise for higher d
    Phi_d = 1 / (1 + np.exp(-(time - t_peak) / width))
    
    # Frequency coordination: complementary decay
    Phi_f = 1 - Phi_d
    
    # Add realistic noise
    np.random.seed(42 + d)
    Phi_d += np.random.normal(0, 0.01, n_points)
    Phi_f += np.random.normal(0, 0.01, n_points)
    
    # Clip to [0, 1]
    Phi_d = np.clip(Phi_d, 0, 1)
    Phi_f = np.clip(Phi_f, 0, 1)
    
    # Coordination capacity (agency spike)
    # Product measure: peaks when Phi_d and Phi_f are balanced
    numerator = 4 * np.sqrt(Phi_d * (1 - Phi_d) * Phi_f * (1 - Phi_f))
    denominator = Phi_d + Phi_f + 1e-10  # Avoid division by zero
    Phi_a = numerator / denominator
    
    # Cumulative coordination action
    S_coord_cumulative = np.cumsum(Phi_a) * dt
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time,
        'Phi_d': Phi_d,
        'Phi_f': Phi_f,
        'Phi_a': Phi_a,
        'S_coord_cumulative': S_coord_cumulative
    })
    
    return df


def add_header(filename, d):
    """Add descriptive header to CSV file."""
    header = f"""# Sample Coordination Data for d={d}
# Generated from continuous homodyne measurement trajectory
# Measurement strength gamma = 1.0 MHz, chi/kappa = 1.0
# Time unit: microseconds (μs)
# Coordination measures: Phi_d (duration), Phi_f (frequency), Phi_a (agency)
# S_coord_cumulative: Running integral of Phi_a
#
# This is a TEMPLATE file. Replace with actual simulation data.
#
"""
    with open(filename, 'r') as f:
        content = f.read()
    
    with open(filename, 'w') as f:
        f.write(header + content)


def main():
    """Generate sample CSV files for all dimensions."""
    
    dimensions = [2, 3, 4, 5, 6, 7, 8]
    
    print("Generating sample coordination data files...")
    
    for d in dimensions:
        filename = f'data/coordination_d{d}.csv'
        
        # Generate sample data
        df = generate_sample_trajectory(d)
        
        # Save to CSV
        df.to_csv(filename, index=False, float_format='%.3f')
        
        # Add descriptive header
        add_header(filename, d)
        
        print(f"  Created {filename} (S_coord = {df['S_coord_cumulative'].iloc[-1]:.3f})")
    
    # Generate fast_gamma_results (protocol sensitivity)
    print("\nGenerating protocol sensitivity data...")
    
    # Sample data showing protocol independence at high d
    gamma_values = np.linspace(0.5, 2.0, 10)
    data = []
    
    for d in dimensions:
        for gamma in gamma_values:
            # S_coord varies more at low d, less at high d
            base_S = 9.0 * d**(-1.787)
            
            if d <= 4:
                # High sensitivity at low d
                variation = 0.2 * base_S * (gamma - 1.0)
            else:
                # Low sensitivity at high d
                variation = 0.03 * base_S * (gamma - 1.0)
            
            S_coord = base_S + variation + np.random.normal(0, 0.01 * base_S)
            
            data.append({
                'dimension': d,
                'gamma': gamma,
                'S_coord': S_coord,
                'peak_time': 5.0 / np.sqrt(d),
                'peak_amplitude': 0.6 * d**(-0.2)
            })
    
    df_gamma = pd.DataFrame(data)
    df_gamma.to_csv('data/fast_gamma_results.csv', index=False, float_format='%.4f')
    
    print(f"  Created data/fast_gamma_results.csv")
    
    print("\nDone! Sample data files created.")
    print("\n⚠️  IMPORTANT: These are TEMPLATE files.")
    print("   Replace with actual QuTiP simulation data for publication.")


if __name__ == '__main__':
    main()
