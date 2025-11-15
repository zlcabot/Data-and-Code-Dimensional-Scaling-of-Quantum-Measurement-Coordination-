"""
event_detection.py - Coordination Measure Computation

Compute Φ_d, Φ_f, Φ_a from quantum trajectories and detect measurement events.
Implements the triadic coordination capacity framework.

CRITICAL: Φ_d is computed from system-apparatus correlation (mutual information),
NOT from system purity. Computing from purity = "duration trap" = zero agency.

Author: Zayin
Date: 2025
License: MIT
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.signal import find_peaks
from scipy.integrate import trapz


class CoordinationMeasures:
    """
    Compute triadic coordination measures from quantum trajectories.
    
    The three temporal aspects:
    - Duration (Φ_d): Actualized correlation (past → present)
    - Frequency (Φ_f): Remaining superposition (future possibilities)
    - Agency (Φ_a): Coordination intensity (organizing present)
    
    Key Principle:
    "Don't track state evolution toward completion; 
     detect coordination events through agency spikes."
    """
    
    def __init__(self, d: int):
        """
        Initialize coordination measure calculator.
        
        Parameters
        ----------
        d : int
            Hilbert space dimension
        """
        self.d = d
        
    def compute_from_trajectory(
        self,
        rho_S_traj: List[np.ndarray],
        rho_A_traj: List[np.ndarray],
        rho_SA_traj: List[np.ndarray],
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Φ_d, Φ_f, Φ_a from full trajectory.
        
        Parameters
        ----------
        rho_S_traj : list of ndarray
            System density matrices over time
        rho_A_traj : list of ndarray
            Apparatus density matrices over time
        rho_SA_traj : list of ndarray
            Joint system-apparatus density matrices over time
        dt : float
            Time step
            
        Returns
        -------
        Phi_d, Phi_f, Phi_a : ndarray
            Arrays of coordination measures over time
        """
        T = len(rho_S_traj)
        
        Phi_d = np.zeros(T)
        Phi_f = np.zeros(T)
        Phi_a = np.zeros(T)
        
        # Compute initial energy uncertainty for normalization
        H_0 = self._construct_hamiltonian()
        Delta_H_0 = self._energy_uncertainty(rho_S_traj[0], H_0)
        
        for t in range(T):
            # DURATION: From system-apparatus CORRELATION
            Phi_d[t] = self.compute_duration(
                rho_S_traj[t], 
                rho_A_traj[t], 
                rho_SA_traj[t]
            )
            
            # FREQUENCY: From system coherence/superposition
            Phi_f[t] = self.compute_frequency(
                rho_S_traj[t],
                H_0,
                Delta_H_0
            )
            
            # AGENCY: From duration-frequency tension
            Phi_a[t] = self.compute_agency(Phi_d[t], Phi_f[t])
        
        return Phi_d, Phi_f, Phi_a
    
    def compute_duration(
        self,
        rho_S: np.ndarray,
        rho_A: np.ndarray,
        rho_SA: np.ndarray
    ) -> float:
        """
        Compute duration coordination Φ_d.
        
        THE RIGHT WAY: Uses mutual information I(S:A)
        THE WRONG WAY: Would use Tr(ρ_S²) - DO NOT DO THIS!
        
        Parameters
        ----------
        rho_S : ndarray
            System reduced density matrix
        rho_A : ndarray
            Apparatus reduced density matrix
        rho_SA : ndarray
            Joint system-apparatus density matrix
            
        Returns
        -------
        Phi_d : float
            Duration coordination [0, 1]
        """
        # von Neumann entropies
        S_S = self._von_neumann_entropy(rho_S)
        S_A = self._von_neumann_entropy(rho_A)
        S_SA = self._von_neumann_entropy(rho_SA)
        
        # Mutual information I(S:A) = S(S) + S(A) - S(SA)
        MI = S_S + S_A - S_SA
        
        # Normalize by maximum possible MI for dimension d
        MI_max = np.log2(self.d)
        
        # Duration = normalized mutual information
        Phi_d = MI / MI_max
        
        # Ensure in valid range [0, 1]
        return np.clip(Phi_d, 0, 1)
    
    def compute_frequency(
        self,
        rho_S: np.ndarray,
        H: np.ndarray,
        Delta_H_0: float
    ) -> float:
        """
        Compute frequency coordination Φ_f.
        
        Measures remaining quantum coherence (superposition).
        
        Parameters
        ----------
        rho_S : ndarray
            System density matrix
        H : ndarray
            Hamiltonian operator
        Delta_H_0 : float
            Initial energy uncertainty (for normalization)
            
        Returns
        -------
        Phi_f : float
            Frequency coordination [0, 1]
        """
        # Current energy uncertainty
        Delta_H = self._energy_uncertainty(rho_S, H)
        
        # Normalize by initial uncertainty
        if Delta_H_0 > 1e-10:
            Phi_f = Delta_H / Delta_H_0
        else:
            Phi_f = 0.0
        
        # Ensure in valid range
        return np.clip(Phi_f, 0, 1)
    
    def compute_agency(self, Phi_d: float, Phi_f: float) -> float:
        """
        Compute agency coordination Φ_a.
        
        Agency measures coordination intensity - peaks when duration and
        frequency are balanced (Φ_d ≈ Φ_f ≈ 0.5).
        
        Formula: Φ_a = 4√[Φ_d(1-Φ_d)·Φ_f(1-Φ_f)] / (Φ_d + Φ_f)
        
        Parameters
        ----------
        Phi_d : float
            Duration coordination
        Phi_f : float
            Frequency coordination
            
        Returns
        -------
        Phi_a : float
            Agency coordination [0, 1]
        """
        # Avoid division by zero
        if Phi_d + Phi_f < 1e-10:
            return 0.0
        
        # Compute "tensions" - how far from extremes
        tension_d = Phi_d * (1 - Phi_d)
        tension_f = Phi_f * (1 - Phi_f)
        
        # Both must be non-negative
        if tension_d < 0 or tension_f < 0:
            return 0.0
        
        # Agency = geometric mean of tensions, normalized
        numerator = 4 * np.sqrt(tension_d * tension_f)
        denominator = Phi_d + Phi_f
        
        Phi_a = numerator / denominator
        
        # Ensure in valid range
        return np.clip(Phi_a, 0, 1)
    
    def _von_neumann_entropy(self, rho: np.ndarray, eps: float = 1e-12) -> float:
        """
        Compute von Neumann entropy S = -Tr(ρ log₂ ρ).
        
        Parameters
        ----------
        rho : ndarray
            Density matrix
        eps : float
            Small number to avoid log(0)
            
        Returns
        -------
        S : float
            von Neumann entropy in bits
        """
        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho)
        
        # Filter out numerical zeros
        eigenvalues = eigenvalues[eigenvalues > eps]
        
        # S = -Σ λ_i log₂(λ_i)
        S = -np.sum(eigenvalues * np.log2(eigenvalues))
        
        return S
    
    def _energy_uncertainty(self, rho: np.ndarray, H: np.ndarray) -> float:
        """
        Compute energy uncertainty ΔH = √(⟨H²⟩ - ⟨H⟩²).
        
        Parameters
        ----------
        rho : ndarray
            Density matrix
        H : ndarray
            Hamiltonian operator
            
        Returns
        -------
        Delta_H : float
            Energy uncertainty
        """
        # Expectation values
        H_avg = np.real(np.trace(rho @ H))
        H2_avg = np.real(np.trace(rho @ H @ H))
        
        # Uncertainty
        variance = H2_avg - H_avg**2
        
        # Ensure non-negative (numerical errors can make it slightly negative)
        variance = max(variance, 0.0)
        
        Delta_H = np.sqrt(variance)
        
        return Delta_H
    
    def _construct_hamiltonian(self) -> np.ndarray:
        """
        Construct simple Hamiltonian for energy uncertainty calculation.
        
        Uses number operator: H = N = Σ n|n⟩⟨n|
        
        Returns
        -------
        H : ndarray
            Hamiltonian operator
        """
        H = np.diag(np.arange(self.d, dtype=float))
        return H


class EventDetector:
    """
    Detect measurement events from agency spikes.
    
    The key insight: measurement is not gradual evolution toward eigenstates,
    but discrete coordination events marked by agency spikes.
    """
    
    def __init__(self, dt: float):
        """
        Initialize event detector.
        
        Parameters
        ----------
        dt : float
            Time step (μs)
        """
        self.dt = dt
    
    def find_agency_peak(
        self,
        Phi_a: np.ndarray,
        min_height: float = 0.1,
        min_prominence: float = 0.05
    ) -> Dict[str, float]:
        """
        Find the main agency peak (measurement event).
        
        Parameters
        ----------
        Phi_a : ndarray
            Agency coordination over time
        min_height : float
            Minimum peak height
        min_prominence : float
            Minimum peak prominence
            
        Returns
        -------
        event_info : dict
            Dictionary containing:
            - peak_idx: Index of peak
            - peak_time: Time of peak (μs)
            - peak_height: Height of peak
            - S_coord: Integrated coordination action
        """
        # Find all peaks
        peaks, properties = find_peaks(
            Phi_a,
            height=min_height,
            prominence=min_prominence
        )
        
        if len(peaks) == 0:
            # No peaks found
            return {
                'peak_idx': -1,
                'peak_time': -1,
                'peak_height': 0.0,
                'S_coord': 0.0
            }
        
        # Get highest peak
        peak_heights = properties['peak_heights']
        main_peak_idx = peaks[np.argmax(peak_heights)]
        
        # Compute coordination action (integrate agency)
        S_coord = self.integrate_agency(Phi_a)
        
        return {
            'peak_idx': int(main_peak_idx),
            'peak_time': main_peak_idx * self.dt,
            'peak_height': float(Phi_a[main_peak_idx]),
            'S_coord': S_coord
        }
    
    def integrate_agency(self, Phi_a: np.ndarray) -> float:
        """
        Compute coordination action S_coord = ∫ Φ_a(t) dt.
        
        Parameters
        ----------
        Phi_a : ndarray
            Agency coordination over time
            
        Returns
        -------
        S_coord : float
            Coordination action
        """
        # Trapezoidal integration
        S_coord = trapz(Phi_a, dx=self.dt)
        
        return S_coord
    
    def find_event_window(
        self,
        Phi_a: np.ndarray,
        peak_idx: int,
        threshold: float = 0.1
    ) -> Tuple[int, int]:
        """
        Find the time window around the agency peak where Φ_a > threshold.
        
        This defines the temporal extent of the coordination event.
        
        Parameters
        ----------
        Phi_a : ndarray
            Agency coordination over time
        peak_idx : int
            Index of peak
        threshold : float
            Threshold for event window
            
        Returns
        -------
        start_idx, end_idx : tuple of int
            Start and end indices of event window
        """
        if peak_idx < 0 or peak_idx >= len(Phi_a):
            return 0, 0
        
        # Find where Φ_a drops below threshold before peak
        start_idx = peak_idx
        while start_idx > 0 and Phi_a[start_idx] > threshold:
            start_idx -= 1
        
        # Find where Φ_a drops below threshold after peak
        end_idx = peak_idx
        while end_idx < len(Phi_a) - 1 and Phi_a[end_idx] > threshold:
            end_idx += 1
        
        return start_idx, end_idx


# Convenience functions
def compute_coordination_measures(
    rho_S_traj: List[np.ndarray],
    rho_A_traj: List[np.ndarray],
    rho_SA_traj: List[np.ndarray],
    d: int,
    dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute all coordination measures from trajectory.
    
    Parameters
    ----------
    rho_S_traj, rho_A_traj, rho_SA_traj : list of ndarray
        Density matrix trajectories
    d : int
        Hilbert space dimension
    dt : float
        Time step
        
    Returns
    -------
    Phi_d, Phi_f, Phi_a : ndarray
        Coordination measures over time
        
    Example
    -------
    >>> Phi_d, Phi_f, Phi_a = compute_coordination_measures(
    ...     rho_S, rho_A, rho_SA, d=2, dt=0.01
    ... )
    """
    measures = CoordinationMeasures(d)
    return measures.compute_from_trajectory(
        rho_S_traj, rho_A_traj, rho_SA_traj, dt
    )


def find_measurement_event(
    Phi_a: np.ndarray,
    dt: float
) -> Dict[str, float]:
    """
    Find the main measurement event from agency trajectory.
    
    Parameters
    ----------
    Phi_a : ndarray
        Agency coordination over time
    dt : float
        Time step
        
    Returns
    -------
    event_info : dict
        Event information (peak time, height, S_coord)
        
    Example
    -------
    >>> event = find_measurement_event(Phi_a, dt=0.01)
    >>> print(f"Event at t={event['peak_time']:.2f} μs")
    """
    detector = EventDetector(dt)
    return detector.find_agency_peak(Phi_a)


if __name__ == "__main__":
    # Example usage
    print("Event Detection Example")
    print("=" * 60)
    
    # Generate example trajectory (would normally use simulate_trajectory.py)
    print("Generating synthetic agency trajectory...")
    
    # Synthetic data: Gaussian peak representing measurement event
    dt = 0.01
    times = np.linspace(0, 100, 10000)
    
    # Create synthetic Φ_a with peak at t=50μs
    peak_time = 50.0
    peak_width = 5.0
    Phi_a = 0.8 * np.exp(-((times - peak_time) / peak_width)**2)
    
    # Detect event
    detector = EventDetector(dt)
    event = detector.find_agency_peak(Phi_a)
    
    print("\n" + "=" * 60)
    print("EVENT DETECTION RESULTS:")
    print(f"  Peak time: {event['peak_time']:.2f} μs")
    print(f"  Peak height (Φ_a): {event['peak_height']:.3f}")
    print(f"  Coordination action (S_coord): {event['S_coord']:.3f}")
    
    # Find event window
    start_idx, end_idx = detector.find_event_window(
        Phi_a, event['peak_idx'], threshold=0.1
    )
    event_duration = (end_idx - start_idx) * dt
    
    print(f"  Event duration: {event_duration:.2f} μs")
    print(f"  Event window: [{start_idx*dt:.2f}, {end_idx*dt:.2f}] μs")
    
    print("\n✓ Example complete!")
    print("  Use with simulate_trajectory.py to analyze real quantum measurements")
    print("=" * 60)
