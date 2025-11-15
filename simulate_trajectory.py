"""
simulate_trajectory.py - Core SME Integration

Generate quantum measurement trajectories using stochastic master equation (SME).
Implements continuous weak measurement of quantum systems coupled to measurement apparatus.

Author: Zayin
Date: 2025
License: MIT
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple, List, Optional
import warnings


class QuantumTrajectory:
    """
    Generate and store quantum measurement trajectories.
    
    Critical Implementation:
    - Tracks SYSTEM-APPARATUS CORRELATION via joint density matrix ρ_SA
    - NOT just system purity (that's the "duration trap")
    - Enables detection of coordination events through agency spikes
    """
    
    def __init__(
        self,
        d: int,
        gamma: float,
        chi: float,
        kappa: float,
        T: float,
        dt: float,
        initial_state: Optional[str] = 'superposition'
    ):
        """
        Initialize quantum trajectory simulator.
        
        Parameters
        ----------
        d : int
            Hilbert space dimension (2-8 recommended)
        gamma : float
            Measurement strength in MHz (0.1 - 10.0 typical)
        chi : float
            Coherent coupling strength (system-apparatus)
        kappa : float
            Decoherence rate (apparatus to environment)
        T : float
            Total simulation time in microseconds (~100 typical)
        dt : float
            Integration timestep in microseconds (~0.01 typical)
        initial_state : str, optional
            Initial system state: 'superposition', 'ground', or 'excited'
        """
        # Store parameters
        self.d = d
        self.gamma = gamma
        self.chi = chi
        self.kappa = kappa
        self.T = T
        self.dt = dt
        
        # Compute number of timesteps
        self.n_steps = int(T / dt)
        self.times = np.linspace(0, T, self.n_steps)
        
        # Initialize state
        self.rho_S = self._initialize_system_state(initial_state)
        self.rho_A = self._initialize_apparatus_state()
        self.rho_SA = np.kron(self.rho_S, self.rho_A)  # Joint state
        
        # Storage for trajectory
        self.rho_S_trajectory = []
        self.rho_A_trajectory = []
        self.rho_SA_trajectory = []
        self.measurement_record = []
        
        # Operators
        self._construct_operators()
        
    def _initialize_system_state(self, state_type: str) -> np.ndarray:
        """Initialize system density matrix."""
        if state_type == 'superposition':
            # Equal superposition: |ψ⟩ = (|0⟩ + |1⟩ + ... + |d-1⟩)/√d
            psi = np.ones(self.d) / np.sqrt(self.d)
            rho = np.outer(psi, psi.conj())
        elif state_type == 'ground':
            # Ground state: |0⟩
            rho = np.zeros((self.d, self.d))
            rho[0, 0] = 1.0
        elif state_type == 'excited':
            # First excited state: |1⟩
            rho = np.zeros((self.d, self.d))
            rho[1, 1] = 1.0
        else:
            raise ValueError(f"Unknown initial state: {state_type}")
        
        return rho
    
    def _initialize_apparatus_state(self) -> np.ndarray:
        """Initialize apparatus in ground state."""
        rho_A = np.zeros((self.d, self.d))
        rho_A[0, 0] = 1.0
        return rho_A
    
    def _construct_operators(self):
        """Construct quantum operators for system and apparatus."""
        # Number operator for system
        self.N_S = np.diag(np.arange(self.d))
        
        # Ladder operators for apparatus
        self.a = np.zeros((self.d, self.d))
        for i in range(self.d - 1):
            self.a[i, i + 1] = np.sqrt(i + 1)
        self.a_dag = self.a.T
        self.N_A = self.a_dag @ self.a
        
        # Pauli-like operators for measurement
        self.sigma_x = self._construct_sigma_x()
        self.sigma_z = np.diag(np.arange(self.d) - (self.d - 1) / 2)
        
        # Identity operators
        self.I_S = np.eye(self.d)
        self.I_A = np.eye(self.d)
        
    def _construct_sigma_x(self) -> np.ndarray:
        """Construct generalized σ_x for dimension d."""
        sigma_x = np.zeros((self.d, self.d))
        for i in range(self.d - 1):
            sigma_x[i, i + 1] = 1.0
            sigma_x[i + 1, i] = 1.0
        return sigma_x
    
    def evolve(self, verbose: bool = True) -> None:
        """
        Evolve the quantum trajectory using SME.
        
        This implements the Lindblad master equation with continuous measurement.
        The key is tracking system-apparatus correlation, not just system purity.
        """
        if verbose:
            print(f"Evolving trajectory: d={self.d}, γ={self.gamma:.2f} MHz, "
                  f"χ/κ={self.chi/self.kappa:.2f}")
        
        # Store initial state
        self.rho_S_trajectory.append(self.rho_S.copy())
        self.rho_A_trajectory.append(self.rho_A.copy())
        self.rho_SA_trajectory.append(self.rho_SA.copy())
        
        # Evolution loop
        for step in range(1, self.n_steps):
            # Evolve one timestep
            self.rho_SA = self._evolve_timestep(self.rho_SA)
            
            # Extract reduced density matrices
            self.rho_S = self._partial_trace_A(self.rho_SA)
            self.rho_A = self._partial_trace_S(self.rho_SA)
            
            # Store
            self.rho_S_trajectory.append(self.rho_S.copy())
            self.rho_A_trajectory.append(self.rho_A.copy())
            self.rho_SA_trajectory.append(self.rho_SA.copy())
            
            # Progress
            if verbose and step % 1000 == 0:
                print(f"  Step {step}/{self.n_steps} ({100*step/self.n_steps:.1f}%)")
        
        if verbose:
            print(f"✓ Evolution complete: {self.n_steps} timesteps")
    
    def _evolve_timestep(self, rho: np.ndarray) -> np.ndarray:
        """
        Evolve density matrix by one timestep using Lindblad equation.
        
        dρ/dt = -i[H, ρ] + κ L[a]ρ + γ L[N_S ⊗ X_A]ρ
        
        where L[c]ρ = c ρ c† - (1/2){c†c, ρ}
        """
        # Hamiltonian (coherent coupling between system and apparatus)
        H = self.chi * np.kron(self.N_S, self.sigma_x)
        
        # Unitary part: -i[H, ρ]
        commutator = H @ rho - rho @ H
        drho_unitary = -1j * commutator
        
        # Dissipation: apparatus couples to environment
        L_a = np.kron(self.I_S, self.a)
        drho_dissipation = self.kappa * self._lindblad_term(L_a, rho)
        
        # Measurement: continuous monitoring
        L_meas = np.kron(self.N_S, self.sigma_x)
        drho_measurement = self.gamma * self._lindblad_term(L_meas, rho)
        
        # Total evolution
        drho_dt = drho_unitary + drho_dissipation + drho_measurement
        
        # Euler integration (could upgrade to RK4 if needed)
        rho_new = rho + self.dt * drho_dt
        
        # Ensure hermiticity and normalization
        rho_new = 0.5 * (rho_new + rho_new.conj().T)
        rho_new = rho_new / np.trace(rho_new)
        
        return rho_new
    
    def _lindblad_term(self, L: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """
        Compute Lindblad superoperator term: L[c]ρ = c ρ c† - (1/2){c†c, ρ}
        """
        L_dag = L.conj().T
        L_dag_L = L_dag @ L
        
        term1 = L @ rho @ L_dag
        term2 = 0.5 * (L_dag_L @ rho + rho @ L_dag_L)
        
        return term1 - term2
    
    def _partial_trace_A(self, rho_SA: np.ndarray) -> np.ndarray:
        """Trace out apparatus to get system reduced density matrix."""
        rho_S = np.zeros((self.d, self.d), dtype=complex)
        
        for i in range(self.d):
            for j in range(self.d):
                # Sum over apparatus indices
                for k in range(self.d):
                    idx1 = i * self.d + k
                    idx2 = j * self.d + k
                    rho_S[i, j] += rho_SA[idx1, idx2]
        
        return rho_S
    
    def _partial_trace_S(self, rho_SA: np.ndarray) -> np.ndarray:
        """Trace out system to get apparatus reduced density matrix."""
        rho_A = np.zeros((self.d, self.d), dtype=complex)
        
        for i in range(self.d):
            for j in range(self.d):
                # Sum over system indices
                for k in range(self.d):
                    idx1 = k * self.d + i
                    idx2 = k * self.d + j
                    rho_A[i, j] += rho_SA[idx1, idx2]
        
        return rho_A
    
    def get_trajectories(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Get full trajectory data.
        
        Returns
        -------
        rho_S_trajectory : list of ndarray
            System reduced density matrices
        rho_A_trajectory : list of ndarray
            Apparatus reduced density matrices
        rho_SA_trajectory : list of ndarray
            Joint system-apparatus density matrices
        """
        return (
            self.rho_S_trajectory,
            self.rho_A_trajectory,
            self.rho_SA_trajectory
        )


# Convenience function
def generate_measurement_trajectory(
    d: int = 2,
    gamma: float = 1.0,
    chi: float = 1.0,
    kappa: float = 1.0,
    T: float = 100.0,
    dt: float = 0.01,
    initial_state: str = 'superposition',
    verbose: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Generate a quantum measurement trajectory.
    
    Parameters
    ----------
    d : int
        Hilbert space dimension (2-8)
    gamma : float
        Measurement strength (MHz)
    chi : float
        Coherent coupling strength
    kappa : float
        Decoherence rate
    T : float
        Total time (μs)
    dt : float
        Timestep (μs)
    initial_state : str
        Initial state type
    verbose : bool
        Print progress messages
    
    Returns
    -------
    rho_S, rho_A, rho_SA : tuple of lists
        Density matrix trajectories
        
    Example
    -------
    >>> rho_S, rho_A, rho_SA = generate_measurement_trajectory(
    ...     d=2, gamma=1.0, chi=1.0, kappa=1.0, T=100, dt=0.01
    ... )
    >>> print(f"Generated {len(rho_S)} timesteps")
    """
    traj = QuantumTrajectory(d, gamma, chi, kappa, T, dt, initial_state)
    traj.evolve(verbose=verbose)
    return traj.get_trajectories()


if __name__ == "__main__":
    # Example usage
    print("Generating example quantum measurement trajectory...")
    print("=" * 60)
    
    rho_S, rho_A, rho_SA = generate_measurement_trajectory(
        d=2,           # Qubit
        gamma=1.0,     # 1 MHz measurement
        chi=1.0,       # Balanced coupling (chi=kappa is optimal!)
        kappa=1.0,     # 1 MHz decoherence
        T=100.0,       # 100 μs
        dt=0.01,       # 0.01 μs timestep
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY:")
    print(f"  Total timesteps: {len(rho_S)}")
    print(f"  System dimension: {rho_S[0].shape[0]}")
    print(f"  Initial purity: {np.trace(rho_S[0] @ rho_S[0]):.4f}")
    print(f"  Final purity: {np.trace(rho_S[-1] @ rho_S[-1]):.4f}")
    print("\n✓ Success! Use event_detection.py to compute Φ_d, Φ_f, Φ_a")
    print("=" * 60)
