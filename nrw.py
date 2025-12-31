"""
Nicolson-Ross-Weir (NRW) Algorithm Implementation

This module implements the NRW method for computing scattering parameters (S-parameters)
from material properties and vice versa. Used for microwave waveguide measurements
to extract complex permittivity and permeability from reflection and transmission data.

Key functions:
- nrw_direct: Forward calculation of S11/S21 from material properties
- nrw_inverse: Inverse calculation of material properties from S-parameters

Supports TE10 mode in rectangular waveguides. Assumes non-magnetic materials (μ_r = 1)
for permittivity extraction in inverse mode.

Based on: Nicolson, A.M. and Ross, G.F., 1970. Measurement of the intrinsic properties
of materials by time-domain techniques. IEEE Transactions on Instrumentation and Measurement.
"""

import torch

# Physical constants
EPS_0 = 8.85418782e-12
MU_0 = 4 * torch.pi * 1e-7

def nrw_direct(d: float, a: float,
                        frequencies: torch.Tensor, 
                        eps_r: torch.Tensor, 
                        mu_r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute S-parameters (S11, S21) for TE10 mode.
    
    Args:
        d: sample thickness (m)
        a: waveguide width (m)
        frequencies: frequency array (Hz)
        eps_r: relative permittivity (complex)
        mu_r: relative permeability (complex)
        
    Returns:
        S11, S21: complex S-parameters
    """
    omega = 2 * torch.pi * frequencies
    k0_squared = omega**2 * EPS_0 * MU_0
    kt_mn = torch.pi / a
    
    beta1_e = torch.sqrt(k0_squared - kt_mn**2)
    beta1_s = torch.sqrt(k0_squared * eps_r * mu_r - kt_mn**2)
    
    Z_e = omega * MU_0 / beta1_e
    Z_s = omega * MU_0 * mu_r / beta1_s
    
    Gamma = (Z_s - Z_e) / (Z_s + Z_e)
    P = torch.exp(-1j * beta1_s * d)
    
    S11 = Gamma * (1 - P**2) / (1 - Gamma**2 * P**2)
    S21 = P * (1 - Gamma**2) / (1 - Gamma**2 * P**2)
    
    return S11, S21

def nrw_inverse(d: float, a: float,
                       frequencies: torch.Tensor,
                       S11: torch.Tensor,
                       S21: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    NRW (Nicolson-Ross-Weir) method for TE10 mode.
    Extract material properties from S-parameters. Might fail for certain configurations.
    """
    V1, V2 = S21 + S11, S21 - S11
    X = (1 - V1 * V2) / (V1 - V2 + 1e-16)
    Gamma = torch.where(torch.abs(X + torch.sqrt(X**2 - 1)) > 1, 
                       X - torch.sqrt(X**2 - 1), 
                       X + torch.sqrt(X**2 - 1))
    P = (V1 - Gamma) / (1 - Gamma * V1)
    beta1_s = -torch.log(P) / (1j * d)
    
    omega = 2 * torch.pi * frequencies
    k0_squared = omega**2 * EPS_0 * MU_0
    kt_mn = torch.pi / a
    beta1_e = torch.sqrt(k0_squared - kt_mn**2)
    
    mu_r = (1 + Gamma) / (1 - Gamma) * beta1_s / beta1_e
    epsilon_r = (beta1_s**2 + kt_mn**2) / (k0_squared * mu_r)
    
    return epsilon_r, mu_r