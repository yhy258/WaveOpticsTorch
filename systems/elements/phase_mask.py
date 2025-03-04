import math
import torch
import torch.nn as nn
import numpy as np



class PhaseMaskRandom(nn.Module):
    def __init__(self, grid, init_type='uniform', std=0.1):
        super().__init__()
        
        if init_type == 'uniform':
            init_phase = 2 * np.pi * torch.rand_like(grid)
        elif init_type == 'zeros':
            init_phase = torch.zeros_like(grid)
        elif init_type == 'normal':
            init_phase = std * torch.randn_like(grid)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        
        # nn.Parameter로 등록 -> 학습 가능
        self.phase = nn.Parameter(init_phase)
        
    def forward(self, field: torch.Tensor):
        """
        field (Tensor; B, C, H, W)
        """
        # Phase mask = e^(j * self.phase)
        phase = torch.exp(1j * self.phase)[None, None] # 1, 1, H, W
        out = field * phase
        return out
    
    
class PhaseMaskZernikeInit(nn.Module):
    def __init__(self, grid, zernike_modes=[(2,0), (2,2)], 
                 coeff_init='random',          
                 ):
        """
        - grid (Tensor; 2, H, W): grid of field (system)
        - zernike_modes: Entire list of Zernike mode (m, n)
        - coeff_init: 'random' -> set each mode's num in random way.
        - max_r: [-max_r, max_r]
        """
        super().__init__()
        self.zernike_modes = zernike_modes
        
        r = torch.sum(grid**2, dim=0) # H, W
        theta = torch.atan2(grid[1], grid[0]) # H, W
        
        total_phase = torch.zeros_like(r)
        for (n, m) in zernike_modes:
            if coeff_init == 'random':
                c_ = 0.1 * torch.randn(1) 
            else:
                c_ = torch.tensor([1.0])
            
            Z = zernike_polynomial(n, m, r, theta)
            total_phase += c_ * Z
    
        self.phase = nn.Parameter(total_phase)
        
        self.register_buffer('r', r)
        self.register_buffer('theta', theta)

    def forward(self, field: torch.Tensor, modulo: bool = False):
        """
        field: torch.complex dtype, shape=(B, C, H, W)
        """
        if modulo:
            phase = torch.remainder(self.phase, 2*np.pi)
        else:
            phase = self.phase
        
        phase_mask = torch.exp(1j * phase)[None, None, :, :]
        
        out = field * phase_mask
        return out
    
class PhaseMaskQuantized(nn.Module):
    """
    - While setting 2D phase as trainable,
    - Let's discretize the phase (emulating SLM, Fab,...)
    """
    def __init__(self, grid, levels=16):
        super().__init__()
        self.levels = levels
        
        init_phase = 2*np.pi*torch.rand_like(grid[0]) # H, W
        self.phase_raw = nn.Parameter(init_phase) 

    def forward(self, field, modulo: bool = False):
        # (1) [0, 2π) Modular
        if modulo:
            phase_mod = torch.remainder(self.phase_raw, 2*np.pi)
        else:
            phase_mod = self.phase_raw
        
        # (2) quantize
        step = 2*np.pi / self.levels
        # round(phase_mod / step) * step
        phase_quant = torch.round(phase_mod / step) * step
        
        # Make phase shifter
        phase = torch.exp(1j * phase_quant)
        
        out = field * phase[None, None, :, :] # 1, 1, H, W
        
        return out
    
    
def zernike_polynomial(n, m, r, theta):
    n_m = n - abs(m)
    if (n_m % 2) != 0:
        return torch.zeros_like(r)
    
    # 방사형 다항식 계산
    kmax = n_m // 2
    R = torch.zeros_like(r)
    for k in range(kmax+1):
        c = ((-1.0)**k 
             * float(math.factorial(n - k))
             / (math.factorial(k) 
                * math.factorial((n + abs(m))//2 - k) 
                * math.factorial((n - abs(m))//2 - k)))
        R += c * (r**(n - 2*k))
        
    # 각 모드별 각도 의존성
    if m >= 0:
        Z = R * torch.cos(m * theta)
    else:
        Z = R * torch.sin(-m * theta)
    
    return Z
