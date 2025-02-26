import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import cart2pol, combinatorial


# Frequency domain pixels.
class Pixels(nn.Module):
    def __init__(self, kx, ky, pixels=None):
        super(Pixels, self).__init__()
        self.kx = kx
        self.ky = ky
        if pixels is None:
            pixels = torch.zeros_like(self.kx)
        self.pixels = nn.Parameter(pixels)

    def forward(self):
        return self.pixels


class SLMPixels(nn.Module): # outcomes phase values.
    def __init__(
        self,
        kx,
        ky,
        min_phase_radians=1.4 * math.pi,
        max_phase_radians=4.6 * math.pi,
        slm_kx=None,
        slm_ky=None,
        slm_pixels=None,
        wrap=False,
        interpolation="nearest",
    ):
        super(SLMPixels, self).__init__()
        self.kx = kx
        self.ky = ky
        self.min_phase_radians = min_phase_radians
        self.max_phase_radians = max_phase_radians
        if slm_kx is None:
            slm_kx = kx
        if slm_ky is None:
            slm_ky = ky
        self.slm_kx = slm_kx
        self.slm_ky = slm_ky
        self.slm_shape = slm_kx.shape
        self.P_shape = kx.shape
        self.slm_pixels = torch.empty(self.slm_shape, device=self.kx.device)
        # select initializer type
        if isinstance(slm_pixels, torch.Tensor):
            self.slm_pixels = slm_pixels
        elif slm_pixels == "random": # set the slm phase in random manner.
            self.random_init()
        elif isinstance(slm_pixels, tuple) or isinstance(slm_pixels, list):
            self.random_init(slm_pixels[0], slm_pixels[-1]) # The provided parameters are bound of phase.
        else: # Flat mirror (0 phase) -> The entire phase (self.slm_pixels.zero_())
            self.flat_init()
        self.slm_pixels = nn.Parameter(self.slm_pixels)
        self.wrap = wrap
        self.interpolation = interpolation

    def flat_init(self):
        """Initialize SLM pixels to be a flat mirror (0 phase)."""
        self.slm_pixels.zero_()

    def random_init(self, min_phase_radians=None, max_phase_radians=None):
        """
        Initialize with uniform random samples at each SLM pixel from range
        min_phase_radians to max_phase_radians.
        """
        if min_phase_radians is None or max_phase_radians is None:
            midpoint = (self.min_phase_radians + self.max_phase_radians) / 2.0
            total_range = min(
                self.max_phase_radians - self.min_phase_radians, 2 * math.pi
            )
            min_phase_radians = midpoint - (total_range / 2.0)
            max_phase_radians = midpoint + (total_range / 2.0)
        assert (
            max_phase_radians >= min_phase_radians
        ), "Max phase must be greater than min"
        self.slm_pixels.uniform_(min_phase_radians, max_phase_radians)

    def forward(self):
        if self.wrap:
            with torch.no_grad():
                self.slm_pixels.data = phase_wrap(
                    self.slm_pixels.data,
                    self.min_phase_radians,
                    self.max_phase_radians,
                )
        else:
            with torch.no_grad():
                self.slm_pixels.clamp_(
                    min=self.min_phase_radians, max=self.max_phase_radians
                )
        P = (
            F.interpolate(
                self.slm_pixels.unsqueeze(0).unsqueeze(0),
                size=self.P_shape,
                mode=self.interpolation,
            )
            .squeeze(0)
            .squeeze(0)
        )
        return P # phase value.


class DefocusedRamps(nn.Module):
    # Defocusing + Phase Ramping.
    def __init__(
        self,
        kx,
        ky,
        fNA=0.8 / 0.55,
        n_immersion=1.33,
        wavelength=0.55,
        nramps=6,
        delta=2374.0,
        defocus=[-50.0, 150.0, -100.0, 50.0, -150.0, 100.0],
    ):
        super(DefocusedRamps, self).__init__()
        self.fNA = fNA
        self.kx = kx / self.fNA
        self.ky = ky / self.fNA
        self.rho, self.theta = cart2pol(self.kx, self.ky)
        self.n_immersion = n_immersion
        self.wavelength = wavelength
        self.nramps = nramps
        self.edges = torch.linspace(
            -math.pi, math.pi, nramps + 1, device=self.kx.device
        ) # split the 2pi phase with the number of ramps
        self.centers = self.edges[:-1] / 2 + self.edges[1:] / 2
        self.delta = nn.Parameter(
            torch.tensor([delta] * nramps, dtype=torch.float32, device=self.kx.device)
        )
        self.defocus = nn.Parameter(
            torch.tensor(defocus, dtype=torch.float32, device=self.kx.device)
        )
        self.rcenter = (self.nramps + 1) ** -0.5
        self.dcenter = (self.rcenter + 1) / 2.0

    def forward(self):
        P = torch.zeros_like(self.kx)

        for ramp in range(self.nramps):
            select = (
                (self.theta >= self.edges[ramp])
                & (self.theta < self.edges[ramp + 1])
                & (
                    (
                        self.ky * torch.sin(self.centers[ramp])
                        + self.kx * torch.cos(self.centers[ramp])
                    )
                    > self.rcenter
                )
                & (self.kx ** 2 + self.ky ** 2 < 1)
            )
            select = select.to(P.dtype)
            P = P + select * (
                self.delta[ramp]
                * (
                    self.ky * torch.cos(self.centers[ramp])
                    - self.kx * torch.sin(self.centers[ramp])
                )
            )
            P = P + select * (
                self.defocus[ramp]
                * (
                    (self.kx - torch.cos(self.centers[ramp]) * self.dcenter) ** 2
                    + (self.ky - torch.sin(self.centers[ramp]) * self.dcenter) ** 2
                )
            )
            P = P - (select * P[select.to(torch.bool)].mean())
        P = P * (self.kx ** 2 + self.ky ** 2 < 1).to(P.dtype)

        return P


class Ramps(DefocusedRamps):
    def __init__(
        self,
        kx,
        ky,
        fNA=0.8 / 0.55,
        n_immersion=1.33,
        wavelength=0.55,
        nramps=6,
        delta=2374.0,
    ):
        super(Ramps, self).__init__(
            kx,
            ky,
            fNA=fNA,
            n_immersion=n_immersion,
            wavelength=wavelength,
            nramps=nramps,
            delta=delta,
            defocus=[0] * nramps,
        )


class PotatoChip(nn.Module):
    def __init__(
        self,
        kx,
        ky,
        fNA=0.8 / 0.55,
        n_immersion=1.33,
        wavelength=0.55,
        d=50.0,
        C0=-146.7,
    ):
        super(PotatoChip, self).__init__()
        self.fNA = fNA
        self.kx = kx / self.fNA
        self.ky = ky / self.fNA
        self.rho, self.theta = cart2pol(self.kx, self.ky)
        self.n_immersion = n_immersion
        self.wavelength = wavelength
        self.k = self.n_immersion / self.wavelength
        self.d = nn.Parameter(torch.tensor(d, dtype=torch.float32))
        self.C0 = nn.Parameter(torch.tensor(C0, dtype=torch.float32))
        self.mask = ((self.kx ** 2 + self.ky ** 2) <= 1).to(torch.float32)

    def forward(self):
        P = self.theta * (
            self.d * torch.sqrt(self.k ** 2 - (self.fNA * self.rho) ** 2) + self.C0
        )
        P = P * self.mask
        return P


class GeneralizedCubic(nn.Module):
    def __init__(
        self,
        kx,
        ky,
        fNA=0.8 / 0.55,
        n_immersion=1.33,
        wavelength=0.55,
    ):
        super(GeneralizedCubic, self).__init__()
        self.fNA = fNA
        self.kx = kx / self.fNA
        self.ky = ky / self.fNA
        self.rho, self.theta = cart2pol(self.kx, self.ky)
        self.n_immersion = n_immersion
        self.wavelength = wavelength
        self.k = self.n_immersion / self.wavelength
        self.mask = ((self.kx ** 2 + self.ky ** 2) <= 1).to(torch.float32)

    def forward(self):
        P = 0
        return P


class ZernikeBases(nn.Module):
    def __init__(
        self,
        kx,
        ky,
        fNA=0.8 / 0.55,
        n_immersion=1.33,
        wavelength=0.55,
        num_coefficients=1,
    ):
        super(ZernikeBases, self).__init__()
        self.fNA = fNA
        # The maximum kx, ky for maximum incident angle is NA/lambda
        # kx and ky are spatial frequency coordinate. Plus, fNA refers to NA/lambda.
        # Therefore, the below code is normalizing spatial frequency using NA/lambda.
        # : [0, 1] normalization.
        self.kx = kx / self.fNA
        self.ky = ky / self.fNA
        self.rho, self.theta = cart2pol(self.kx, self.ky)
        self.n_immersion = n_immersion
        self.wavelength = wavelength
        self.k = self.n_immersion / self.wavelength
        self.num_coefficients = num_coefficients
        # initialize coefficients used to weight combination of zernike bases
        self.coefficients = nn.Parameter(
            torch.rand(
                self.num_coefficients,
                dtype=torch.float32,
                device=self.kx.device,
            )
        )
        # construct zernike bases to combine during forward pass
        self.zernike_bases = []
        radial_pairs = filter(
            lambda p: p[0] >= p[1] and (p[0] - p[1]) % 2 == 0,
            ((n, m) for n in range(num_coefficients) for m in range(num_coefficients)),
        )
        for (n, m) in radial_pairs:
            radial_polynomial = ZernikeBases.radial_polynomial(n, m)
            for sinusoid in [torch.cos, torch.sin]:
                basis = radial_polynomial(self.rho) * sinusoid(m * self.theta)
                self.zernike_bases.append(basis)
                if len(self.zernike_bases) >= self.num_coefficients:
                    break
        self.mask = ((self.kx ** 2 + self.ky ** 2) <= 1).to(torch.float32)

    def radial_polynomial(cls, n, m):
        """Returns a function calculating the specified radial polynomial."""

        def R(rho):
            sum = 0
            if (n - m) % 2 == 0:
                for k in range((n - m) / 2):
                    sum += (
                        rho ** (n - 2 * k)
                        * (-(1 ** k))
                        * combinatorial(n - k, k)
                        * combinatorial(n - 2 * k, (n - m) / 2 - k)
                    )
            return sum

        return R

    ### After initializing zernike polynonmials, allocate the specified values.
    def forward(self):
        P = torch.zeros_like(self.kx)
        for j, basis in enumerate(self.zernike_bases):
            P = P + (self.coefficients[j] * basis)
        P = P * self.mask
        return P


# Pre-process the given phase where the phase value is outside of bounded phase. 
def phase_wrap(phase_mask_radians, min_phase_radians, max_phase_radians):
    idx = phase_mask_radians < min_phase_radians
    # < min phase. -> modular operation. + n*2pi.
    phase_mask_radians[idx] = phase_mask_radians[idx] + 2 * math.pi * (
        1 + (min_phase_radians - phase_mask_radians[idx]) // (2 * math.pi)
    ) # > max phase. -> modular operation. - n*2pi.
    idx = phase_mask_radians > max_phase_radians
    phase_mask_radians[idx] = phase_mask_radians[idx] - 2 * math.pi * (
        1 + (phase_mask_radians[idx] - max_phase_radians) // (2 * math.pi)
    )
    return phase_mask_radians
