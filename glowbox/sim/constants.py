"""Physical constants and literature defaults for the NV-centre model.

Every value carries its unit in the name and a literature source. docs/physics.md is the spec that the QuTiP
reference and the fast surrogate both follow; this module is its numeric half. Frequencies are in Hz (cycles per
second, *not* rad/s) unless a name says ``_RAD_S``. Rates of incoherent processes are in 1/s.

Values marked "default" are typical-sample choices, not constants of nature; they seed ``EnvParams`` and are meant
to be overridden per experiment.
"""

from __future__ import annotations

import numpy as np

# --- Fundamental constants -------------------------------------------------------------------------------------

#: Bohr magneton over Planck's constant, Hz/T. CODATA 2018 (mu_B/h = 13.996 244 936 1 GHz/T).
MU_B_OVER_H_HZ_PER_T: float = 13.9962449361e9

#: NV ground-state electron g-factor (dimensionless). Doherty et al., Phys. Rep. 528, 1 (2013) quote g ~ 2.003,
#: isotropic to ~1e-4; 2.0028 is the commonly used value. Anisotropy is neglected here.
G_FACTOR_NV: float = 2.0028

#: Electron gyromagnetic ratio gamma_e = g mu_B / h, Hz/T (~28.03 GHz/T). Derived from the two values above.
GAMMA_E_HZ_PER_T: float = G_FACTOR_NV * MU_B_OVER_H_HZ_PER_T

# --- Ground-state spin Hamiltonian ------------------------------------------------------------------------------

#: Ground-state (3A2) zero-field splitting D at room temperature, Hz. Doherty et al., Phys. Rep. 528, 1 (2013);
#: Acosta et al., Phys. Rev. Lett. 104, 070801 (2010) measure dD/dT ~ -74 kHz/K near 300 K, which this
#: temperature-free model ignores (fold it into drift if needed).
D_GS_HZ: float = 2.870e9

#: Default transverse (strain/electric) splitting E, Hz. Default: 0. Ensembles typically show E of order
#: 0.1-few MHz from local strain (Barry et al., Rev. Mod. Phys. 92, 015004 (2020)); set per sample.
E_GS_DEFAULT_HZ: float = 0.0

#: Excited-state (3E) zero-field splitting at room temperature, Hz. Fuchs et al., Phys. Rev. Lett. 101, 117601
#: (2008). Only used to locate the excited-state level anticrossing (ESLAC, B_z ~ D_ES / gamma_e ~ 51 mT), above
#: which the zero-field rate model below stops being valid.
D_ES_HZ: float = 1.42e9

# --- Crystal geometry -------------------------------------------------------------------------------------------

#: The four NV symmetry axes, unit vectors in the cubic crystal frame (x,y,z = [100],[010],[001]). Order fixes the
#: index of ``EnvParams.orientation_populations``. The sign of an axis is a convention (spin-1 physics is
#: unchanged under axis -> -axis up to ms=+1 <-> -1 relabelling); these four have an even number of minus signs.
NV_AXES: np.ndarray = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]
) / np.sqrt(3.0)
NV_AXES.setflags(write=False)

# --- Optical cycle: seven-level rate model ----------------------------------------------------------------------
# Tetienne et al., New J. Phys. 14, 103033 (2012), Table 1: fits of four single NVs at B = 0. We use the mean of
# the four columns (NV C, D, E, F). Level numbering follows the paper:
#   |1> ground ms=0, |2> ground ms=-1, |3> ground ms=+1,
#   |4> excited ms=0, |5> excited ms=-1, |6> excited ms=+1, |7> metastable singlet (1E/1A1 merged).
# Optical transitions are spin-conserving with spin-independent radiative rate k_r (k41 = k52 = k63 = k_r); the
# pump rate is k_14 = k_25 = k_36 = beta * k_r with beta the optical pumping parameter (Tetienne's definition).
# Robledo et al., New J. Phys. 13, 025013 (2011) bound spin-flip optical transitions to <~2% (cited by Tetienne),
# which justifies neglecting them. Units: 1/s.

#: Radiative decay 3E -> 3A2, spin-conserving, 1/s (table: 67.7, 63.2, 63.7, 69.1 us^-1; mean 65.9).
K_RADIATIVE_PER_S: float = 65.9e6
#: Intersystem crossing excited ms=0 (|4>) -> singlet (|7>), 1/s (6.4, 10.8, 9.3, 5.2; mean 7.9).
K_ISC_MS0_PER_S: float = 7.9e6
#: Intersystem crossing excited ms=+-1 (|5>,|6>, each) -> singlet, 1/s (50.7, 60.7, 53.0, 48.6; mean 53.3).
K_ISC_MS1_PER_S: float = 53.3e6
#: Singlet (|7>) -> ground ms=0 (|1>), 1/s (0.7, 0.8, 0.9, 1.5; mean 0.98).
K_SINGLET_TO_MS0_PER_S: float = 0.98e6
#: Singlet (|7>) -> ground ms=-1 and, separately, -> ms=+1 (k72 = k73, per level), 1/s (0.6, 0.4, 0.5, 1.4; mean
#: 0.73).
K_SINGLET_TO_MS1_PER_S: float = 0.73e6

# --- Relaxation defaults ----------------------------------------------------------------------------------------

#: Default longitudinal spin relaxation time, s. Room-temperature T1 is a few ms, phonon-limited: Jarmola et al.,
#: Phys. Rev. Lett. 108, 197601 (2012). Dense ensembles can be shorter.
T1_DEFAULT_S: float = 5e-3
#: Default inhomogeneous dephasing time T2*, s. Ensemble values span ~0.1-10 us depending on nitrogen density and
#: 13C content (Barry et al., Rev. Mod. Phys. 92, 015004 (2020), Sec. IV). Default choice, not a measurement.
T2_STAR_DEFAULT_S: float = 1e-6
#: Default Hahn-echo coherence time T2, s. Ensembles: ~1-100s of us (Barry et al. 2020). Default choice.
T2_DEFAULT_S: float = 10e-6

# --- Drive / optics defaults ------------------------------------------------------------------------------------

#: Default Rabi frequency at full microwave amplitude for an NV whose axis is perpendicular to the MW field, Hz.
#: Typical loop/stripline setups give 1-20 MHz. Default choice.
RABI_FULL_SCALE_DEFAULT_HZ: float = 10e6
#: Default optical pumping parameter beta = k_pump / k_r at full laser power (Tetienne et al. 2012 definition).
#: beta ~ 1 is near optical saturation. Default choice.
OPTICAL_BETA_FULL_SCALE_DEFAULT: float = 0.1
#: Default photon collection (and detection) efficiency, dimensionless. Confocal ensemble setups: ~1e-3-1e-2
#: (Barry et al. 2020, Sec. V). Default choice.
COLLECTION_EFFICIENCY_DEFAULT: float = 3e-3
#: Default number of NVs in the excitation/collection volume. Default choice (a small ensemble).
N_NV_DEFAULT: float = 1e3
#: Default background count rate at the detector, counts/s. Default choice.
BACKGROUND_DEFAULT_CPS: float = 1e4

#: Default time grid of pulse sequences and fluorescence bins, ns (project brief: start with 10 ns).
GRID_DEFAULT_NS: int = 10


def b_projections_hz(b_field_t: np.ndarray) -> np.ndarray:
    """Return (B_parallel, |B_perp|) for each NV axis, each times gamma_e, in Hz.

    ``b_field_t`` is a crystal-frame field vector in tesla, shape (3,) or (..., 3). Output shape (..., 4, 2).
    """
    b = np.asarray(b_field_t, dtype=np.float64)
    par = b @ NV_AXES.T  # (..., 4) tesla
    tot2 = np.sum(b * b, axis=-1, keepdims=True)
    perp = np.sqrt(np.clip(tot2 - par * par, 0.0, None))
    return GAMMA_E_HZ_PER_T * np.stack([par, perp], axis=-1)
