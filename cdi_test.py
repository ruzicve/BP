"""Unit tests for cdi.py"""
import pytest
import numpy as np
from cdi import (
    phase_object, diffraction, get_rough_support, EmptySupportError,
    DivergenceError, sanity_check, to_grayscale, oversample,
    add_gaussian_noise, P_M, P_S, ER_rule, HIO_rule, generate_guess,
    shrinkwrap
)


# ==========================================
# 1. Tests for phase_object
# ==========================================

def test_phase_image_magnitude():
    """A pure phase object must have an amplitude of exactly 1.0 where
    the input is positive, and 0.0 otherwise."""
    test_img = np.array([[-1.0, 0.0],
                         [0.5, 1.0]])
    phase_obj = phase_object(test_img)
    amplitudes = np.abs(phase_obj)

    # Background should have an amplitude of 0
    assert np.allclose(amplitudes[0, :], 0.0)
    # Object should have an amplitude of 1
    assert np.allclose(amplitudes[1, :], 1.0)


def test_phase_image_scaling():
    """Tests if the phase is scaled correctly up to the maximum phase shift."""
    test_img = np.array([-5.0, 0.0, 5.0, 10.0])
    max_phase = np.pi / 4
    phase_obj = phase_object(test_img, max_phase=max_phase)
    phases = np.angle(phase_obj)

    # Max value in input (10.0) should correspond to max_phase
    assert phases[3] == pytest.approx(max_phase)
    # Positive values should fall strictly between 0 and max_phase
    assert np.all(phases[2:] > 0)
    assert np.all(phases[2:] <= max_phase)


def test_phase_image_empty():
    """Tests if an input with no positive values returns an array of zeros."""
    blank_img = np.array([[-1.0, -0.5],
                          [0.0, -2.0]])
    phase_obj = phase_object(blank_img)

    assert np.all(phase_obj == 0.0 + 0.0j)


# ==========================================
# 2. Tests for diffraction
# ==========================================

def test_diffraction_normalization():
    """Tests if the diffraction pattern is properly normalized to a max of 1.0."""
    test_img = np.random.rand(32, 32)
    sqrt_I = diffraction(test_img)

    assert np.max(sqrt_I) == pytest.approx(1.0)
    assert np.min(sqrt_I) >= 0.0


# ==========================================
# 3. Tests for Preprocessing & Utilities
# ==========================================

def test_to_grayscale():
    """Tests proper channel handling and dimension rejection."""
    img_2d = np.random.rand(10, 10)
    assert to_grayscale(img_2d).shape == (10, 10)

    img_rgb = np.random.rand(10, 10, 3)
    assert to_grayscale(img_rgb).shape == (10, 10)

    img_rgba = np.random.rand(10, 10, 4)
    assert to_grayscale(img_rgba).shape == (10, 10)

    with pytest.raises(ValueError):
        to_grayscale(np.random.rand(10))  # 1D array should fail


def test_oversample():
    """Tests if padding correctly expands the image to the target ratio."""
    # A 10x10 support mask has an area of 100
    img = np.ones((10, 10))
    support = np.ones((10, 10))

    # ratio of 4.0 -> target area 400 -> 20x20 bounding box
    padded_img, padded_supp = oversample(img, support, oversampling=4.0)
    assert padded_img.shape == (20, 20)
    assert padded_supp.shape == (20, 20)


def test_get_rough_support_bounds():
    """Tests if the support mask correctly bounds the object with the specified margin."""
    test_img = np.zeros((20, 20))
    test_img[10:12, 10:12] = 1.0  # 2x2 object in center

    margin = 2
    mask = get_rough_support(test_img, margin=margin)

    # Expected bounds: y_min=8, y_max=13, x_min=8, x_max=13
    assert mask[8, 8] == 1.0
    assert mask[13, 13] == 1.0
    assert mask[7, 10] == 0.0
    assert np.sum(mask) == 36  # 6x6 square


def test_get_rough_support_empty():
    """Tests if the function raises EmptySupportError for a completely blank image."""
    blank_img = np.zeros((10, 10))
    with pytest.raises(EmptySupportError):
        get_rough_support(blank_img)


def test_add_gaussian_noise_zero_sigma():
    """Tests if sigma=0 returns the identical image."""
    img = np.random.rand(10, 10)
    assert np.array_equal(add_gaussian_noise(img, 0.0), img)


def test_sanity_check():
    """Tests if the sanity check catches NaN and Inf errors."""
    sanity_check(0.5, 1, 1)  # Should pass silently
    with pytest.raises(DivergenceError):
        sanity_check(np.nan, 1, 1)
    with pytest.raises(DivergenceError):
        sanity_check(np.inf, 1, 1)


# ==========================================
# 4. Tests for Core Physics & Projections
# ==========================================

def test_P_M_magnitude_constraint():
    """Tests that P_M strictly enforces the Fourier magnitude constraint."""
    sqrt_I = np.random.rand(16, 16)
    g_guess = np.random.rand(16, 16) + 1j * np.random.rand(16, 16)

    g_new, _ = P_M(g_guess, sqrt_I)
    G_new_fourier = np.fft.fft2(g_new)

    # Mathematical guarantee: FFT of the new guess must equal target magnitude
    assert np.allclose(np.abs(G_new_fourier), sqrt_I)


def test_P_S_mask():
    """Tests that P_S zeroes out anything outside the support mask."""
    g = np.ones((10, 10))
    support = np.zeros((10, 10))
    support[4:6, 4:6] = 1.0

    g_new = P_S(g, support)
    assert np.sum(g_new) == 4.0
    assert g_new[0, 0] == 0.0


def test_generate_guess_magnitude():
    """Tests if the random initial guess honors the target diffraction magnitude."""
    sqrt_I = np.random.rand(16, 16)
    guess = generate_guess(sqrt_I)
    guess_fourier = np.fft.fft2(guess)

    assert np.allclose(np.abs(guess_fourier), sqrt_I)


# ==========================================
# 5. Tests for Update Rules & Branches
# ==========================================

def test_ER_rule_is_real():
    """Tests if ER_rule enforces strict positivity when is_real is True."""
    g = np.random.rand(16, 16) + 1j * np.random.rand(16, 16)
    g -= (0.5 + 0.5j)  # Introduce negative real components

    sqrt_I = np.random.rand(16, 16)
    support = np.ones((16, 16))

    g_new, _ = ER_rule(g, sqrt_I, support, is_real=True, is_phase=False)

    assert np.all(np.isreal(g_new))
    assert np.all(np.real(g_new) >= 0.0)


def test_ER_rule_is_phase():
    """Tests if ER_rule enforces an amplitude of 1.0 inside the support."""
    g = np.random.rand(16, 16) + 1j * np.random.rand(16, 16)
    sqrt_I = np.random.rand(16, 16)
    support = np.zeros((16, 16))
    support[5:10, 5:10] = 1.0

    g_new, _ = ER_rule(g, sqrt_I, support, is_real=False, is_phase=True)

    # Inside support, amplitude should be forced to 1.0
    inside = g_new[support == 1.0]
    assert np.allclose(np.abs(inside), 1.0)

    # Outside support, ER forces 0.0
    outside = g_new[support == 0.0]
    assert np.allclose(np.abs(outside), 0.0)


def test_HIO_beta_update():
    """Tests the HIO background update rule outside the support mask."""
    g = np.ones((16, 16), dtype=np.complex128)
    sqrt_I = np.random.rand(16, 16)
    support = np.zeros((16, 16))  # All background
    beta = 0.9

    g_mod, _ = P_M(g, sqrt_I)
    g_new, _ = HIO_rule(g, sqrt_I, beta, support, is_real=False, is_phase=False)

    # In HIO, background updates as: g_previous - beta * g_mod
    expected = g - beta * g_mod
    assert np.allclose(g_new, expected)


def test_shrinkwrap():
    """Tests if shrinkwrap correctly generates a boolean mask threshold."""
    g = np.zeros((20, 20))
    g[10, 10] = 100.0  # Single bright spot

    # At tau 0.1, pixels > 10% of the max blurred value become the mask
    new_support = shrinkwrap(g, sigma=1.0, threshold_ratio=0.1)

    assert new_support.dtype == bool
    assert new_support[10, 10] == True
    assert new_support[0, 0] == False