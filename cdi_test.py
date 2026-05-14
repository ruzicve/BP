import pytest
import numpy as np
from cdi import (scale_to_pi, phase_object, diffraction,
                 get_rough_support, EmptySupportError)


def test_scale_to_pi_bounds():
    """Tests if the function strictly bounds data between -pi and pi."""
    test_img = np.array([[0, 50], [100, 200]], dtype=float)
    scaled = scale_to_pi(test_img)

    assert np.min(scaled) == pytest.approx(-np.pi)
    assert np.max(scaled) == pytest.approx(np.pi)


def test_scale_to_pi_flat_image():
    """Tests the edge case where the image is
     a single solid color (range = 0)."""
    flat_img = np.ones((10, 10))
    scaled = scale_to_pi(flat_img)

    # According to your logic, it should return -pi everywhere
    assert np.all(scaled == pytest.approx(-np.pi))


def test_phase_image_magnitude():
    """A pure phase object must have an amplitude of exactly 1.0 everywhere."""
    test_img = np.random.rand(64, 64)
    phase_obj = phase_object(test_img)

    amplitudes = np.abs(phase_obj)
    # np.allclose handles tiny floating-point rounding errors
    assert np.allclose(amplitudes, 1.0)


def test_diffraction_normalization():
    """Tests if the diffraction pattern is properly
     normalized to a max of 1.0."""
    test_img = np.random.rand(32, 32)
    sqrt_I = diffraction(test_img)

    assert np.max(sqrt_I) == pytest.approx(1.0)
    assert np.min(sqrt_I) >= 0.0  # Intensity can never be negative


def test_get_rough_support_bounds():
    """Tests if the support mask correctly bounds the object with
     the specified margin."""
    # Create a 20x20 background of zeros
    test_img = np.zeros((20, 20))
    # Create a 2x2 object in the center (y: 10-11, x: 10-11)
    test_img[10:12, 10:12] = 1.0

    margin = 2
    mask = get_rough_support(test_img, margin=margin)

    # Expected bounds:
    # y_min = 10 - 2 = 8, y_max = 11 + 2 = 13
    # x_min = 10 - 2 = 8, x_max = 11 + 2 = 13
    assert mask[8, 8] == 1.0  # Inside margin
    assert mask[13, 13] == 1.0  # Inside margin

    # Check pixels just outside the expected box are 0
    assert mask[7, 10] == 0.0
    assert mask[14, 10] == 0.0

    # Check total area: a 6x6 square (8 to 13 inclusive) = 36 pixels
    assert np.sum(mask) == 36


def test_get_rough_support_empty():
    """Tests if the function raises EmptySupportError for a completely
    blank image."""
    blank_img = np.zeros((10, 10))

    # Pytest context manager to check if the specific exception is raised
    with pytest.raises(EmptySupportError):
        get_rough_support(blank_img)
