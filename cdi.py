"""This file defines tools for CDI reconstruction"""
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.color import rgb2gray, rgba2rgb
from scipy.ndimage import gaussian_filter

# --- SETTING UP LOGGING ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    file_handler = logging.FileHandler("reconstruction.log")
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


class CDIError(Exception):
    """Base class for all CDI-related exceptions."""

    pass


class EmptySupportError(CDIError):
    """Raised when the support mask contains zero pixels."""

    pass


class DivergenceError(CDIError):
    """Raised when the error metric explodes to NaN or Infinity."""

    pass


def sanity_check(error_val, cycle, iteration, phase="HIO"):
    """Check if the error metric has exploded to NaN or infinity."""
    if np.isnan(error_val) or np.isinf(error_val):
        logger.error(f"Divergence detected in {phase} at cycle {cycle},"
                     f" iteration {iteration}!")
        raise DivergenceError(f"Math diverged to {error_val}. "
                              f"Check your FFT scaling or beta value.")


# ---PREPROCESSING---

# ------GRAYSCALE---
def to_grayscale(image):
    """Convert an image to grayscale.

    inputs: image: np.ndarray (2D grayscale or 3D RGB/RGBA)
    returns: grayscale image: np.ndarray
    """
    # If it's already a 2D image, just ensure it's a float
    if image.ndim == 2:
        logger.info(f"Image successfully converted to grayscale.")
        return image.astype(np.float64)

    # If it's a 3D image, check the number of channels
    if image.ndim == 3:
        if image.shape[-1] == 4:
            # Convert RGBA to RGB to drop the alpha channel
            image = rgba2rgb(image)
        return rgb2gray(image).astype(np.float64)
    # anything else is not supported
    logger.error(f"Failed converting image to grayscale.")
    raise ValueError(f"Unsupported image dimensions: {image.shape}")


# ------SUPPORT ESTIMATE---
def get_rough_support(image, margin=5):
    """ Generate a binary (0, 1) map of the minimum rectangle
     that encompasses the whole object.

    inputs: image: np.ndarray
            margin: safety margin, int
    returns: mask: rough rectangular support, np.ndarray
    """
    h, w = image.shape
    mask = np.zeros((h, w), dtype=np.float64)

    # Coordinates with signal
    coords = np.argwhere(image > 0)

    if coords.size == 0:
        logger.error("Failed to find signal in the image.")
        raise EmptySupportError("Image contains only zeros."
                                " Cannot generate support.")

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Safety margin
    y_start = max(0, y_min - margin)
    y_end = min(h, y_max + margin + 1)
    x_start = max(0, x_min - margin)
    x_end = min(w, x_max + margin + 1)

    mask[y_start:y_end, x_start:x_end] = 1
    logger.info(f"Generated rectangular support with bounds: "
                f"{coords.min(axis=0)} to {coords.max(axis=0)}")
    return mask


# ------OVERSAMPLING---
def oversample(image, rough_support, oversampling=4.0):
    """
    Pad an image based on the area of a rough support to achieve at least
    the specified 2D oversampling ratio in both the x and the y axes.

    inputs: image, np.ndarray
            a rough support, np.ndarray
            oversampling ratio, float (>2 for phase reconstruction)
    """
    n_support_pixels = np.count_nonzero(rough_support)
    n_total_target = n_support_pixels * oversampling
    side_length = int(np.ceil(np.sqrt(n_total_target)))

    h_orig, w_orig = image.shape[:2]
    h_final = max(side_length, h_orig)
    w_final = max(side_length, w_orig)

    total_pad_h = h_final - h_orig
    total_pad_w = w_final - w_orig

    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top
    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left

    pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))

    padded_img = np.pad(image, pad_width, mode='constant', constant_values=0)
    padded_supp = np.pad(rough_support, pad_width,
                         mode='constant', constant_values=0)
    logger.info(f"Oversampling of image set to > {oversampling}.")
    return padded_img, padded_supp


# ------NOISE---
def add_gaussian_noise(image, sigma):
    """Add Gaussian noise to an image array.

    inputs: image, np.ndarray
            standard deviation of the gaussian filter, float
    returns: image with noise, np.ndarray
    """
    if sigma <= 0:
        return image
    noise = np.random.normal(0, sigma, image.shape)
    logger.info(f"Added gaussian noise with sigma={sigma}")
    return image + noise


# ------PHASE OBJECTS---
def scale_to_pi(image):
    """Linearly scale an image array to have the range [-pi, pi].¨

    inputs: image, np.ndarray
    returns: scaled image, np.ndarray
    """
    data_float = image.astype(np.float64)

    # Identify the range of the input
    source_min = np.min(data_float)
    source_max = np.max(data_float)
    source_range = source_max - source_min

    # Handle the case where the image is a single solid color
    if source_range == 0:
        return np.full_like(data_float, -np.pi)

    # Constrain the range from -pi to pi
    target_min = -np.pi
    target_range = 2 * np.pi

    normalized_data = (data_float - source_min) / source_range
    scaled_pi = (normalized_data * target_range) + target_min

    return scaled_pi


def phase_object(image):
    """Convert a simulated 2D object into a phase object.

    inputs: image, np.ndarray
    returns: phase image, np.ndarray
    """
    scaled_image = scale_to_pi(image)
    phase_image = np.exp(scaled_image * 1j)
    logger.info("Image successfully converted to phase object.")
    return phase_image


# --- SIMULATING DIFFRACTION FROM A DIGITAL IMAGE ---
def diffraction(image):
    """Calculate FT of an image and returns the intensity distribution.

    input: 2D image of an object, np.ndarray
    returns: 2D intensity distribution, np.ndarray
    """
    G = fft2(image)
    G_shifted = fftshift(G)
    sqrt_I = np.abs(G_shifted)
    sqrt_I = sqrt_I/np.max(sqrt_I)
    logger.info("Diffraction distribution successfully generated.")
    return ifftshift(sqrt_I)


# ---PROJECTIONS ---
def P_M(g, sqrt_I):
    """" Return Fourier magnitude constraint projection and calculate error."""
    G = fft2(g)
    abs_G = np.abs(G)
    error = np.linalg.norm(abs_G - sqrt_I) / np.linalg.norm(sqrt_I)
    G_new = sqrt_I * np.exp(1j * np.angle(G))
    return ifft2(G_new), error


def P_S(g, support):
    """Return Real space support contraint projection."""
    return g*support


# --- ITERATIVE PHASE RETRIEVAL ---
# ------ UPDATE RULES ---
def ER_rule(g, sqrt_I, support):
    """Calculate the ER update rule.

    inputs:
        g, np.ndarray
        sqrt_I, np.ndarray
        support, np.ndarray
    returns:
        g_new, np.ndarray
        error, np.ndarray
    """
    g_mod, error = P_M(g, sqrt_I)
    return P_S(g_mod, support), error


def HIO_rule(g, sqrt_I, beta, support, is_real):
    """Calculate the HIO update rule.

    inputs:
        g, np.ndarray
        sqrt_I, np.ndarray
        beta, float
        support, np.ndarray
    returns:
        g_new, np.ndarray
        error, np.ndarray
    """
    g_mod, error = P_M(g, sqrt_I)
    mask = (support == 1)

    if not is_real:
        g_new = mask * constrained_inside + (~mask) * (g - beta * g_mod)
    else:
        # Enforce positivity inside support
        g_real = np.real(g_mod)
        good_pixels = mask & (g_real > 0)

        g_new = np.empty_like(g, dtype=np.complex128)
        g_new[good_pixels] = g_real[good_pixels]
        g_new[~good_pixels] = g[~good_pixels] - beta * g_mod[~good_pixels]
    return g_new, error

# ------INITIAL GUESS---
def generate_guess(sqrt_I):
    """Generate an initial guess for the phase reconstruction."""
    random_phase = np.exp(1j * np.random.uniform(-np.pi, np.pi, sqrt_I.shape))
    # Initial guess spectrum G constrained by magnitude in the Fourier space
    G = sqrt_I * random_phase
    # Shift to real space
    return ifft2(G)


# ------SHRINKWRAP---
def shrinkwrap(g, sigma=2.0, threshold_ratio=0.1):
    """Update the support mask based on the current reconstruction.

    inputs:
        g: The current real-space reconstruction, np.ndarray
        sigma: Standard deviation for Gaussian filter, float
        threshold_ratio: Fraction of the maximum intensity to use
          as threshold, float

    returns:
        np.ndarray: A binary mask (bool) representing the updated support.
        """
    mod_g = np.abs(g)
    blurred = gaussian_filter(mod_g, sigma=sigma)
    tau = threshold_ratio * np.max(blurred)
    new_support = blurred > tau

    return new_support


# -----THE LOOP---
def cdi_loop(sqrt_I, init_supp,g=None, snapshots=None, total_cycles = 5,
             beta =0.9, hio_iter=80, er_iter=20, sigma=2.0, tau = 0.1,
             is_real=False, use_sw=True):
    """Run HIO, ER and shrinkwrap on an image and save the state of the phase reconstruction after
       select cycles.

    inputs:
        sqrt_I: sqrt of measured intensity distribution, np.ndarray
        init_supp: initial support estimate, np.ndarray
        snapshots: when to save progress, list
        total_cycles, int
        beta: HIO parameter  -  values between 0 and 1, float
        hio_iter: iterations of HIO in each cycle, int
    returns:
      g: final reconstruction , np.ndarray
      error_metric, list
      history: saved snapshots of the reconstructed image, dictionary
      - keys are the snapshot iterations, the values are ndarrays
      history_supp: saved snapshots of the evolving support, dictionary
    """
    if snapshots is None:
      snapshots = [0, 1, 2, 3, 4]
    history = {}
    history_sup = {}
    error_metric = []
    support = init_supp

    if g is None:
      g = generate_guess(sqrt_I)

    for k in range(total_cycles):

        for i in range(hio_iter):
            g_new, error = HIO_rule(g, sqrt_I, beta, support, is_real)
            g = g_new
            error_metric.append(error)

            sanity_check(error, cycle=k, iteration=i, phase="HIO")

        if k in snapshots:
            history[2*k] = g.copy()

        for j in range(er_iter):
            g_new, error = ER_rule(g, sqrt_I, support)
            g = g_new
            error_metric.append(error)

            sanity_check(error, cycle=k, iteration=j, phase="ER")

        if k in snapshots:
            history[2*k + 1] = g.copy()
            history_sup[k] = support.copy()

        if use_sw:
            support = shrinkwrap(g, sigma, tau)

    logger.info(f"Reconstruction successful with {total_cycles} cycles of HIO: {hio_iter},"
                f" beta={beta}, ER: {er_iter}, shrinkwrap: sigma={sigma}, tau={tau}")
    return g, error_metric, history, history_sup


# ---GENERATOR VERSION---
def cdi_loop_generator(sqrt_I, init_supp, total_cycles=5, beta=0.9,
                       hio_iter=80, er_iter=20, sigma=2.0, tau=0.1,
                       is_real=False, use_sw=True):
    """Run HIO, ER and shrinkwrap on an image and save the state of the phase
       reconstruction after each cycle.
    inputs:
        sqrt_I: sqrt of measured intensity distribution, np.ndarray
        init_supp: initial support estimate, np.ndarray
        total_cycles, int
        beta: HIO parameter  -  values between 0 and 1, float
        hio_iter: iterations of HIO in each cycle, int
    returns:
      g: final reconstruction , np.ndarray
      error_metric, list
      history: saved snapshots of the reconstructed image, dictionary
      - keys are the snapshot iterations, the values are ndarrays
      history_supp: saved snapshots of the evolving support, dictionary
    """
    g = generate_guess(sqrt_I)
    support = init_supp.copy()
    error_metric = []
    for k in range(total_cycles):
        for i in range(hio_iter):
            g, error = HIO_rule(g, sqrt_I, beta, support, is_real)
            sanity_check(error, cycle=k, iteration=i, phase="HIO")
            error_metric.append(error)

        for j in range(er_iter):
            g, error = ER_rule(g, sqrt_I, support)
            sanity_check(error, cycle=k, iteration=j, phase="ER")
            error_metric.append(error)

        if use_sw:
            support = shrinkwrap(g, sigma=sigma, threshold_ratio=tau)

        logger.info(f"Yielding result of iteration: {k}.")
        yield k, g.copy(), support.copy(), error_metric[:]


# ---VISUALISATION OF RESULTS
def save_comprehensive_snapshot(cycle, img_hio, img_er, support, errors, output_dir,
                                prefix="demo", colour="magma"):
    """
    Plots and saves a 1x4 figure containing:
    Post-HIO, Post-ER, Shrinkwrap Support, and Error Metric history.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{prefix.capitalize()} - Cycle {cycle}", fontsize=16)

    # Post-HIO
    im0 = axes[0].imshow(np.abs(img_hio), cmap=colour)
    axes[0].set_title("Post-HIO")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Post-ER
    im1 = axes[1].imshow(np.abs(img_er), cmap=colour)
    axes[1].set_title("Post-ER")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Support
    im2 = axes[2].imshow(support, cmap="gray")
    axes[2].set_title("Support (Shrinkwrap)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Error Metric
    axes[3].plot(errors, color='blue')
    axes[3].set_title("Error Metric")
    axes[3].set_xlabel("Iterations")
    axes[3].set_ylabel("Error")
    axes[3].set_yscale("log")  # Log scale is usually best for CDI convergence

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{prefix}_cycle_{cycle:04d}.png")
    plt.savefig(save_path)
    plt.close(fig)
    logger.info(f"Saved comprehensive snapshot for cycle {cycle} to {save_path}")

