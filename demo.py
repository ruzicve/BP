"""Demonstration of usage for tools in cdi.py"""
from cdi import *
from skimage import data
from scipy.ndimage import binary_dilation

# =====================================================================
# 1. PROCESSING A REAL IMAGE USING cdi_loop_generator
# =====================================================================
logger.info("--- Starting Cameraman Reconstruction ---")
out_dir_cam = "results_cameraman_1"

# Setup Object
cameraman_img = data.camera()

# Estimate support before converting to phase
# (background goes to 1+0j otherwise)
rough_supp_cam = get_rough_support(to_grayscale(cameraman_img))

# Oversample
pad_img_cam, pad_supp_cam = oversample(cameraman_img, rough_supp_cam,
                                       oversampling=4.0)

# Simulate Diffraction
sqrt_I_cam = diffraction(pad_img_cam)

# Initialize and run the generator
generator_cam = cdi_loop_generator(
    sqrt_I=sqrt_I_cam,
    init_supp=pad_supp_cam,
    total_cycles=15,
    beta=0.9,
    is_real=True,
    use_sw=False
)

# Iterate through the generator, processing and
# plotting every cycle on the fly
for cycle, g_er, support, current_errors in generator_cam:
    save_comprehensive_snapshot(cycle, g_er, g_er, support, current_errors,
                                out_dir_cam, prefix="cameraman", colour="gray")

logger.info(f"Cameraman results saved to {out_dir_cam}/")

# =====================================================================
# 2. PROCESSING AN UPLOADED IMAGE (star) USING cdi_loop_generator
# =====================================================================
logger.info("--- Starting Uploaded Image Reconstruction ---")
out_dir_upload = "results_uploaded"
my_image_path = "star.png"

try:
    # Load user image
    my_img = plt.imread(my_image_path)

    # Preprocessing - grayscale
    gray_img = to_grayscale(my_img)

    # Estimate initial support
    rough_supp_my = get_rough_support(gray_img)

    # Preprocessing - oversampling
    pad_img_my, pad_supp_my = oversample(gray_img, rough_supp_my,
                                         oversampling=4.0)

    # Get diffraction pattern
    # (or replace all preceding steps by loadig actual
    # experimental diffraction data)
    sqrt_I_my = diffraction(pad_img_my)

    # Run the reconstruction (generator version)
    generator_my = cdi_loop_generator(
        sqrt_I=sqrt_I_my,
        init_supp=pad_supp_my,
        total_cycles=10,
        beta=0.9
    )

    # Save results on the fly
    for cycle, g_er, support, current_errors in generator_my:
        save_comprehensive_snapshot(cycle, g_er, g_er, support, current_errors,
                                    out_dir_upload, prefix="uploaded")

    print(f"Uploaded image results saved to {out_dir_upload}/")

except FileNotFoundError:
    print(f"Note: Could not find '{my_image_path}'. Place an image with this"
          f" name in the directory to test Section 3.")

# =====================================================================
# 3. PROCESSING A PHASE OBJECT (L) USING cdi_loop
# =====================================================================
logger.info("--- Starting Phase Object Reconstruction ---")
out_dir_L = "results_phase"

# Setup Object
L_img = np.zeros((128, 128), dtype=float)

# Draw an asymmetrical 'L' shape
L_img[30:100, 40:60] = 1.0  # Vertical bar
L_img[80:100, 60:110] = 1.0  # Horizontal bar extending right

# Create an asymmetric support by slightly dilating the true shape.
# (Simulates a tight support mask)
L_support = binary_dilation(L_img, iterations=3).astype(float)

phase_L = phase_object(L_img)
pad_img_L, pad_supp_L = oversample(phase_L, L_support,
                                    oversampling=4.0)

# Simulate Diffraction
sqrt_I_star = diffraction(pad_img_L)

# Define when to take snapshots (here every 2nd cycle)
total_cycles_L = 100
snapshots_L = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]

# Run the standard loop
g_final_L, errors_L, history_L, history_sup_L = cdi_loop(
    sqrt_I=sqrt_I_star,
    init_supp=pad_supp_L,
    snapshots=snapshots_L,
    total_cycles=total_cycles_L,
    beta=0.7,
    use_sw=False
)

# Plot the saved snapshots
iters_per_cycle = 80 + 20
for k in snapshots_L:
    img_hio = history_L[2 * k]
    img_er = history_L[2 * k + 1]
    support = history_sup_L[k]
    current_errors = errors_L[:(k + 1) * iters_per_cycle]

    save_comprehensive_snapshot(k, img_hio, img_er, support, current_errors,
                                out_dir_L, prefix="phase_L")
logger.info(f"Star results saved to {out_dir_L}/")
