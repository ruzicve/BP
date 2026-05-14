from cdi import *
from skimage import data
from skimage.morphology import star

# =====================================================================
# 1. PROCESSING A GEOMETRICAL OBJECT (Star) USING cdi_loop
# =====================================================================
logger.info("--- Starting Star Object Reconstruction ---")
out_dir_star = "results_star"

# Setup Object
star_img = star(100).astype(float)
rough_supp_star = get_rough_support(star_img)
pad_img_star, pad_supp_star = oversample(star_img, rough_supp_star,
                                         oversampling=4.0)

# Simulate Diffraction
sqrt_I_star = diffraction(pad_img_star)

# Define when to take snapshots (here every 2nd cycle)
total_cycles_star = 6
snapshots_star = [0, 2, 4]

# Run the standard loop
g_final_star, errors_star, history_star, history_sup_star = cdi_loop(
    sqrt_I=sqrt_I_star,
    init_supp=pad_supp_star,
    snapshots=snapshots_star,
    total_cycles=total_cycles_star,
    beta=0.9, hio_iter=80, er_iter=20
)

# Plot the saved snapshots
iters_per_cycle = 80 + 20
for k in snapshots_star:
    img_hio = history_star[2 * k]
    img_er = history_star[2 * k + 1]
    support = history_sup_star[k]
    current_errors = errors_star[:(k + 1) * iters_per_cycle]

    save_comprehensive_snapshot(k, img_hio, img_er, support, current_errors,
                                out_dir_star, prefix="star")
logger.info(f"Star results saved to {out_dir_star}/")

# =====================================================================
# 2. PROCESSING A PHASE OBJECT (Cameraman) USING cdi_loop_generator
# =====================================================================
logger.info("--- Starting Cameraman Phase Object Reconstruction ---")
out_dir_cam = "results_cameraman"

# Setup Object
cameraman_img = data.camera()

# Estimate support before converting to phase
# (background goes to 1+0j otherwise)
rough_supp_cam = get_rough_support(to_grayscale(cameraman_img))

# Convert to phase object
phase_cam = phase_object(cameraman_img)

# Oversample
pad_img_cam, pad_supp_cam = oversample(phase_cam, rough_supp_cam,
                                       oversampling=4.0)

# Simulate Diffraction
sqrt_I_cam = diffraction(pad_img_cam)

# Initialize and run the generator
generator_cam = cdi_loop_generator(
    sqrt_I=sqrt_I_cam,
    init_supp=pad_supp_cam,
    total_cycles=10,
    beta=0.9, hio_iter=800, er_iter=200
)

# Iterate through the generator, processing and
# plotting every cycle on the fly
for cycle, g_er, support, current_errors in generator_cam:
    save_comprehensive_snapshot(cycle, g_er, g_er, support, current_errors,
                                out_dir_cam, prefix="cameraman_phase")

logger.info(f"Cameraman results saved to {out_dir_cam}/")

# =====================================================================
# 3. PROCESSING EXPERIMENTAL DATA / UPLOADED IMAGE
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
        beta=0.9,
        hio_iter=400,
        er_iter=100
    )

    # Save results on the fly
    for cycle, g_er, support, current_errors in generator_my:
        save_comprehensive_snapshot(cycle, g_er, g_er, support, current_errors,
                                    out_dir_upload, prefix="uploaded")

    print(f"Uploaded image results saved to {out_dir_upload}/")

except FileNotFoundError:
    print(f"Note: Could not find '{my_image_path}'. Place an image with this"
          f" name in the directory to test Section 3.")