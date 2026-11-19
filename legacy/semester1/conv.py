import os
import h5py
import numpy as np
from PIL import Image

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
input_folder = r"C:\Users\shawa\Desktop\final project\dataset\all"
output_folder = r"C:\Users\shawa\Desktop\final project\dataset\output"
resize_to = (256, 256)
save_format = "png"
# -------------------------------------------------------

# Mapping numeric labels to names
label_map = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary"
}

# Create output directories
for cls in label_map.values():
    os.makedirs(os.path.join(output_folder, cls, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, cls, "masks"), exist_ok=True)

# Iterate over .mat files
for file_name in os.listdir(input_folder):
    if not file_name.endswith(".mat"):
        continue

    file_path = os.path.join(input_folder, file_name)

    with h5py.File(file_path, 'r') as f:
        # Access fields (MATLAB stores them in structs under "cjdata")
        cjdata = f["cjdata"]

        label = int(np.array(cjdata["label"])[0][0])
        image = np.array(cjdata["image"])
        mask = np.array(cjdata["tumorMask"])

    # Normalize MRI image to 0–255
    im = image.astype(np.float32)
    im = 255 * (im - im.min()) / (im.max() - im.min() + 1e-8)
    im = im.astype(np.uint8)

    # Mask → 0 or 255
    mask = (mask.astype(np.uint8) * 255)

    # Convert to PIL
    im_pil = Image.fromarray(im)
    mask_pil = Image.fromarray(mask)

    # Resize
    if resize_to:
        im_pil = im_pil.resize(resize_to)
        mask_pil = mask_pil.resize(resize_to)

    class_name = label_map[label]
    base = file_name.replace(".mat", "")

    # Save files
    img_out = os.path.join(output_folder, class_name, "images", f"{base}.{save_format}")
    mask_out = os.path.join(output_folder, class_name, "masks", f"{base}.{save_format}")

    im_pil.save(img_out)
    mask_pil.save(mask_out)

    print(f"Saved → {img_out}")

print("\n✔ Conversion complete! All MATLAB v7.3 files processed successfully.")
