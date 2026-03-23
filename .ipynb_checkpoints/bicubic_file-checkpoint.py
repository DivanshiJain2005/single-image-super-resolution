import os
from PIL import Image

# Original HR folder (relative path from /workspace/EPGDUN)
hr_dir = "./data/DIV2K/DIV2K_valid_HR"

# Base LR folder
lr_base_dir = "data/DIV2K/DIV2K_valid_LR_bicubic"

# Scales you want to generate
scales = [2, 3, 4]

# Create LR folders if they don't exist
for scale in scales:
    lr_dir = os.path.join(lr_base_dir, f"X{scale}")
    os.makedirs(lr_dir, exist_ok=True)

# List all HR images
hr_images = [f for f in os.listdir(hr_dir) if f.endswith('.png')]

for img_name in hr_images:
    hr_path = os.path.join(hr_dir, img_name)

    # Open HR image
    hr_img = Image.open(hr_path)

    # Remove extension from original name
    base_name = os.path.splitext(img_name)[0]

    for scale in scales:
        lr_dir = os.path.join(lr_base_dir, f"X{scale}")
        # Rename according to EPGDUN convention
        lr_name = f"{base_name}x{scale}.png"
        lr_path = os.path.join(lr_dir, lr_name)

        # Compute LR size
        lr_size = (hr_img.width // scale, hr_img.height // scale)

        # Resize using bicubic
        lr_img = hr_img.resize(lr_size, resample=Image.BICUBIC)

        # Save LR image
        lr_img.save(lr_path)

        print(f"Saved {lr_path}")

print("All LR images generated with proper DIV2K naming!")
