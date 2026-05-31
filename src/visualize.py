"""
Visualization — draws segmentation overlay on CT scan.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_NAMES = [
    "background","aorta","gallbladder","spleen",
    "left_kidney","right_kidney","liver","stomach","pancreas"
]

CLASS_COLORS = {
    1: (255, 80,  80,  160),  # aorta
    2: (80,  255, 80,  160),  # gallbladder
    3: (80,  80,  255, 160),  # spleen
    4: (255, 255, 80,  160),  # left kidney
    5: (255, 165, 0,   160),  # right kidney
    6: (200, 80,  200, 160),  # liver
    7: (80,  220, 220, 160),  # stomach
    8: (255, 180, 180, 160),  # pancreas
}

def draw_segmentation_overlay(pil_image, seg_output, class_colors=None, class_names=None) -> Image.Image:
    """Draw coloured organ masks over the original CT slice."""
    colors = class_colors or CLASS_COLORS
    names  = class_names  or CLASS_NAMES

    # Convert original to RGB at display size
    base = pil_image.convert("RGB").resize((448, 448), Image.BILINEAR)
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    pred_mask = seg_output.get("pred_mask")
    if pred_mask is None:
        return base

    # Resize mask to display size
    mask_img = Image.fromarray(pred_mask.astype(np.uint8)).resize(
        base.size, Image.NEAREST)
    mask_np  = np.array(mask_img)

    for class_id, color in colors.items():
        if len(color) == 3:
            color = (*color, 160)
        class_mask = (mask_np == class_id).astype(np.uint8) * 255
        if class_mask.sum() == 0:
            continue
        colored = Image.new("RGBA", base.size, color[:3])
        alpha   = Image.fromarray(
            (class_mask * (color[3] / 255)).astype(np.uint8), mode="L")
        overlay.paste(colored, mask=alpha)

    # Composite
    result = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

    # Add legend
    draw = ImageDraw.Draw(result)
    y = 8
    for class_id, color in colors.items():
        name = names[class_id] if class_id < len(names) else str(class_id)
        if np.array(Image.fromarray(
                (np.array(Image.fromarray(pred_mask.astype(np.uint8))
                          .resize(base.size, Image.NEAREST)) == class_id)
                .astype(np.uint8)*255).convert("L")).sum() > 0:
            draw.rectangle([8, y, 22, y+12], fill=color[:3])
            draw.text((26, y), name, fill=(255,255,255))
            y += 16

    return result
