"""
Visualization — draws segmentation overlay with organ labels.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = [
    "background","aorta","gallbladder","spleen",
    "left_kidney","right_kidney","liver","stomach","pancreas"
]

CLASS_COLORS = {
    1: (255, 80,  80,  170),   # aorta — red
    2: (80,  230, 80,  170),   # gallbladder — green
    3: (80,  80,  255, 170),   # spleen — blue
    4: (255, 230, 60,  170),   # left kidney — yellow
    5: (255, 140, 0,   170),   # right kidney — orange
    6: (180, 80,  200, 170),   # liver — purple
    7: (60,  210, 210, 170),   # stomach — cyan
    8: (255, 160, 160, 170),   # pancreas — pink
}

def draw_segmentation_overlay(pil_image, seg_output,
                               class_colors=None, class_names=None) -> Image.Image:
    colors = class_colors or CLASS_COLORS
    names  = class_names  or CLASS_NAMES

    display_size = (512, 512)
    base    = pil_image.convert("RGB").resize(display_size, Image.BILINEAR)
    overlay = Image.new("RGBA", display_size, (0,0,0,0))

    pred_mask = seg_output.get("pred_mask")
    if pred_mask is None:
        return base

    mask_resized = np.array(
        Image.fromarray(pred_mask.astype(np.uint8)).resize(display_size, Image.NEAREST)
    )

    # Draw filled organ regions
    for cid, color in colors.items():
        region = (mask_resized == cid).astype(np.uint8)
        if region.sum() == 0:
            continue
        layer = Image.new("RGBA", display_size, (0,0,0,0))
        pixels = layer.load()
        ys, xs = np.where(region)
        for y, x in zip(ys.tolist(), xs.tolist()):
            pixels[x, y] = color
        overlay = Image.alpha_composite(overlay, layer)

    result = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    draw   = ImageDraw.Draw(result)

    # Legend — top left
    y = 10
    detected = [(cid, color) for cid, color in colors.items()
                if (mask_resized == cid).sum() > 0]
    for cid, color in detected:
        name = names[cid] if cid < len(names) else str(cid)
        draw.rectangle([10, y, 26, y+14], fill=color[:3], outline=(255,255,255))
        draw.text((32, y), name, fill=(255,255,255))
        y += 18

    # Confidence labels on each organ centroid
    organ_masks = seg_output.get("organ_masks", {})
    for cid, color in colors.items():
        if cid >= len(names): continue
        name = names[cid]
        region = (mask_resized == cid)
        if region.sum() < 100: continue
        ys_r, xs_r = np.where(region)
        cx, cy = int(xs_r.mean()), int(ys_r.mean())
        conf = organ_masks.get(name, {}).get("confidence", 0)
        if conf > 0:
            label = f"{conf:.2f}"
            draw.text((cx-12, cy-8), label, fill=(255,255,255))

    return result
