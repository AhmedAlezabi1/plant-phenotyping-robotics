"""
cv_pipeline.py

All CV pipeline functions extracted from Task 8 notebook.
Used by run_task12_all.py to detect root tips from a plate image.

Usage:
    from cv_pipeline import run_pipeline
    tips, plate_pixel_size = run_pipeline("path/to/image.png", model)
"""

import os
import numpy as np
import cv2
import networkx as nx

from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt



# 1. CROP PETRI DISH


def crop_petri_dish(img_gray):
    """
    Detects and crops the Petri dish region using brightness + Otsu.
    Returns cropped image and crop box (minr, minc, maxr, maxc).
    """
    h0, w0 = img_gray.shape
    blur = cv2.GaussianBlur(img_gray, (9, 9), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = (thresh == 255)
    mask = remove_small_objects(mask, min_size=5000)
    labeled = label(mask)
    regions = regionprops(labeled)

    if not regions:
        crop_size = min(h0, w0)
        sy = (h0 - crop_size) // 2
        sx = (w0 - crop_size) // 2
        return img_gray[sy:sy+crop_size, sx:sx+crop_size], (sy, sx, sy+crop_size, sx+crop_size)

    largest = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest.bbox
    expand = 18
    minr = max(minr - expand, 0)
    minc = max(minc - expand, 0)
    maxr = min(maxr + expand, h0)
    maxc = min(maxc + expand, w0)

    return img_gray[minr:maxr, minc:maxc], (minr, minc, maxr, maxc)



# 2. CLAHE + PADDING


def apply_clahe(img_u8, clip=2.0, grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(img_u8)


def pad_to_multiple(img, patch_size=192):
    h, w = img.shape
    new_h = ((h + patch_size - 1) // patch_size) * patch_size
    new_w = ((w + patch_size - 1) // patch_size) * patch_size
    y_off = (new_h - h) // 2
    x_off = (new_w - w) // 2
    padded = cv2.copyMakeBorder(
        img, y_off, new_h - h - y_off, x_off, new_w - w - x_off,
        cv2.BORDER_CONSTANT, value=0
    )
    return padded, {
        "orig_shape": (h, w),
        "padded_shape": (new_h, new_w),
        "offset": (y_off, x_off)
    }



# 3. PATCH-BASED U-NET INFERENCE


def patchify_image(img, patch_size=192, step=96):
    patches, positions = [], []
    h, w = img.shape
    for i in range(0, h - patch_size + 1, step):
        for j in range(0, w - patch_size + 1, step):
            patches.append(img[i:i+patch_size, j:j+patch_size])
            positions.append((i, j))
    return np.array(patches), positions


def unpatchify_avg(patches, positions, full_h, full_w, patch_size=192):
    recon = np.zeros((full_h, full_w), dtype=np.float32)
    count = np.zeros((full_h, full_w), dtype=np.float32)
    for patch, (i, j) in zip(patches, positions):
        recon[i:i+patch_size, j:j+patch_size] += patch
        count[i:i+patch_size, j:j+patch_size] += 1
    count[count == 0] = 1
    return recon / count


def predict_mask(model, padded_img, patch_size=192, step=96, threshold=0.20):
    patches, positions = patchify_image(padded_img, patch_size, step)
    patches_in = patches[..., np.newaxis].astype(np.float32) / 255.0
    preds = model.predict(patches_in, batch_size=1, verbose=0)[..., 0]
    restored = unpatchify_avg(preds, positions, padded_img.shape[0], padded_img.shape[1], patch_size)
    return (restored > threshold).astype(np.uint8)



# 4. RESTORE MASK TO ORIGINAL SIZE


def restore_mask(pred_mask, crop_box, pad_info, orig_shape):
    minr, minc, maxr, maxc = crop_box
    crop_h, crop_w = maxr - minr, maxc - minc
    y_off, x_off = pad_info["offset"]
    unpadded = pred_mask[y_off:y_off + crop_h, x_off:x_off + crop_w]
    restored = np.zeros(orig_shape, dtype=np.uint8)
    restored[minr:maxr, minc:maxc] = unpadded
    return restored



# 5. POST-PROCESSING (all 7 cleaning functions from Task 8)


def find_density_band(projection, keep_ratio=0.95):
    total = projection.sum()
    if total == 0:
        return 0, len(projection) - 1
    cum = np.cumsum(projection)
    low = np.searchsorted(cum, (1 - keep_ratio) * total)
    high = np.searchsorted(cum, keep_ratio * total)
    return max(0, low), min(len(projection) - 1, high)


def clean_root_mask(mask, keep_ratio_x=0.93, keep_ratio_y=0.90, padding=40):
    H, W = mask.shape
    col_sum = mask.sum(axis=0)
    row_sum = mask.sum(axis=1)
    x_min, x_max = find_density_band(col_sum, keep_ratio_x)
    y_min, y_max = find_density_band(row_sum, keep_ratio_y)
    x_min0 = max(0, x_min - padding)
    x_max0 = min(W - 1, x_max + padding)
    y_min0 = max(0, y_min - padding)
    y_max0 = min(H - 1, y_max + padding)

    lbl = label(mask)
    for r in regionprops(lbl):
        xmin, ymin, xmax, ymax = r.bbox[1], r.bbox[0], r.bbox[3], r.bbox[2]
        if not (xmax < x_min0 or xmin > x_max0) and not (ymax < y_min0 or ymin > y_max0):
            x_min0 = min(x_min0, xmin - padding)
            x_max0 = max(x_max0, xmax + padding)
            y_min0 = min(y_min0, ymin - padding)
            y_max0 = max(y_max0, ymax + padding)

    x_min0 = max(0, x_min0)
    x_max0 = min(W - 1, x_max0)
    y_min0 = max(0, y_min0)
    y_max0 = min(H - 1, y_max0)

    cleaned = np.zeros_like(mask)
    cleaned[y_min0:y_max0, x_min0:x_max0] = mask[y_min0:y_max0, x_min0:x_max0]
    return cleaned


def orientation_filter(mask, angle_limit=40, aspect_limit=1.2, ecc_limit=0.5, skel_limit=3):
    labeled = label(mask)
    output = np.zeros_like(mask)
    for region in regionprops(labeled):
        comp = (labeled == region.label)
        ecc = region.eccentricity
        maj = region.major_axis_length
        minr = region.minor_axis_length + 1e-5
        aspect = maj / minr
        skel_len = skeletonize(comp).sum()
        angle = abs(region.orientation * 180 / np.pi)
        angle = min(angle, 180 - angle)
        if angle <= angle_limit and aspect >= aspect_limit and ecc >= ecc_limit and skel_len >= skel_limit:
            output[comp] = 1
    return output


def margin_clean(mask, top_frac=0.15, side_px=840, side_small_area=500):
    H, W = mask.shape
    out = mask.copy()
    top_cut = int(top_frac * H)
    out[:top_cut, :] = 0
    lbl = label(out)
    for r in regionprops(lbl):
        minr, minc, maxr, maxc = r.bbox
        touches_left = (minc < side_px)
        touches_right = (maxc > (W - side_px))
        if (touches_left or touches_right) and (r.area < side_small_area):
            out[lbl == r.label] = 0
    return out


def keep_components_near_top_start(mask, expected_plants=5, start_slack_px=80, late_small_area=2000):
    lbl = label(mask)
    regions = regionprops(lbl)
    if not regions:
        return mask
    starts = sorted([(r.bbox[0], r.label) for r in regions], key=lambda x: x[0])
    k = min(expected_plants, len(starts))
    crown_y = max(s for s, _ in starts[:k])
    threshold_y = crown_y + start_slack_px
    out = np.zeros_like(mask)
    for r in regions:
        comp = (lbl == r.label)
        if r.bbox[0] <= threshold_y or r.area >= late_small_area:
            out[comp] = 1
    return out


def remove_small_close_to_plants(mask, plant_count=5, small_area=500, near_px=40):
    lbl = label(mask)
    regions = regionprops(lbl)
    if not regions:
        return mask
    regions_sorted = sorted(regions, key=lambda r: r.area, reverse=True)
    plants = regions_sorted[:min(plant_count, len(regions_sorted))]
    plant_mask = np.zeros_like(mask, dtype=bool)
    plant_labels = set()
    for r in plants:
        plant_mask[lbl == r.label] = True
        plant_labels.add(r.label)
    if not plant_mask.any():
        return mask
    dist_to_plant = distance_transform_edt(~plant_mask)
    out = mask.copy()
    for r in regions:
        if r.label in plant_labels or r.area >= small_area:
            continue
        comp = (lbl == r.label)
        if dist_to_plant[comp].min() <= near_px:
            out[comp] = 0
    return out


def remove_small_isolated(mask, small_area=800, isolation_px=150):
    lbl = label(mask)
    regions = regionprops(lbl)
    if len(regions) <= 1:
        return mask
    out = mask.copy()
    for r in regions:
        if r.area >= small_area:
            continue
        comp = (lbl == r.label)
        others = mask & (~comp)
        if not others.any():
            continue
        dist = distance_transform_edt(~others)
        if dist[comp].min() >= isolation_px:
            out[comp] = 0
    return out


def postprocess_mask(mask):
    """Full post-processing pipeline from Task 8."""
    m = clean_root_mask(mask)
    m = orientation_filter(m)
    m = margin_clean(m, top_frac=0.15, side_px=840, side_small_area=500)
    m = keep_components_near_top_start(m, expected_plants=5, start_slack_px=80, late_small_area=2000)
    m = remove_small_close_to_plants(m, plant_count=5, small_area=1000, near_px=800)
    m = remove_small_isolated(m, small_area=200, isolation_px=150)
    return m



# 6. SPLIT INTO 5 PLANTS (Task 6)


def split_into_5_plants(mask):
    m = (mask > 0).astype(np.uint8)
    m = remove_small_objects(m.astype(bool), min_size=15).astype(np.uint8)
    H, W = m.shape

    col_sum = m.sum(axis=0)
    xs = np.where(col_sum >= 3)[0]
    if len(xs) == 0:
        xL, xR = int(0.2 * W), int(0.8 * W)
    else:
        xL = max(0, int(xs.min()) - 30)
        xR = min(W - 1, int(xs.max()) + 30)

    centers = np.linspace(xL, xR, 5).astype(int).tolist()
    lbl = label(m, connectivity=2)
    regs = regionprops(lbl)

    slot_masks = [np.zeros_like(m, dtype=np.uint8) for _ in range(5)]
    for r in regs:
        _, cx = r.centroid
        slot = int(np.argmin([abs(cx - c) for c in centers]))
        slot_masks[slot][lbl == r.label] = 1

    return slot_masks



# 7. ROOT TIP EXTRACTION (Task 7)


def build_skeleton_graph(skel):
    ys, xs = np.where(skel)
    G = nx.Graph()
    for y, x in zip(ys, xs):
        G.add_node((y, x))
    for y, x in zip(ys, xs):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (ny, nx_) in G:
                    G.add_edge((y, x), (ny, nx_), w=float(np.hypot(dy, dx)))
    return G


def pick_main_skeleton(skel_full, min_nodes=40):
    lab = label(skel_full, connectivity=2)
    if lab.max() == 0:
        return None
    best_lab, best_key = None, None
    for k in range(1, lab.max() + 1):
        ys, xs = np.where(lab == k)
        if len(xs) < min_nodes:
            continue
        key = (int(ys.max() - ys.min()), int(len(xs)))
        if best_key is None or key > best_key:
            best_key = key
            best_lab = k
    if best_lab is None:
        best_lab = max(range(1, lab.max() + 1), key=lambda k: int(np.sum(lab == k)))
    return (lab == best_lab)


def extract_root_tip(plant_mask, min_pixels=50):
    """
    Extract the primary root tip (bottommost skeleton endpoint).

    Args:
        plant_mask: binary mask for one plant

    Returns:
        (tip_y, tip_x) in pixel coordinates, or None
    """
    if plant_mask.sum() < min_pixels:
        return None

    skel_full = skeletonize(plant_mask > 0)
    if skel_full.sum() < 10:
        return None

    skel = pick_main_skeleton(skel_full)
    if skel is None or skel.sum() < 10:
        return None

    G = build_skeleton_graph(skel)
    if G.number_of_nodes() < 10:
        return None

    endpoints = [n for n, d in dict(G.degree()).items() if d == 1]
    if not endpoints:
        return None

    # Tip = bottommost endpoint (largest y value)
    tip = max(endpoints, key=lambda n: n[0])
    return tip  # (y, x)



# 8. MAIN PIPELINE FUNCTION


def run_pipeline(image_path, model, patch_size=192):
    """
    Run full CV pipeline on a plate image.

    Args:
        image_path: str — path to plate image
        model: loaded Keras U-Net model
        patch_size: int (default 192)

    Returns:
        tips: list of dicts, each with:
            - 'plant_id': int (1-5)
            - 'tip_x': float — x pixel relative to cropped plate
            - 'tip_y': float — y pixel relative to cropped plate
        plate_pixel_size: int — plate size in pixels (for conversion factor)
    """
    # Load image
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    orig_shape = img_gray.shape
    print(f"  Loaded: {os.path.basename(image_path)} ({orig_shape})")

    # Step 1: Crop Petri dish
    cropped, crop_box = crop_petri_dish(img_gray)
    minr, minc, maxr, maxc = crop_box
    plate_pixel_size = int((cropped.shape[0] + cropped.shape[1]) / 2)
    print(f"  Cropped: {cropped.shape} → plate ~{plate_pixel_size}px")

    # Step 2: CLAHE + Pad
    enhanced = apply_clahe(cropped)
    padded, pad_info = pad_to_multiple(enhanced, patch_size)
    print(f"  Padded: {padded.shape}")

    # Step 3: U-Net inference
    step = patch_size // 2
    pred_mask = predict_mask(model, padded, patch_size, step, threshold=0.20)
    print(f"  Prediction done")

    # Step 4: Restore to original size
    restored = restore_mask(pred_mask, crop_box, pad_info, orig_shape)

    # Step 5: Post-process
    cleaned = postprocess_mask(restored)
    print(f"  Post-processing done")

    # Step 6: Split into 5 plants
    plant_masks = split_into_5_plants(cleaned)

    # Step 7: Extract root tips
    tips = []
    for plant_idx, pmask in enumerate(plant_masks):
        tip = extract_root_tip(pmask)
        if tip is not None:
            tip_y, tip_x = tip  # in original image coordinates

            # Convert to crop-relative coordinates
            # (plate top-left in image = crop origin = robot plate origin)
            crop_tip_x = tip_x - minc
            crop_tip_y = tip_y - minr

            tips.append({
                "plant_id": plant_idx + 1,
                "tip_x": float(crop_tip_x),
                "tip_y": float(crop_tip_y),
            })
            print(f"  Plant {plant_idx+1}: tip at crop pixel ({crop_tip_x:.0f}, {crop_tip_y:.0f})")
        else:
            print(f"  Plant {plant_idx+1}: no tip detected")

    print(f"  Total tips found: {len(tips)}")
    return tips, plate_pixel_size
