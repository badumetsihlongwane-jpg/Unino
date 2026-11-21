#!/usr/bin/env python3
"""
Stitch NWU_Full_Tour 6-face cubemaps (0.jpg..5.jpg) into dual-tier WebP 2:1 equirectangular 360 panoramas:
  - Tier 1 (Preview): assets/tour360/<id>_preview.webp     (1536x768, ~70-120 KB — instant low-data load)
  - Tier 2 (HD Zoom): assets/tour360_hd/<id>_hd.webp       (4096x2048, ~450-850 KB — hosted on Cloudflare R2, loaded on zoom/HD tap)

Face convention (confirmed from NWU_Full_Tour/library_aerial_view_54):
  0 = Front (+Z), 1 = Right (+X), 2 = Back (-Z), 3 = Left (-X), 4 = Top (+Y), 5 = Bottom (-Y)
"""
import os
import numpy as np
from PIL import Image

SCENES = [
    ("library_aerial_view_54", "library_aerial"),
    ("main_gate_1888", "main_gate"),
    ("north_east_aerial_vi_57", "north_east_aerial"),
    ("north_west_aerial_vi_59", "north_west_aerial"),
    ("north_eastern_overvi_58", "north_eastern_overview"),
    ("north_eastern_overvi_1171", "north_eastern_high"),
]

PREVIEW_W, PREVIEW_H, PREVIEW_Q = 1536, 768, 74
HD_W, HD_H, HD_Q = 4096, 2048, 82


def sample_bilinear(img_arr, u, v):
    h, w, _ = img_arr.shape
    x = np.clip(u, 0.0, 1.0) * (w - 1)
    y = np.clip(v, 0.0, 1.0) * (h - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]

    c00 = img_arr[y0, x0]
    c10 = img_arr[y0, x1]
    c01 = img_arr[y1, x0]
    c11 = img_arr[y1, x1]

    c0 = c00 * (1.0 - fx) + c10 * fx
    c1 = c01 * (1.0 - fx) + c11 * fx
    return (c0 * (1.0 - fy) + c1 * fy).astype(np.uint8)


def stitch_equirect(faces, out_w, out_h):
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h
    lon = (xs * 2.0 - 1.0) * np.pi
    lat = (0.5 - ys) * np.pi

    cos_lat = np.cos(lat)[:, None]
    sin_lat = np.sin(lat)[:, None]
    sin_lon = np.sin(lon)[None, :]
    cos_lon = np.cos(lon)[None, :]

    dx = cos_lat * sin_lon
    dy = np.broadcast_to(sin_lat, (out_h, out_w))
    dz = cos_lat * cos_lon

    ax = np.abs(dx)
    ay = np.abs(dy)
    az = np.abs(dz)

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    m0 = (az >= ax) & (az >= ay) & (dz > 0)
    out[m0] = sample_bilinear(faces[0], (dx[m0] / az[m0] + 1.0) * 0.5, (-dy[m0] / az[m0] + 1.0) * 0.5)

    m1 = (ax > az) & (ax >= ay) & (dx > 0)
    out[m1] = sample_bilinear(faces[1], (-dz[m1] / ax[m1] + 1.0) * 0.5, (-dy[m1] / ax[m1] + 1.0) * 0.5)

    m2 = (az >= ax) & (az >= ay) & (dz <= 0)
    out[m2] = sample_bilinear(faces[2], (-dx[m2] / az[m2] + 1.0) * 0.5, (-dy[m2] / az[m2] + 1.0) * 0.5)

    m3 = (ax > az) & (ax >= ay) & (dx <= 0)
    out[m3] = sample_bilinear(faces[3], (dz[m3] / ax[m3] + 1.0) * 0.5, (-dy[m3] / ax[m3] + 1.0) * 0.5)

    m4 = (ay > ax) & (ay > az) & (dy > 0)
    out[m4] = sample_bilinear(faces[4], (dx[m4] / ay[m4] + 1.0) * 0.5, (dz[m4] / ay[m4] + 1.0) * 0.5)

    m5 = (ay > ax) & (ay > az) & (dy <= 0)
    out[m5] = sample_bilinear(faces[5], (dx[m5] / ay[m5] + 1.0) * 0.5, (-dz[m5] / ay[m5] + 1.0) * 0.5)

    return Image.fromarray(out)


def main():
    os.makedirs("assets/tour360", exist_ok=True)
    os.makedirs("assets/tour360_hd", exist_ok=True)
    total_preview = 0.0
    total_hd = 0.0

    for folder, scene_id in SCENES:
        src = os.path.join("NWU_Full_Tour", folder)
        if not os.path.isdir(src):
            print(f"Skipping missing {src}")
            continue
        faces = {
            i: np.asarray(Image.open(os.path.join(src, f"{i}.jpg")).convert("RGB"), dtype=np.float32)
            for i in range(6)
        }
        # Adapt HD size to source resolution so 1024px faces don't bloat needlessly (3072x1536 vs 4096x2048)
        face_w = faces[0].shape[1]
        hd_w, hd_h = (HD_W, HD_H) if face_w >= 2048 else (3072, 1536)

        hd_img = stitch_equirect(faces, hd_w, hd_h)
        hd_path = os.path.join("assets/tour360_hd", f"{scene_id}_hd.webp")
        hd_img.save(hd_path, "WEBP", quality=HD_Q, method=5)
        hd_kb = os.path.getsize(hd_path) / 1024
        total_hd += hd_kb

        preview_img = hd_img.resize((PREVIEW_W, PREVIEW_H), Image.Resampling.LANCZOS)
        preview_path = os.path.join("assets/tour360", f"{scene_id}_preview.webp")
        preview_img.save(preview_path, "WEBP", quality=PREVIEW_Q, method=6)
        prev_kb = os.path.getsize(preview_path) / 1024
        total_preview += prev_kb

        print(f"{scene_id:24s} preview={prev_kb:6.1f} KB ({PREVIEW_W}x{PREVIEW_H}) | hd={hd_kb:6.1f} KB ({hd_w}x{hd_h})")

    print(f"TOTAL preview (bundled): {total_preview/1024:.2f} MB | TOTAL HD (for R2): {total_hd/1024:.2f} MB")


if __name__ == "__main__":
    main()
