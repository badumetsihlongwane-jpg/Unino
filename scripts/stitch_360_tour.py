#!/usr/bin/env python3
"""
Stitch NWU_Full_Tour 6-face cubemaps (0.jpg..5.jpg) into lightweight, seamless
2:1 equirectangular 360 panoramas in assets/tour360/.

Face convention (confirmed from NWU_Full_Tour/library_aerial_view_54):
  0 = Front (+Z)
  1 = Right (+X)
  2 = Back  (-Z)
  3 = Left  (-X)
  4 = Top   (+Y)
  5 = Bottom(-Y)
"""
import os
import sys
import numpy as np
from PIL import Image

SCENES = [
    ("library_aerial_view_54", "library_aerial.jpg"),
    ("main_gate_1888", "main_gate.jpg"),
    ("north_east_aerial_vi_57", "north_east_aerial.jpg"),
    ("north_west_aerial_vi_59", "north_west_aerial.jpg"),
    ("north_eastern_overvi_58", "north_eastern_overview.jpg"),
    ("north_eastern_overvi_1171", "north_eastern_high.jpg"),
]

OUT_W = 2560
OUT_H = 1280
JPEG_QUALITY = 82


def sample_bilinear(img_arr, u, v):
    """Bilinear sample img_arr (H, W, 3) at normalized coords u,v in [0, 1]."""
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


def cubemap_to_equirect(src_dir, dst_path, out_w=OUT_W, out_h=OUT_H):
    faces = {}
    for i in range(6):
        fp = os.path.join(src_dir, f"{i}.jpg")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing face {fp}")
        faces[i] = np.asarray(Image.open(fp).convert("RGB"), dtype=np.float32)

    # Spherical grid: lon in [-pi, pi] (center = Front/+Z = face 0), lat in [+pi/2, -pi/2]
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h
    lon = (xs * 2.0 - 1.0) * np.pi          # -pi (Back) .. 0 (Front) .. +pi (Back)
    lat = (0.5 - ys) * np.pi                # +pi/2 (Up) .. -pi/2 (Down)

    cos_lat = np.cos(lat)[:, None]
    sin_lat = np.sin(lat)[:, None]
    sin_lon = np.sin(lon)[None, :]
    cos_lon = np.cos(lon)[None, :]

    # 3D ray direction: +Z Front (face 0), +X Right (face 1), +Y Up (face 4)
    dx = cos_lat * sin_lon
    dy = np.broadcast_to(sin_lat, (out_h, out_w))
    dz = cos_lat * cos_lon

    ax = np.abs(dx)
    ay = np.abs(dy)
    az = np.abs(dz)

    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # Face 0: +Z (Front) -> u = (dx/az + 1)/2, v = (-dy/az + 1)/2
    m0 = (az >= ax) & (az >= ay) & (dz > 0)
    out[m0] = sample_bilinear(faces[0], (dx[m0] / az[m0] + 1.0) * 0.5, (-dy[m0] / az[m0] + 1.0) * 0.5)

    # Face 1: +X (Right) -> u = (-dz/ax + 1)/2, v = (-dy/ax + 1)/2
    m1 = (ax > az) & (ax >= ay) & (dx > 0)
    out[m1] = sample_bilinear(faces[1], (-dz[m1] / ax[m1] + 1.0) * 0.5, (-dy[m1] / ax[m1] + 1.0) * 0.5)

    # Face 2: -Z (Back) -> u = (-dx/az + 1)/2, v = (-dy/az + 1)/2
    m2 = (az >= ax) & (az >= ay) & (dz <= 0)
    out[m2] = sample_bilinear(faces[2], (-dx[m2] / az[m2] + 1.0) * 0.5, (-dy[m2] / az[m2] + 1.0) * 0.5)

    # Face 3: -X (Left) -> u = (dz/ax + 1)/2, v = (-dy/ax + 1)/2
    m3 = (ax > az) & (ax >= ay) & (dx <= 0)
    out[m3] = sample_bilinear(faces[3], (dz[m3] / ax[m3] + 1.0) * 0.5, (-dy[m3] / ax[m3] + 1.0) * 0.5)

    # Face 4: +Y (Top) -> u = (dx/ay + 1)/2, v = (dz/ay + 1)/2
    m4 = (ay > ax) & (ay > az) & (dy > 0)
    out[m4] = sample_bilinear(faces[4], (dx[m4] / ay[m4] + 1.0) * 0.5, (dz[m4] / ay[m4] + 1.0) * 0.5)

    # Face 5: -Y (Bottom) -> u = (dx/ay + 1)/2, v = (-dz/ay + 1)/2
    m5 = (ay > ax) & (ay > az) & (dy <= 0)
    out[m5] = sample_bilinear(faces[5], (dx[m5] / ay[m5] + 1.0) * 0.5, (-dz[m5] / ay[m5] + 1.0) * 0.5)

    Image.fromarray(out).save(dst_path, "JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
    kb = os.path.getsize(dst_path) / 1024
    print(f"Stitched {dst_path} ({out_w}x{out_h}, {kb:.1f} KB)")


def main():
    os.makedirs("assets/tour360", exist_ok=True)
    for folder, out_name in SCENES:
        src = os.path.join("NWU_Full_Tour", folder)
        if not os.path.isdir(src):
            print(f"Skipping missing {src}")
            continue
        cubemap_to_equirect(src, os.path.join("assets/tour360", out_name))


if __name__ == "__main__":
    main()
