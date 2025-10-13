import os
import argparse
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import tifffile
import cv2

from dataset import RingBeamDataset, read_image
from model import build_model

# ---------------------------
# Polar utilities (adapted from preprocess.py, minimized)
# ---------------------------
def _contour_centroid(img: np.ndarray, threshold: float) -> tuple[float, float, float]:
    h, w = img.shape
    max_val = float(img.max())
    if max_val > 0:
        norm_img = (img / max_val * 255).astype(np.uint8)
        binary_thresh = int((threshold / max_val) * 255)
    else:
        norm_img = img.astype(np.uint8)
        binary_thresh = 0
    _, binary = cv2.threshold(norm_img, binary_thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (w - 1) / 2.0, (h - 1) / 2.0, min(w, h) / 2.0
    main_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(main_contour)
    return float(x), float(y), float(radius)


def _bilinear_sample(image: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    h, w = image.shape
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    Ia = image[y0c, x0c]
    Ib = image[y1c, x0c]
    Ic = image[y0c, x1c]
    Id = image[y1c, x1c]
    wa = (x1 - xs) * (y1 - ys)
    wb = (x1 - xs) * (ys - y0)
    wc = (xs - x0) * (y1 - ys)
    wd = (xs - x0) * (ys - y0)
    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return out.astype(np.float32)


def _polar_resample(img: np.ndarray, cx: float, cy: float, r_bins: int, s_bins: int, rmin: float = 0.0, rmax: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    if rmax is None:
        rmax = float(min(cx, cy, w - 1 - cx, h - 1 - cy))
    rs = np.linspace(rmin, rmax, r_bins, dtype=np.float32)
    th = np.linspace(0.0, 2 * math.pi, s_bins, endpoint=False, dtype=np.float32)
    Rg, Tg = np.meshgrid(rs, th, indexing="ij")
    xs = cx + Rg * np.cos(Tg)
    ys = cy + Rg * np.sin(Tg)
    resampled_image = _bilinear_sample(img, ys, xs)
    return resampled_image, rs


def main():
    parser = argparse.ArgumentParser(description="Extract features from the beam dataset (with TIFF crop + color).")
    parser.add_argument('--data_dir', type=str, default='dataset', help='Root directory of the dataset.')
    parser.add_argument('--output_dir', type=str, default='features', help='Directory to save the extracted features.')
    parser.add_argument('--encoder', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_small'],
                        help='Encoder model to use for feature extraction.')
    parser.add_argument('--classes', nargs='+', help='List of class names to process.')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size.')
    parser.add_argument('--stride', type=int, default=128, help='Stride for patch extraction.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--crop', action='store_true', help='Enable cropping.')
    parser.add_argument('--crop_x', type=int, default=1550)
    parser.add_argument('--crop_y', type=int, default=900)
    parser.add_argument('--crop_w', type=int, default=1024)
    parser.add_argument('--crop_h', type=int, default=1024)
    parser.add_argument('--threshold', type=float, default=8300.0, help='Sum threshold for filtering dark patches.')
    # Polarization options
    parser.add_argument('--use_polar', action='store_true', help='Apply polarization (polar transform) before patching/encoding.')
    parser.add_argument('--r_bins', type=int, default=512, help='Radial resolution for polar transform.')
    parser.add_argument('--s_bins', type=int, default=1024, help='Angular resolution for polar transform.')
    parser.add_argument('--polar_threshold', type=float, default=0.0, help='Threshold used to find ring contour for initial radius.')
    parser.add_argument('--signal_ratio', type=float, default=0.2, help='Ratio factor for focusing signal region in radius (0~1).')

    args = parser.parse_args()
    stride = args.stride if args.stride else args.patch_size

    output_dir = args.output_dir


    if args.classes:
        class_names = args.classes
    else:
        class_names = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]
        if not class_names:
            print(f"[ERROR] No class subdirectories found in {args.data_dir}. Please specify classes via the --classes argument.")
            return
    print(f"[INFO] Processing classes: {class_names}")

    print(f"[INFO] Starting feature extraction using encoder '{args.encoder}'.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(
        encoder_name=args.encoder,
        num_classes=len(class_names), 
        in_channels=3,
        pretrained=True,
        freeze_encoder=True
    )
    encoder = model.encoder.to(device)
    encoder.eval()

    # Save under features/polar/<encoder> when polar is enabled, else features/<encoder>
    features_output_dir = os.path.join(output_dir, 'polar', args.encoder) if args.use_polar else os.path.join(output_dir, args.encoder)
    for name in class_names:
        os.makedirs(os.path.join(features_output_dir, name), exist_ok=True)
    
    print(f"[INFO] Features will be saved to: {features_output_dir}")

    dataset = RingBeamDataset(
        root_path=args.data_dir,
        target_size=None,
        classes=class_names,
        crop=args.crop,
        crop_coords=(args.crop_x, args.crop_y, args.crop_w, args.crop_h)
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)


    with torch.no_grad():
        for img_full, label, fname in tqdm(data_loader, desc="Extracting Features (TIFF-Color Patch)"):
            label = label.item()
            fname = fname[0]
            class_name = class_names[label]

            # Retrieve grayscale image (H, W)
            img_full = img_full.squeeze(0).squeeze(0).numpy()

            # Optional polar transform before color mapping
            if args.use_polar:
                # Step 1: estimate center and initial radius
                cx, cy, initial_rmax = _contour_centroid(img_full, args.polar_threshold)
                # Step 2: coarse polar to find signal region
                polar_coarse, rs_coarse = _polar_resample(img_full, cx, cy, args.r_bins, args.s_bins, rmin=0.0, rmax=initial_rmax)
                radial_mean_profile = polar_coarse.mean(axis=1)
                # Heuristic focusing range using ratio of mean
                signal_threshold = float(radial_mean_profile.mean()) * float(args.signal_ratio)
                significant_indices = np.where(radial_mean_profile > signal_threshold)[0]
                if len(significant_indices) > 0:
                    r_idx_min = int(significant_indices.min())
                    r_idx_max = int(significant_indices.max())
                    focused_rmin = float(rs_coarse[r_idx_min])
                    focused_rmax = float(rs_coarse[r_idx_max])
                else:
                    focused_rmin = 0.0
                    focused_rmax = float(initial_rmax)
                # Step 3: final polar image for feature extraction
                polar_img, _ = _polar_resample(img_full, cx, cy, args.r_bins, args.s_bins, rmin=focused_rmin, rmax=focused_rmax)
                work_img = polar_img
            else:
                work_img = img_full

            # Normalize and colorize to RGB for encoders trained on 3-channel images
            img_min, img_max = float(work_img.min()), float(work_img.max())
            if img_max > img_min:
                img_norm = (work_img - img_min) / (img_max - img_min)
            else:
                img_norm = np.zeros_like(work_img, dtype=np.float32)
            img_8bit = (img_norm * 255).astype(np.uint8)

            img_color = cv2.applyColorMap(img_8bit, cv2.COLORMAP_JET)
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

            ph, pw = args.patch_size, args.patch_size
            patches = []
            H, W, _ = img_color.shape
            for i in range(0, H, stride):
                for j in range(0, W, stride):
                    if i + ph > H or j + pw > W:
                        continue
                    patch = img_color[i:i+ph, j:j+pw, :]
                    patches.append(patch)

            if len(patches) == 0:
                continue

            tensors = [torch.tensor(p, dtype=torch.float32).permute(2, 0, 1) / 255.0 for p in patches]
            batch_tensor = torch.stack(tensors).to(device)
            patch_sums = batch_tensor.sum(dim=(1, 2, 3))
            mask = patch_sums >= args.threshold
            batch_tensor = batch_tensor[mask]

            if batch_tensor.shape[0] == 0:
                print(f"[WARN] All patches filtered out for {fname}")
                continue

            features = encoder(batch_tensor).cpu().squeeze()
            if len(features.shape) == 1:
                features = features.unsqueeze(0)

            save_path = os.path.join(features_output_dir, class_name, f"{os.path.splitext(fname)[0]}_features.pt")
            torch.save({'features': features, 'label': label}, save_path)
            print(f"[INFO] Saved features: {save_path}")

    print("\nFeature extraction completed successfully!")


def fetch_tif_files(data_dir):
    tif_files = glob.glob(os.path.join(data_dir, '**', '*.tif'), recursive=True)
    for f in tif_files:
        print(f)
    return tif_files


if __name__ == '__main__':
    main()
