import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import tifffile
import cv2

from dataset import RingBeamDataset, read_image
from model import build_model


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
    parser.add_argument('--crop_y', type=int, default=1000)
    parser.add_argument('--crop_w', type=int, default=1024)
    parser.add_argument('--crop_h', type=int, default=1024)
    parser.add_argument('--threshold', type=float, default=8300.0, help='Sum threshold for filtering dark patches.')

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

    features_output_dir = os.path.join(output_dir, args.encoder)
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

            img_full = img_full.squeeze(0).squeeze(0).numpy()

            img_norm = (img_full - img_full.min()) / (img_full.max() - img_full.min())
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
