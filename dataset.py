import os
import re
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile

def read_image(path, normalize=True, max_value=65535.0):
    img_np = tifffile.imread(path).astype(np.float32)
    if normalize:
        img_np /= max_value
    return img_np

def downsample_image_np(img_np, target_size):
    if img_np.shape[0] > target_size[0] and img_np.shape[1] > target_size[1]:
        stride_y = img_np.shape[0] // target_size[0]
        stride_x = img_np.shape[1] // target_size[1]
        return img_np[::stride_y, ::stride_x][:target_size[0], :target_size[1]]
    return img_np

class RingBeamDataset(Dataset):
    def __init__(self, root_path, classes: list[str], target_size=(256, 256),
                 normalize=True, crop=False, crop_coords=(0, 0, 0, 0)):
        self.root_path = root_path
        self.target_size = target_size
        self.normalize = normalize
        self.max_value = 4096.0 if normalize else 1.0
        self.crop = crop
        self.crop_coords = crop_coords
        self.class_names = classes
        self.num_classes = len(classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.items = []
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_path, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue
            # :흰색_확인_표시: 하위 폴더까지 재귀적으로 순회
            for root, _, files in os.walk(class_dir):
                for fname in files:
                    if fname.lower().endswith(('.tiff', '.tif')):
                        fpath = os.path.join(root, fname)
                        self.items.append((fpath, self.class_to_idx[class_name]))
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = read_image(path, normalize=self.normalize, max_value=self.max_value)
        if self.crop:
            x, y, w, h = self.crop_coords
            img_h, img_w = img.shape
            if w == -1:
                w = img_w - x
            if h == -1:
                h = img_h - y
            img = img[y:y+h, x:x+w]
        if self.target_size is not None:
            img = downsample_image_np(img, self.target_size)
        tensor = torch.from_numpy(img.copy()).unsqueeze(0)
        return tensor, int(label), os.path.basename(path)

class BeamFeatureDataset(Dataset):
    def __init__(self, features_root_dir, classes: list[str], method='resize'):
        self.features_root_dir = features_root_dir
        self.method = method
        self.items = []

        self.class_names = classes
        self.num_classes = len(classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_dir = os.path.join(self.features_root_dir, class_name)
            print(class_dir)
            if not os.path.isdir(class_dir):
                continue

            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith('.pt'):
                    self.items.append((os.path.join(class_dir, fname), self.class_to_idx[class_name]))

        print(f"[INFO] Loaded {len(self.items)} .pt files from {features_root_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        data = torch.load(path, weights_only=True)
      
        feature_tensor = data['features'] 
        fname = os.path.basename(path)

        return feature_tensor, label, fname