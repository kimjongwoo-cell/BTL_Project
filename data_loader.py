import os
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from skimage.transform import rotate

from dataset import RingBeamDataset, BeamFeatureDataset, read_image, downsample_image_np

def collate_fn_patch(batch):
    features, labels, fnames = zip(*batch)
    lengths = torch.tensor([len(f) for f in features])
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return features, labels, fnames, lengths

def build_image_loaders(data_dir, source_classes, target_classes, label_map, target_size, batch_size, num_workers, seed, use_augmentation, test_frac=0.1, val_frac=0.2, split_method='ratio', train_samples_per_class=None):
    print("[INFO] Building loaders from raw image data.")
    full_ds = RingBeamDataset(root_path=data_dir, classes=source_classes, target_size=target_size, normalize=True)
    base_items = full_ds.items

    augment_factor = 10 if use_augmentation else 1
    augmented_items_with_labels = []
    for path, original_label in base_items:
        target_label = label_map[original_label]
        for i in range(augment_factor):
            augmented_items_with_labels.append(((path, i), target_label))

    if split_method == 'ratio':
        print("[INFO] Splitting data by ratio.")
        augmented_items = [item for item, label in augmented_items_with_labels]
        target_labels = [label for item, label in augmented_items_with_labels]

        items_trainval, items_test, y_trainval, y_test = train_test_split(
            augmented_items, target_labels, test_size=test_frac, stratify=target_labels, random_state=seed
        )
        items_train, items_val, y_train, y_val = train_test_split(
            items_trainval, y_trainval, test_size=val_frac, stratify=y_trainval, random_state=seed
        )
    elif split_method == 'fixed':
        if not train_samples_per_class:
            raise ValueError("train_samples_per_class must be set for 'fixed' split method.")
        print(f"[INFO] Splitting data with a fixed number of {train_samples_per_class} training samples per class.")
        
        class_items = {i: [] for i in range(len(target_classes))}
        for item, label in augmented_items_with_labels:
            class_items[label].append(item)

        items_train, items_val, items_test = [], [], []
        y_train, y_val, y_test = [], [], []

        for label, items_for_class in sorted(class_items.items()):
            np.random.shuffle(items_for_class)
            
            class_train_items = items_for_class[:train_samples_per_class]
            remainder = items_for_class[train_samples_per_class:]
            
            if len(remainder) > 1:
                class_val_items, class_test_items = train_test_split(remainder, test_size=0.5, random_state=seed)
            else:
                class_val_items, class_test_items = remainder, []

            items_train.extend(class_train_items)
            items_val.extend(class_val_items)
            items_test.extend(class_test_items)
            y_train.extend([label] * len(class_train_items))
            y_val.extend([label] * len(class_val_items))
            y_test.extend([label] * len(class_test_items))

    train_counts, val_counts, test_counts = Counter(y_train), Counter(y_val), Counter(y_test)
    print(f"- Train: {len(items_train)} samples. Distribution: {{ {', '.join([f'{target_classes[k]}: {v}' for k, v in sorted(train_counts.items())])} }}")
    print(f"- Validation: {len(items_val)} samples. Distribution: {{ {', '.join([f'{target_classes[k]}: {v}' for k, v in sorted(val_counts.items())])} }}")
    print(f"- Test: {len(items_test)} samples. Distribution: {{ {', '.join([f'{target_classes[k]}: {v}' for k, v in sorted(test_counts.items())])} }}")

    def _apply_augmentation(img, aug_idx):
        if aug_idx == 1: return np.fliplr(img)
        if aug_idx == 2: return np.flipud(img)
        if aug_idx == 3: return rotate(img, angle=90, resize=False, preserve_range=True)
        if aug_idx == 4: return rotate(img, angle=270, resize=False, preserve_range=True)
        return img

    class AugmentedSubset(Dataset):
        def __init__(self, items, labels):
            self.items = items
            self.labels = labels

        def __len__(self): return len(self.items)

        def __getitem__(self, idx):
            (path, aug_idx), label = self.items[idx], self.labels[idx]
            img = read_image(path, normalize=full_ds.normalize, max_value=full_ds.max_value)
            if full_ds.target_size is not None:
                img = downsample_image_np(img, full_ds.target_size)
            img = _apply_augmentation(img, aug_idx)
            tensor = torch.from_numpy(img.copy()).unsqueeze(0)
            return tensor, int(label), os.path.basename(path)

    train_ds, val_ds, test_ds = AugmentedSubset(items_train, y_train), AugmentedSubset(items_val, y_val), AugmentedSubset(items_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, Counter(y_train)

def build_feature_loaders(features_dir, source_classes, target_classes, label_map, batch_size, num_workers, seed, method='resize', test_frac=0.1, val_frac=0.2, split_method='ratio', train_samples_per_class=None):
    print(f"[INFO] Building loaders from pre-extracted features in: {features_dir}")
    full_ds = BeamFeatureDataset(features_root_dir=features_dir, classes=source_classes, method=method)
  
    indices = list(range(len(full_ds)))
    original_labels = [full_ds[i][1] for i in indices]
    target_labels = [label_map[l] for l in original_labels]

    class_indices = {}
    for i, label in enumerate(target_labels):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(indices[i])

    train_indices, val_indices, test_indices = [], [], []

    if split_method == 'ratio':
        print("[INFO] Splitting data by ratio.")
        for label, indices_for_class in sorted(class_indices.items()):
            n_samples_class = len(indices_for_class)
            
            test_split_idx = int(n_samples_class * (1 - test_frac))
            class_trainval_indices = indices_for_class[:test_split_idx]
            class_test_indices = indices_for_class[test_split_idx:]
            
            val_split_idx = int(len(class_trainval_indices) * (1 - val_frac))
            class_train_indices = class_trainval_indices[:val_split_idx]
            class_val_indices = class_trainval_indices[val_split_idx:]
            
            train_indices.extend(class_train_indices)
            val_indices.extend(class_val_indices)
            test_indices.extend(class_test_indices)
    
    elif split_method == 'fixed':
        if not train_samples_per_class:
            raise ValueError("train_samples_per_class must be set for 'fixed' split method.")
        print(f"[INFO] Splitting data with a fixed number of {train_samples_per_class} training samples per class.")
        for label, indices_for_class in sorted(class_indices.items()):
            np.random.shuffle(indices_for_class)
            
            class_train_indices = indices_for_class[:train_samples_per_class]
            remainder = indices_for_class[train_samples_per_class:]
            
            if len(remainder) > 1:
                # Split remainder into validation and test sets (9:5 ratio)
                class_val_indices, class_test_indices = train_test_split(remainder, test_size=0.5, random_state=seed)
            else:
                class_val_indices, class_test_indices = remainder, []

            train_indices.extend(class_train_indices)
            val_indices.extend(class_val_indices)
            test_indices.extend(class_test_indices)

    
    train_indices.sort()
    val_indices.sort()
    test_indices.sort()

    class LabelRemapper(Dataset):
        def __init__(self, subset, label_map):
            self.subset = subset
            self.label_map = label_map
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            feature, original_label, fname = self.subset[idx]
            return feature, self.label_map[original_label], fname

    train_ds = LabelRemapper(Subset(full_ds, train_indices), label_map)
    val_ds = LabelRemapper(Subset(full_ds, val_indices), label_map)
    test_ds = LabelRemapper(Subset(full_ds, test_indices), label_map)

    train_target_labels = [target_labels[i] for i in train_indices]
    val_target_labels = [target_labels[i] for i in val_indices]
    test_target_labels = [target_labels[i] for i in test_indices]

    train_counts, val_counts, test_counts = Counter(train_target_labels), Counter(val_target_labels), Counter(test_target_labels)
    print(f"- Train: {len(train_ds)} samples. Distribution: {{ {', '.join([f'{target_classes[k]}: {v}' for k, v in sorted(train_counts.items())])} }}")
    print(f"- Validation: {len(val_ds)} samples. Distribution: {{ {', '.join([f'{target_classes[k]}: {v}' for k, v in sorted(val_counts.items())])} }}")
    print(f"- Test: {len(test_ds)} samples. Distribution: {{ {', '.join([f'{target_classes[k]}: {v}' for k, v in sorted(test_counts.items())])} }}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn_patch if method == 'patch' else None)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_patch if method == 'patch' else None)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn_patch if method == 'patch' else None)
    return train_loader, val_loader, test_loader, Counter(train_target_labels)
