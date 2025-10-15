import os, json, time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset, DataLoader

from data_loader import collate_fn_patch
from dataset import RingBeamDataset, BeamFeatureDataset, read_image, downsample_image_np
from model import build_model, get_encoder_feature_dim, FeatureClassifier, PatchClassifier
from train import run_training
from evaluate import run_final_evaluation
from utils import seed_everything, load_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset', help="Root directory for raw data (used if --features_dir is not set).")
    parser.add_argument('--features_dir', type=str, default=None, help="Root directory for pre-extracted features. If set, this overrides --data_dir.")
    parser.add_argument('--exp_dir', type=str, default='./experiments')
    parser.add_argument('--encoder', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'efficientnet_b0', 'mobilenet_v3_small'])
    parser.add_argument('--classes', nargs='+', required=True, help='List of class names to train on. Use "+" to group classes, e.g., Ref Particle+Crack')
    parser.add_argument('--method', type=str, default='resize', choices=['resize', 'patch'], help="Method for feature-based training.")
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max', 'abmil','transmil'], help="Pooling method for patch-based training.")
    parser.add_argument('--unfreeze_encoder', action='store_true', help='If set, the encoder is not frozen and will be fine-tuned (image mode only).')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--use_class_weights', action='store_true')
    parser.add_argument('--target_h', type=int, default=256, help="(Image mode only)")
    parser.add_argument('--target_w', type=int, default=256, help="(Image mode only)")
    parser.add_argument('--use_augmentation', action='store_true', help='Use data augmentation (image mode only)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--k_folds', type=int, default=5, help="Number of folds for cross-validation.")

    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    target_classes = args.classes
    num_classes = len(target_classes)

    source_classes = []
    for group in target_classes:
        source_classes.extend(group.split('+'))

    label_map = {}
    for i, source_class in enumerate(source_classes):
        for j, target_group in enumerate(target_classes):
            if source_class in target_group.split('+'):
                label_map[i] = j
                break
    
    if args.features_dir:
        full_ds = BeamFeatureDataset(features_root_dir=args.features_dir, classes=source_classes, method=args.method)
        all_labels = [label_map[full_ds[i][1]] for i in range(len(full_ds))]
    else:
        full_ds = RingBeamDataset(root_path=args.data_dir, classes=source_classes, target_size=(args.target_h, args.target_w), normalize=True)
        all_labels = [label_map[full_ds[i][1]] for i in range(len(full_ds))]

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    fold_accuracies = []
    fold_f1_scores = []

    for fold, (train_val_indices, test_indices) in enumerate(skf.split(np.arange(len(full_ds)), all_labels)):
        print(f"--- Fold {fold + 1}/{args.k_folds} ---")

        # Split train_val_indices into train and validation
        train_val_labels = [all_labels[i] for i in train_val_indices]
        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed) # 20% for validation
        try:
            train_indices, val_indices = next(skf_inner.split(train_val_indices, train_val_labels))
            train_indices = train_val_indices[train_indices]
            val_indices = train_val_indices[val_indices]
        except ValueError: # Handle small number of samples
            val_size = len(train_val_indices) // 5
            train_indices = train_val_indices[val_size:]
            val_indices = train_val_indices[:val_size]

        if args.features_dir:
            train_ds = Subset(full_ds, train_indices)
            val_ds = Subset(full_ds, val_indices)
            test_ds = Subset(full_ds, test_indices)
            
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn_patch if args.method == 'patch' else None)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_patch if args.method == 'patch' else None)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_patch if args.method == 'patch' else None)

            num_features = get_encoder_feature_dim(args.encoder)
            if args.method == 'patch':
                model = PatchClassifier(num_features, num_classes, pooling_method=args.pooling).to(device)
            else: # resize
                model = FeatureClassifier(num_features, num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            train_ds = Subset(full_ds, train_indices)
            val_ds = Subset(full_ds, val_indices)
            test_ds = Subset(full_ds, test_indices)

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            model = build_model(
                encoder_name=args.encoder, num_classes=num_classes, in_channels=1,
                pretrained=True, freeze_encoder=not args.unfreeze_encoder
            ).to(device)
            if not args.unfreeze_encoder:
                optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        criterion = nn.BCELoss()
        
        original_exp_dir = args.exp_dir
        args.exp_dir = os.path.join(original_exp_dir, f"fold_{{fold+1}}")

        best_model_path, run_dir, writer = run_training(args, model, train_loader, val_loader, criterion, optimizer, device)

        if os.path.exists(best_model_path):
            load_checkpoint(best_model_path, model=model, map_location=device)
        
        test_acc, test_f1 = run_final_evaluation(model, test_loader, criterion, device, target_classes, writer)
        fold_accuracies.append(test_acc)
        fold_f1_scores.append(test_f1)

        summary_path = os.path.join(run_dir, "summary.json")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        summary["test_acc"] = float(test_acc)
        summary["test_f1"] = float(test_f1)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        args.exp_dir = original_exp_dir # Reset for next fold

    print("\n--- Overall Summary ---")
    print(f"Fold accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"Fold F1-scores: {[f'{f1:.4f}' for f1 in fold_f1_scores]}")
    print(f"Average test accuracy over {args.k_folds}-folds: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    print(f"Average test F1-score over {args.k_folds}-folds: {np.mean(fold_f1_scores):.4f} (+/- {np.std(fold_f1_scores):.4f})")

if __name__ == "__main__":
    main()