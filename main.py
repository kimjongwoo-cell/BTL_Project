import os, json, time
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from data_loader import build_feature_loaders, build_image_loaders
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
    parser.add_argument('--classes', nargs='+', required=True, help='List of class names to train on. Use '+' to group classes, e.g., Ref Particle+Crack')
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
    parser.add_argument('--split_method', type=str, default='ratio', choices=['ratio', 'fixed'], help="Method for splitting data.")
    parser.add_argument('--train_samples_per_class', type=int, default=None, help="Number of training samples per class (used with --split_method fixed).")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    
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
        print(f"[INFO] Training with pre-extracted features from {args.features_dir}")
        train_loader, val_loader, test_loader, label_counts = build_feature_loaders(
            features_dir=args.features_dir,
            source_classes=source_classes,
            target_classes=target_classes,
            label_map=label_map,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            method=args.method,
            split_method=args.split_method,
            train_samples_per_class=args.train_samples_per_class
        )
        num_features = get_encoder_feature_dim(args.encoder)
        
        if args.method == 'patch':
            model = PatchClassifier(num_features, num_classes, pooling_method=args.pooling).to(device)
            print(f"[INFO] Model: PatchClassifier built for '{args.encoder}' features with '{args.pooling}' pooling.")
        else: # resize
            model = FeatureClassifier(num_features, num_classes).to(device)
            print(f"[INFO] Model: FeatureClassifier built for '{args.encoder}' features.")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    else:
        train_loader, val_loader, test_loader, label_counts = build_image_loaders(
            args.data_dir, 
            source_classes=source_classes,
            target_classes=target_classes,
            label_map=label_map,
            target_size=(args.target_h, args.target_w), 
            batch_size=args.batch_size, 
            num_workers=args.num_workers, 
            seed=args.seed, 
            use_augmentation=args.use_augmentation,
            split_method=args.split_method,
            train_samples_per_class=args.train_samples_per_class
        )
        model = build_model(
            encoder_name=args.encoder, num_classes=num_classes, in_channels=1,
            pretrained=True, freeze_encoder=not args.unfreeze_encoder
        ).to(device)
        print(f"[INFO] Model: Full image classifier built with '{args.encoder}' encoder.")
        if not args.unfreeze_encoder:
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = nn.BCELoss()

    best_model_path, run_dir, writer = run_training(args, model, train_loader, val_loader, criterion, optimizer, device)

    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model=model, map_location=device)
    
    test_acc = run_final_evaluation(model, test_loader, criterion, device, target_classes, writer)

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    
    summary["test_acc"] = float(test_acc)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
