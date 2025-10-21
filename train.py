import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import AverageMeter, save_checkpoint, load_checkpoint, plot_confusion_matrix_image
from evaluate import run_one_epoch

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    pbar = tqdm(loader, desc="[Train] Batch", unit="batch", leave=False)
    for data_items in loader:
        if len(data_items) == 4:  # Patch-based loader
            inputs, labels, fnames, lengths = data_items
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            outputs = model(inputs, lengths)
        else:  # Image or resize-feature loader
            inputs, labels = data_items[0], data_items[1]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

        if isinstance(criterion, nn.BCEWithLogitsLoss):
            outs = outputs.view(-1)
            y = labels.float().view(-1)
            loss = criterion(outs, y)
            probs = torch.sigmoid(outs)
            preds = (probs >= 0.5).long()
            y_true = y.long()
        elif isinstance(criterion, nn.BCELoss):
            outs = outputs.view(-1)
            y = labels.float().view(-1)
            loss = criterion(outs, y)
            preds = (outs >= 0.5).long()
            y_true = y.long()
        else:
            loss = criterion(outputs, labels)
            preds = outputs.argmax(1)
            y_true = labels.long()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        n = y_true.size(0)
        batch_acc = (preds == y_true).float().mean().item()
        loss_meter.update(loss.item(), n)
        acc_meter.update(batch_acc, n)
        pbar.set_postfix(step_loss=f"{loss_meter.avg:.4f}", step_acc=f"{acc_meter.avg:.4f}")
    pbar.close()
    return loss_meter.avg, acc_meter.avg

def run_training(args, model, train_loader, val_loader, criterion, optimizer, device):
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    sanitized_classes = '_'.join(c.replace('+', '-') for c in args.classes)
    run_name = f"{sanitized_classes}_{args.encoder}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = os.path.join(args.exp_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, 'checkpoints'); os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tb'))
    start_epoch, best_val_acc, no_improve = 1, 0.0, 0

    if args.resume:
        ckpt = load_checkpoint(args.resume, model=model, optimizer=optimizer, scheduler=scheduler, map_location=device)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, y_true_val, y_pred_val = run_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        writer.add_scalar('train/loss', tr_loss, epoch)
        writer.add_scalar('train/acc', tr_acc, epoch)
        writer.add_scalar('val/loss', va_loss, epoch)
        writer.add_scalar('val/acc', va_acc, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        cm_img = plot_confusion_matrix_image(y_true_val, y_pred_val, class_names=args.classes)
        writer.add_image('val/confusion_matrix', cm_img, epoch, dataformats='HWC')

        print(f"[{epoch:03d}/{args.epochs}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={va_loss:.4f} acc={va_acc:.4f}")

        is_best = va_acc > best_val_acc
        if is_best:
            best_val_acc = va_acc
            save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'best_val_acc': best_val_acc, 'config': vars(args)}, ckpt_dir, "model_best.pth.tar")
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= args.early_stop:
            print(f"Early stopping at epoch {epoch}.")
            break
    
    best_model_path = os.path.join(ckpt_dir, "model_best.pth.tar")
    
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"best_val_acc": best_val_acc, "config": vars(args)}, f, indent=2, ensure_ascii=False)

    writer.close()
    print(f"Run dir: {run_dir}")
    return best_model_path, run_dir, writer