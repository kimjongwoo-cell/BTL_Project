import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

from utils import AverageMeter, plot_confusion_matrix_image

def run_one_epoch(model, loader, criterion, device):
    model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    all_preds, all_labels = [], []

    with torch.no_grad():
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

            n = y_true.size(0)
            batch_acc = (preds == y_true).float().mean().item()
            loss_meter.update(loss.item(), n)
            acc_meter.update(batch_acc, n)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(y_true.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])
    return loss_meter.avg, acc_meter.avg, all_labels, all_preds

def run_final_evaluation(model, test_loader, criterion, device, target_classes, writer):
    print("Running final evaluation on test set.")
    te_loss, te_acc, y_true, y_pred = run_one_epoch(model, test_loader, criterion, device)
    
    report_dict = classification_report(y_true, y_pred, target_names=target_classes, output_dict=True, zero_division=0)
    f1_score = report_dict['weighted avg']['f1-score']

    print(f"[TEST] loss={te_loss:.4f}, acc={te_acc:.4f}, f1={f1_score:.4f}")

    print("\n" + "="*50)
    print(" " * 17, "Test Classification Report")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=target_classes, digits=4, zero_division=0))
    
    print("\n" + "="*50)
    print(" " * 17, "Test Confusion Matrix")
    print("="*50)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(target_classes))))
    cm_df = pd.DataFrame(cm, index=target_classes, columns=target_classes)
    print(cm_df)
    print("\n")

    cm_img_test = plot_confusion_matrix_image(y_true, y_pred, class_names=target_classes)
    if writer:
        writer.add_image('test/confusion_matrix', cm_img_test, 0, dataformats='HWC')
    
    return te_acc, f1_score
