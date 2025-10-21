import os, random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import matplotlib as mpl

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def save_checkpoint(state, checkpoint_dir, filename):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    return path

def load_checkpoint(path, model=None, optimizer=None, scheduler=None, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None: model.load_state_dict(ckpt['state_dict'])
    if optimizer is not None and 'optimizer' in ckpt: optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and 'scheduler' in ckpt: scheduler.load_state_dict(ckpt['scheduler'])
    return ckpt

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1): 
        self.val = val 
        self.sum += val * n 
        self.count += n 
        self.avg = self.sum / self.count

def accuracy(output, target):
    with torch.no_grad():
        pred = output.argmax(1)
        correct = (pred == target).sum().item()
        return correct / target.size(0)

def plot_confusion_matrix_image(y_true, y_pred, class_names):
    try:
        mpl.rc('font', family='Malgun Gothic')
        mpl.rc('axes', unicode_minus=False)
    except Exception as e:
        print(f"폰트 설정 경고: {e}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True', xlabel='Pred')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh/2 else "black")
    fig.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    image = np.array(image)
    return image