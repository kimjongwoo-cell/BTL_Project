import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class BeamClassifier(nn.Module):

    def __init__(self, encoder, num_features, num_classes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def build_model(encoder_name: str, num_classes: int, in_channels: int = 1, pretrained: bool = True, freeze_encoder: bool = True):
    encoder = None
    num_features = 0
    weights = 'DEFAULT' if pretrained else None

    if encoder_name == 'resnet18':
        encoder = models.resnet18(weights=weights)
        num_features = encoder.fc.in_features
        encoder.fc = nn.Identity()

    elif encoder_name == 'resnet50':
        encoder = models.resnet50(weights=weights)
        num_features = encoder.fc.in_features
        encoder.fc = nn.Identity()

    elif encoder_name == 'efficientnet_b0':
        encoder = models.efficientnet_b0(weights=weights)
        num_features = encoder.classifier[1].in_features
        encoder.classifier = nn.Identity()

    elif encoder_name == 'mobilenet_v3_small':
        encoder = models.mobilenet_v3_small(weights=weights)
        num_features = encoder.classifier[0].in_features
        encoder.classifier = nn.Identity()
        encoder = nn.Sequential(
            encoder.features,
            encoder.avgpool
        )

    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")

    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False

    model = BeamClassifier(encoder, num_features, num_classes)
    return model

def get_encoder_feature_dim(encoder_name: str) -> int:
    if encoder_name == 'resnet18':
        return 512
    elif encoder_name == 'resnet50':
        return 2048
    elif encoder_name == 'efficientnet_b0':
        return 1280
    elif encoder_name == 'mobilenet_v3_small':
        return 576
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")

class FeatureClassifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.classifier(x)

class PatchClassifier(nn.Module):
    """Classifier for a sequence of patch features."""
    def __init__(self, num_features: int, num_classes: int, pooling_method: str = 'mean'):
        super().__init__()
        self.pooling_method = pooling_method
        self.num_features = num_features

        if self.pooling_method == 'abmil':
            self.mil = GatedAttention(num_classes)
        elif self.pooling_method == 'transmil':
            self.mil = TransMIL(num_classes)   
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x, lengths=None):
        # x shape: (batch_size, num_patches, feature_dim)
        if lengths is not None:
            mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]

        if self.pooling_method == 'mean':
            if lengths is not None:
                x = x * mask.unsqueeze(-1)
                pooled_features = torch.sum(x, dim=1) / lengths.unsqueeze(-1)
            else:
                pooled_features = torch.mean(x, dim=1)
            return self.classifier(pooled_features)
    
        elif self.pooling_method == 'max':
            if lengths is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), -1e9)
            pooled_features, _ = torch.max(x, dim=1)
            return self.classifier(pooled_features)
    
        elif self.pooling_method == 'abmil':
            return self.mil(x)
       
        elif self.pooling_method == 'transmil':
            return self.mil(x) 
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.M = 500
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # KxM

        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class GatedAttention(nn.Module):
    def __init__(self,out_dim):
        super(GatedAttention, self).__init__()
        self.M = 512
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.M),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix U
            nn.Sigmoid()
        )

        self.attention_w = nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)

        self.classifier = nn.Sequential(
            nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        A_V = self.attention_V(x)  # KxL
        A_U = self.attention_U(x)  # KxL
        A = self.attention_w(A_V * A_U) # element wise multiplication # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, x)  # ATTENTION_BRANCHESxM

        Y_prob = self.classifier(Z)
        Y_hat = torch.ge(Y_prob, 0.5).float()
 
        return Y_prob

    import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)

        self._fc2 = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        h = x #[B, n, 1024]
        
        h = self._fc1(h) #[B, n, 512]
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        return logits

