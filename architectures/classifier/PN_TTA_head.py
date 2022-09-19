from logging import setLoggerClass
import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.utils import L2SquareDist
from torch import Tensor

class PN_TTA_head(nn.Module):
    r"""The metric-based protypical classifier from ``Prototypical Networks for Few-shot Learning''.

    Args:
        metric: Whether use cosine or enclidean distance.
        scale_cls: The initial scale number which affects the following softmax function.
        learn_scale: Whether make scale number learnable.
        normalize: Whether normalize each spatial dimension of image features before average pooling.
    """
    def __init__(
        self, 
        metric: str = "cosine", 
        scale_cls: int =10.0, 
        learn_scale: bool = True, 
        normalize: bool = True,
        lambd=0
        ) -> None:
        super().__init__()
        assert metric in ["cosine", "enclidean"]
        if learn_scale:
            self.scale_cls = nn.Parameter(
                torch.FloatTensor(1).fill_(scale_cls), requires_grad=True
            )    
        else:
            self.scale_cls = scale_cls
        self.metric = metric
        self.normalize = normalize
        self.lambd = lambd



    def forward(self, features_test: Tensor, features_train: Tensor, 
                way: int, shot: int, patch_query:Tensor, patch_support:Tensor) -> Tensor:
        r"""Take batches of few-shot training examples and testing examples as input,
            output the logits of each testing examples.

        Args:
            features_test: Testing examples. size: [batch_size, num_query, c, h, w]
            features_train: Training examples which has labels like:[abcdabcdabcd].
                            size: [batch_size, way*shot, c, h, w]
            way: The number of classes of each few-shot classification task.
            shot: The number of training images per class in each few-shot classification
                  task.
        Output:
            classification_scores: The calculated logits of testing examples.
                                   size: [batch_size, num_query, way]
        """
        batchsize, _, cropNum, c, w, h = patch_support.shape
        if features_train.dim() == 5:
            if self.normalize:
                features_train=F.normalize(features_train, p=2, dim=2, eps=1e-12)
                patch_support = patch_support.reshape([batchsize, shot, way, cropNum, c, w, h])
                patch_support = patch_support.permute(0, 2, 1, 3, 4, 5, 6)
                patch_support = patch_support.reshape([batchsize, way, -1 ,c, w, h])
                patch_center=F.normalize(patch_support.mean(2), p=2, dim=-3, eps=1e-12)
            features_train = F.adaptive_avg_pool2d(features_train, 1).squeeze_(-1).squeeze_(-1)
            patch_center = F.adaptive_avg_pool2d(patch_center, 1).squeeze_(-1).squeeze_(-1)
        assert features_train.dim() == 3

        batch_size = features_train.size(0)
        if self.metric == "cosine":
            features_train = F.normalize(features_train, p=2, dim=2, eps=1e-12)
            patch_center = F.normalize(patch_center, p=2, dim=-1, eps=1e-12)
        #prototypes: [batch_size, way, c]
        prototypes = torch.mean(features_train.reshape(batch_size, shot, way, -1),dim=1)
        # prototypes = prototypes.reshape(batch_size*way, -1)

        prototypes = self.lambd*prototypes + (1-self.lambd)*patch_center
        
        prototypes = F.normalize(prototypes, p=2, dim=2, eps=1e-12)

        patch_query = patch_query.mean(2)
        if self.normalize:
            features_test=F.normalize(features_test, p=2, dim=2, eps=1e-12)
            patch_query = F.normalize(patch_query, p=2, dim=2, eps=1e-12)
        if features_test.dim() == 5:
            features_test = F.adaptive_avg_pool2d(features_test, 1).squeeze_(-1).squeeze_(-1)
            patch_query = F.adaptive_avg_pool2d(patch_query, 1).squeeze_(-1).squeeze_(-1)
        assert features_test.dim() == 3

        features_test = self.lambd * features_test + (1-self.lambd) * patch_query

        if self.metric == "cosine":
            features_test = F.normalize(features_test, p=2, dim=2, eps=1e-12)
            #[batch_size, num_query, c] * [batch_size, c, way] -> [batch_size, num_query, way]
            classification_scores = self.scale_cls * torch.bmm(features_test, prototypes.transpose(1, 2))

        elif self.metric == "euclidean":
            classification_scores = -self.scale_cls * L2SquareDist(features_test, prototypes)
        return classification_scores

def create_model(metric: str = "cosine", 
        scale_cls: int =10.0, 
        learn_scale: bool = True, 
        normalize: bool = True):
    return PN_TTA_head(metric, scale_cls, learn_scale, normalize)
