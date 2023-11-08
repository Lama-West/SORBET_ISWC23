
import torch
from torch import nn, Tensor


class CosineSimilarityLoss(nn.Module):

    def __init__(self, loss_fct = nn.MSELoss()):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fct = loss_fct


    def forward(self, source_embeddings, target_embeddings, labels: Tensor):
        output = torch.cosine_similarity(source_embeddings, target_embeddings)
        return self.loss_fct(output, labels.view(-1))