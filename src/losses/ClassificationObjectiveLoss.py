
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from globals import Globals


class ClassificationObjectiveLoss(nn.Module):

    def __init__(self, loss_fct = nn.BCELoss()):
        super(ClassificationObjectiveLoss, self).__init__()
        self.loss_fct = loss_fct

        self.fc1 = nn.Linear(384*3, 1).to(Globals.device)
        self.activation = nn.Sigmoid()


    def forward(self, source_embeddings, target_embeddings, labels: Tensor):

        x = self.fc1(torch.cat((source_embeddings, target_embeddings, torch.abs(target_embeddings - source_embeddings)), dim=1))
        # x =  self.fc1(torch.abs(target_embeddings - source_embeddings))

        output = self.activation(x).view((len(labels)))

        labels = labels.to(torch.float)

        return self.loss_fct(output, labels)
    
    def forward_mlp(self, source_embeddings, target_embeddings):
        x = self.fc1(torch.cat((source_embeddings, target_embeddings, torch.abs(target_embeddings - source_embeddings)), dim=1))
        # x =  self.fc1(torch.abs(target_embeddings - source_embeddings))
        output = self.activation(x)

        return output