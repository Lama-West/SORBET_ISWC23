
import torch
from torch import nn, Tensor
from transformers import AutoTokenizer, AutoModel

import json

with open("config.json", 'r') as f:
    config = json.load(f)

from globals import Globals


class SubclassLoss(nn.Module):

    def __init__(self):
        super(SubclassLoss, self).__init__()

        self.distance_loss = nn.MSELoss()

        self._compute_subclass_embeddings()


        
    def _compute_subclass_embeddings(self):
        sbert_tokenizer = AutoTokenizer.from_pretrained(config["General"]["model"])
        sbert_model = AutoModel.from_pretrained(config["General"]["model"], output_hidden_states=True)
        
        def _sbert_mean_pooling(model_output):
                token_embeddings = model_output[0] #First element of model_output contains all token embeddings
                token_embeddings = token_embeddings[:,1:-1] # Remove CLS and SEP tokens
                return torch.mean(token_embeddings, 1)



        encoded_input = sbert_tokenizer("<", padding=True, truncation=True, return_tensors='pt')

        model_output = sbert_model(**encoded_input)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings = _sbert_mean_pooling(model_output)
        self.subclass_relation = sentence_embeddings.to(Globals.device).detach()

    def _get_mean_embedding(self, model_output, mask):
        return torch.mean(model_output[mask.squeeze()], dim=0).view(1,-1)

    def forward(self, model_output, subclass_pair):
        """ loss function

        Args:
            model_output (tensor): output tensor of the model
            subclass_pair (Tuple): tuple in the form of (parent_mask, relation_mask, child_mask)

        Returns:
            _type_: loss
        """
        parent = self._get_mean_embedding(model_output, subclass_pair[0])
        child = self._get_mean_embedding(model_output, subclass_pair[2])

        return self.distance_loss(child + self.subclass_relation, parent)