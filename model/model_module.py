import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat
from model.transformers import (
    GraphAutoregTransformer,
    GraphMaskedTransformer,
    GraphRandomMaskedTransformer,
    EdgeSearchAutoregTransformer,
    EdgeSearchRandomMaskedTransformer
)

class GraphModelModule(pl.LightningModule):
    
    def __init__(self,
                 lr=1e-4,
                 type='autoreg',
                 graph_input='graph_embed',
                 **model_kwargs):
        
        super().__init__()
        self.save_hyperparameters()

        # graph embedding models
        if self.hparams.type == 'autoreg' and self.hparams.graph_input == 'graph_embed':
            self.transformer = GraphAutoregTransformer(**model_kwargs)
        elif self.hparams.type == 'masked' and self.hparams.graph_input == 'graph_embed':
            self.transformer = GraphMaskedTransformer(**model_kwargs)
        elif self.hparams.type == 'random_masked' and self.hparams.graph_input == 'graph_embed':
            self.transformer = GraphRandomMaskedTransformer(**model_kwargs)

        # edge token models
        elif self.hparams.type == 'autoreg' and self.hparams.graph_input == 'edge_token':
            self.transformer = EdgeSearchAutoregTransformer(**model_kwargs)
        elif self.hparams.type == 'random_masked' and self.hparams.graph_input == 'edge_token':
            self.transformer = EdgeSearchRandomMaskedTransformer(**model_kwargs)
        
        else:
            raise ValueError('unknown model type')

    def forward(self, batch):
        '''
        model forward + loss/acc computation wrapper
        '''
        out = self.transformer(batch)
        result_dict = self.transformer.generate_result_dict(batch, out)
        loss_dict, acc_dict = self.transformer.calc_loss_acc(result_dict)

        return result_dict, loss_dict, acc_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        result_dict, loss_dict, acc_dict = self.forward(batch)
        for loss in loss_dict.keys():
            self.log('train/%s'%loss, loss_dict[loss], on_epoch=True)
        for acc in acc_dict.keys():
            self.log('train/%s'%acc, acc_dict[acc], on_epoch=True)
        return sum(loss_dict.values())

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            result_dict, loss_dict, acc_dict = self.forward(batch)
        for loss in loss_dict.keys():
            self.log('val/%s'%loss, loss_dict[loss], prog_bar=True)
        for acc in acc_dict.keys():
            self.log('val/%s'%acc, acc_dict[acc], prog_bar=True)
        return result_dict['pred']

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            result_dict, loss_dict, acc_dict = self.forward(batch)
        for loss in loss_dict.keys():
            self.log('test/%s'%loss, loss_dict[loss])
        for acc in acc_dict.keys():
            self.log('test/%s'%acc, acc_dict[acc])
        return result_dict['pred']