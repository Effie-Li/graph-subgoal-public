import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat

from model.layers import Attention, MLP

class BaseTransformerEncoder(nn.Module):

    '''
    specifies core transformer encoder sublayer modules
    the exact sequence composition, token embeddings, attention masks, 
    output extraction, and loss/acc are implemented in subclasses
    '''

    def __init__(self,
                 n_graph,
                 architecture,
                 embed_dim,
                 n_heads,
                 mlp_dim,
                 out_dim,
                 share_attn_weight=False,
                 dropout=0.0,
                 **kwargs,
                 ):
        
        '''
        args
        ----
        n_graph : int
            number of graphs to learn
        architecture : str
            str indicator of sublayer ordering, e.g.:
            'a.f.a.f.a.f': a standard 3-layer transformer encoder, 
            'a.a.ff': two attention layers followed by an mlp with 2 hidden layers
        embed_dim : int
            size of latent token embedding
        n_heads : int
            number of attention heads in each layer
        mlp_dim : int
            size of the hidden layer in mlp sublayer
        out_dim : int
            number of final output units
        share_attn_weight : bool
            whether to share weights across attention layers
        dropout : float
            proportion of weights to drop out during training
        '''
        
        super().__init__()

        self.n_graph = n_graph
        self.embed_dim = embed_dim
        self.architecture = architecture
        self.n_layer = len(self.architecture.split('.'))
        self.share_attn_weight = share_attn_weight
        
        self.init_encoder(n_heads, embed_dim, mlp_dim, dropout)
        
        self.out = nn.Linear(embed_dim, out_dim)

    def init_encoder(self, n_heads, embed_dim, mlp_dim, dropout):

        if self.share_attn_weight:
            shared_attn_layer = Attention(embed_dim=embed_dim, 
                                          n_heads=n_heads, 
                                          dropout=dropout)

        self.encoder = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in self.architecture.split('.'):
            if layer == 'a': # attention layer
                self.encoder.append(shared_attn_layer if self.share_attn_weight else
                                    Attention(embed_dim=embed_dim, 
                                              n_heads=n_heads, 
                                              dropout=dropout))
                self.norms.append(nn.LayerNorm(embed_dim))
            elif layer[0] == 'f': # feedforward layer
                self.encoder.append(MLP(in_dim=embed_dim, 
                                        hidden_dim=[mlp_dim]*len(layer), 
                                        out_dim=embed_dim, 
                                        dropout=dropout))
                self.norms.append(nn.LayerNorm(embed_dim))
            else:
                raise NotImplementedError('unknown encoder layer indicator')

    def input_embed(self, batch):
        '''
        args
        ----
        batch : dict
            contains tensors relevant for the task
        
        returns
        -------
        tensor : shape (bsz, seq_len, embed_dim)
        '''
        raise NotImplementedError()

    def create_attn_mask(self, seq_len, device):
        '''
        creates a bool mask (seq_len, seq_len) or (bsz, n_head, seq_len, seq_len)
        '''
        raise NotImplementedError()

    def _encoder_forward(self, x, attn_mask, return_attn=False, return_reps=False):

        attn_weights = []
        token_reps = []
        for i, layer in enumerate(self.encoder):

            if type(layer) == Attention:
                x1, attn = layer(x, x, x, attn_mask=attn_mask, return_attn=True)
                x = x + x1
                if return_attn:
                    attn_weights.append(attn) # attn at each layer is (bsz, n_heads, tgt_len, src_len)
            
            elif type(layer) == MLP:
                x = x + layer(x)
            
            x = self.norms[i](x)

            if return_reps:
                token_reps.append(x)

        attn_weights = torch.stack(attn_weights, dim=1) if return_attn else None
        token_reps = torch.stack(token_reps, dim=1) if return_reps else None
        return x, attn_weights, token_reps

    def forward(self, batch):

        x = self.input_embed(batch)
        bsz, seq_len, _ = x.shape
        attn_mask = self.create_attn_mask(seq_len, x.device)
        for i, layer in enumerate(self.encoder):
            if type(layer) == Attention:
                x = x + layer(x, x, x, attn_mask=attn_mask)
            elif type(layer) == MLP:
                x = x + layer(x)
            x = self.norms[i](x)
        x = self.out(x) # (batch, seq_len, out_dim)
        return x

    @torch.no_grad()
    def get_attn_and_reps(self, batch):
        '''
        forward() alternative that returns attention weights and token representations
        '''

        input_embed = self.input_embed(batch)
        bsz, seq_len, _ = input_embed.shape
        attn_mask = self.create_attn_mask(seq_len, input_embed.device)

        x, attn_weights, token_reps = self._encoder_forward(input_embed, 
                                                            attn_mask=attn_mask, 
                                                            return_attn=True, 
                                                            return_reps=True)
        
        x = self.out(x) # (batch, seq_len, out_dim)

        token_reps = torch.cat([input_embed.unsqueeze(1), token_reps], dim=1)
        return x, attn_weights, token_reps

    def generate_result_dict(self, batch, out):

        '''
        creates result dict, including prediction target

        args
        ----
        batch : dict of tensors
            contains task data
        out : tensor, shape (bsz, n_output, out_dim)
            model output states
        '''

        raise NotImplementedError()

    def calc_loss_acc(self, result_dict):
            
        '''
        calculates loss and accuracy from result dict

        args
        ----
        result_dict : dict
            contains model output and target tensors

        returns
        -------
        loss_dict : dict
            contains loss values
        acc_dict : dict
            contains accuracy values
        '''

        raise NotImplementedError()


class GraphAutoregTransformer(BaseTransformerEncoder):

    '''
    predicts shortest paths autoregressively (from start to goal)
                                                              eos
                                             s   n_0 n_1        g
                                             |    |   |         |
    sequence composes of:    G,     s, g,   sos,  s, n_0, ..., n_n, g, pad, pad
                         (context) (probe) (rest)      (output)
    output starts from the sos token and produced with future masking
    '''

    def __init__(self, 
                 n_graph_node,
                 pad_index,
                 **kwargs):
        
        super().__init__(**kwargs)

        self.node_embeddings = nn.Embedding(num_embeddings=n_graph_node, 
                                            embedding_dim=self.embed_dim, 
                                            padding_idx=pad_index)
        # sequence parsing embeddings / meta tokens
        self.graph_context_token = nn.Embedding(self.n_graph, self.embed_dim)
        self.start_token = nn.Embedding(1, self.embed_dim)
        self.goal_token = nn.Embedding(1, self.embed_dim)

        self.cross_entropy_loss = nn.CrossEntropyLoss(
            ignore_index=self.node_embeddings.padding_idx, 
            reduction='none'
        )

    def embed_graph_connectivity(self, batch):
        # graph context encoding token (like a [cls] token)
        graph_embedding = self.graph_context_token(batch['graph_context'].unsqueeze(-1)) # (batch, 1, embed_dim)
        return graph_embedding
    
    def input_embed(self, batch):
        '''
        specifies the input sequence and generate input embeddings
        start/goal probe tokens and output tokens are concatenated with meta token mebeddings

        args
        ----
        batch : dict
            'graph_context' : tensor, shape (batch, )
            'start_goal_probe' : tensor, shape (batch, 1, 2)
            'shortest_path' : tensor, shape (batch, max_path_len)

        returns
        -------
        x : tensor, shape (batch, 4 + max_path_len, embed_dim)
        '''
        device = batch['start_goal_probe'].device
        bsz = batch['start_goal_probe'].shape[0]

        graph_embedding = self.embed_graph_connectivity(batch)

        # embed start/goal probe
        probe_embeddings = self.node_embeddings(batch['start_goal_probe']) # (batch, 2, embed_dim)
        probe_meta_embeddings = torch.stack(
            [self.start_token(torch.tensor(0).to(device)), self.goal_token(torch.tensor(0).to(device))],
            dim = 0
        )
        probe_embeddings += probe_meta_embeddings

        # sos token
        sos_embedding = torch.zeros(bsz, 1, self.embed_dim).to(device) # (batch, 1, embed_dim)

        # embed output tokens
        path_embeddings = self.node_embeddings(batch['shortest_path']) # (batch, max_path_len, embed_dim)
        
        input_embeddings = torch.cat(
            [graph_embedding, probe_embeddings, sos_embedding, path_embeddings], dim=-2
        ) # (batch, 4 + max_path_len, embed_dim)

        # some useful batch metadata/masks
        # 1. index of start of shortest path prediction
        #    used to create future mask and select appropriate output states
        self.sos_index = 3
        # 2. sequence pad token indicator mask, used as src_key_padding_mask
        self.batch_pad_mask = torch.zeros(input_embeddings.shape[:2]).bool()
        self.batch_pad_mask[:, -path_embeddings.shape[1]:] = batch['shortest_path'] == self.node_embeddings.padding_idx

        return input_embeddings

    def create_attn_mask(self, seq_len, device):

        # **True** values indicate that the corresponding position **is** allowed to attend
        # shape (target_seq_len, source_seq_len), a causal mask starting from self.output_start

        attention_mask = torch.ones((seq_len, seq_len), dtype=bool)
        # no source token can peak into output tokens
        attention_mask[:self.sos_index, self.sos_index:] = 0
        # future masking for output tokens
        n_output = seq_len - self.sos_index
        attention_mask[self.sos_index:, self.sos_index:] = torch.tril(torch.ones(n_output, n_output))
        attention_mask = attention_mask

        bsz, seq_len = self.batch_pad_mask.shape
        attention_mask = repeat(attention_mask, 'nq nk -> b nq nk', b=bsz)
        attention_mask = torch.mul(attention_mask, ~self.batch_pad_mask.unsqueeze(1)) # no attention to <pad>

        return attention_mask.to(device)
    
    def forward_rollout(self, batch, return_attn=False, return_reps=False):

        '''
        autoregressive rollout
        '''
        
        input_embed = self.input_embed(batch)
        device = input_embed.device
        bsz, seq_len, _ = input_embed.shape
        attn_mask = self.create_attn_mask(seq_len, input_embed.device)
        batch_pred = torch.zeros((bsz, seq_len, self.out.out_features), device=device)

        for n_input_token in range(self.sos_index+1, seq_len+1):

            output_so_far = batch_pred[:, self.sos_index:n_input_token-1, 2:].argmax(-1)
            new_input_embed = self.node_embeddings(output_so_far)
            x = torch.cat([
                input_embed[:, :self.sos_index+1], new_input_embed
            ], dim=1) # (bsz, n_input_token, embed_dim)

            x, attn_weights, token_reps = self._encoder_forward(
                x, 
                attn_mask=attn_mask[:, :n_input_token, :n_input_token], 
                return_attn=return_attn,
                return_reps=return_reps
            )
            out = self.out(x)
            
            if n_input_token == self.sos_index+1:
                batch_pred[:, :n_input_token] = out
            else:
                batch_pred[:, n_input_token-1] = out[:, -1]

        if return_reps:
            input_embed = torch.cat([input_embed[:, :self.sos_index+1], new_input_embed], dim=1).unsqueeze(1)
            token_reps = torch.cat((input_embed, token_reps), dim=1)
        else:
            token_reps = None

        return batch_pred, attn_weights if return_attn else None, token_reps

    def generate_result_dict(self, batch, out):

        '''
        creates result dict, including prediction target

        args
        ----
        batch : dict of tensors
            contains task data
        out : tensor, shape (bsz, n_output, out_dim)
            model output states
        '''

        # extract output states starting from the sos probe token, 
        # excluding output state of last token (unused)
        out = out[:, self.sos_index:-1, :] # (batch, max_path_len, 2 + n_node)

        # slice eos/path predictions
        eos_pred = out[:, :, :2] # (batch, max_path_len, 2)
        path_pred = out[:, :, 2:] # (batch, max_path_len, 8)

        # path target is shortest path
        path_target = batch['shortest_path']

        # create eos target
        path_len = (path_target != self.cross_entropy_loss.ignore_index).sum(-1)
        eos_target = F.one_hot(path_len - 1, num_classes=eos_pred.shape[1])
        pad_mask = path_target == self.cross_entropy_loss.ignore_index
        # mask pad tokens after eos to be ignore_index
        eos_target.masked_fill_(pad_mask, value=self.cross_entropy_loss.ignore_index)

        result_dict = {'eos_pred': eos_pred, 'eos_target': eos_target.long(),
                       'pred': path_pred, 'path_target': path_target,
                       'pad_mask': pad_mask, 'path_len': path_len}

        return result_dict

    def calc_loss_acc(self, result_dict):

        loss_dict, acc_dict = {}, {}

        # path prediction
        path_pred, path_target = result_dict['pred'], result_dict['path_target']
        bsz, n = path_target.shape
        loss_path = self.cross_entropy_loss(
            path_pred.reshape(bsz * n, -1),
            path_target.reshape(bsz * n)
        )
        loss_dict['loss_path'] = loss_path.reshape(bsz, n).mean(1).mean(0)

        acc_path = (path_pred.detach().argmax(-1) == path_target)
        # no need to do any pad masking here because pad_index = n_output_classes so acc for pad tokens will always be 0
        acc_path = (acc_path.sum(1) / result_dict['path_len']).mean(0)
        acc_dict['acc_path'] = acc_path

        # eos prediction
        eos_pred, eos_target = result_dict['eos_pred'], result_dict['eos_target']
        bsz, n = eos_target.shape
        loss_eos = self.cross_entropy_loss(
            eos_pred.reshape(bsz * n, -1),
            eos_target.flatten()
        )
        loss_dict['loss_eos'] = loss_eos.reshape(bsz, n).mean(1).mean(0)

        acc_eos = (eos_pred.detach().argmax(-1) == eos_target)
        acc_eos = (acc_eos.sum(1) / result_dict['path_len']).mean(0)
        acc_dict['acc_eos'] = acc_eos
        acc_dict['acc'] = (acc_path + acc_eos) / 2
        
        return loss_dict, acc_dict

class GraphMaskedTransformer(BaseTransformerEncoder):

    '''
    predicts shortest paths in a masked way (all-to-all attention)
                                                                              
                                               n0     n1
                                               |       |
    sequence composes of:    G,      s,      <mask>, <mask>,   g,  pad
                         (context) (start)      (masked)     (goal)
    output taken from masked tokens
    '''

    def __init__(self, 
                 n_graph_node,
                 pad_index,
                 max_pathlen=6,
                 **kwargs):
        
        super().__init__(**kwargs)

        self.node_embeddings = nn.Embedding(num_embeddings=n_graph_node, 
                                            embedding_dim=self.embed_dim, 
                                            padding_idx=pad_index)
        # sequence parsing embeddings / meta tokens
        self.graph_context_token = nn.Embedding(self.n_graph, self.embed_dim)
        self.start_token = nn.Embedding(1, self.embed_dim)
        self.goal_token = nn.Embedding(1, self.embed_dim)
        self.output_tokens = nn.Embedding(1, self.embed_dim)

        self.rel_pos_to_start_enc = nn.Embedding(max_pathlen+1, self.embed_dim, 
                                                 padding_idx=max_pathlen)
        self.rel_pos_to_goal_enc = nn.Embedding(max_pathlen+1, self.embed_dim, 
                                                padding_idx=max_pathlen)
    
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            ignore_index=self.node_embeddings.padding_idx, 
            reduction='none'
        )
    
    def embed_graph_connectivity(self, batch):
        # graph context encoding token (like a [cls] token)
        graph_embedding = self.graph_context_token(batch['graph_context'].unsqueeze(-1)) # (batch, 1, embed_dim)
        return graph_embedding

    def set_batch_output_mask(self):
        # mask out all intermediate nodes
        # mask is of size (batch, max_path_len)
        self.batch_output_mask = self.path_intermediate_node_mask

    def embed_and_mask_path(self, batch):
        device = batch['start_goal_probe'].device
        bsz = batch['start_goal_probe'].shape[0]

        # 1. embed ground-truth shortest_path
        path_embeddings = self.node_embeddings(batch['shortest_path'])

        # 2.1 generate/update masks for start, goal, and intermediate nodes for the current batch
        self.max_path_len = path_embeddings.shape[1]
        self.path_start_mask = F.one_hot(torch.zeros(bsz).long(), num_classes=self.max_path_len).bool().to(device)
        self.path_goal_mask = F.one_hot(batch['path_len'].long() - 1, num_classes=self.max_path_len).bool().to(device)
        self.path_pad_mask = (batch['shortest_path'] == self.node_embeddings.padding_idx)
        self.path_intermediate_node_mask = ~ (self.path_start_mask | self.path_goal_mask | self.path_pad_mask) # True marks the intermediate tokens

        # 2.2 add start/goal embedding to start/goal nodes
        start_embedding = self.start_token(torch.zeros(bsz, self.max_path_len).long().to(device)) # (bsz, max_path_len, embed_dim)
        start_embedding.masked_fill_(repeat(~self.path_start_mask, 'b n -> b n d', d=self.embed_dim), 0)
        goal_embedding = self.goal_token(torch.zeros(bsz, self.max_path_len).long().to(device)) # (bsz, max_path_len, embed_dim)
        goal_embedding.masked_fill_(repeat(~self.path_goal_mask, 'b n -> b n d', d=self.embed_dim), 0)
        path_embeddings += start_embedding
        path_embeddings += goal_embedding

        # 2.3 mask out intermediate nodes (replace with output token embedding)
        self.set_batch_output_mask()
        output_embed = self.output_tokens(torch.tensor(0).to(device))
        path_embeddings = torch.where(repeat(self.batch_output_mask, 'b n -> b n d', d=self.embed_dim),
                                      output_embed,
                                      path_embeddings)
        
        # 3. add positional information to nodes on path
        # e.g., if shortest path is [[0, 7, 4, 2, p, p]], 
        #       rel_pos_to_start is [[0, 1, 2, 3, p, p]]
        #       rel_pos_to_goal is  [[3, 2, 1, 0, p, p]]
        rel_pos_to_start = torch.arange(self.max_path_len).to(device)
        rel_pos_to_start = repeat(rel_pos_to_start, 'n -> b n', b = bsz)
        rel_pos_to_start.masked_fill_(self.path_pad_mask, self.rel_pos_to_start_enc.padding_idx)
        rel_pos_to_goal = batch['path_len'].unsqueeze(-1) - 1 - rel_pos_to_start
        rel_pos_to_goal.masked_fill_(self.path_pad_mask, self.rel_pos_to_goal_enc.padding_idx)

        path_embeddings += self.rel_pos_to_start_enc(rel_pos_to_start)
        path_embeddings += self.rel_pos_to_goal_enc(rel_pos_to_goal)

        return path_embeddings
    
    def input_embed(self, batch):
        '''
        specifies the input sequence and generate input embeddings
        start/goal probe tokens and output tokens are concatenated with meta token mebeddings

        args
        ----
        batch : dict
            'graph_context' : tensor, shape (batch, )
            'start_goal_probe' : tensor, shape (batch, 1, 2)
            'shortest_path' : tensor, shape (batch, max_path_len)

        returns
        -------
        x : tensor, shape (batch, max_n_edges + max_path_len, embed_dim)
        '''
        graph_embedding = self.embed_graph_connectivity(batch)
        path_embeddings = self.embed_and_mask_path(batch)
        
        input_embeddings = torch.cat(
            [graph_embedding, path_embeddings], dim=-2
        ) # (batch, 1 + max_path_len, embed_dim)

        # sequence pad token indicator mask for creating the attention mask
        self.batch_pad_mask = torch.zeros(input_embeddings.shape[:2]).bool()
        self.batch_pad_mask[:, -path_embeddings.shape[1]:] = self.path_pad_mask

        return input_embeddings

    def create_attn_mask(self, seq_len, device):

        bsz = self.batch_pad_mask.shape[0]
        attention_mask = torch.ones((bsz, seq_len, seq_len), dtype=bool)
        attention_mask = torch.mul(attention_mask, ~self.batch_pad_mask.unsqueeze(1)) # no attention to <pad>

        return attention_mask.to(device)

    def generate_result_dict(self, batch, out):

        pred = out[:, -self.max_path_len:, :] # (batch, max_path_len, n_node)

        # create target (only count the intermediate nodes)
        target = batch['shortest_path'].detach().clone()
        target.masked_fill_(~self.batch_output_mask, self.cross_entropy_loss.ignore_index)

        result_dict = {'pred': pred, 'target': target}

        return result_dict

    def calc_loss_acc(self, result_dict):
        
        loss_dict, acc_dict = {}, {}

        pred, target = result_dict['pred'], result_dict['target']
        bsz, n = target.shape
        loss = self.cross_entropy_loss(
            pred.reshape(bsz * n, -1),
            target.reshape(bsz * n)
        )
        loss_dict['loss'] = loss.reshape(bsz, n).mean(1).mean(0)

        n_output = self.batch_output_mask.sum(-1)
        acc = (pred.detach().argmax(-1) == target).sum(1) / n_output
        acc = acc.mean(0)
        acc_dict['acc'] = acc

        return loss_dict, acc_dict

class GraphRandomMaskedTransformer(GraphMaskedTransformer):

    '''
    randomly mask out some intermediate nodes instead of all intermediate nodes
                                                                              
                                               n0
                                               | 
    sequence composes of:    G,      s,      <mask>, n1,   g,  pad
                         (context) (start)               (goal)
    '''

    def __init__(self,
                 **kwargs):
        
        super().__init__(**kwargs)
        self.mask_all = False # turn on for inference loop, set to False for training
    
    def set_batch_output_mask(self):
        # mask out some intermediate nodes
        # mask is of size (batch, max_path_len)

        if self.mask_all: # inference loop
            self.batch_output_mask = self.path_intermediate_node_mask
        else: # used in training
            sampled_nodes = [ # uniformly sample 1 to n nodes to predict for each path
                torch.multinomial(
                    row.float(), 
                    num_samples=torch.randint(low=1, high=row.sum()+1, size=(1,)).item()
                )
                for row in self.path_intermediate_node_mask
            ]
            self.batch_output_mask = torch.stack([ # add all node onehots and convert into a bool mask
                F.one_hot(nodes, num_classes=self.max_path_len).sum(0).bool()
                for nodes in sampled_nodes
            ])

class EdgeSearchAutoregTransformer(GraphAutoregTransformer):

    '''
    predicts shortest paths autoregressively (from start to goal)
                                                                                        eos
                                                                      s   n_0 n_1        g
                                                                      |    |   |         |
    sequence composes of:    e_0, e_1, ..., e_n, pad, pad,   s, g,   sos,  s, n_0, ..., n_n, g, pad, pad
                                    (edge tokens)          (probe) (rest)          (output)
    output starts from the sos token and produced with future masking
    '''

    def embed_graph_connectivity(self, batch):
        # embed graph edge tokens
        edge_embeddings = self.node_embeddings(batch['edges']) # (batch, max_n_edges, 2, embed_dim)
        graph_embedding = edge_embeddings.sum(-2) # (batch, max_n_edges, embed_dim)
        return graph_embedding

    def input_embed(self, batch):
        input_embeddings = super().input_embed(batch)

        # modify two things
        max_graph_edges = batch['edges'].shape[1]
        self.sos_index = max_graph_edges + 2
        self.batch_pad_mask[:, :max_graph_edges] = batch['edges'][...,0] == self.node_embeddings.padding_idx

        return input_embeddings

class EdgeSearchRandomMaskedTransformer(GraphRandomMaskedTransformer):

    '''
    randomly mask out some intermediate nodes instead of all intermediate nodes
                                                                              
                                               n0
                                               | 
    sequence composes of:    G,      s,      <mask>, n1,   g,  pad
                         (context) (start)               (goal)
    '''

    def embed_graph_connectivity(self, batch):
        # embed graph edge tokens
        edge_embeddings = self.node_embeddings(batch['edges']) # (batch, max_n_edges, 2, embed_dim)
        graph_embedding = edge_embeddings.sum(-2) # (batch, max_n_edges, embed_dim)
        return graph_embedding

    def input_embed(self, batch):
        input_embeddings = super().input_embed(batch)

        # modify two things
        max_graph_edges = batch['edges'].shape[1]
        self.batch_pad_mask[:, :max_graph_edges] = batch['edges'][...,0] == self.node_embeddings.padding_idx

        return input_embeddings