import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

class Attention(nn.Module):

    '''
    Multi-head attention mechanism.
    '''

    def __init__(self, embed_dim, n_heads=1, dropout=0., project_out=False):
        '''
        args
        ----
        embed_dim : int,
            the input and out dim for each item in sequence
        n_heads : int
            number of attention heads
        dropout : float
            dropout prob applied to the attention weights
        project_out: bool
            whether to project each token after weighted attention through another linear layer
        '''
        super().__init__()

        self.n_heads = n_heads
        dim_head = embed_dim // n_heads
        assert dim_head * n_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = dim_head ** -0.5

        self.q_project = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_project = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_project = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        
        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v, attn_mask=None, return_attn=False):
        '''
        args
        ----
        q : torch.tensor
            shape `(batch, n_target_item, embed_dim)`
        k/v : torch.tensor
            shape `(batch, n_source_item, embed_dim)`
        attn_mask : torch.bool
            shape `(n_target_item, n_source_item)` or `(batch, n_target_item, n_source_item)`
            positions with ``True`` are allowed to attend while ``False`` are marked with -inf
        return_attn : bool
            whether to return attention weights
        
        returns
        -------
        out : torch.tensor
            output, shape `(batch, n_target_item, embed_dim)`
        attn : torch.tensor
            attention weights (if return_attn), shape `(n_target_item, n_source_item)`
        '''

        bsz, q_len, _ = q.shape
        _, k_len, _ = k.shape

        # check attention mask
        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            assert attn_mask.shape == (q_len, k_len) or attn_mask.shape == (bsz, q_len, k_len)

        # project q/k/v
        q = self.q_project(q) # (batch, n_items, embed_dim)
        k = self.k_project(k)
        v = self.v_project(v)
        # (batch, n_items, n_heads x dim_head) -> (batch, n_heads, n_items, dim_head)
        q, k, v = map(lambda x: rearrange(x, 'b n (h d) -> b h n d', h=self.n_heads), (q,k,v))

        attn = torch.matmul(q, k.transpose(-1,-2)) * self.scale # (batch, n_heads, n_target_items, n_source_items)
        if attn_mask is not None:
            if attn_mask.dim()==3:
                attn_mask = attn_mask.unsqueeze(1) # add an n_head dim for broadcast add
            attn_mask = attn_mask.to(attn.device)
            # mark -inf where mask==False
            ninf_mask = torch.zeros_like(attn_mask, dtype=q.dtype, device=attn.device)
            ninf_mask.masked_fill_(attn_mask==False, float('-inf'))
            attn += ninf_mask
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(self.attn_dropout(attn), v) # (batch, n_heads, n_target_items, n_source_items) x (batch, n_heads, n_source_item, dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)') # (batch, n_target_items, n_heads x dim_head)
        out = self.to_out(out) # (batch, n_target_items, embed_dim)
        return (out, attn) if return_attn else out

class MLP(nn.Module):

    '''
    n-layer MLP with relu activation in the hidden layer
    '''

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.):
        
        super().__init__()

        if type(hidden_dim) == int:
            hidden_dim = [hidden_dim]
        hidden_dim = [in_dim] + hidden_dim

        net = []
        for i in range(1, len(hidden_dim)):
            net.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            net.append(nn.ReLU())
            net.append(nn.Dropout(dropout))
        net.append(nn.Linear(hidden_dim[-1], out_dim))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class TransformerEncoderLayer(nn.Module):

    '''
    single encoder block, consisting of a self-attention layer and an MLP
    '''

    def __init__(self, embed_dim, n_heads, mlp_hid_dim, dropout=0.):
        super().__init__()
        self.self_attn = Attention(embed_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_hid_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask=None):
        '''
        args
        ----
        x : tensor
            shape (batch, max_len, embed_dim)
        attn_mask : bool tensor
            - (n_items, n_items) or (batch, n_items, n_items)
            - mask for attention operation (e.g., causal future mask)
            - True will be attended, False will be masked with -inf
        '''
        x = self.norm1(x + self.self_attn(x, x, x, attn_mask=attn_mask))
        x = self.norm2(x + self.mlp(x))
        return x