from transformers import PretrainedConfig
from typing import Optional, Tuple

class MyMindConfig(PretrainedConfig):
    model_type = "mymind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
import math

# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, hidden_size:int, eps:float=1e-8):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x:torch.Tensor):
        rms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True)+self.eps)
        return x*rms.type_as(x)
    
    def forward(self, x:torch.Tensor):
        return self._norm(x)*self.weights
    

# Rope
def precompute_freqs_cis(dim:int, 
                         end:int = 32*1024, 
                         rope_base: float= 1e6,
                         rope_scaling:Optional[dict]= None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2).float() / dim))

# 计算corr_dim
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow= (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4),
            rope_scaling.get("beta_slow", 1)
        )
        
        corr_dim = next((i for i in range(dim//2) if 2*math.pi/freqs[i]>orig_max), dim//2)

    # 计算power
        power = torch.arange(0, dim//2, device=freqs.device).float()/max(dim//2-1, 1)
    # 计算beta
        beta = (beta_fast ** power) * (beta_slow ** (1 - power))
    # 计算scale
        scale = torch.where(torch.arange(dim//2, device=freqs.device)<corr_dim,
                            (beta*factor-beta+1)/(beta*factor),
                            1./factor)
        
        freqs *= scale 
    
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    # 返回一个cos和sin
    freqs_cos = torch.cat([torch.cos(freqs),torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs),torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]],dim=-1)


    q_embed = (q*cos.unsqueeze(unsqueeze_dim))+(rotate_half(q)*sin.unsqueeze(unsqueeze_dim))
    k_embed = (k*cos.unsqueeze(unsqueeze_dim))+(rotate_half(k)*sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x:torch.Tensor, n_rep:int)->torch.Tensor:
    batch_size, seq_len, head_nums, head_dim = x.shape
    if n_rep==1:
        return x
    return (
        x[:,:,:,None,:]
        .expand(batch_size, seq_len, head_nums, n_rep, head_dim)
        .reshape(batch_size, seq_len, head_nums*n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args:MyMindConfig):
        super().__init__()
        self.num_key_value_heads = (
            args.num_key_value_heads if args.num_key_value_heads is not None
            else args.num_attention_heads
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0, (
            "num_attention_head must be divisible by num_key_value_heads"
        )

        self.hidden_dim = args.hidden_size
        self.n_rep = args.num_attention_heads//args.num_key_value_heads
        self.num_attention_heads = args.num_attention_heads
        self.head_dim = self.hidden_dim//self.num_attention_heads
        

        self.q_proj = nn.Linear(self.hidden_dim, self.head_dim*self.num_attention_heads, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.head_dim*self.num_attention_heads, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.head_dim*self.num_attention_heads, bias=False)
        self.o_proj = nn.Linear(self.head_dim*self.num_attention_heads, self.hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        self.flash = (hasattr(torch.nn.functional, 'scaled_dot_product_attention') 
                      and args.flash_attention)
        
    def forward(self, x:torch.Tensor, 
                pos_emb:Tuple[torch.Tensor, torch.Tensor], 
                past_kv:Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
                use_kv_cache:bool=False,
                attention_mask:Optional[torch.Tensor]=None,)->torch.Tensor:
        batch_size, seq_len, _= x.shape
        xq, xk, xv= self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # reshape for multi-head attention
        xq= xq.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        xk= xk.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        xv= xv.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)

        # q, k rotary embedding
        cos, sin = pos_emb
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])