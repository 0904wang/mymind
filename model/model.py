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
import torch.nn.functional as F
import math
from transformers.activations import ACT2FN

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
        self.k_proj = nn.Linear(self.hidden_dim, self.num_key_value_heads*self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_key_value_heads*self.head_dim, bias=False)
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
                attention_mask:Optional[torch.Tensor]=None,):
        batch_size, seq_len, _= x.shape
        xq, xk, xv= self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # reshape for multi-head attention
        xq= xq.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        xk= xk.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        xv= xv.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # q, k rotary embedding
        cos, sin = pos_emb
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])
        if past_kv is not None:
            xk = torch.cat([past_kv[0], xk], dim=1)
            xv = torch.cat([past_kv[1], xv], dim=1)
        past_kv = (xk, xv) if use_kv_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len>1 and (attention_mask is None or
                                         torch.all(attention_mask==1)):
            attn_mask = (
                None 
                if attention_mask is None 
                else attention_mask.view(batch_size, 1, 1, -1).expand(
                    batch_size, self.num_attention_heads, seq_len, -1
                ).bool()
            )
            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask, 
                                                    self.dropout if self.training else 0.,
                                                    is_causal=True)
        else:
            score = xq @ xk.transpose(-1, -2)/math.sqrt(self.head_dim)
            
            # 创建因果掩码 (causal mask)
            kv_seq_len = xk.size(2)  # key 的序列长度
            past_len = kv_seq_len - seq_len
            causal_mask = torch.tril(
                torch.ones(seq_len, kv_seq_len, device=score.device, dtype=torch.bool),
                diagonal=past_len,
            )
            score = score.masked_fill(causal_mask[None, None, :, :] == 0, float("-inf"))
            
            # 如果有 padding mask，也应用
            if attention_mask is not None:
                if attention_mask.size(-1) != kv_seq_len:
                    pad_len = kv_seq_len - attention_mask.size(-1)
                    if pad_len > 0:
                        attention_mask = F.pad(attention_mask, (pad_len, 0), value=1)
                padding_mask = attention_mask[:, None, None, :].bool()
                score = score.masked_fill(~padding_mask, float("-inf"))
        
            score = F.softmax(score.float(), dim = -1)
            score = self.attn_dropout(score)
            output = score@xv
        output = output.transpose(1, 2).contiguous().view(batch_size,
                                                          seq_len,
                                                          self.head_dim*self.num_attention_heads)
        output = self.o_proj(output)
        return output, past_kv
    
# FFN with Gating
class FeedForward(nn.Module):
    def __init__(self, args:MyMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size = 8/3*args.hidden_size
            intermediate_size = int(64*((intermediate_size+64-1)//64))
        else:
            intermediate_size = args.intermediate_size

        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        self.activation = ACT2FN[args.hidden_act]

    def forward(self, x:torch.Tensor):
        return self.dropout(
            self.down_proj(self.activation(self.up_proj(x))*self.gate_proj(x))
        )
    

class MymindBlock(nn.Module):
    def __init__(self, args:MyMindConfig, layer_id: int):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_dim = args.hidden_size
        self.attn = Attention(args)

        self.layer_id = layer_id
        self.input_ln = RMSNorm(self.hidden_dim, eps = args.rms_norm_eps)
        self.post_ln = RMSNorm(self.hidden_dim, eps = args.rms_norm_eps)
        self.ffn = FeedForward(args)

    def forward(self, 
                hidden_state: torch.Tensor, 
                pos_emb: Tuple[torch.Tensor, torch.Tensor],
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,
                use_kv_cache: bool=False,
                attention_mask: Optional[torch.Tensor]=None,
                ):
        residual = hidden_state
        hidden_state, past_kv = self.attn(
            self.input_ln(hidden_state),
            pos_emb,
            past_kv,
            use_kv_cache,
            attention_mask
        )
        hidden_state = residual + hidden_state
        hidden_state = hidden_state+self.ffn(self.post_ln(hidden_state))
        return hidden_state, past_kv
    

class MymindModel(nn.Module):
    def __init__(self, args:MyMindConfig):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.hidden_dim = args.hidden_size
        
        self.embeddings = nn.Embedding(args.vocab_size,
                                       args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)

        self.layers = nn.ModuleList(
            [MymindBlock(args, i) for i in range(args.num_hidden_layers)]
        )
        self.norm = RMSNorm(args.hidden_size, eps = args.rms_norm_eps)

        # Rope预计算
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=args.hidden_size//args.num_attention_heads,
            end=args.max_position_embeddings,
            rope_base=args.rope_theta,
            rope_scaling=args.rope_scaling
        )

        self.register_buffer(
            "freqs_cos", freqs_cos, persistent=False
        )
        self.register_buffer(
            "freqs_sin", freqs_sin, persistent=False
        )

    def forward(self,
                input_ids: Optional[torch.Tensor]=None,
                attention_mask: Optional[torch.Tensor]=None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None,
                use_kv_cache: bool=False,
                **kwargs,
                ):
        batch_size, seq_len = input_ids.shape
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        
        past_key_values = past_key_values or [None]*len(self.layers)
        
        start_pos = (
            past_key_values[0][0].size(1) if past_key_values[0] is not None else 0
        )

        hidden_state = self.dropout(
            self.embeddings(input_ids)
        )

        pos_emb = (
            self.freqs_cos[start_pos: start_pos + seq_len],
            self.freqs_sin[start_pos: start_pos + seq_len],
        )

        present_key_values = []
        for layer_idx, (layer, past_kv) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_state, present_kv = layer(
                hidden_state,
                pos_emb,
                past_kv,
                use_kv_cache,
                attention_mask,
            )
            present_key_values.append(present_kv)

        hidden_state = self.norm(hidden_state)

        return hidden_state, present_key_values
    