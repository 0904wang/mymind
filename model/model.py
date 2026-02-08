from transformers import PretrainedConfig
from typing import Optional, Tuple, Union

class MyMindConfig(PretrainedConfig):
    model_type = "mymind"

    def __init__(
        self,
        dropout: float = 0.0,
        
        # 1. 适配 Qwen Tokenizer 的关键设置
        # Qwen 词表大小通常是 151936 (151643 + padding)
        vocab_size: int = 151936,
        # 强制指定 <|endoftext|> (ID: 151643) 为 BOS/EOS，这对预训练至关重要
        bos_token_id: int = 151643,
        eos_token_id: int = 151643,
        
        hidden_act: str = "silu",
        
        # 2. 模型尺寸 (Base Model Scale: ~2B)
        # 2048 维度配合 24 层是经典的 2B 模型架构，在 5090 上训练效率极高
        hidden_size: int = 2048,
        # Intermediate Size 通常为 Hidden * 3.5 左右 (SwiGLU)
        # MoE 的 FFN 维度可以稍微大一点
        intermediate_size: int = 5632, 
        
        max_position_embeddings: int = 8192, # 8K 上下文
        
        num_attention_heads: int = 16,   # Head Dim = 2048 / 16 = 128
        num_hidden_layers: int = 24,
        num_key_value_heads: int = 8,    # GQA (Grouped Query Attention) 2:1，节省显存
        
        rms_norm_eps: float = 1e-06,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        
        ############ MoE 核心配置 (DeepSeek-V2/V3 风格) ############
        use_moe: bool = True,
        
        # 激活参数控制：每次只用 2 个专家 + 1 个共享专家
        num_experts_per_tok: int = 2,  
        
        # 总专家数：8 个路由专家
        # 总参数量 ≈ Base + 8 * Experts
        n_routed_experts: int = 8,     
        
        # 共享专家：1 个 (关键！这能显著稳定训练，减少 MoE 常见的“专家坍缩”问题)
        n_shared_experts: int = 1,     
        
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.01,  # 负载均衡损失系数，0.01 是经验值
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
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 0.5,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
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
        if end>orig_max:
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
    
class MoEGate(nn.Module):
    def __init__(self, args:MyMindConfig):
        super().__init__()
        self.args = args
        self.top_k = args.num_experts_per_tok
        self.n_routed_experts = args.n_routed_experts
        
        self.scoring_func = args.scoring_func
        self.alpha = args.aux_loss_alpha
        self.seq_aux = args.seq_aux

        self.norm_topk_prob = args.norm_topk_prob
        self.gating_dim = args.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_state:torch.Tensor):
        batch_size, seq_len, _ = hidden_state.shape
        hidden_state = hidden_state.view(-1, hidden_state.size(-1))

        logits = F.linear(hidden_state, self.weight, None)

        if self.scoring_func=="softmax":
            score = F.softmax(logits, dim=-1)
        elif self.scoring_func=="sigmoid":
            score = torch.sigmoid(logits)
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func}")
        
        topk_weights, topk_indices = torch.topk(
            score, self.top_k, dim=-1, sorted=False
        )


        # 归一化
        if self.top_k>1 and self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True)+1e-20

            topk_weights = topk_weights/denominator
        
        # 计算辅助损失
        if self.training and self.alpha>0.:
            scores_for_aux = score
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_indices.view(batch_size, -1)
            
            # 序列级别的辅助损失
            if self.seq_aux:
                scores_for_aux = scores_for_aux.view(batch_size, seq_len, self.n_routed_experts)
                ce = torch.zeros(
                    batch_size, self.n_routed_experts, device=hidden_state.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(batch_size, seq_len*aux_topk, device=hidden_state.device)
                )

                ce = ce.div(seq_len*aux_topk/self.n_routed_experts)
                aux_loss= (ce*scores_for_aux.mean(dim=1)).sum(dim=-1).mean()*self.alpha
            
            # 批次级别的辅助损失
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().sum(dim=0)
                Pi = scores_for_aux.mean(dim=0)
                fi = ce*self.n_routed_experts
                aux_loss = (fi*Pi).sum()*self.alpha
        else:
            aux_loss = hidden_state.new_tensor(0.0)
        
        return topk_weights, topk_indices, aux_loss

class MoEFeedForward(nn.Module):
    def __init__(self, args:MyMindConfig):
        super().__init__()
        self.args = args
        self.expert_ffns = nn.ModuleList(
            [FeedForward(args) for _ in range(args.n_routed_experts)]
        )
        self.gate = MoEGate(args)
        if args.n_shared_experts>0:
            self.shared_ffns = nn.ModuleList(
                [FeedForward(args) for _ in range(args.n_shared_experts)]
            )

    def forward(self, hidden_state:torch.Tensor):
        identity = hidden_state
        orin_shape = hidden_state.shape
        batch_size, seq_len, hidden_dim = orin_shape

        topk_weights, topk_indices, aux_loss = self.gate(hidden_state)

        hidden_state = hidden_state.view(-1, hidden_dim)
        flat_topk_indices = topk_indices.view(-1)

        if self.training and aux_loss>0.:
            hidden_state = hidden_state.repeat_interleave(
                self.args.num_experts_per_tok, dim=0
            ) 
            y = torch.zeros_like(hidden_state)

            for i, expert in enumerate(self.expert_ffns):
                mask = flat_topk_indices == i
                if mask.any():
                    y[mask] = expert(hidden_state[mask])
            y = (y.view(*topk_weights.shape, -1) * topk_weights.unsqueeze(-1)).sum(dim=1)
            y = y.view(orin_shape)
        else:
            y = self.moe_infer(hidden_state, flat_topk_indices, topk_weights.view(-1, 1)).view(orin_shape)
        
        # 添加共享专家的输出
        if self.args.n_shared_experts > 0:
            for shared_expert in self.shared_ffns:
                y = y + shared_expert(identity)
        
        self.aux_loss = aux_loss
        return y
                


    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 使用cache，创建一个和x形状相同的零张量
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序，最后是[0,0,0,1,1,2,2,2,...]这样的顺序
        # 分拣
        idxs = flat_expert_indices.argsort()
        # 统计每个专家被分配到的token数量
        # 打包
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个token对应的专家索引
        token_idxs = idxs // self.args.num_experts_per_tok
        # 对每个打包好的包进行处理
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前包的起始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 取出当前包对应的专家
            expert = self.expert_ffns[i]
            # 取出token对应的原始id
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 取出token对应的数据
            expert_tokens = x[exp_token_idx]
            # 计算专家输出，一次性处理当前包的所有token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将结果散点加到缓存中对应位置
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache

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
        
        # 根据配置选择FFN类型
        if args.use_moe:
            self.ffn = MoEFeedForward(args)
        else:
            self.ffn = FeedForward(args)
        self.use_moe = args.use_moe

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
        
        # 如果使用MoE，返回aux_loss
        aux_loss = self.ffn.aux_loss if self.use_moe else None
        return hidden_state, past_kv, aux_loss
    

class MymindModel(nn.Module):
    def __init__(self, args:MyMindConfig):
        super().__init__()
        self.args = args
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
        aux_loss = None
        for layer_idx, (layer, past_kv) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_state, present_kv, layer_aux_loss = layer(
                hidden_state,
                pos_emb,
                past_kv,
                use_kv_cache,
                attention_mask,
            )
            present_key_values.append(present_kv)
            # 累积aux_loss
            if layer_aux_loss is not None:
                aux_loss = layer_aux_loss if aux_loss is None else aux_loss + layer_aux_loss

        hidden_state = self.norm(hidden_state)

        return hidden_state, present_key_values, aux_loss


from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
class MymindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MyMindConfig
    def __init__(self, config: MyMindConfig):
        self.config = config
        super().__init__(config)
        self.model = MymindModel(config)
        self.lm_head = nn.Linear(self.config.hidden_size,
                                 self.config.vocab_size, bias=False)
        self.model.embeddings.weight = self.lm_head.weight
        self.post_init()
    
    def forward(self, 
                input_ids:Optional[torch.Tensor]=None,
                attention_mask:Optional[torch.Tensor]=None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]=None,
                use_kv_cache: bool=False,
                labels: Optional[torch.Tensor]=None,
                logits_to_keep: Union[int, torch.Tensor]=0,
                **args):
        hidden_state, past_kv, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_kv_cache=use_kv_cache,
            **args
        )

        # 计算loss时需要完整的logits（用于trainer外部loss计算）
        # 推理时可以只计算部分logits以提高效率
        if labels is not None or logits_to_keep == 0:
            logits = self.lm_head(hidden_state)
        else:
            slice_indices = (
                slice(-logits_to_keep, None) if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(hidden_state[:, slice_indices, :])
        
        # 如果提供了labels，计算loss（用于HuggingFace标准训练）
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
            # 添加aux_loss
            if aux_loss is not None:
                loss = loss + aux_loss
        
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_kv,
            hidden_states=(hidden_state,),
        )
        # 为trainer添加aux_loss属性
        output.aux_loss = aux_loss if aux_loss is not None else torch.tensor(0.0, device=logits.device)
        
        return output
    
