"""History adapters around the original RoPE blocks; no backbone replacement."""
import torch
from torch import nn
from torch.nn import functional as F


class CrossAdapter(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)

    def forward(self, current, memory, available):
        q = self.norm(current)
        # Missing blocks are zeroed before encoding and masked after attention.
        # Avoid fully masked softmax rows, which can create NaNs.
        update = self.attention(q, memory, memory, need_weights=False)[0]
        return current + update * available[:, None, None]


def history_positions(current_length, device):
    quarters = torch.arange(96, device=device, dtype=torch.float32)
    if current_length == 24:
        return -24 + (quarters + .5) / 4 - .5
    if current_length == 96:
        return quarters - 96
    raise ValueError('Expected 24 or 96 current positions')


def concatenated_block(block, tokens, condition, positions, valid_keys):
    """Use the original QKV/out/MLP weights with explicit physical RoPE positions."""
    s1, a1, g1, s2, a2, g2 = block.modulation(condition).chunk(6, dim=-1)
    x = block.norm1(tokens) * (1 + a1[:, None]) + s1[:, None]
    batch, length, width = x.shape
    attention = block.attn
    qkv = attention.qkv(x).reshape(batch, length, 3, attention.n_heads, attention.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attention.rope(q, positions), attention.rope(k, positions)
    update = F.scaled_dot_product_attention(q, k, v, attn_mask=valid_keys[:, None, None, :],
                                           dropout_p=attention.dropout if block.training else 0.)
    update = attention.out(update.transpose(1, 2).reshape(batch, length, width))
    tokens = tokens + g1[:, None] * block.dropout(update)
    return tokens + g2[:, None] * block.dropout(block.mlp(block.norm2(tokens) * (1 + a2[:, None]) + s2[:, None]))
