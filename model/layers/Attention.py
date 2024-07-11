import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Attention Block was redesigned with reference to 
https://github.com/hyunwoongko/transformer.git
"""


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.linear_q = nn.Linear(hidden_dim, hidden_dim)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output linear transformation layer
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, bias=None, mask=None):
        batch_size, seq_len, hidden_dim  = x.size()  # b, s, h
        k = self.linear_k(x)  # (batch_size, seq_len, hidden_dim)
        if bias is not None:
            q = self.linear_q(x) + self.linear_q(bias)
        else:
            q = self.linear_q(x)  # (batch_size, seq_len, hidden_dim)
        v = self.linear_v(x)  # (batch_size, seq_len, hidden_dim)

        # Split the vector dimension of each head
        k = k.view(batch_size, self.num_heads, seq_len, self.head_dim)
        q = q.view(batch_size, self.num_heads, seq_len, self.head_dim)
        v = v.view(batch_size, self.num_heads, seq_len, self.head_dim)

        # Calculating attention score
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention weight
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        # Output linear transformation layer
        attn_output = self.linear_out(attn_output)
        attn_output = self.dropout(attn_output)

        return attn_output, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.self_attentions = nn.ModuleList([SelfAttention(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.linear1s = nn.ModuleList([nn.Linear(hidden_dim, 4 * hidden_dim) for _ in range(num_layers)])
        self.linear2s = nn.ModuleList([nn.Linear(4 * hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, bias=None, mask=None):
        attn_weights = []
        for i, (self_attention, linear1, linear2, layer_norm1, layer_norm2) in enumerate(zip(self.self_attentions, self.linear1s, self.linear2s, self.layer_norms1, self.layer_norms2)):
            # Self-Attention
            attn_output, attn_weight = self_attention(x, bias, mask)
            attn_weights.append(attn_weight)
            # Add&Norm
            x = layer_norm1(x + self.dropout(attn_output))

            # Feedforward Network
            ff_output = linear2(F.relu(linear1(x)))

            # Add&Norm
            x = layer_norm2(x + self.dropout(ff_output))

        return x, torch.stack(attn_weights)



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.query_linear = nn.Linear(d_model, d_model)
        # print(d_model, self.query_linear.weight.shape)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # Reshape to split heads
        query = query.view(batch_size * self.num_heads, -1, self.d_model // self.num_heads)
        key = key.view(batch_size * self.num_heads, -1, self.d_model // self.num_heads)
        value = value.view(batch_size * self.num_heads, -1, self.d_model // self.num_heads)

        # Calculate attention scores
        scores = torch.bmm(query, key.transpose(1, 2))
        scores = scores / (self.d_model // self.num_heads) ** 0.5

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        weights = nn.functional.softmax(scores, dim=-1)

        # Apply attention weights to values
        attended = torch.bmm(weights, value)

        # Reshape back to original shape
        attended = attended.view(batch_size, -1, self.d_model)

        # Linear projection to get final output
        output = self.output_linear(attended)

        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, d_ff, dropout):
        super().__init__()

        self.self_attention = MultiHeadAttention(num_heads, d_model)
        self.self_attention_norm = nn.LayerNorm(d_model)

        self.enc_attention = MultiHeadAttention(num_heads, d_model)
        self.enc_attention_norm = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            # nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.feed_forward_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, memo2, tgt_mask=None, memory_mask=None):

        # Self-Attention on Decoder Inputs
        tgt2 = self.self_attention_norm(tgt + self.dropout(self.self_attention(tgt, tgt, tgt, tgt_mask)))

        # Attention on Encoder Outputs
        tgt3 = self.enc_attention_norm(tgt2 + self.dropout(self.enc_attention(tgt2, memory, memory, memory_mask)))
        if memo2 is not None:
            tgt3_5 = self.enc_attention_norm(tgt2 + self.dropout(self.enc_attention(tgt2, memo2, memo2, memory_mask)))
            tgt3 = (tgt3 + tgt3_5)/2

        # Feed-Forward Network
        tgt4 = self.feed_forward_norm(tgt3 + self.dropout(self.feed_forward(tgt3)))

        return tgt4

class TransformerDecoder(nn.Module):
    """
    Redesigned TransformerDecoder
    """
    def __init__(self, num_layers, num_heads, d_model, d_ff, dropout):
        super().__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList([TransformerDecoderLayer(num_heads, d_model, d_ff, dropout) for _ in range(num_layers)])
        self.norm_att1 = nn.LayerNorm(d_model)
        self.norm_att2 = nn.LayerNorm(d_model)
        self.norm_att3 = nn.LayerNorm(d_model)

        self.norm_att = nn.LayerNorm(d_model)

        # self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, memo2=None, embedding=None,tgt_mask=None, memory_mask=None):

        for i in range(self.num_layers):
            if embedding is not None:
                norm_tgt = self.norm_att(tgt + embedding)
            else:
                norm_tgt = self.norm_att(tgt)
            norm_memo1 = self.norm_att(memory)
            if memo2 is not None:
                norm_memo2 = self.norm_att(memo2)
            else:
                norm_memo2 = norm_memo1

            tgt = self.layers[i](norm_tgt, norm_memo1, norm_memo2, tgt_mask=tgt_mask, memory_mask=memory_mask)

        output = tgt
        return output


