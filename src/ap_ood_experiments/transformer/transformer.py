import math
import torch
from torch import nn


class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_key, d_value):
        super(EncoderMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_value = d_value

        self.k = nn.Linear(d_model, n_heads * d_key)
        self.q = nn.Linear(d_model, n_heads * d_key)
        self.v = nn.Linear(d_model, n_heads * d_value)
        self.o = nn.Linear(n_heads * d_value, d_model)

    def forward(self, x, attention_mask):
        B, S, D = x.shape

        k = self.k(x)
        q = self.q(x)
        v = self.v(x)

        k = k.reshape([B, S, self.n_heads, self.d_key]).transpose(1, 2)  # B, NH, S, D
        q = q.reshape([B, S, self.n_heads, self.d_key]).transpose(1, 2)
        v = v.reshape([B, S, self.n_heads, self.d_value]).transpose(1, 2)

        attention_mask_2d = self.get_2d_mask(attention_mask)

        attn_output = nn.functional.scaled_dot_product_attention(q, k, v, attention_mask_2d)  # scale is set correctly by sdpa

        attn_output = attn_output.transpose(1, 2).reshape([B, S, self.n_heads * self.d_value])
        return self.o(attn_output)

    def get_2d_mask(self, attention_mask):
        B, S = attention_mask.shape
        key_mask = attention_mask.bool().unsqueeze(1)  # (B, 1, S)
        mask = key_mask.expand(-1, S, -1)  # (B, S, S)
        return mask.unsqueeze(1)  # (B, 1, S, S)


class TransformerFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(TransformerFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.relu(x)
        return self.linear2(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, d_key, d_value, p_drop):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_value = d_value
        self.p_drop = p_drop
        

        self.multi_head_attention = EncoderMultiHeadAttention(d_model, n_heads, d_key, d_value)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=p_drop)
        self.feedfoward = TransformerFeedForward(d_model, d_ff)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attention_mask):
        x_mha = self.multi_head_attention(x, attention_mask=attention_mask)
        x_mha = self.dropout(x_mha)
        x = self.ln1(x + x_mha)
        x_ff = self.feedfoward(x)
        x_ff = self.dropout(x_ff)
        return self.ln2(x + x_ff)



class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, d_model, d_ff, n_heads, d_key, d_value, p_drop):
        super(TransformerEncoder, self).__init__()
        self.n_layer = n_layer
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_value = d_value
        self.p_drop = p_drop
        
        self.layers = nn.ModuleList([TransformerEncoderLayer(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            d_key=d_key,
            d_value=d_value,
            p_drop=p_drop,
        ) for _ in range(self.n_layer)])


    def forward(self, x, attention_mask):
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x


class DecoderMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_key, d_value):
        super(DecoderMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_value = d_value

        self.k = nn.Linear(d_model, n_heads * d_key)
        self.q = nn.Linear(d_model, n_heads * d_key)
        self.v = nn.Linear(d_model, n_heads * d_value)
        self.o = nn.Linear(n_heads * d_value, d_model)

    def forward(self, x, attention_mask):
        B, S, D = x.shape

        k = self.k(x)
        q = self.q(x)
        v = self.v(x)

        k = k.reshape([B, S, self.n_heads, self.d_key]).transpose(1, 2)  # B, NH, S, D
        q = q.reshape([B, S, self.n_heads, self.d_key]).transpose(1, 2)
        v = v.reshape([B, S, self.n_heads, self.d_value]).transpose(1, 2)

        attention_mask_2d = self.get_2d_mask(attention_mask)

        attn_output = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask_2d, is_causal=True)  # scale is set correctly by sdpa

        attn_output = attn_output.transpose(1, 2).reshape([B, S, self.n_heads * self.d_value])
        return self.o(attn_output)

    def get_2d_mask(self, attention_mask):
        B, S = attention_mask.shape
        key_mask = attention_mask.bool().unsqueeze(1)  # (B, 1, S)
        mask = key_mask.expand(-1, S, -1)  # (B, S, S)
        return mask.unsqueeze(1)  # (B, 1, S, S)

class DecoderCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_key, d_value):
        super(DecoderCrossAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_value = d_value

        self.k = nn.Linear(d_model, n_heads * d_key)
        self.q = nn.Linear(d_model, n_heads * d_key)
        self.v = nn.Linear(d_model, n_heads * d_value)
        self.o = nn.Linear(n_heads * d_value, d_model)

    def forward(self, x, attention_mask, cross_attention_weights, cross_attention_mask):
        B, Sd, D = x.shape
        _, Se, D = cross_attention_weights.shape

        k = self.k(cross_attention_weights)
        q = self.q(x)
        v = self.v(cross_attention_weights)

        k = k.reshape([B, Se, self.n_heads, self.d_key]).transpose(1, 2)
        q = q.reshape([B, Sd, self.n_heads, self.d_key]).transpose(1, 2)
        v = v.reshape([B, Se, self.n_heads, self.d_value]).transpose(1, 2)

        attention_mask_2d = self.get_2d_mask(attention_mask, cross_attention_mask)

        attn_output = nn.functional.scaled_dot_product_attention(q, k, v, attention_mask_2d)  # scale is set correctly by sdpa

        attn_output = attn_output.transpose(1, 2).reshape([B, Sd, self.n_heads * self.d_value])
        return self.o(attn_output)

    def get_2d_mask(self, attention_mask, cross_attention_mask):
        B, Sd = attention_mask.shape
        B, Se = cross_attention_mask.shape
        return torch.ones([B, 1, Sd, 1], device=cross_attention_mask.device).bool() & cross_attention_mask.reshape([B, 1, 1, Se]).bool()


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, d_key, d_value, p_drop):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_value = d_value
        self.p_drop = p_drop

        self.multi_head_attention = DecoderMultiHeadAttention(d_model, n_heads, d_key, d_value)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=p_drop)
        self.cross_attention = DecoderCrossAttention(d_model, n_heads, d_key, d_value)
        self.ln2 = nn.LayerNorm(d_model)
        self.feedfoward = TransformerFeedForward(d_model, d_ff)
        self.ln3 = nn.LayerNorm(d_model)
    
    def forward(self, x, attention_mask, cross_attention_weights, cross_attention_mask):
        x_mha = self.multi_head_attention(x, attention_mask=attention_mask)
        x_mha = self.dropout(x_mha)
        x = self.ln1(x + x_mha)
        x_ca = self.cross_attention(x, attention_mask=attention_mask, cross_attention_weights=cross_attention_weights, cross_attention_mask=cross_attention_mask)
        x_ca = self.dropout(x_ca)
        x = self.ln2(x + x_ca)
        x_ff = self.feedfoward(x)
        x_ff = self.dropout(x_ff)
        return self.ln3(x + x_ff)


class TransformerDecoder(nn.Module):
    def __init__(self, n_layer, d_model, d_ff, n_heads, d_key, d_value, p_drop):
        super(TransformerDecoder, self).__init__()
        self.n_layer = n_layer
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_value = d_value
        self.p_drop = p_drop

        self.layers = nn.ModuleList([TransformerDecoderLayer(
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            d_key=d_key,
            d_value=d_value,
            p_drop=p_drop,
        ) for _ in range(self.n_layer)])

    def forward(self, x, attention_mask, cross_attention_weights, cross_attention_mask):
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, cross_attention_weights=cross_attention_weights, cross_attention_mask=cross_attention_mask)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, n_layer=6, d_model=512, d_ff=2048, n_heads=8, d_key=64, d_value=64, p_drop=0.1):
        super(Transformer, self).__init__()
        self.n_layer = n_layer
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.d_key = d_key
        self.d_value = d_value
        self.p_drop = p_drop
        self.vocab_size = vocab_size

        self.input_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.output_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pe = PositionalEncoding(d_model, max_len=1024)
        self.dropout = nn.Dropout(p=p_drop)
        self.linear_out = nn.Linear(self.d_model, self.vocab_size)

        self.encoder = TransformerEncoder(
            n_layer=n_layer,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            d_key=d_key,
            d_value=d_value,
            p_drop=p_drop,
        )

        self.decoder = TransformerDecoder(
            n_layer=n_layer,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            d_key=d_key,
            d_value=d_value,
            p_drop=p_drop,
        )

    def input_embed(self, input_ids):
        embedded_input = self.input_embedding(input_ids)
        embedded_input = self.pe(embedded_input)
        embedded_input = self.dropout(embedded_input)
        return embedded_input
    
    def output_embed(self, input_ids):
        embedded_input = self.output_embedding(input_ids)
        embedded_input = self.pe(embedded_input)
        embedded_input = self.dropout(embedded_input)
        return embedded_input

    def forward(self, input_ids, attention_mask, target_input_ids, target_attention_mask):
        embedded_input = self.input_embed(input_ids)
        embedded_target = self.output_embed(target_input_ids)

        encoded_input = self.encoder(x=embedded_input, attention_mask=attention_mask)
        output = self.decoder(x=embedded_target,
                              attention_mask=target_attention_mask,
                              cross_attention_weights=encoded_input,
                              cross_attention_mask=attention_mask)
        predicted_ids = self.linear_out(output)
        return predicted_ids
