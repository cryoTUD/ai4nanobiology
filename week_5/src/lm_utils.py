import numpy as np 
import torch
import torch.nn as nn
import base64

def get_bpe_encoder(bpe_path):
    import tiktoken
    bpe_ranks = {}
    with open(bpe_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                token_b64, rank_str = line.split()
                bpe_ranks[base64.b64decode(token_b64)] = int(rank_str)

    enc = tiktoken.Encoding(
        name="gpt2_offline",
        pat_str=r"""'(?:[ sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s""",
        mergeable_ranks=bpe_ranks,
        special_tokens={"<|endoftext|>": 50256}
    )

    return enc

class SelfAttentionHead(nn.Module):
    def __init__(self, d_input, d_model,
                 project_output=True):
        """
        project_output : if True, apply an internal W_O mapping d_model -> d_input
                         (standalone head). Set False when used inside
                         MultiHeadAttention, where a single shared projection is
                         applied after concatenating the heads.
        """
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.project_output = project_output

        self.W_Q = nn.Linear(d_input, d_model)
        self.W_K = nn.Linear(d_input, d_model)
        self.W_V = nn.Linear(d_input, d_model)
        self.W_O = nn.Linear(d_model, d_input) if project_output else None


    def forward(self, x, forward_mask=None, verbose=False, return_attention_weights=False):
        Q = self.W_Q(x); K = self.W_K(x); V = self.W_V(x)
        if verbose:
            print("Query shape:", Q.shape, " Key:", K.shape, " Value:", V.shape)

        A = Q @ K.transpose(-1, -2) / np.sqrt(self.d_model)
        if forward_mask is not None:
            A = A.masked_fill(~forward_mask, float("-inf"))
        A = A - A.amax(dim=-1, keepdim=True)
        A = torch.exp(A)
        A = A / A.sum(dim=-1, keepdim=True)

        output = A @ V                       # (batch, seq, d_model)
        if self.W_O is not None:
            output = self.W_O(output)        # -> (batch, seq, d_input)

        if return_attention_weights:
            return output, A
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_input, d_model,
                 dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_input = d_input
        self.d_model = d_model
        self.d_head = d_model // n_heads

        # each head projects d_input -> d_head and does NOT apply its own W_O
        self.heads = nn.ModuleList([
            SelfAttentionHead(d_input, self.d_head, project_output=False)
            for _ in range(n_heads)
        ])
        # single shared output projection after concatenation: d_model -> d_input
        self.project_out = nn.Linear(d_model, d_input)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, forward_mask=None, verbose=False, return_attention_weights=False):
        head_outputs = []
        attn_weights = []
        for head in self.heads:
            if return_attention_weights:
                out_h, A_h = head(x, forward_mask, verbose, return_attention_weights)
                attn_weights.append(A_h)
            else:
                out_h = head(x, forward_mask, verbose)
            head_outputs.append(out_h)          # each (batch, seq, d_head)

        # concat heads along feature axis -> (batch, seq, d_model)
        concat = torch.cat(head_outputs, dim=-1)
        output = self.project_out(concat)       # -> (batch, seq, d_input)
        output = self.dropout(output)

        if return_attention_weights:
            attn = torch.stack(attn_weights, dim=1)
            return output, attn
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, n_heads, d_input, d_model, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, d_input, d_model)
        self.norm1 = nn.LayerNorm(d_input)
        self.ffn = FeedForward(d_input, d_ff)
        self.norm2 = nn.LayerNorm(d_input)

    def forward(self, x, forward_mask=None, return_attention_weights=False):
        if return_attention_weights:
            attn_output, attns = self.attention(x, 
                                                forward_mask=forward_mask,
                                                return_attention_weights=return_attention_weights
            )
        else:
            attn_output = self.attention(x, 
                                         forward_mask=forward_mask,
            )
        x = self.norm1(x + attn_output)  # Residual connection + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)    # Residual connection + LayerNorm
        if return_attention_weights:
            return x, attns
        else:
            return x
        
class LanguageModel(nn.Module):
    def __init__(self, n_layers, n_head, d_model, d_input, d_ff, vocab_size, context_length):
        super().__init__()
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_model = d_model 
        self.d_input = d_input 
        self.d_ff = d_ff 

        self.token_embedding = nn.Embedding(vocab_size, d_input) 
        self.positional_embedding = nn.Embedding(context_length, d_input)


        
        self.transformer_layers = nn.ModuleList([TransformerBlock(n_head, d_input, d_model, d_ff) 
                                        for _ in range(n_layers)])
        self.output_head = nn.Linear(d_input, vocab_size)

    def forward(self, idx, causal=False, verbose=False, return_attention_weights=False):
        batch, seq = idx.shape
        if causal:
            mask = torch.tril(torch.ones((seq, seq), dtype=torch.bool, device=idx.device))
        else:
            mask = None
        layer_attentions = []
        token_emb = self.token_embedding(idx)
        positions = torch.arange(idx.shape[1], device=idx.device)
        pos_emb = self.positional_embedding(positions)
        x = token_emb + pos_emb
        layer_out = x
        if return_attention_weights:
            for layer in self.transformer_layers:
                layer_out, layer_attn = layer(layer_out, 
                                              forward_mask=mask,
                                              return_attention_weights=return_attention_weights
                )
                layer_attentions.append(layer_attn)
            
            out = self.output_head(layer_out)
            return out, layer_attentions
        else: 
            for layer in self.transformer_layers:
                layer_out = layer(layer_out,
                                  forward_mask=mask,
                )
            out = self.output_head(layer_out)
            return out
    
    def generate(self, idx, max_new_tokens, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.positional_embedding.num_embeddings:]
            out = self(idx_cond, causal=True)
            next_token_logits = out[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            if top_k is not None:
                top_k_probs, top_k_indices = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter(1, top_k_indices, top_k_probs)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            idx = torch.cat((idx, next_token.unsqueeze(1)), dim=1)
        return idx
            
        
            
            
       