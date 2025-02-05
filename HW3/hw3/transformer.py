import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def sliding_window_attention(q, k, v, window_size, padding_mask=None):
    """
    Computes the simple sliding window attention from 'Longformer: The Long-Document Transformer'.
    This implementation is meant for multihead attention on batched tensors. It should work for both single and multi-head attention.
    :param q - the query vectors. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param k - the key vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param v - the value vectors.  #[Batch, *, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :param window_size - size of sliding window. Must be an even number.
    :param padding_mask - a mask that indicates padding with 0.  #[Batch, SeqLen]
    :return values - the output values. #[Batch, SeqLen, Dims] or [Batch, num_heads, SeqLen, Dims]
    :return attention - the attention weights. #[Batch, SeqLen, SeqLen] or [Batch, num_heads, SeqLen, SeqLen]
    """
    assert window_size % 2 == 0, "window size must be an even number"

    bt_size, seq_len, embed_dim = q.shape[0], q.shape[-2], q.shape[-1]
    num_heads = q.shape[1] if q.dim() > 3 else None

    k_padded = F.pad(k, (0, 0, window_size // 2, window_size // 2)).to(q.device)

    k_indices = torch.arange(k_padded.shape[-2], device=q.device).unfold(
        0, window_size + 1, 1
    )
    k_indices = k_indices.unsqueeze(0)
    if q.dim() > 3:
        k_indices = k_indices.unsqueeze(0).repeat(bt_size, num_heads, 1, 1)
        k_indices = k_indices.unsqueeze(-1).expand(-1, -1, -1, -1, embed_dim)
        k_padded = k_padded.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
    else:
        k_indices = k_indices.repeat(bt_size, 1, 1)
        k_indices = k_indices.unsqueeze(-1).expand(-1, -1, -1, embed_dim)
        k_padded = k_padded.unsqueeze(1).expand(-1, seq_len, -1, -1)

    k_windows = torch.gather(k_padded, -2, k_indices).to(q.device)

    q_reshaped = q.unsqueeze(-2)
    atten_scores = torch.matmul(q_reshaped, k_windows.transpose(-1, -2))
    atten_scores = atten_scores / math.sqrt(embed_dim)
    atten_scores = atten_scores.squeeze(-2)

    idx = torch.arange(seq_len, device=q.device).unsqueeze(1) + torch.arange(
        -window_size // 2, window_size // 2 + 1, device=q.device
    ).unsqueeze(0)
    idx = idx.clamp(0, seq_len - 1).unsqueeze(0)

    if q.dim() > 3:
        idx = idx.unsqueeze(0).expand(bt_size, num_heads, -1, -1)
        atten_scores = torch.zeros(
            (bt_size, num_heads, seq_len, seq_len), device=q.device
        ).scatter_add_(3, idx, atten_scores)
    else:
        idx = idx.expand(bt_size, -1, -1)
        atten_scores = torch.zeros(
            (bt_size, seq_len, seq_len), device=q.device
        ).scatter_add_(2, idx, atten_scores)

    atten_scores[atten_scores == 0.0] = float("-inf")
    atten_scores = atten_scores.detach()

    if padding_mask is not None:
        if q.dim() > 3:
            padding_mask = padding_mask.unsqueeze(-2).unsqueeze(-2)
        else:
            padding_mask = padding_mask.unsqueeze(-2)

        atten_scores = atten_scores.masked_fill_(padding_mask == 0, float("-inf"))
        atten_scores = atten_scores.masked_fill_(
            padding_mask.transpose(-1, -2) == 0, float("-inf")
        )

    attention = F.softmax(atten_scores, dim=-1)
    attention[torch.isnan(attention)] = 0.0
    values = torch.matmul(attention, v)

    return values, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads, window_size):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        # Stack all weight matrices 1...h together for efficiency
        # "bias=False" is optional, but for the projection we learned, there is no theoretical justification to use bias
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation of the paper if you would like....
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, padding_mask, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, 3*Dims]

        q, k, v = qkv.chunk(3, dim=-1)  # [Batch, Head, SeqLen, Dims]

        values, attention = sliding_window_attention(
            q, k, v, self.window_size, padding_mask
        )

        # Concatenate heads and project output
        values = values.permute(0, 2, 1, 3).reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, window_size, dropout=0.1):
        """
        :param embed_dim: the dimensionality of the input and output
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param num_heads: the number of heads in the multi-head attention
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            embed_dim, embed_dim, num_heads, window_size
        )
        self.feed_forward = PositionWiseFeedForward(embed_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask):
        """
        :param x: the input to the layer of shape [Batch, SeqLen, Dims]
        :param padding_mask: the padding mask of shape [Batch, SeqLen]
        :return: the output of the layer of shape [Batch, SeqLen, Dims]
        """

        attn_output = self.dropout(self.self_attn(x, padding_mask))
        x = self.norm1(x + attn_output)
        ff_output = self.dropout(self.feed_forward(x))
        x = self.norm2(x + ff_output)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        hidden_dim,
        max_seq_length,
        window_size,
        dropout=0.1,
    ):
        """
        :param vocab_size: the size of the vocabulary
        :param embed_dim: the dimensionality of the embeddings and the model
        :param num_heads: the number of heads in the multi-head attention
        :param num_layers: the number of layers in the encoder
        :param hidden_dim: the dimensionality of the hidden layer in the feed-forward network
        :param max_seq_length: the maximum length of a sequence
        :param window_size: the size of the sliding window
        :param dropout: the dropout probability

        """
        super(Encoder, self).__init__()
        self.encoder_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(embed_dim, hidden_dim, num_heads, window_size, dropout)
                for _ in range(num_layers)
            ]
        )

        self.classification_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence, padding_mask):
        """
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the logits  [Batch]
        """
        output = None

        x = self.encoder_embedding(sentence)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, padding_mask)
        output = self.classification_mlp(x[:, 0])

        return output

    def predict(self, sentence, padding_mask):
        """
        :param sententence #[Batch, max_seq_len]
        :param padding mask #[Batch, max_seq_len]
        :return: the binary predictions  [Batch]
        """
        logits = self.forward(sentence, padding_mask)
        preds = torch.round(torch.sigmoid(logits))
        return preds
