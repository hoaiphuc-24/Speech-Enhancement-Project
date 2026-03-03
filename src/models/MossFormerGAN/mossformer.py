import torch
import torch.nn as nn
from .modules import DepthwiseConv1DBlock, FeedForwardNetwork

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(self.dropout(attn_probs), V)
        return output

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(attn_output)
        return output

class ConvolutionModule(nn.Module):
    """Convolution Module commonly used in Conformer/MossFormer"""
    def __init__(self, channels, kernel_size=15, expansion_factor=2, dropout=0.1):
        super(ConvolutionModule, self).__init__()
        inner_channels = channels * expansion_factor
        
        self.pointwise_conv1 = nn.Conv1d(channels, inner_channels * 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(inner_channels, inner_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=inner_channels, bias=True)
        self.batch_norm = nn.BatchNorm1d(inner_channels)
        self.swish = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(inner_channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x expected shape: [Batch, Channels, Time]
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x

class MossFormerBlock(nn.Module):
    """
    A unified block representing MossFormer characteristics:
    - Multi-scale/Gated Self-Attention
    - Convolutional enhancement
    - Feed-forward networks (Macaron style)
    """
    def __init__(self, d_model, num_heads=4, ffn_expansion=4, conv_kernel_size=15, dropout=0.1):
        super(MossFormerBlock, self).__init__()
        
        # First FFN (Macaron net half-step)
        self.ffn1 = FeedForwardNetwork(d_model, d_model * ffn_expansion, dropout, activation='swish')
        self.norm1 = nn.LayerNorm(d_model)
        
        # Self Attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Convolutional Module
        self.conv_module = ConvolutionModule(d_model, conv_kernel_size, expansion_factor=2, dropout=dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Second FFN
        self.ffn2 = FeedForwardNetwork(d_model, d_model * ffn_expansion, dropout, activation='swish')
        self.norm4 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x is [B, T, C]
        
        # 1. First FFN (half step)
        residual = x
        x = self.norm1(x)
        x = residual + 0.5 * self.dropout(self.ffn1(x))
        
        # 2. Self Attention
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.attention(x, x, x, mask))
        
        # 3. Convolutional Module
        residual = x
        x = self.norm3(x)
        # Conv module expects [B, C, T]
        x = x.transpose(1, 2)
        x = self.conv_module(x)
        x = x.transpose(1, 2)
        x = residual + self.dropout(x)
        
        # 4. Second FFN (half step)
        residual = x
        x = self.norm4(x)
        x = residual + 0.5 * self.dropout(self.ffn2(x))
        
        return x
