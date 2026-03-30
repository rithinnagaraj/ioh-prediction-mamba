import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 32, max_len: int = 1500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        
        return x

class IOHTransformer(nn.Module):
    def __init__(self, model_dim=64, num_heads=8):
        super(IOHTransformer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm_attn = nn.LayerNorm(model_dim)

        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.ReLU(),
            nn.Linear(model_dim * 4, model_dim)
        )
        self.layer_norm_ffn = nn.LayerNorm(model_dim)

    def forward(self, x):
        # x shape: (Batch, seq_len, input_dim)
        attn_output, _ = self.multihead_attn(x, x, x)  # (Batch, seq_len, model_dim)
        x = self.layer_norm_attn(x + attn_output)  # Residual connection + LayerNorm

        ffn_output = self.ffn(x)  # (Batch, seq_len, model_dim)
        x = self.layer_norm_ffn(x + ffn_output)  # Residual connection + LayerNorm

        return x

class IOHPredictor(nn.Module):
    def __init__(self, input_dim = 4, model_dim_1 = 32, model_dim_2 = 64, num_heads = 4):
        super(IOHPredictor, self).__init__()
        self.input_projection_1 = nn.Linear(input_dim, model_dim_1)
        self.pos_embed = PositionalEncoding(d_model=model_dim_1)
        self.transformer_1 = IOHTransformer(model_dim=model_dim_1, num_heads=num_heads)

        self.conv1d = nn.Conv1d(model_dim_1, model_dim_2, kernel_size=5, stride=2)

        # self.input_projection_2 = nn.Linear(model_dim_1, model_dim_2)
        self.transformer_2 = IOHTransformer(model_dim=model_dim_2, num_heads=num_heads)

        # extra
        # self.input_projection_3 = nn.Linear(model_dim_2, 64)
        # self.transformer_3 = IOHTransformer(model_dim=64, num_heads=num_heads)

        self.output_projection = nn.Sequential(
            nn.Linear(model_dim_2+5, model_dim_2 // 2),
            nn.ELU(),
            nn.Linear(model_dim_2 // 2, 1)
        )

        # self.output_projection = nn.Sequential(
        #     nn.Linear(64+5, 64 // 2),
        #     nn.ELU(),
        #     nn.Linear(64 // 2, 1)
        # )
    
    def forward(self, x_seq, x_static):
        x = self.input_projection_1(x_seq) # (B, 900, 4)
        x = self.pos_embed(x)  # (B, 900, 4) -> (B, 900, model_dim)
        x = self.transformer_1(x)  # (Batch, 900, 32)

        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)  # (B, 448, 64)
        print(x.shape)
        x = self.transformer_2(x)  # (Batch, 448, 64)

        # x = self.input_projection_3(x)
        # x = self.transformer_3(x)  # (Batch, 448, 64)

        x = torch.mean(x, dim=1)  # (B, 64)
        x = torch.concat([x, x_static], dim=-1)  # (B, 64 + 5)

        output = self.output_projection(x)  # (B, 1)
        return output.squeeze(-1)  # (B,)

if __name__ == "__main__":
    model = IOHPredictor(input_dim=4, model_dim_1=32, model_dim_2=64, num_heads=4)
    dummy_seq = torch.randn(2, 900, 4)  # (Batch, seq_len, input_dim)
    dummy_static = torch.randn(2, 5)# (Batch, static_dim)

    output = model(dummy_seq, dummy_static)
    print("Output shape:", output.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")