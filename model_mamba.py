import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("Please install mamba_ssm and causal-conv1d to use this model.")

class IOHMambaPredictor(nn.Module):
    def __init__(self, input_dim=4, model_dim_1=64, model_dim_2=64):
        super(IOHMambaPredictor, self).__init__()
        
        # 1. Initial Projection 
        self.input_projection_1 = nn.Linear(input_dim, model_dim_1)
        self.norm_1 = nn.LayerNorm(model_dim_1)
        
        # 2. Mamba Block 1
        self.mamba_1 = Mamba(
            d_model=model_dim_1,
            d_state=16,
            d_conv=4,
            expand=2
        )

        # 3. Projection to larger dimension
        self.input_projection_2 = nn.Linear(model_dim_1, model_dim_2)
        self.norm_2 = nn.LayerNorm(model_dim_2)
        
        # 4. Mamba Block 2
        self.mamba_2 = Mamba(
            d_model=model_dim_2,
            d_state=16,
            d_conv=4,
            expand=2
        )
        
        # 5. Final Output Projection (Late Fusion - Identical to Transformer)
        self.output_projection = nn.Sequential(
            nn.Linear(model_dim_2 + 5, model_dim_2 // 2),
            nn.ELU(),
            nn.Linear(model_dim_2 // 2, 1)
        )
    
    def forward(self, x_seq, x_static):
        # Sequence: (B, 900, 4)
        x = self.input_projection_1(x_seq) # (B, 900, 64)
        x = self.norm_1(x)
        x = self.mamba_1(x)                # (B, 900, 64)
        
        x = self.input_projection_2(x)     # (B, 900, 64)
        x = self.norm_2(x)
        x = self.mamba_2(x)                # (B, 900, 64)

        # 1. Pool the time-series sequence
        x = torch.mean(x, dim=1)           # (B, 64)

        # 2. Concatenate static demographics
        x = torch.concat([x, x_static], dim=-1)  # (B, 69)

        # 3. Final prediction
        output = self.output_projection(x) # (B, 1)
        
        return output.squeeze(-1)          # (B)

if __name__ == "__main__":
    # Mamba requires CUDA. It will crash on CPU due to the custom kernels.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = IOHMambaPredictor().to(device)
    
    # Calculate trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Mamba Model Parameters: {total_params:,}")
    
    if device.type == "cuda":
        dummy_seq = torch.randn(32, 900, 4).to(device)
        dummy_static = torch.randn(32, 5).to(device)
        
        out = model(dummy_seq, dummy_static)
        print(f"Output Shape: {out.shape} (Expected: [32])")
    else:
        print("Warning: CUDA not detected. Mamba forward pass skipped.")