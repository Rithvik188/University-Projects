"""
evaluate_transferability.py
Loads the trained EGNN and runs it on blind_test_data.pt (new simulation, seed=42).
Uses CUDA (RTX 3060) for inference.
Outputs: Pearson R on the unseen data, internal_transfer_parity.png
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from torch_geometric.loader import DataLoader

from model import EGNNModel

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Inference device: {device}")

# ── Load blind test set ────────────────────────────────────────────────────────
print("Loading blind_test_data.pt…")
dataset = torch.load('blind_test_data.pt', weights_only=False)
loader  = DataLoader(dataset, batch_size=16, shuffle=False)
print(f"  {len(dataset)} graphs loaded")

# ── Load trained model ─────────────────────────────────────────────────────────
print("Loading glass_gnn_model.pth…")
model = EGNNModel(in_features=2, hidden_dim=64, num_layers=4).to(device)
model.load_state_dict(torch.load('glass_gnn_model.pth',
                                  map_location=device,
                                  weights_only=True))
model.eval()

# ── Inference ──────────────────────────────────────────────────────────────────
all_preds   = []
all_targets = []

print("Running inference…")
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        preds = model(batch)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch.y.cpu().numpy())

all_preds   = np.array(all_preds)
all_targets = np.array(all_targets)

# ── Metrics ────────────────────────────────────────────────────────────────────
pearson_r, p_val = pearsonr(all_preds, all_targets)
mse              = np.mean((all_preds - all_targets) ** 2)

print("\n" + "=" * 50)
print("Internal Transferability Results (seed=42 test set)")
print(f"  Pearson R  : {pearson_r:.4f}  (p={p_val:.2e})")
print(f"  Test MSE   : {mse:.4f}")
print("=" * 50 + "\n")

# ── Parity plot ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

hb = ax.hexbin(all_targets, all_preds, gridsize=60, cmap='plasma', mincnt=1)
fig.colorbar(hb, ax=ax, label='Counts')

lo = min(all_targets.min(), all_preds.min())
hi = max(all_targets.max(), all_preds.max())
ax.plot([lo, hi], [lo, hi], 'w--', lw=2, label='y = x (perfect)')

ax.set_xlabel('Actual Propensity (seed=42 system)', fontsize=12)
ax.set_ylabel('Predicted Propensity (EGNN)',          fontsize=12)
ax.set_title(
    f'Internal Transferability — Unseen Simulation\n'
    f'Pearson $R$ = {pearson_r:.3f}  |  MSE = {mse:.4f}',
    fontsize=13
)
ax.legend(fontsize=11)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('internal_transfer_parity.png', dpi=300)
print("Saved internal_transfer_parity.png")
