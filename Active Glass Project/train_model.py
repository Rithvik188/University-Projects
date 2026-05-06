import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

from model import EGNNModel

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load data
    print("Loading data from glass_graph_data.pt...")
    dataset = torch.load('glass_graph_data.pt', weights_only=False)
    
    # 2. Shuffle and split (80% / 20%) -> 160 trains, 40 tests
    torch.manual_seed(42) # For reproducibility
    indices = torch.randperm(len(dataset)).tolist()
    
    train_indices = indices[:160]
    test_indices = indices[160:]
    
    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]
    
    # 3. DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 4. Model, Optimizer, Loss Function
    model = EGNNModel(in_features=2, hidden_dim=64, num_layers=4).to(device)
    
    # Optimizer: Adam with learning rate 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    # Loss: Mean Squared Error (MSE)
    criterion = nn.MSELoss()

    # Tracking metrics
    train_losses = []
    test_losses = []
    
    epochs = 100
    print("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for data in train_loader:
            data = data.to(device)
            
            optimizer.zero_grad()
            out = model(data)
            
            # Compute loss
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
            # Loss is averaged over nodes per batch. 
            total_train_loss += loss.item() * data.num_graphs
            
        avg_train_loss = total_train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)
        
        # Evaluation step at each epoch
        model.eval()
        total_test_loss = 0
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)
                total_test_loss += loss.item() * data.num_graphs
                
        avg_test_loss = total_test_loss / len(test_dataset)
        test_losses.append(avg_test_loss)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss (MSE): {avg_train_loss:.4f} | Test Loss (MSE): {avg_test_loss:.4f}")

    # 5. Final Detailed Evaluation (Pearson R)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            all_preds.extend(out.cpu().numpy())
            all_targets.extend(data.y.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate Pearson Correlation Coefficient
    pearson_r, _ = pearsonr(all_preds, all_targets)
    final_test_mse = test_losses[-1]
    
    print("\n" + "=" * 40)
    print("Training Complete!")
    print(f"Final Test MSE: {final_test_mse:.4f}")
    print(f"Pearson R (Propensity correlation): {pearson_r:.4f}")
    print("=" * 40 + "\n")

    # 6. Save Model
    print("Saving model weights to glass_gnn_model.pth...")
    torch.save(model.state_dict(), 'glass_gnn_model.pth')
    
    # 7. Generate and Document Plotted Results
    print("Generating learning_results.png plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Loss Curve
    ax1.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    ax1.plot(test_losses, label='Test Loss', color='red', linewidth=2)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.set_title('Learning Curve', fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Parity Plot
    # Hexbin or scatter to visualize density
    hb = ax2.hexbin(all_targets, all_preds, gridsize=50, cmap='viridis', mincnt=1)
    cb = fig.colorbar(hb, ax=ax2, label='Counts')
    
    # Diagonal line of perfect prediction
    min_val = min(np.min(all_targets), np.min(all_preds))
    max_val = max(np.max(all_targets), np.max(all_preds))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y = x')
    
    ax2.set_xlabel('Actual Propensity', fontsize=12)
    ax2.set_ylabel('Predicted Propensity', fontsize=12)
    ax2.set_title(f'Parity Plot (Test Set)\nPearson R = {pearson_r:.3f}', fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('learning_results.png', dpi=300)
    print("Saved learning_results.png successfully!")

if __name__ == '__main__':
    train()
