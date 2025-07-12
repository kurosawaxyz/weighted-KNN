import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(style='darkgrid')
from tqdm import tqdm
from sklearn.datasets import load_iris
from palmerpenguins import load_penguins
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import argparse
from omegaconf import OmegaConf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from torchinfo import summary

from Wknn.nn import WKnn
from Wknn.capacity import compute_capacities_improved, capacity_regularization
from Wknn.hyperparam import Hyperparam
from Wknn.dataloader import dynamic_generate_positive_gaussian_data


def train(X: torch.Tensor, y: torch.Tensor, config: Hyperparam):
    """Training loop with improvements"""
    torch.manual_seed(42)
    
    d = X.shape[1]
    num_classes = len(torch.unique(y))
    
    model = WKnn(d, num_classes, config)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Count the number of parameters in the pipeline
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the model: {num_params}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=10
    )
    
    losses = []
    accuracies = []
    
    for epoch in tqdm(range(config.epochs), desc="Training"):
        model.train()

        idx = torch.randperm(X.size(0))[:config.batch_size]
        X = X[idx]
        y = y[idx]
        
        # Forward pass
        prediction_scores, capacities = model(X, y)
        
        # Main loss (cross-entropy)
        main_loss = F.cross_entropy(prediction_scores, y)
        
        # Regularization
        reg_loss = capacity_regularization(capacities, model.subsets, config)
        
        # Total loss
        total_loss = main_loss + config.capacity_reg * reg_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(total_loss.item())
        
        # Compute accuracy
        with torch.no_grad():
            predicted_labels = prediction_scores.argmax(dim=1)
            accuracy = (predicted_labels == y).float().mean().item()
            accuracies.append(accuracy)
        
        scheduler.step(total_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {total_loss.item():.4f} | "
                  f"Accuracy: {accuracy:.2%}")
    
    return model, losses, accuracies

def visualize_results(model, losses, accuracies, X, y, save_path=None):
    """Enhanced visualization of results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    ax1.plot(losses, label='Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(accuracies, label='Accuracy', color='blue')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.grid(True)
    ax2.legend()
    
    # Capacity visualization
    with torch.no_grad():
        latent = model.encoder(X)
        increments = model.capacity_generator(latent)
        capacities = compute_capacities_improved(increments, model.inclusion_mat, model.subsets)
    
    subset_names = ['∅' if not s else str(s) for s in model.subsets]
    
    ax3.bar(range(len(capacities)), capacities.numpy())
    ax3.set_xlabel('Subset Index')
    ax3.set_ylabel('Capacity Value')
    ax3.set_title('Learned Capacities')
    ax3.grid(True)
    
    # Feature importance
    d = X.shape[1]
    importance = torch.zeros(d)
    for i, subset in enumerate(model.subsets):
        for feature in subset:
            importance[feature] += capacities[i]
    
    ax4.bar(range(d), importance.numpy())
    ax4.set_xlabel('Feature Index')
    ax4.set_ylabel('Importance Score')
    ax4.set_title('Feature Importance')
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Print capacity details
    print("\n" + "="*50)
    print("LEARNED CAPACITIES:")
    print("="*50)
    for name, val in zip(subset_names, capacities):
        print(f"{name:>12}: {val.item():.4f}")
    
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE:")
    print("="*50)
    for i, imp in enumerate(importance):
        print(f"Feature {i}: {imp.item():.4f}")

    if save_path:
        plt.savefig(save_path)
        print(f"Results saved to {save_path}")

# Main execution
if __name__ == "__main__":
    # Argument parser for command line options
    parser = argparse.ArgumentParser(description="Train Choquet XAI Classifier")
    parser.add_argument('--config', type=str, default="config/config.yml", help='Path to config file')
    parser.add_argument('--data', type=str, default="iris", help='Dataset name (e.g., iris)')
    args = parser.parse_args()

    # Load the configuration file
    cfg = OmegaConf.load(args.config)
    # Configuration
    config = Hyperparam(**cfg)

    data_name = args.data.lower()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Create directory for saving checkpoints
    checkpoint_dir = f"checkpoints/{timestamp}_{data_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Checkpoint directory created at:", checkpoint_dir)
    
    if data_name == "iris":

        # Generate sample data
        torch.manual_seed(42)
        iris_data = load_iris()
        iris = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
        iris['target'] = iris_data.target
        
        # Proper data preprocessing
        scaler = StandardScaler()
        data = scaler.fit_transform(iris.iloc[:, :-1].values)
        labels = iris.iloc[:, -1].values
    
    elif data_name == "penguins":
        penguins = load_penguins()
        
        # Remove missing values first
        penguins = penguins.dropna()
        
        # Encode categorical variables
        le = LabelEncoder()
        
        # Encode island
        penguins['island'] = le.fit_transform(penguins['island'])
        
        # Encode sex
        penguins['sex'] = le.fit_transform(penguins['sex'])
        
        # Encode species (target variable)
        penguins['species'] = le.fit_transform(penguins['species'])

        # MEMORY FIX: Limit features to prevent exponential growth
        # Select only the most important features 
        # feature_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        # tmp = np.random.choice(range(len(penguins)), 100)

        # X = penguins.iloc[tmp][feature_columns]
        # y = penguins.iloc[tmp]['species']

        X = penguins.drop(columns=['species'])
        y = penguins['species']

        # Alternative: If you want to use all features, reduce the model complexity
        # by limiting the maximum subset size in your WKnn model
        
        # Proper data preprocessing
        scaler = StandardScaler()
        data = scaler.fit_transform(X)
        labels = y.values
        
    # Debug information
    print(f"Using dataset: {data_name}")
    print(f"Dataset shape: {data.shape}")
    print(f"Number of features: {data.shape[1]}")
    print(f"Number of possible subsets: {2**data.shape[1]}")
    print(f"Memory estimate: ~{2**data.shape[1] * 8} bytes for subset storage")

    X = torch.tensor(data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Train model
    print("Training Choquet XAI Classifier...")
    model, losses, accuracies = train(X, y, config)
    
    # Visualize results
    visualize_results(model, losses, accuracies, X, y, save_path=f"{checkpoint_dir}/results.png")

    with torch.no_grad():
        model.eval()
        print("Evaluating final model...")
        prediction_scores, capacities = model(X, y)
        predicted_labels = prediction_scores.argmax(dim=1)
        accuracy = (predicted_labels == y).float().mean().item()

        print(f"Final Training Accuracy: {accuracy:.2%}")

    # Save the model
    model_save_path = f"{checkpoint_dir}/model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    summa = summary(model, input_data=(X, y), device="cuda" if torch.cuda.is_available() else "cpu")
    # Save model summary to a file
    with open(f"{checkpoint_dir}/model_summary.txt", "w") as f:
        f.write(str(summa))

    r = []
    for i in range(len(capacities)):
        r.append({
            'index': i,
            'label': model.subsets[i],
            'capacity_value': capacities[i].item(),
        })

    # Save results to a txt file
    results_df = pd.DataFrame(r)
    results_df.to_csv(os.path.join(checkpoint_dir, 'best_capacity.txt'), index=False)