import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, test_loader, device='cuda', threshold=0.2):
    """
    Evaluate model performance with MAE, MSE and Accuracy metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test dataset
        device: Computation device ('cuda' or 'cpu')
        threshold: Distance threshold (in meters) for accuracy calculation
    
    Returns:
        Dictionary containing MAE, MSE, Accuracy and raw predictions/targets
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X1, X2, X3, X4, y in test_loader:
            # Move data to device
            X1, X2, X3, X4 = X1.to(device), X2.to(device), X3.to(device), X4.to(device)
            y = y.to(device)
            
            # Get predictions
            outputs = model(X1, X2, X3, X4)
            
            # Store results
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # Calculate Euclidean distance for each prediction
    distances = np.sqrt(np.sum((predictions - targets)**2, axis=1))
    
    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    
    # Accuracy = percentage of predictions within threshold distance
    correct = np.sum(distances <= threshold)
    accuracy = correct / len(distances)
    
    results = {
        'MAE': mae,
        'MSE': mse,
        'Accuracy': accuracy,
        'Threshold': threshold,
        'Predictions': predictions,
        'Targets': targets,
        'Distances': distances
    }
    
    # Print summary
    print(f"\nEvaluation Results:")
    print(f"- MAE: {mae:.4f} meters")
    print(f"- MSE: {mse:.4f} meters²")
    print(f"- Accuracy (@{threshold}m): {accuracy:.2%} ({correct}/{len(distances)})")
    print(f"- Max distance: {np.max(distances):.4f}m")
    print(f"- Min distance: {np.min(distances):.4f}m")
    print(f"- Median distance: {np.median(distances):.4f}m")
    
    return results


# Example usage:
if __name__ == "__main__":
    # Assuming you have defined model and test_loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Evaluate with default 0.2m threshold
    eval_results = evaluate_model(model, test_loader, device)
    
    # You can also specify custom threshold (e.g., 0.1m for stricter evaluation)
    # eval_results = evaluate_model(model, test_loader, device, threshold=0.1)