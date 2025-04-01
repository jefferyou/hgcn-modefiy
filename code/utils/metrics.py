import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


def calc_f1(y_true, y_pred, sigmoid=False, average="micro"):
    """
    Calculate F1 score for classification
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels or probabilities
        sigmoid: Whether predictions are from sigmoid (multi-label) or softmax (multi-class)
        average: How to average F1 scores ('micro', 'macro', 'weighted', etc.)
        
    Returns:
        Tuple of (micro_f1, macro_f1)
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    if not sigmoid:
        # Multi-class classification with softmax
        y_true = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
        y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    else:
        # Multi-label classification with sigmoid
        y_pred = (y_pred > 0.5).astype(int)
    
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=1)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=1)
    
    return micro_f1, macro_f1


def evaluate_unsupervised(model, val_edges, device="cpu"):
    """
    Evaluate unsupervised model on validation edges
    
    Args:
        model: The model to evaluate
        val_edges: Tuple of (val_node1, val_node2) for validation
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    val_node1, val_node2 = val_edges
    val_node1 = val_node1.to(device)
    val_node2 = val_node2.to(device)
    
    with torch.no_grad():
        # Get model outputs
        loss, scores, _, _ = model(val_node1, val_node2)
        
        # Calculate MRR (Mean Reciprocal Rank)
        # This assumes the model returns scores for positive edges
        # and negative edges in a format where higher score means more likely
        mrr = torch.mean(1.0 / (torch.argsort(
            torch.argsort(-scores)  # Negative to sort in descending order
        ) + 1).float())
    
    return {
        "loss": loss.item(),
        "mrr": mrr.item()
    }


def evaluate_supervised(model, val_nodes, val_labels, sigmoid=False, device="cpu"):
    """
    Evaluate supervised model on validation nodes
    
    Args:
        model: The model to evaluate
        val_nodes: Validation node IDs
        val_labels: Validation labels
        sigmoid: Whether model uses sigmoid (multi-label) or softmax (multi-class)
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    val_nodes = torch.LongTensor(val_nodes).to(device)
    val_labels = val_labels.to(device)
    
    with torch.no_grad():
        # Get model predictions
        logits = model(val_nodes)
        preds = model.predict(logits)
        loss = model.loss(logits, val_labels)
        
        # Calculate F1 scores
        micro_f1, macro_f1 = calc_f1(val_labels, preds, sigmoid=sigmoid)
    
    return {
        "loss": loss.item(),
        "micro_f1": micro_f1,
        "macro_f1": macro_f1
    }


def print_metrics(metrics, epoch=None, prefix="", use_tqdm=False):
    """
    Print evaluation metrics in a nice format
    
    Args:
        metrics: Dictionary of metrics
        epoch: Current epoch (optional)
        prefix: Prefix string (e.g., 'Train', 'Val')
        use_tqdm: Whether to use tqdm for printing
    """
    message = prefix
    
    if epoch is not None:
        message += f" Epoch: {epoch:4d}"
        
    for k, v in metrics.items():
        if isinstance(v, float):
            message += f" | {k}: {v:.5f}"
        else:
            message += f" | {k}: {v}"
    
    if use_tqdm:
        from tqdm import tqdm
        tqdm.write(message)
    else:
        print(message)
