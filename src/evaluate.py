import torch
from tqdm import tqdm

def eval_fn(data_loader, model, loss_fn):
    model.eval()
    total_loss = 0.0
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        for images, masks in tqdm(data_loader, desc="Evaluation"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).long()
            
            # AJOUT : Squeeze
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            
            # Forward pass
            outputs = model(images)
            
            # Calcul de la loss
            loss = loss_fn(outputs, masks)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss

