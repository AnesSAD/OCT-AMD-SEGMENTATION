import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from src.evaluate import eval_fn

def plot_multiclass_segmentation(loader, model, num_classes=8, class_names=None):
    """
    Visualise les résultats de segmentation multi-classes
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if class_names is None:
        class_names = [f'Classe {i}' for i in range(num_classes)]
    
    # Créer une colormap personnalisée
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    cmap = ListedColormap(colors)
    
    model.eval()
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            # Prédiction
            outputs = model(images)
            pred_masks = torch.argmax(outputs, dim=1)
            
            # Convertir en numpy
            image = images[0].cpu().numpy()
            
            # AJOUT : Squeeze pour enlever les dimensions inutiles
            if image.shape[0] == 1:  # Si 1 channel
                image = image.squeeze(0)  # (1, H, W) -> (H, W)
            else:  # Si 3 channels
                image = image.transpose(1, 2, 0)
            
            true_mask = masks[0].cpu().numpy()
            
            # AJOUT : Squeeze le mask aussi si nécessaire
            while true_mask.ndim > 2:
                true_mask = true_mask.squeeze(0)  # (1, H, W) -> (H, W)
            
            pred_mask = pred_masks[0].cpu().numpy()
            
            # AJOUT : Squeeze pred_mask aussi
            while pred_mask.ndim > 2:
                pred_mask = pred_mask.squeeze(0)
            
            # Créer la figure
            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
            
            # 1. Image originale
            axs[0, 0].imshow(image, cmap='gray')
            axs[0, 0].set_title('Image Originale', fontsize=14, fontweight='bold')
            axs[0, 0].axis('off')
            
            # 2. Masque vrai
            im1 = axs[0, 1].imshow(true_mask, cmap=cmap, vmin=0, vmax=num_classes-1)
            axs[0, 1].set_title('Masque Vrai', fontsize=14, fontweight='bold')
            axs[0, 1].axis('off')
            
            # 3. Masque prédit
            im2 = axs[0, 2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=num_classes-1)
            axs[0, 2].set_title('Masque Prédit', fontsize=14, fontweight='bold')
            axs[0, 2].axis('off')
            
            # 4. Overlay vrai
            axs[1, 0].imshow(image, cmap='gray')
            axs[1, 0].imshow(true_mask, cmap=cmap, alpha=0.5, vmin=0, vmax=num_classes-1)
            axs[1, 0].set_title('Overlay Vrai', fontsize=14, fontweight='bold')
            axs[1, 0].axis('off')
            
            # 5. Overlay prédit
            axs[1, 1].imshow(image, cmap='gray')
            axs[1, 1].imshow(pred_mask, cmap=cmap, alpha=0.5, vmin=0, vmax=num_classes-1)
            axs[1, 1].set_title('Overlay Prédit', fontsize=14, fontweight='bold')
            axs[1, 1].axis('off')
            
            # 6. Différences (erreurs)
            diff = (true_mask != pred_mask).astype(int)
            axs[1, 2].imshow(diff, cmap='Reds', vmin=0, vmax=1)
            axs[1, 2].set_title('Erreurs (Rouge)', fontsize=14, fontweight='bold')
            axs[1, 2].axis('off')
            
            # Colorbar avec labels
            cbar = plt.colorbar(im2, ax=axs.ravel().tolist(), 
                               orientation='horizontal', 
                               fraction=0.05, pad=0.05)
            cbar.set_ticks(range(num_classes))
            cbar.set_ticklabels(class_names)
            
            plt.tight_layout()
            plt.show()
            
            # Statistiques
            print("-" * 50)
            for i, name in enumerate(class_names):
                true_count = np.sum(true_mask == i)
                pred_count = np.sum(pred_mask == i)
                correct = np.sum((true_mask == i) & (pred_mask == i))
                if true_count > 0:
                    accuracy = 100 * correct / true_count
                    print(f"{name:10s} | Vrai: {true_count:6d} | Prédit: {pred_count:6d} | Précision: {accuracy:5.1f}%")
            
            break



def train_fn(data_loader, model, optimizer, loss_fn):
    model.train()
    total_loss = 0.0
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for images, masks in tqdm(data_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE).long()
        
        # AJOUT : Squeeze pour enlever la dimension du canal si présente
        if masks.ndim == 4:  # Si (batch, 1, H, W)
            masks = masks.squeeze(1)  # Devient (batch, H, W)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)  # (batch, 7, H, W)
        
        # Calcul de la loss
        l = loss_fn(outputs, masks)  # masks doit être (batch, H, W)
        
        # Backward pass
        l.backward()
        optimizer.step()
        
        total_loss += l.item()
    
    return total_loss / len(data_loader)

import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

def training(config, working_directory, experiment_name, 
             dloader_train, dloader_test, model, optimizer, loss_fn):
    """
    Fonction d'entraînement complète pour segmentation multi-classes
    
    Args:
        config: Dictionnaire de configuration
        working_directory: Répertoire de travail
        experiment_name: Nom de l'expérience
        dloader_train: DataLoader d'entraînement
        dloader_test: DataLoader de test
        model: Modèle de segmentation
        optimizer: Optimiseur
        loss_fn: Fonction de perte
    
    Returns:
        training_loss, testing_loss: Listes des pertes
    """
    epochs = config['epochs']
    best_test_loss = np.inf
    training_loss = []
    testing_loss = []
    
    # Créer le répertoire pour l'expérience
    experiment_dir = os.path.join(working_directory, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(experiment_dir)
    
    # Scheduler (optionnel)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"\n{'='*60}")
    print(f"Début de l'entraînement: {experiment_name}")
    print(f"{'='*60}")
    print(f"Epochs: {epochs}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Nombre de classes: {config.get('num_classes', 8)}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Training
        train_loss = train_fn(dloader_train, model, optimizer, loss_fn)
        
        # Validation/Testing
        test_loss = eval_fn(dloader_test, model, loss_fn)  # Pas besoin d'optimizer ici
        
        # Log dans TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        # Sauvegarder les pertes
        training_loss.append(train_loss)
        testing_loss.append(test_loss)
        
        # Afficher les résultats
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        
        # Sauvegarder le meilleur modèle
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            
            # Sauvegarder le modèle complet
            model_path = os.path.join(experiment_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'config': config
            }, model_path)
            
            print(f"✅ MODÈLE SAUVEGARDÉ (Test Loss: {test_loss:.4f})")
            
            # Visualiser les prédictions du meilleur modèle
            print("Génération des visualisations...")
            plot_multiclass_segmentation(dloader_test, model, config.get('num_classes', 8),['ILM','NFL','IPL','INL','OPL','ISM','OS','BM'])
        
        # Update learning rate avec scheduler
        scheduler.step(test_loss)
        
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"Entraînement terminé!")
    print(f"Meilleure Test Loss: {best_test_loss:.4f}")
    print(f"Modèle sauvegardé dans: {experiment_dir}")
    print(f"{'='*60}\n")
    
    return training_loss, testing_loss