import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def create_model(config):
    ENCODER = config['encoder']
    ENCODER_WEIGHTS = config['weights']
    LR = config['lr']
    NUM_CLASSES = config.get('nb_classes', 8)  
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loss = nn.CrossEntropyLoss() 
    
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=NUM_CLASSES, 
        activation=None, 
        in_channels=1
    )
    
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    return model, loss, optimizer