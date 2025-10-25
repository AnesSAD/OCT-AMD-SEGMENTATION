import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
  
class UNET(nn.Module):
    
    """
    UNET(nn.Module)

    This is a PyTorch implementation of the U-Net architecture, widely used for image segmentation tasks.

    🧠 Overview:
    - Composed of a contracting path (encoder) and an expansive path (decoder)
    - Skip connections are used to concatenate high-resolution features from the encoder to the decoder
    - Particularly effective for biomedical image segmentation and tasks requiring precise localization

    🔧 Parameters:
    - in_channels (int): Number of input channels (e.g., 1 for grayscale images, 3 for RGB)
    - out_channels (int): Number of output channels (e.g., 1 for binary segmentation)
    - features (list[int]): List of feature map sizes for each encoder level (default: [64, 128, 256, 512])

    🧱 Components:
    - `DoubleConv`: A building block composed of two convolutional layers (3x3), each followed by BatchNorm and ReLU activation
    - `MaxPool2d`: Reduces the spatial resolution by half at each encoder step
    - `ConvTranspose2d`: Performs upsampling in the decoder path
    - `final_conv`: A 1x1 convolution that reduces the final feature map to the desired number of output channels

    ⚙️ Forward Pass:
    - The input is passed through a sequence of `DoubleConv` blocks and downsampled using max pooling
    - Features are stored at each encoder level for skip connections
    - A bottleneck block processes the deepest features
    - Decoder upsamples the features using `ConvTranspose2d` and concatenates them with corresponding skip connections
    - Final output is generated with a 1x1 convolution, preserving the original spatial dimensions

    📈 Output:
    - Returns a segmentation map of shape `(batch_size, out_channels, height, width)`
    """

    def __init__(self, in_channels=1, out_channels=1, features=[64,128,256,512]):
      super(UNET, self).__init__()

      self.downs = nn.ModuleList()
      self.ups = nn.ModuleList()
      self.pool = nn.MaxPool2d(kernel_size=2,stride=2)


      # ENCODER
      for feature in features:
        self.downs.append(DoubleConv(in_channels,feature))
        in_channels = feature

      # DECODER
      for feature in features[::-1] :
        self.ups.append(nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2))
        self.ups.append(DoubleConv(feature*2,feature))

      self.bottleneck = DoubleConv(features[-1],features[-1]*2)
      self.final_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)

    def forward(self,x):

      skip_connections = []

      for down in self.downs:

        x = down(x)
        skip_connections.append(x)
        x= self.pool(x)
      x = self.bottleneck(x)

      skip_connections = skip_connections[::-1]

      for idx in range(0,len(self.ups),2):
        x = self.ups[idx](x)
        skip_connection = skip_connections[idx//2]

        if x.shape != skip_connection.shape:
          x = TF.resize(x,skip_connection.shape[2:])

        concat_skip = torch.cat((skip_connection,x),dim=1)

      x = self.ups[idx+1](concat_skip)

      return self.final_conv(x)

class DoubleConv(nn.Module):
    """
    DoubleConv(nn.Module)

    This class defines a basic building block used in U-Net architectures: two consecutive convolutional layers,
    each followed by batch normalization and a ReLU activation function.

    📌 Purpose:
    - Extract and refine spatial features in an image
    - Used in both the encoder and decoder parts of U-Net

    🔧 Parameters:
    - in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB)
    - out_channels (int): Number of output channels produced by the block

    🧠 Architecture:
    - Two 2D convolution layers (3x3 kernel, stride=1, padding=1, no bias)
    - Each followed by BatchNorm2d and ReLU activation
    - Preserves spatial dimensions (height and width)

    🌀 Forward Pass:
    - Takes an input tensor of shape (B, in_channels, H, W)
    - Returns a tensor of shape (B, out_channels, H, W)

    ✅ Example:
      block = DoubleConv(1, 64)
      out = block(input_tensor)
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, 3, 1, 1, bias=False), # 3 : convolutional core 3*3, 1 : no convolution, 1: border pixels
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)

def create_model(config):
    """
    create_model(config: dict) -> Tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer]

    Initializes a U-Net++ segmentation model using `segmentation_models_pytorch`, along with the
    binary cross-entropy loss and Adam optimizer.

    📌 Purpose:
    - Create a ready-to-train segmentation model with preconfigured architecture, loss, and optimizer

    🔧 Parameters:
    - config (dict): Dictionary containing configuration options. Required keys:
        - 'encoder': name of the encoder (e.g., 'resnet34', 'efficientnet-b0', etc.)
        - 'weights': pretrained weights for the encoder (e.g., 'imagenet')
        - 'lr': learning rate (float)

    🚀 Returns:
    - model (torch.nn.Module): U-Net++ model instance
    - loss (torch.nn.Module): Binary cross-entropy loss with logits (BCEWithLogitsLoss)
    - optimizer (torch.optim.Optimizer): Adam optimizer for the model's parameters

    🖥️ Device:
    - Automatically uses GPU if available; falls back to CPU otherwise

    🧠 Notes:
    - Input images are assumed to have 1 channel
    - Output is a single-channel mask (binary segmentation)
    - Activation is set to `None` because the final activation (sigmoid) is usually applied during inference

    ✅ Example:
      model, loss_fn, optimizer = create_model(config)
    """
    ENCODER = config['encoder']
    ENCODER_WEIGHTS = config['weights']
    LR = config['lr']
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = torch.nn.BCEWithLogitsLoss()
    model = smp.UnetPlusPlus(encoder_name=ENCODER, #Unet++ used here but can also use Unet 
                           encoder_weights = ENCODER_WEIGHTS,
                           classes = 1,
                           activiation = None,
                           in_channels = 1)

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)

    return model, loss, optimizer
     

def test():
    """test()

    Performs a simple forward pass through the U-Net model to verify that it runs without errors.

    📌 Purpose:
    - Check whether the UNET model is correctly defined and can process a sample input
    - Useful for debugging model architecture or tensor shape mismatches

    🚀 What it does:
    - Creates a dummy input tensor of shape (1, 1, 161, 161), simulating a single grayscale image
    - Initializes a U-Net model with 1 input and 1 output channel
    - Passes the input through the model to ensure it produces an output without errors

    ⚠️ Note:
    - This is not a unit test with assertions. It's a basic smoke test to ensure forward propagation works.

    ✅ Example usage:
      test()
    """
    x = torch.randn((1,1,161,161))
    model = UNET(in_channels=1,out_channels=1)
    preds = model(x)
  
  
 