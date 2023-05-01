import pytorch_lightning as pl
import torch.nn as nn

class ConvAutoencoder(pl.LightningModule):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #encoded  torch.Size([7, 16, 16, 16])
            nn.Flatten(),
            nn.Linear(16*16*16, self.latent_dim) # batch, latent_dim
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 16*16*16),
            nn.Unflatten(dim=1, unflattened_size=(16, 16, 16)),
            nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=13, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Linear(64, 64)
        )
        # Mean square error loss
        self.loss_fn = nn.MSELoss()
    
    def forward(self, x):
        encoded = self.encoder(x)
        print(" encoded ", encoded.shape)
        decoded = self.decoder(encoded)
        return decoded
    
    def training_step(self, batch, batch_idx):
        x = batch
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        loss = self.loss_fn(decoded, x)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer