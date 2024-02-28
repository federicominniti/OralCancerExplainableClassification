import torch
import torchvision
from pytorch_lightning import LightningModule
import tensorboard as tb

class ImageVAE(LightningModule):
    def __init__(self, latent_dim=64, lr = 0.0004, max_epochs = 100):
        super().__init__()
        self.save_hyperparameters()

        # Encoder
        self.encoder_block1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Dropout(0.15)
        )
        self.maxpooling_3_2 = torch.nn.MaxPool2d(3, stride = 2, return_indices=True)

        self.encoder_block2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Dropout(0.15)
        )

        self.encoder_block3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Dropout(0.15)
        )

        self.maxpooling_2_2 = torch.nn.MaxPool2d(2, stride = 2, return_indices=True)

        self.encoder_block4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.Dropout(0.15),

            torch.nn.Flatten(),
        )

        self.layer1 = torch.nn.Linear(512 * 2 * 2, latent_dim)
        self.layer2 = torch.nn.Linear(512 * 2 * 2, latent_dim)

        # Decoder
        self.decoder_block1 = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512 * 2 * 2),
            torch.nn.ReLU(),
            torch.nn.Unflatten(1, (512,2,2)),

            torch.nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.Dropout(0.15),
        )
        self.maxunpool_2_2 = torch.nn.MaxUnpool2d(2, stride=2, padding=0)

        self.decoder_block2 = torch.nn.Sequential(        
            torch.nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Dropout(0.15)
        )
        
        self.maxunpool_3_2 = torch.nn.MaxUnpool2d(3, stride=2, padding=0)

        self.decoder_block3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 8, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Dropout(0.15),
        )

        self.decoder_block4 = torch.nn.Sequential(     
            torch.nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()

        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        encoded1 = self.encoder_block1(x)
        encodedM1, index1 = self.maxpooling_3_2(encoded1)
        encoded2 = self.encoder_block2(encodedM1)
        encodedM2, index2 = self.maxpooling_3_2(encoded2)
        encoded3 = self.encoder_block3(encodedM2)
        encodedM3, index3 = self.maxpooling_2_2(encoded3)
        x = self.encoder_block4(encodedM3)

        mu =  self.layer1(x)
        sigma = torch.exp(self.layer2(x))
        x = mu + sigma*self.N.sample(mu.shape)

        decoded = self.decoder_block1(x)
        decoded = self.maxunpool_2_2(decoded, index3, output_size=encoded3.size())
        decoded = self.decoder_block2(decoded)
        decoded = self.maxunpool_3_2(decoded, index2, output_size=encoded2.size())
        decoded = self.decoder_block3(decoded)
        decoded = self.maxunpool_3_2(decoded, index1, output_size=encoded1.size())
        x = self.decoder_block4(decoded)

        return x, mu, sigma

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train") 

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val") 
        
    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def _common_step(self, batch, batch_idx, stage):
        x, _ = batch
        reconstruction, mu, sigma = self.forward(x)
        
        reconstruction_loss = self.loss(reconstruction, x)
        
        kl_loss = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        
        loss = reconstruction_loss + kl_loss

        self.log(f"{stage}_recon_loss", reconstruction_loss, on_step=True, on_epoch=True)
        self.log(f"{stage}_kl_loss", kl_loss, on_step=True, on_epoch=True)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

