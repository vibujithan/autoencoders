import numpy as np
import torch
from decoder import Decoder
from encoder import Encoder
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


class AE:

    def __init__(self, encoded_dim):
        self.encoded_dim = encoded_dim

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {self.device}')

        self.encoder = Encoder(encoded_dim=encoded_dim).to(self.device)
        self.decoder = Decoder(encoded_dim=encoded_dim).to(self.device)

    def train(self, train_loader, lr=0.0005, weight_decay=1e-06):
        self.encoder.train()
        self.decoder.train()

        params = [{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}]
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        loss_func = torch.nn.MSELoss()

        train_loss = []
        for batch in train_loader:
            image_batch = batch[0]
            image_batch = image_batch.to(self.device)
            encoded_data = self.encoder(image_batch)
            decoded_data = self.decoder(encoded_data)

            loss = loss_func(decoded_data, image_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    def test(self, dataloader):
        self.encoder.eval()
        self.decoder.eval()

        loss_func = torch.nn.MSELoss()

        with torch.no_grad():
            conc_out = []
            conc_label = []
            for batch in dataloader:
                image_batch = batch[0]
                image_batch = image_batch.to(self.device)
                encoded_data = self.encoder(image_batch)
                decoded_data = self.decoder(encoded_data)

                conc_out.append(decoded_data.cpu())
                conc_label.append(image_batch.cpu())

            conc_out = torch.cat(conc_out)
            conc_label = torch.cat(conc_label)
            val_loss = loss_func(conc_out, conc_label)
        return val_loss.data.numpy()

    def plot_ae_outputs(self, dataloader, n=10):
        self.encoder.eval()
        self.decoder.eval()

        fig = plt.figure(figsize=(20, 4))
        targets = next(iter(dataloader))[0]
        with torch.no_grad():
            for i in range(n):
                ax = fig.add_subplot(2, n, i + 1)
                img = targets[i].unsqueeze(0).to(self.device)

                rec_img = self.decoder(self.encoder(img))

                ax.imshow(img.cpu().squeeze().numpy(), vmin=0, vmax=1, cmap='gist_gray')
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                if i == n // 2:
                    ax.set_title('Original images')

                ax = fig.add_subplot(2, n, i + 1 + n)
                ax.imshow(rec_img.cpu().squeeze().numpy(), vmin=0, vmax=1, cmap='gist_gray')
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                if i == n // 2:
                    ax.set_title('Reconstructed images')

        plt.close()
        return fig
