import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader

class EncoderMrs(nn.Module):
    def __init__(self, features):
        super(EncoderMrs, self).__init__()
        self.features = features
        self.enc1 = nn.Linear(784, 512)
        self.enc2 = nn.Linear(512, 256)
        self.enc3 = nn.Linear(256, self.features)

        self.mu = nn.Linear(self.features, self.features)
        self.logvar = nn.Linear(self.features, self.features)
    
    def reparameterize(self, mu, log_var):
        device = mu.device
        return torch.randn(self.features, device=device) * torch.exp(0.5*log_var) + mu

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = self.enc3(x)

        mu = self.mu(x)
        log_var = self.logvar(x)

        encoding = self.reparameterize(mu, log_var)

        return mu, log_var, encoding


class DecoderMrs(nn.Module):
    def __init__(self, features):
        super(DecoderMrs, self).__init__()
        self.features = features
        self.dec1 = nn.Linear(self.features, 256)
        self.dec2 = nn.Linear(256, 512)
        self.dec3 = nn.Linear(512, 784)

    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.sigmoid(self.dec3(x))

        return x


class VaeMrs(nn.Module):
    def __init__(self, features):
        super(VaeMrs, self).__init__()
        self.encoder = EncoderMrs(features)
        self.decoder = DecoderMrs(features)

    def forward(self, x):
        mu, log_var, encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)

        return reconstruction, mu, log_var


def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def fit(model, dataloader, optimizer, criterion, device=torch.device("cpu")):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(60000/dataloader.batch_size)):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss


def main():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Train and Test data
    train_data = datasets.FashionMNIST(
        root='.data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    # Learning parameters
    epochs = 200
    batch_size = 128
    lr = 0.001

    # Training data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False
    )

    vae_mrs = VaeMrs(features=100).to(device)
    
    optimizer = optim.RMSprop(vae_mrs.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum')

    train_loss = []
    for epoch in range(epochs):
        print("Epoch "+str(epoch+1)+" of "+str(epochs)+"\n")
        train_epoch_loss = fit(vae_mrs, train_loader, optimizer, criterion, device=device)
        train_loss.append(train_epoch_loss)
        print("Train Loss: "+str(train_epoch_loss)+"\n")

    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    torch.save(vae_mrs, './saved_models/vae_mrs')


if __name__ == "__main__":
    from tqdm import tqdm
    main()