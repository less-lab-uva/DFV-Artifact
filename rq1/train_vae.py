import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
# from tqdm import tqdm


class Encoder(nn.Module):
    def __init__(self, features):
        super(Encoder, self).__init__()
        self.features = features
        self.enc1 = nn.Linear(784, 24)
        self.enc2 = nn.Linear(24, self.features)

        self.mu = nn.Linear(self.features, self.features)
        self.logvar = nn.Linear(self.features, self.features)
    
    def reparameterize(self, mu, log_var):
        device = mu.device
        return torch.randn(self.features, device=device) * torch.exp(0.5*log_var) + mu

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = self.enc2(x)

        mu = self.mu(x)
        log_var = self.logvar(x)

        encoding = self.reparameterize(mu, log_var)

        return mu, log_var, encoding


class Decoder(nn.Module):
    def __init__(self, features):
        super(Decoder, self).__init__()
        self.features = features
        self.dec1 = nn.Linear(self.features, 24)
        self.dec2 = nn.Linear(24, 784)

    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = self.__relu_approx_sigmoid__(self.dec2(x))

        return x

    def __relu_approx_sigmoid__(self, x):
        device = x.device
        I = torch.eye(784, device=device)
        I_ = -I
        ones = torch.ones(784, device=device)
        return F.relu(F.relu(x @ (0.25*I) + (ones-0.5)) @ I_ + ones) @ I_ + ones


class Vae(nn.Module):
    def __init__(self, features):
        super(Vae, self).__init__()
        self.encoder = Encoder(features)
        self.decoder = Decoder(features)

    def forward(self, x):
        mu, log_var, encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)

        return reconstruction, mu, log_var


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(784,24)
        self.fc2 = nn.Linear(24,24)
        self.fc3 = nn.Linear(24,10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ModifiedDnn(nn.Module):
    def __init__(self, decoder, dnn):
        super(ModifiedDnn, self).__init__()
        self.decoder = decoder
        self.dnn = dnn

    def forward(self, z):
        x = self.decoder(z)
        x = self.dnn(x)

        return x


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


def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device=torch.device("cpu")):
    
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()

        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
        training_loss /= len(train_loader)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output,targets)
            valid_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader)

        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss, valid_loss, num_correct / num_examples))


def main():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Train and Test data
    train_data = datasets.FashionMNIST(
        root='.data/FashionMNIST',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    test_data = datasets.FashionMNIST(
        root='.data/FashionMNIST',
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    # Learning parameters
    epochs = 200
    batch_size = 128
    lr = 0.001

    # Training and Test data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    ) 

    ### VAE
    latent_space = 2
    vae = Vae(features=latent_space).to(device)
    
    optimizer = optim.RMSprop(vae.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum')

    train_loss = []
    for epoch in range(epochs):
        print("Epoch "+str(epoch+1)+" of "+str(epochs)+"\n")
        train_epoch_loss = fit(vae, train_loader, optimizer, criterion, device=device)
        train_loss.append(train_epoch_loss)
        print("Train Loss: "+str(train_epoch_loss)+"\n")

    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    torch.save(vae.state_dict(), './saved_models/vae')

    ### Network
    network = Network().to(device)
    optimizer = optim.RMSprop(network.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epochs = 20

    train(network, optimizer, criterion, train_loader, test_loader, epochs, device=device)
    torch.save(network.state_dict(), './saved_models/network')

    vae = vae.to(torch.device("cpu"))
    network = network.to(torch.device("cpu"))

    ### Save network as onnx
    if not os.path.exists('./saved_models/onnx'):
        os.makedirs('./saved_models/onnx')

    batch_size = 1
    x = torch.randn(batch_size,784,requires_grad=True)
    torch.onnx.export(network, x, './saved_models/onnx/network.onnx', export_params=True, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})

    ### Create modifiedDnn ###
    modifiedDnn = ModifiedDnn(vae.decoder, network)

    ### Save modifiedDnn as onnx
    batch_size = 1
    x = torch.randn(batch_size,latent_space,requires_grad=True)
    torch.onnx.export(modifiedDnn, x, './saved_models/onnx/modifiedDnn.onnx', export_params=True, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})


if __name__ == "__main__":
    from tqdm import tqdm
    main()