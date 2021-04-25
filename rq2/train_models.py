import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.types import Device

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import time
import os


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### VAE 1 layer ###

class Encoder1(nn.Module):
    def __init__(self, features, neurons):
        super(Encoder1, self).__init__()
        self.features = features
        self.neurons = neurons
        self.enc1 = nn.Linear(784, self.neurons)
        self.enc2 = nn.Linear(self.neurons, self.features)

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


class Decoder1(nn.Module):
    def __init__(self, features, neurons):
        super(Decoder1, self).__init__()
        self.features = features
        self.neurons = neurons
        self.dec1 = nn.Linear(self.features, self.neurons)
        self.dec2 = nn.Linear(self.neurons, 784)

    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = F.sigmoid(self.dec2(x))

        return x


class Vae1(nn.Module):
    def __init__(self, features, neurons):
        super(Vae1, self).__init__()
        self.encoder = Encoder1(features, neurons)
        self.decoder = Decoder1(features, neurons)

    def forward(self, x):
        mu, log_var, encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)

        return reconstruction, mu, log_var

### VAE 2 layers ###

class Encoder2(nn.Module):
    def __init__(self, features, neurons):
        super(Encoder2, self).__init__()
        self.features = features
        self.neurons = neurons
        self.enc1 = nn.Linear(784, self.neurons)
        self.enc2 = nn.Linear(self.neurons, self.neurons)
        self.enc3 = nn.Linear(self.neurons, self.features)

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


class Decoder2(nn.Module):
    def __init__(self, features, neurons):
        super(Decoder2, self).__init__()
        self.features = features
        self.neurons = neurons
        self.dec1 = nn.Linear(self.features, self.neurons)
        self.dec2 = nn.Linear(self.neurons, self.neurons)
        self.dec3 = nn.Linear(self.neurons, 784)

    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.sigmoid(self.dec3(x))

        return x


class Vae2(nn.Module):
    def __init__(self, features, neurons):
        super(Vae2, self).__init__()
        self.encoder = Encoder2(features, neurons)
        self.decoder = Decoder2(features, neurons)

    def forward(self, x):
        mu, log_var, encoding = self.encoder(x)
        reconstruction = self.decoder(encoding)

        return reconstruction, mu, log_var

### VAE 4 layers ###

class Encoder4(nn.Module):
    def __init__(self, features, neurons):
        super(Encoder4, self).__init__()
        self.features = features
        self.neurons = neurons
        self.enc1 = nn.Linear(784, self.neurons)
        self.enc2 = nn.Linear(self.neurons, self.neurons)
        self.enc3 = nn.Linear(self.neurons, self.neurons)
        self.enc4 = nn.Linear(self.neurons, self.neurons)
        self.enc5 = nn.Linear(self.neurons, self.features)

        self.mu = nn.Linear(self.features, self.features)
        self.logvar = nn.Linear(self.features, self.features)
    
    def reparameterize(self, mu, log_var):
        device = mu.device
        return torch.randn(self.features, device=device) * torch.exp(0.5*log_var) + mu

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = self.enc5(x)

        mu = self.mu(x)
        log_var = self.logvar(x)

        encoding = self.reparameterize(mu, log_var)

        return mu, log_var, encoding


class Decoder4(nn.Module):
    def __init__(self, features, neurons):
        super(Decoder4, self).__init__()
        self.features = features
        self.neurons = neurons
        self.dec1 = nn.Linear(self.features, self.neurons)
        self.dec2 = nn.Linear(self.neurons, self.neurons)
        self.dec3 = nn.Linear(self.neurons, self.neurons)
        self.dec4 = nn.Linear(self.neurons, self.neurons)
        self.dec5 = nn.Linear(self.neurons, 784)

    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.sigmoid(self.dec5(x))

        return x


class Vae4(nn.Module):
    def __init__(self, features, neurons):
        super(Vae4, self).__init__()
        self.encoder = Encoder4(features, neurons)
        self.decoder = Decoder4(features, neurons)

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
    # latent_space = int(sys.argv[1])
    # number_layer = int(sys.argv[2])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    batch_size = 128
    lr = 0.001

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

    ### Network
    network = Network().to(device)
    optimizer = optim.RMSprop(network.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epochs = 1

    train(network, optimizer, criterion, train_loader, test_loader, epochs, device=device)
    
    if not os.path.exists('./saved_models'):
        os.makedirs('./saved_models')

    torch.save(network.state_dict(), './saved_models/network')

    network = network.to(torch.device("cpu"))

    ### Save network as onnx
    batch_size = 1
    x = torch.randn(batch_size,784,requires_grad=True)
    torch.onnx.export(network, x, './saved_models/network.onnx', export_params=True, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})


    ### VAEs
    epochs = 1

    for latent_space in [1,2,4,8,16,32]:

        if not os.path.exists('./saved_models/latent_space{}'.format(latent_space)):
            os.makedirs('./saved_models/latent_space{}'.format(latent_space))
        for number_layer in [1,2,4]:

            if not os.path.exists('./saved_models/latent_space{}/number_layer{}'.format(latent_space, number_layer)):
                os.makedirs('./saved_models/latent_space{}/number_layer{}'.format(latent_space, number_layer))
            for number_neuron in [16,32,64,128,256]:

                if not os.path.exists('./saved_models/latent_space{}/number_layer{}/vae'.format(latent_space, number_layer)):
                    os.makedirs('./saved_models/latent_space{}/number_layer{}/vae'.format(latent_space, number_layer))
                if not os.path.exists('./saved_models/latent_space{}/number_layer{}/decoder'.format(latent_space, number_layer)):
                    os.makedirs('./saved_models/latent_space{}/number_layer{}/decoder'.format(latent_space, number_layer))
                if not os.path.exists('./saved_models/latent_space{}/number_layer{}/onnx'.format(latent_space, number_layer)):
                    os.makedirs('./saved_models/latent_space{}/number_layer{}/onnx'.format(latent_space, number_layer))

                if number_layer == 1:
                    vae = Vae1(features=latent_space, neurons=number_neuron).to(device)
                elif number_layer == 2:
                    vae = Vae2(features=latent_space, neurons=number_neuron).to(device)
                elif number_layer == 4:
                    vae = Vae4(features=latent_space, neurons=number_neuron).to(device)

                optimizer = optim.RMSprop(vae.parameters(), lr=lr)
                criterion = nn.MSELoss(reduction='sum')
                train_loss = []

                start_t = time.time()
                for epoch in range(epochs):
                    print("Epoch "+str(epoch+1)+" of "+str(epochs)+"\n")
                    train_epoch_loss = fit(vae, train_loader, optimizer, criterion, device=device)
                    train_loss.append(train_epoch_loss)
                end_t = time.time()
                duration = end_t - start_t

                print('Duration: '+str(duration)+'\n')
                print("Train Loss: "+str(train_epoch_loss)+"\n")
                
                vae = vae.to(torch.device('cpu'))

                torch.save(vae.state_dict(), './saved_models/latent_space'+str(latent_space)+'/number_layer'+str(number_layer)+'/vae/vae'+str(number_neuron))

                ### Create modifiedDnn ###
                modifiedDnn = ModifiedDnn(vae.decoder, network)

                ### Save onnx model ###
                batch_size = 1
                x = torch.randn(batch_size,latent_space,requires_grad=True)
                torch.onnx.export(modifiedDnn, x, './saved_models/latent_space'+str(latent_space)+'/number_layer'+str(number_layer)+'/onnx/modifiedDnn'+str(number_neuron)+'.onnx', export_params=True, do_constant_folding=True, input_names=['input'], output_names=['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})



if __name__ == "__main__":
    from tqdm import tqdm
    main()