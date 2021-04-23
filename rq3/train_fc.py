import numpy as np
import os
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pathlib import Path
from torchvision.datasets.folder import default_loader
from torchvision.utils import make_grid, save_image
from typing import List

NAME = "fc"
Path(f"samples_{NAME}").mkdir(exist_ok=True, parents=True)


class Dronet(data.Dataset):
    def __init__(
        self, root, transform=None, target_transform=None, loader=default_loader
    ):
        self.root = root
        self.samples = []

        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader

        for subdir in sorted(os.listdir(root)):
            experiment_dir = os.path.join(root, subdir)
            if not os.path.isdir(experiment_dir):
                continue
            has_steering = os.path.exists(
                os.path.join(experiment_dir, "sync_steering.txt")
            )
            has_labels = os.path.exists(os.path.join(experiment_dir, "labels.txt"))
            assert has_steering or has_labels, (
                "Neither steerings nor labels found in %s" % experiment_dir
            )
            assert not (has_steering and has_labels), (
                "Both steerings and labels found in %s" % experiment_dir
            )
            if has_steering:
                steering_ground_truth = np.loadtxt(
                    os.path.join(experiment_dir, "sync_steering.txt"),
                    usecols=0,
                    delimiter=",",
                    skiprows=1,
                )
                label_ground_truth = np.ones_like(steering_ground_truth) * float("nan")
            if has_labels:
                label_ground_truth = np.loadtxt(
                    os.path.join(experiment_dir, "labels.txt"), usecols=0
                )
                steering_ground_truth = np.ones_like(label_ground_truth) * float("nan")
            img_dir = os.path.join(experiment_dir, "images")
            files = (
                name
                for name in os.listdir(img_dir)
                if os.path.isfile(os.path.join(img_dir, name))
                and os.path.splitext(name)[1] in [".png", ".jpg"]
            )
            for frame_number, fname in enumerate(
                sorted(files, key=lambda fname: int(re.search(r"\d+", fname).group()))
            ):
                img_path = os.path.join(img_dir, fname)
                target = np.array(
                    [
                        steering_ground_truth[frame_number],
                        label_ground_truth[frame_number],
                    ],
                    dtype=np.float32,
                )
                self.samples.append((img_path, target))

    def process_sample(self, sample):
        return self.loader(sample)

    def __getitem__(self, index):
        sample_, target = self.samples[index]
        sample = self.process_sample(sample_)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transforms(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(
            tmp, repr(self.transform).replace("\n", "\n" + " " * len(tmp))
        )
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(
            tmp, repr(self.target_transform).replace("\n", "\n" + " " * len(tmp))
        )
        return fmt_str


class Model(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: List = None, **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Linear(200 * 200, 2048),
            nn.ELU(),
            nn.Linear(2048, 1024),
            nn.ELU(),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            # nn.Linear(512, 512),
            # nn.ELU(),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)

        # Build Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ELU(),
            # nn.Linear(512, 512),
            # nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 1024),
            nn.ELU(),
            nn.Linear(1024, 2048),
            nn.ELU(),
            nn.Linear(2048, 200 * 200),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = x.flatten(1)
        result = self.encoder(x)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        y = self.decoder(z)
        y = y.view(-1, 1, 200, 200)
        return y

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    def sample(self, num_samples: int) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(self.fc_var.weight.device)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)[0]


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z):
        return self.model.decode(z)


def get_data_transform(
    transform_config,
    default_height=224,
    default_width=224,
    default_crop_height=None,
    default_crop_width=None,
):
    data_transforms = []
    is_grayscale = transform_config.get("grayscale", False)
    is_bgr = transform_config.get("bgr", False)
    assert not is_grayscale or not is_bgr, "Cannot be both grayscale and bgr"
    if is_grayscale:
        data_transforms.append(transforms.Grayscale(num_output_channels=1))
    data_transforms.append(transforms.ToTensor())
    if is_bgr:
        data_transforms.append(transforms.Lambda(lambda t: t[[2, 1, 0]]))
    if not transform_config.get("presized", True):
        data_transforms.append(transforms.ToPILImage())
        resize_height = transform_config.get("height", default_height)
        resize_width = transform_config.get("width", default_width)
        data_transforms.append(transforms.Resize((resize_height, resize_width)))

        if default_crop_height is not None or default_crop_width is not None:
            default_crop_height = default_crop_height or resize_height
            default_crop_width = default_crop_width or resize_width
            crop_height = transform_config.get("crop_height", default_crop_height)
            crop_width = transform_config.get("crop_width", default_crop_width)
            data_transforms.append(transforms.CenterCrop((crop_height, crop_width)))
        data_transforms.append(transforms.ToTensor())
    data_transforms.append(
        transforms.Lambda(lambda t: t * transform_config.get("max_value", 1.0))
    )
    transform_normalize_mean = transform_config.get("mean", None)
    if transform_normalize_mean is not None:
        data_transforms.append(
            transforms.Normalize(
                mean=transform_normalize_mean,
                std=transform_config.get("std", [1.0] * len(transform_normalize_mean)),
            )
        )
    transform = transforms.Compose(data_transforms)
    return transform


def load_data(path, batch_size=225, shuffle=False):
    transform = get_data_transform(
        {"grayscale": True},
        default_height=240,
        default_width=320,
        default_crop_height=200,
        default_crop_width=200,
    )

    dataset = Dronet(path, transform=transform)

    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        pin_memory=True,
    )


def loss_fn(recons, x, mu, log_var, kld_weight):
    recons_loss = F.mse_loss(recons, x)
    # recons_loss = F.binary_cross_entropy(recons, x)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )
    loss = recons_loss + kld_weight * kld_loss
    return loss, recons_loss, kld_loss


def train(model, train_loader, num_epochs=300, device=torch.device("cpu")):
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95, verbose=True)
    # kld_weight = 0.01 * train_loader.batch_size / len(train_loader.dataset)
    kld_weight = train_loader.batch_size / len(train_loader.dataset)
    all_losses = []
    start_t = time.time()
    z = torch.randn(100, model.latent_dim).to(device)
    grid = make_grid(model.decode(z).detach().cpu(), 10)
    save_image(grid, f"samples_{NAME}/z0.png")
    for epoch in range(1, num_epochs + 1):
        epoch_start_t = time.time()
        losses = []
        for i, (x, y) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            x = x.to(device)
            recons, x, mu, log_var = model(x)
            loss, rloss, kloss = loss_fn(recons, x, mu, log_var, kld_weight)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            if (i % 10) == 0:
                iter_end_t = time.time()
                print(
                    f"{epoch} {i} [{iter_end_t - epoch_start_t:.4f}]: {losses[-1]:.4f} {np.mean(losses[-10:]):.4f} {rloss.item():.4f} {kloss.item():.4f}"
                )
                # grid = make_grid(model.sample(100).detach().cpu(), 10)
                # save_image(grid, f"samples_{NAME}/{epoch}_{i}.png")
        epoch_end_t = time.time()
        print(f"epoch time: {epoch_end_t - epoch_start_t} seconds")
        print(f"total time: {epoch_end_t - start_t} seconds")
        all_losses.append(losses)
        lr_scheduler.step()
        grid = make_grid(model.sample(100).detach().cpu(), 10)
        save_image(grid, f"samples_{NAME}/{epoch}.png")
        grid = make_grid(model.decode(z).detach().cpu(), 10)
        save_image(grid, f"samples_{NAME}/z{epoch}.png")
    end_t = time.time()
    print(f"total time: {end_t - start_t} seconds")
    return model


def main():
    train_loader = load_data(
        "Data/dronet.200/training/", batch_size=64, shuffle=True
    )
    print(train_loader.dataset)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    model = Model(512).to(device)
    model = train(model, train_loader, device=device)
    model = model.to(torch.device("cpu"))
    torch.onnx.export(
        model,
        torch.randn(1, 1, 200, 200),
        f"dronet_{NAME}_vae.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": [0], "output": [0]},
    )
    torch.onnx.export(
        Decoder(model),
        torch.randn(1, model.latent_dim),
        f"dronet_{NAME}_vae_decoder.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": [0], "output": [0]},
    )


if __name__ == "__main__":
    main()
