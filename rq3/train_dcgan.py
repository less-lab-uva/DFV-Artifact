import numpy as np
import os
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.dataloader import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from pathlib import Path
from torchvision.datasets.folder import default_loader
from torchvision.utils import make_grid, save_image
from typing import List

NAME = "dcgan"
Path(f"samples_{NAME}").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}/iter").mkdir(exist_ok=True, parents=True)
Path(f"samples_{NAME}/epoch").mkdir(exist_ok=True, parents=True)


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


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        img_size = img_shape[1]
        assert img_shape[1] == img_shape[2]

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            # nn.Tanh(),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        img_size = img_shape[1]
        assert img_shape[1] == img_shape[2]

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_shape[0], 16, bn=True),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = int(np.ceil(img_size / 2 ** 4))
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


def train(
    generator: Generator,
    discriminator: Discriminator,
    data_loader: DataLoader,
    num_epochs=300,
    device=torch.device("cpu"),
    lr=0.0002,
    b1=0.5,
    b2=0.999,
    sample_interval=100,
):
    loss_fn = torch.nn.BCELoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    start_t = time.time()

    latent_z = torch.randn(100, generator.latent_dim).to(device)
    grid = make_grid(generator(latent_z).detach().cpu(), 10)
    save_image(grid, f"samples_{NAME}/epoch/0.png")
    for epoch in range(1, num_epochs + 1):
        epoch_start_t = time.time()
        for i, (imgs, _) in enumerate(data_loader):
            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1, requires_grad=False, device=device)
            fake = torch.zeros(imgs.size(0), 1, requires_grad=False, device=device)

            # Configure input
            real_imgs = imgs.to(device)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.size(0), generator.latent_dim, device=device)

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = loss_fn(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = loss_fn(discriminator(real_imgs), valid)
            fake_loss = loss_fn(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch,
                    num_epochs,
                    i + 1,
                    len(data_loader),
                    d_loss.item(),
                    g_loss.item(),
                )
            )

            batches_done = (epoch - 1) * len(data_loader) + i
            if batches_done % sample_interval == 0:
                save_image(
                    gen_imgs.data[:25],
                    f"samples_{NAME}/iter/%d.png" % batches_done,
                    nrow=5,
                    # normalize=True,
                )
        epoch_end_t = time.time()
        print(f"epoch time: {epoch_end_t - epoch_start_t} seconds")
        print(f"total time: {epoch_end_t - start_t} seconds")
        grid = make_grid(generator(latent_z).detach().cpu(), 10)
        save_image(grid, f"samples_{NAME}/epoch/{epoch}.png")

    end_t = time.time()
    print(f"total time: {end_t - start_t} seconds")
    return generator, discriminator


def main():
    train_loader = load_data(
        "Data/dronet.200/training/", batch_size=64, shuffle=True
    )
    print(train_loader.dataset)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    generator = Generator(512, (1, 200, 200)).to(device)
    discriminator = Discriminator(generator.img_shape).to(device)
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    generator, discriminator = train(
        generator, discriminator, train_loader, device=device
    )

    # model = Model(512).to(device)
    # model = train(model, train_loader, device=device)

    generator = generator.to(torch.device("cpu"))
    discriminator = discriminator.to(torch.device("cpu"))
    torch.onnx.export(
        generator,
        torch.randn(1, generator.latent_dim),
        f"dronet_{NAME}_generator.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": [0], "output": [0]},
        opset_version=11,
    )
    torch.onnx.export(
        discriminator,
        torch.randn(1, *generator.img_shape),
        f"dronet_{NAME}_discriminator.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": [0], "output": [0]},
        opset_version=11,
    )


if __name__ == "__main__":
    main()
