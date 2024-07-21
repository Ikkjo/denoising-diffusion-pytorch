import os
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from utils import setup_logging, get_data, save_images, plot_images
from unet import UNet, UNetCFG
from blocks import EMA
from ddpm import Diffusion, DiffusionCFG, logging
from torch.utils.tensorboard.writer import SummaryWriter


def get_cuda_if_available():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_cfg(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNetCFG(num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = DiffusionCFG(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    len_data = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * len_data + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            sampled_images = diffusion.sample(
                model, n=len(labels), labels=labels)
            ema_sampled_images = diffusion.sample(
                ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join(
                "results", args.run_name, f"{epoch}.jpg"))
            save_images(ema_sampled_images, os.path.join(
                "results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(model.state_dict(), os.path.join(
                "models", args.run_name, f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(
                "models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(
                "models", args.run_name, f"optim.pt"))


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    data_len = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(),
                              global_step=epoch*data_len + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])

        save_images(sampled_images, os.path.join(
            "results", args.run_name, f"{epoch}.jpg"))

        torch.save(model.state_dict(), os.path.join(
            "models", args.run_name, f"ckpt.pt"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--device",
        description="Device to use during training. \
                     Can be \"cuda\" or \"cpu\".",
        default=get_cuda_if_available()
    )

    args = parser.parse_args()
    train(args)
