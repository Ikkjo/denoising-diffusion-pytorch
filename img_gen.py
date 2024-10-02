import torch
from unet import UNet
from ddpm import Diffusion
from utils import plot_images


def main():
    device = "cuda"
    model = UNet().to(device)
    ckpt = torch.load("unconditional_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    x = diffusion.sample(model, n=16)
    plot_images(x)


if __name__ == "__main__":
    main()
