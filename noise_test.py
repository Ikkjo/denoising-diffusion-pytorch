import torch
from torchvision.utils import save_image
from ddpm import Diffusion
from utils import get_data
import argparse

# args.batch_size = 1  # 5
# args.image_size = 64
# args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def main(args):

    dataloader = get_data(args)

    diff = Diffusion(device=get_device(), img_size=args.img_size)

    image = next(iter(dataloader))[0]
    t = torch.Tensor([50, 100, 150, 200, 300, 600, 700, 999]).long()

    noised_image, _ = diff.noise_images(image, t)
    save_image(noised_image.add(1).mul(0.5), "noise.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int,
                        description="Batch size", default=1)
    parser.add_argument("-s", "--image-size", type=int,
                        description="Image size", default=64)
    parser.add_argument("-d", "--dataset-path", type=str)
    args = parser.parse_args()

    main(args)

