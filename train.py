import os
import argparse
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CartoonDataset
from models.generator import GeneratorUNet
from models.discriminator import Discriminator

def train(opt):
    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    # 1. Create Dataset & Dataloader
    train_dataset = CartoonDataset(root_dir=opt.data_root, mode="train", image_size=opt.image_size)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)

    # 2. Instantiate Models
    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)

    # 3. Define Losses
    # Pix2Pix typically uses either BCE or MSE for the adversarial loss
    criterion_GAN = nn.MSELoss() if opt.use_lsgan else nn.BCEWithLogitsLoss()
    # L1 for reconstruction of the cartoon
    criterion_L1 = nn.L1Loss()

    # 4. Define Optimisers (Adam)
    optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # If we have a checkpoint to resume from
    start_epoch = 1
    if opt.checkpoint_path and os.path.isfile(opt.checkpoint_path):
        print("[INFO] Loading checkpoint:", opt.checkpoint_path)
        checkpoint = torch.load(opt.checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_G.load_state_dict(checkpoint["optimG"])
        optimizer_D.load_state_dict(checkpoint["optimD"])
        start_epoch = checkpoint["epoch"] + 1

    # 5. Training Loop
    for epoch in range(start_epoch, opt.epochs + 1):
        generator.train()
        discriminator.train()

        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{opt.epochs}]", leave=False)
        for real_imgs, cartoon_imgs in loop:
            real_imgs = real_imgs.to(device)
            cartoon_imgs = cartoon_imgs.to(device)

            # --- Train Discriminator ---
            # (A) Generate fake cartoon
            fake_cartoon = generator(real_imgs)

            # (B) Discriminator on real pair
            pred_real = discriminator(real_imgs, cartoon_imgs)
            if opt.use_lsgan:
                loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            else:
                loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

            # (C) Discriminator on fake pair
            pred_fake = discriminator(real_imgs, fake_cartoon.detach())
            if opt.use_lsgan:
                loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            else:
                loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

            loss_D = (loss_D_real + loss_D_fake) * 0.5

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # --- Train Generator ---
            # (A) Discriminator on fake, generator tries to fool it
            pred_fake_for_G = discriminator(real_imgs, fake_cartoon)
            if opt.use_lsgan:
                loss_G_gan = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            else:
                loss_G_gan = criterion_GAN(pred_fake_for_G, torch.ones_like(pred_fake_for_G))

            # (B) L1 pixel loss vs. ground-truth cartoon
            loss_G_l1 = criterion_L1(fake_cartoon, cartoon_imgs) * opt.lambda_l1

            # (C) Total generator loss
            loss_G = loss_G_gan + loss_G_l1

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # Update progress bar
            loop.set_postfix({
                "lossD": loss_D.item(),
                "lossG": loss_G.item(),
                "GAN": loss_G_gan.item(),
                "L1": loss_G_l1.item()
            })

        # End of epoch, save checkpoint
        if epoch % opt.save_interval == 0 or epoch == opt.epochs:
            checkpoint = {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optimG": optimizer_G.state_dict(),
                "optimD": optimizer_D.state_dict()
            }
            save_path = f"checkpoints/epoch_{epoch}.pt"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(checkpoint, save_path)
            torch.save(generator.state_dict(), "checkpoints/generator_last.pt")
            print(f"[INFO] Epoch {epoch}: saved checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data", help="root folder of dataset")
    parser.add_argument("--epochs", type=int, default=100, help="num of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--lambda_l1", type=float, default=100.0, help="weight for L1 loss")
    parser.add_argument("--use_lsgan", action="store_true", help="use MSE (LSGAN) instead of BCE for adversarial")
    parser.add_argument("--image_size", type=int, default=256, help="image dimension to resize")
    parser.add_argument("--checkpoint_path", type=str, default="", help="path to load checkpoint from")
    parser.add_argument("--save_interval", type=int, default=10, help="save checkpoint every x epochs")

    opt = parser.parse_args()
    train(opt)
