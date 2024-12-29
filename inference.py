import os
import argparse
import torch
from PIL import Image
import torchvision.transforms as transforms
from models.generator import GeneratorUNet

def inference(opt):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running inference on {device}")

    # 1. Load the trained generator
    generator = GeneratorUNet().to(device)
    generator.load_state_dict(torch.load(opt.generator_path, map_location=device))
    generator.eval()

    # 2. Define the same transform as training
    transform = transforms.Compose([
        transforms.Resize((opt.image_size, opt.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # 3. Process each image in input_dir
    os.makedirs(opt.output_dir, exist_ok=True)
    for fname in os.listdir(opt.input_dir):
        if not fname.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
            continue
        img_path = os.path.join(opt.input_dir, fname)
        img = Image.open(img_path).convert("RGB")

        # Transform
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Generate cartoon
        with torch.no_grad():
            fake_cartoon = generator(img_tensor)

        # Denormalise to [0,1]
        output = (fake_cartoon[0].cpu() * 0.5 + 0.5).clamp(0,1)
        out_pil = transforms.ToPILImage()(output)
        save_path = os.path.join(opt.output_dir, fname)
        out_pil.save(save_path)
        print(f"[INFO] Saved cartoonised {fname} to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_path", type=str, default="checkpoints/generator_last.pt",
                        help="Path to the trained generator weights")
    parser.add_argument("--input_dir", type=str, default="new_real_images",
                        help="Folder containing real images to cartoonise")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Folder to save generated cartoon images")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Resolution for resizing input images")
    opt = parser.parse_args()

    inference(opt)
