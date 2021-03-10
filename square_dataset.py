import tensorflow as tf
import cv2, glob, os
from tqdm import tqdm
import argparse

def get_img_paths(input_dir):
    # Get JPEG images
    img_paths = glob.glob(os.path.join(input_dir, "**/*.jpeg"), recursive=True) 
    img_paths += glob.glob(os.path.join(input_dir, "**/*.jpg"), recursive=True) 

    # Get PNG images
    img_paths += glob.glob(os.path.join(input_dir, "**/*.png"), recursive=True)
    return img_paths


def crop(img, max_pct_bar=0.2, margin=6):
    stop_w = 0
    for w in range(img.shape[0]):
        if img[w].any():
            stop_w = w
            break

    if stop_w > 0:
        # Percentage of image covered in black bar
        pct_cover = (stop_w + margin) * 2 / img.shape[0]
        
        # If too much of the image is covered, return nothing
        if pct_cover > max_pct_bar:
            return
        
    # Crop width
    img = img[stop_w + margin : -stop_w-margin]
    
    stop_h = 0
    for h in range(img.shape[1]):
        # Breaks once non-black strip found
        if img[:, h].any():
            stop_h = h
            break

    if stop_h > 0:
        # Percentage of image covered in black bar
        pct_cover = (stop_h + margin) * 2 / img.shape[1]

        # If too much of the image is covered, return nothing
        if pct_cover > max_pct_bar:
            return

    # Crop width
    img = img[:, stop_h + margin : -stop_h-margin]

    return img


def resize(img, img_dim):
    img = cv2.resize(img, (img_dim, img_dim))
    return img

def main(args):
    img_paths = get_img_paths(args.in_dir)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    idx = 0
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        img = crop(img)
        if img is not None:
            img = resize(img, args.img_dim)
            img_name_prefix = img_path.split("/")[-1]
        
            img_name = os.path.join(args.out_dir, str(idx) + "_" + img_name_prefix.split(".")[0] + ".jpg")
            cv2.imwrite(img_name, img)
            idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dim", type=int, default=512,
            help="Image output dimension size.")
    parser.add_argument("--out_dir", default="square_imgs",
            help="Directory to output square images.")
    parser.add_argument("--in_dir", required=True,
            help="Input directory with images.")
    main(parser.parse_args())
