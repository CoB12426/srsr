import cv2
import os
from tqdm import tqdm

def resize_image(image_path, output_image_path, width, height):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    resized_image = cv2.resize(image, (width, height))
    
    cv2.imwrite(output_image_path, resized_image)
    #print(f"Resized image saved as {output_image_path}")

resize_w, resize_h = 32, 64

input_dir = '/home/aki/anaconda3/envs/psr/dataset/Market-1501-v15.09.15/gt_bbox'
output_dir = f'{input_dir.replace('/gt_bbox', '')}/resized_{str(resize_w)}{str(resize_h)}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
imgs = os.listdir(input_dir)
print(len(imgs))
for img in tqdm(imgs):
    if img.endswith('.jpg'):
        resize_image(f'{input_dir}/{img}', f'{output_dir}/{img}', resize_w, resize_h)