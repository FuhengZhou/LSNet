import time
import cv2
import os
import torch
from torchvision import transforms
import argparse
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np

from LSNet import model

transform = transforms.Compose([
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor()
])

def main(args):

    model_my = model().cuda()
    model_my.load_state_dict(torch.load(args.model_path))
    model_my.eval()

    img_path = args.img_path

    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    if os.path.isdir(img_path):
        img_directory = []
        for file in os.listdir(img_path):
            if file.lower().endswith(supported_extensions):
                img_directory.append(os.path.join(img_path, file))
    elif os.path.isfile(img_path) and img_path.endswith(supported_extensions):
        img_directory = [img_path]
    else:
        assert False, "Error: The path is neither a directory nor a recognized image format."

    for img_path in img_directory:
        file_name = img_path.split('/')[-1].split('.')[0]

        time0 = time.time()
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = transform(Image.fromarray(img)).unsqueeze(0).cuda()

        with torch.no_grad():
            output = model_my(img)
        time1 = time.time()

        #output_image = output.squeeze()

        output_image = output.clip(0, 1)

        output_image = output_image[0].cpu().detach().numpy() * 255.0

        # rgb = np.clip(gen, 0, 255).astype(np.uint8)

        output_image = output_image.transpose(1, 2, 0).astype(np.uint8)
        #output_image = output_image.astype(np.uint8)
        image = Image.fromarray(output_image, 'RGB')

        # 保存为PNG
        image.save(os.path.join(args.save_path, file_name + '.png'))
        #output_image.save(os.path.join(args.save_path, file_name + '.png'))
        print(f'{img_path} interference time: {time1 - time0}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('--model_path', type=str, default='weight.pth',
                        help='trained model path')
    parser.add_argument('--img_path', type=str, default='LSUI/new_test/input',
                        help='interfering image path')
    parser.add_argument('--save_path', type=str, default='out',
                        help='save image path')
    args = parser.parse_args()

    main(args)

