import os
import numpy as np
from skimage import color, io

import torch
import torch.nn.functional as F

from PIL import Image
from models import ColorEncoder, ColorUNet

import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def Lab2RGB_out(img_lab):
    img_lab = img_lab.detach().cpu()
    img_l = img_lab[:,:1,:,:]
    img_ab = img_lab[:,1:,:,:]
    # print(torch.max(img_l), torch.min(img_l))
    # print(torch.max(img_ab), torch.min(img_ab))
    img_l = img_l + 50
    pred_lab = torch.cat((img_l, img_ab), 1)[0,...].numpy()
    # grid_lab = utils.make_grid(pred_lab, nrow=1).numpy().astype("float64")
    # print(grid_lab.shape)
    out = (np.clip(color.lab2rgb(pred_lab.transpose(1, 2, 0)), 0, 1)* 255).astype("uint8")
    return out

def RGB2Lab(inputs):
    return color.rgb2lab(inputs)

def Normalize(inputs):
    l = inputs[:, :, 0:1]
    ab = inputs[:, :, 1:3]
    l = l - 50
    lab = np.concatenate((l, ab), 2)

    return lab.astype('float32')

def numpy2tensor(inputs):
    out = torch.from_numpy(inputs.transpose(2,0,1))
    return out

def tensor2numpy(inputs):
    out = inputs[0,...].detach().cpu().numpy().transpose(1,2,0)
    return out

def preprocessing(inputs):
    # input: rgb, [0, 255], uint8
    # print(inputs) -- <PIL.Image.Image image mode=RGB size=335x313>

    # RGB2Lab(inputs).shape, RGB2Lab(inputs).min(), RGB2Lab(inputs).max()
    # ((313, 335, 3), -0.0024549378619953544, 100.0)
    # L: [0, 100.0], A: [-127, 127], B: [-127, 127]
    img_lab = Normalize(RGB2Lab(inputs))
    img = np.array(inputs, 'float32') # [0, 255]
    img = numpy2tensor(img)
    img_lab = numpy2tensor(img_lab)
    # img.size() -- [3, 313, 335], [0, 255,0]
    # img_lab-L [-50.0, 50.0], A[-127, 127], B[127, 127]
    return img.unsqueeze(0), img_lab.unsqueeze(0)


def load_color_net(model, state_dict):
    """Load model."""

    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)

if __name__ == "__main__":
    device = "cuda"

    model_name = 'Color2Embed_1_4.5w'
    ckpt_path = 'experiments/Color2Embed_1/045000.pt'
    test_dir_path = 'test_datasets/gray2color/exemplar_based/'
    out_dir_path = 'results/gray2color/exemplar_based/' + model_name
    imgs_num = len(os.listdir(test_dir_path)) // 2
    imgsize = 256
    
    mkdirs(out_dir_path)

    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    colorEncoder = ColorEncoder().to(device)
    # colorEncoder.load_state_dict(ckpt["colorEncoder"])
    load_color_net(colorEncoder, ckpt["colorEncoder"])

    colorEncoder.eval()

    colorUNet = ColorUNet().to(device)
    colorUNet.load_state_dict(ckpt["colorUNet"])
    colorUNet.eval()


    imgs = []
    imgs_lab = []
    cos_d_avg = 0

    for i in range(imgs_num):
        idx = i
        print('Image', idx, 'Input Image', 'in%d.JPEG'%idx, 'Ref Image', 'ref%d.JPEG'%idx)

        img_path = os.path.join(test_dir_path, '%03d_in.png'%idx)
        ref_img_path = os.path.join(test_dir_path, '%03d_ref.png'%idx)

        img1 = Image.open(img_path).convert("RGB")
        width, height = img1.size
        img2 = Image.open(ref_img_path).convert("RGB")

        img1, img1_lab = preprocessing(img1)
        img2, img2_lab = preprocessing(img2)

        img1 = img1.to(device)
        img1_lab = img1_lab.to(device)
        img2 = img2.to(device)
        # img2_lab = img2_lab.to(device)

        # print('-------',torch.max(img1_lab[:,:1,:,:]), torch.min(img1_lab[:,1:,:,:]))

        with torch.no_grad():
            img2_resize = F.interpolate(img2 / 255., size=(imgsize, imgsize), mode='bilinear', recompute_scale_factor=False, align_corners=False)

            # img2_resize.size() -- [1, 3, 256, 256], [0.0, 1.0]
            color_vector = colorEncoder(img2_resize)
            # color_vector.size() -- [1, 512, 1, 1]

            img1_L_resize = F.interpolate(img1_lab[:,:1,:,:] / 50., size=(imgsize, imgsize), mode='bilinear', recompute_scale_factor=False, align_corners=False)
            # img1_L_resize.size() -- [1, 1, 256, 256], [-1.0, 1.0]
            fake_ab = colorUNet((img1_L_resize, color_vector))
            fake_ab = F.interpolate(fake_ab*110, size=(height, width), mode='bilinear', recompute_scale_factor=False, align_corners=False)
            fake_img = torch.cat((img1_lab[:,:1,:,:], fake_ab), dim=1)
            fake_img = Lab2RGB_out(fake_img)


            out_img_path = os.path.join(out_dir_path, 'in%d_ref%d.png'%(idx, idx))
            io.imsave(out_img_path, fake_img)

            # re_img, re_img_lab = preprocessing(fake_img)
            # re_img = re_img.to(device)
            # re_img_resize = F.interpolate(re_img / 255., size=(imgsize, imgsize), mode='bilinear', recompute_scale_factor=False, align_corners=False)
            # re_color_vector = colorEncoder(re_img_resize)

            # cos_d = torch.cosine_similarity(color_vector, re_color_vector, dim=1)
            # cos_d_avg += cos_d

    print('Average Cosine Distance is: ', cos_d_avg/imgs_num)