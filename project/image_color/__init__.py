"""Image Color Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
import time
from tqdm import tqdm
import torch

import redos
import todos

from . import data, color
from einops import rearrange
import torch.nn.functional as F

import pdb


def get_model():
    """Create model."""

    model_path = "models/image_color.pth"
    cdir = os.path.dirname(__file__)
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path

    model = color.ColorModel()

    # todos.model.load(model.encoder, "/tmp/color_encoder.pth")
    # todos.model.load(model.decoder, "/tmp/color_transform.pth")
    # torch.save(model.state_dict(), "/tmp/image_color.pth")

    todos.model.load(model, checkpoint)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/image_color.torch"):
        model.save("output/image_color.torch")

    return model, device


def model_forward(model, device, g_input_tensor, c_input_tensor):
    g_input_tensor = g_input_tensor.to(device)
    c_input_tensor = c_input_tensor.to(device)

    torch.cuda.synchronize()
    with torch.jit.optimized_execution(False):
        with torch.no_grad():
            output_tensor = model(g_input_tensor, c_input_tensor)
    torch.cuda.synchronize()
    return output_tensor


def image_client(name, grey_input_files, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    grey_filenames = todos.data.load_files(grey_input_files)
    for filename in grey_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.color(filename, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(grey_filenames)} tasks for {name}.")


def image_server(name, host="localhost", port=6379):
    # load model
    model, device = get_model()

    def do_service(input_file, output_file, targ):
        print(f"  clean {input_file} ...")
        try:
            input_tensor = todos.data.load_rgba_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except Exception as e:
            print("exception: ", e)
            return False

    return redos.image.service(name, "image_color", do_service, host, port)


def image_predict(grey_input_files, color_input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    # load files
    grey_filenames = todos.data.load_files(grey_input_files)
    color_filenames = todos.data.load_files(color_input_files)

    # start predict
    progress_bar = tqdm(total=len(grey_filenames))
    for g_filename, c_filename in zip(grey_filenames, color_filenames):
        progress_bar.update(1)

        # orig input
        g_input_tensor = todos.data.load_tensor(g_filename)
        c_input_tensor = todos.data.load_tensor(c_filename)

        predict_tensor = model_forward(model, device, g_input_tensor, c_input_tensor)
        output_file = f"{output_dir}/{os.path.basename(g_filename)}"

        H, W = g_input_tensor.size(2), g_input_tensor.size(3)
        c_input_tensor = F.interpolate(
            c_input_tensor,
            size=(H, W),
            mode="bilinear",
            recompute_scale_factor=False,
            align_corners=False,
        )

        todos.data.save_tensor([g_input_tensor, c_input_tensor, predict_tensor], output_file)
