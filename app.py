import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

import pytorch_lightning as pl

from flask import Flask, render_template, request

from models import ConvAutoencoder
from utils import write_image, read_numpy, read_image, resize_image, transform_numpy_to_tensor


app = Flask(__name__)
latent_dim = 512
last_model = ConvAutoencoder(latent_dim)
ckpt_location = 'eursat_ae_512_10epochs.ckpt'
checkpoint = torch.load(ckpt_location, map_location=torch.device('cpu'))
last_model.load_state_dict(checkpoint)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['photo']
    filename = file.filename
    file_dir = 'static/img/' + filename
    file.save(file_dir)
    #if filename.endswith(".jpeg"):
    #    image = read_image(file_dir)
    #else:
    #print(request.files)
    image = read_numpy(file_dir)
    #print(file_dir)
    #print(image)

    image = image*(3.5 / 1e4)
    image_shape = image.shape
    print(f"np shape {image_shape}")

    if image_shape[2] != 13:
        print("not enough channels for model")
        return render_template('index.html')

    image = resize_image(image)

    show = False
    if show:
        image_to_show = image[:,:, [4,3,2]]
        print(f"shape image to show, {image_to_show.shape}")
        to_show_name = f"{filename}_todisplay.jpeg".replace(".npy", "")
        print(to_show_name)
        toshow_filedir = write_image(image_to_show, to_show_name)
        return render_template('index.html', filename=to_show_name)

    image_tensor = transform_numpy_to_tensor(image).float()
    image_tensor = torch.unsqueeze(image_tensor, 0)
    print(f"tensor shape {image_tensor.shape}")
    print(f"tenso type {image_tensor.type()}")

    latent_code = last_model.encoder(image_tensor)
    print(f"latent {latent_code}")
    return render_template('index.html', filename=filename)





