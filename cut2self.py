import numpy as np
import os
import glob
from matplotlib import cm
from matplotlib import pyplot as plt
import cv2
import util
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, models
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import trange
from time import sleep
from scipy.io import loadmat
import torchvision.datasets as dset
from torch.utils.data import sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from partialconv2d import PartialConv2d
from model import self2self
import matplotlib.pyplot as plt
from array import *
from partialconv2d import PartialConv2d
from model import self2self
import glob
from skimage.draw import disk, ellipse, polygon

def image_loader(image, device, p1, p2):
    """load image, returns cuda tensor"""
    loader = T.Compose([T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),
                        T.RandomVerticalFlip(torch.round(torch.tensor(p2))),
                        T.ToTensor()
                        ])
    image = Image.fromarray(image.astype(np.uint8))
    image = loader(image).float()
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    return image.to(device)
if __name__ == "__main__":
    USE_GPU = True
    dtype = torch.float32
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('using device:', device)
    model = self2self(3)
    # Path to the directory
    path = "/"
    # Extract the list of filenames
    files = glob.glob(path + '*', recursive=False)
    folder_list = []
    # Loop to print the filenames
    for filename in files:
        folder_list.append(filename)
    image_list = []
    image_folder = []
    # noisy image path
    for filepath in glob.iglob("/*.png"):
        image_list.append(filepath)
    print(len(image_list))
    i=0
    for i in range(len(image_list)):
        img=np.array(Image.open(image_list[i]))
        #img = np.stack((img1,) * 3, axis=-1)
        print("Start new image running")
        print(img.shape)
        learning_rate = 1e-4
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        w, h, c = img.shape
        NPred = 100
        length = 4
        n_holes = round(((0.3 * h * w)/100)/(length * length))
        print(n_holes)
        slice_avg = torch.tensor([1, 3, w, h]).to(device)
        for itr in range(200000):
            mask = np.ones([img.shape[0], img.shape[1], img.shape[2]])
            for n in range(n_holes):
                y = np.random.randint(w)
                x = np.random.randint(h)
                y1 = np.clip(y - length // 2, 0, h)
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)
                mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            img_input = img
            y = img
            p1 = np.random.uniform(size=1)
            p2 = np.random.uniform(size=1)
            img_input_tensor = image_loader(img_input, device, p1, p2)
            y = image_loader(y, device, p1, p2)
            mask = np.expand_dims(np.transpose(mask, [2, 0, 1]), 0)
            mask = torch.tensor(mask).to(device, dtype=torch.float32)
            model.train()
            img_input_tensor = img_input_tensor * mask
            output = model(img_input_tensor, mask)
            loader = T.Compose([T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),
                            T.RandomVerticalFlip(torch.round(torch.tensor(p2)))])
            if itr == 0:
                slice_avg = loader(output)
            else:
                slice_avg = slice_avg * 0.99 + loader(output) * 0.01
            loss = torch.sum(abs(output - y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(torch.max(output), torch.max(y))
            print("iteration %d, loss = %.4f" % (itr + 1, loss.item()*100 ))
            if (itr + 1) % 1000 == 0:
                model.eval()
                img_array = []
                sum_preds = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
                for j in range(NPred):
                    mask = np.ones([img.shape[0], img.shape[1], img.shape[2]])
                    img_input = img * mask
                    img_input_tensor = image_loader(img_input, device, 0.1, 0.1)
                    mask = np.expand_dims(np.transpose(mask, [2, 0, 1]), 0)
                    mask = torch.tensor(mask).to(device, dtype=torch.float32)
                    output_test = model(img_input_tensor, mask)
                    sum_preds[:, :, :] += np.transpose(output_test.detach().cpu().numpy(), [2, 3, 1, 0])[:, :, :, 0]
                    img_array.append(np.transpose(output_test.detach().cpu().numpy(), [2, 3, 1, 0])[:, :, :, 0])
                if i == i:
                   k=i
                   print("k= "+str(k)+" image saving done")
                   # calculate avg
                   average = np.squeeze(np.uint8(np.clip(np.average(img_array, axis=0), 0, 1) * 255))
                   write_img = Image.fromarray(average)
                   write_img.save(folder_list[k]+"/avg-" + str(itr + 1) + ".png")

                k=k+i

    i+1
