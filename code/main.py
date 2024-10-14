import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision
import pandas as pd
import time
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm,trange
from tqdm import tqdm,trange
from torchvision import transforms


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )
        self.loss_function = nn.BCELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 200),
            nn.LeakyReLU(0.02),
            nn.LayerNorm(200),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )
        self.optimiser = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, D, inputs, targets):  # 用分类器的损失来训练生成
        g_output = self.forward(inputs)  # 生成器generator的输出
        d_output = D.forward(g_output)  # 分类器discriminator的输出
        loss = D.loss_function(d_output, targets)
        self.counter += 1
        if self.counter % 10 == 0:
            self.progress.append(loss.item())
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()


class MnistDataset(Dataset):
    def __init__(self, csv_file):
        #[60000 rows x 785 columns] (60000, 785) <class 'pandas.core.frame.DataFrame'>
        self.data_df = pd.read_csv(csv_file, header=None)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = self.data_df.iloc[index, 0] #5
        image_values = torch.tensor(self.data_df.iloc[index, 1:].values,dtype=torch.float32) / 255.0
        return image_values, label

    def plot_image(self, index):
        img = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title("label = " + str(self.data_df.iloc[index, 0]))
        plt.imshow(img, interpolation='none', cmap='Blues')


def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data

def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data

def show_loss_curve(prog,title):
    plt.figure(figsize=(8, 8),dpi=70)
    x=np.linspace(1,len(prog),len(prog))
    plt.plot(x,prog)
    plt.title(title)
    plt.savefig('./MAIN/{}.jpg'.format(title))
    plt.close()
    # plt.show()

def get_lab_1(size):
    return torch.ones(size,dtype=torch.float32)

def get_lab_0(size):
    return torch.zeros(size,dtype=torch.float32)

def get_img_tensor(pd,a):
    np_img = np.array(pd[a].reshape(1, -1), dtype=np.float32)[0]  # (784,)
    t_img = torch.tensor(np_img, dtype=torch.float32)  # torch.Size([784]) <class 'torch.Tensor'>
    return t_img

def show_tensor(img):
    plt.imshow(img.detach().numpy().reshape((28, 28)))
    plt.show()

def save_tensor(t_fake,idx):
    images = t_fake.reshape(t_fake.shape[0], 28, 28)#torch.Size([81,28,28])
    fig, axs = plt.subplots(9, 9, figsize=(10, 10))
    for i in range(t_fake.shape[0]):
        row = i // 9
        col = i % 9
        axs[row, col].imshow(images[i], cmap='gray')
        axs[row, col].axis('off')
    plt.tight_layout()
    plt.savefig('./MAIN/fake_photo{:d}.jpg'.format(idx))
    plt.close()

if __name__ == '__main__':

    dataset=MnistDataset('../../dataset/MNIST_CSV/mnist_train.csv')
    loader_data=DataLoader(dataset,batch_size=81,shuffle=True,drop_last=False)

    D=Discriminator().cuda()
    G=Generator().cuda()

    idx=0
    for epoch in range(80):
        for (imgs,labs) in tqdm(loader_data,desc='train {}/{}'.format(epoch+1,80)):
            D.train(
                imgs.cuda(),                                #torch.Size([81, 784])
                get_lab_1(size=(imgs.shape[0],1)).cuda(),   #torch.Size([81, 1])
            )
            D.train(
                G.forward(generate_random_seed(size=(imgs.shape[0],100)).cuda()).detach().cuda(),
                get_lab_0(size=(imgs.shape[0],1)).cuda(),
            )
            G.train(
                D,
                generate_random_seed(size=(imgs.shape[0],100)).cuda(),
                get_lab_1(size=(imgs.shape[0],1)).cuda(),
            )
            t_fake = G.forward(generate_random_seed(size=(imgs.shape[0],100)).cuda())#torch.Size([81, 784])
            if idx%500==0:
                save_tensor(t_fake.detach().cpu(), idx)
            idx+=1
        torch.save(D,'./MAIN/Discriminator_epoch_{}.pth'.format(epoch+1))
        torch.save(G,'./MAIN/Generator_epoch_{}.pth'.format(epoch+1))

    show_loss_curve(D.progress[0::],'Discriminator')
    show_loss_curve(G.progress[0::],'Generator')