import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch


os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=32, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=31, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


def load_data(src, data_dir='dataset'):
    folder_src = os.path.join(os.path.join(data_dir, src), 'images')
    transform = {
        'train': transforms.Compose(
            [transforms.Resize((int(opt.img_size/8*9), int(opt.img_size/8*9))),
                transforms.RandomCrop(opt.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize((opt.img_size, opt.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    }

    source_data = datasets.ImageFolder(
        root=folder_src, transform=transform['train'])
    
    return source_data


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.nc = 32
        self.init_size = opt.img_size // 16
        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        self.linear = nn.Sequential(
            nn.Linear(opt.latent_dim + opt.n_classes, 8 * self.nc * self.init_size**2),
            nn.BatchNorm1d(8 * self.nc * self.init_size**2),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, True),
        )
         
        self.conv_blocks = nn.Sequential(
            # state size. (nc*8) x 14 x 14
            nn.BatchNorm2d(self.nc * 8),
            nn.ConvTranspose2d(self.nc * 8, self.nc * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nc * 4),
            nn.LeakyReLU(0.1, True),
            # state size. (nc*4) x 28 x 28
            nn.ConvTranspose2d(self.nc * 4, self.nc * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nc * 2),
            nn.LeakyReLU(0.1, True),
            # state size. (nc*2) x 56 x 56
            nn.ConvTranspose2d(self.nc * 2, self.nc, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nc),
            nn.LeakyReLU(0.1, True),
            # state size. (nc) x 112 x 112
            nn.ConvTranspose2d(self.nc, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (opt.channels) x 224 x 224
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and noise to produce input
        gen_input = torch.cat((self.label_embedding(labels), noise), -1)
        tmp = self.linear(gen_input)
        tmp = tmp.view(tmp.shape[0], self.nc * 8, self.init_size, self.init_size)
        img = self.conv_blocks(tmp)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.nc = 32
        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        self.conv_blocks = nn.Sequential(
            # input is (opt.channels) x 224 x 224
            nn.Conv2d(opt.channels, self.nc, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (nc) x 112 x 112
            nn.Conv2d(self.nc, self.nc * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nc * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (nc*2) x 56 x 56
            nn.Conv2d(self.nc * 2, self.nc * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nc * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (nc*4) x 28 x 28
            nn.Conv2d(self.nc * 4, self.nc * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nc * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (nc*8) x 14 x 14
            nn.AvgPool2d((int(opt.img_size/16), int(opt.img_size/16)))
        )
        self.fc = nn.Sequential(
            nn.Linear(self.nc * 8 + opt.n_classes, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.nc * 8, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, opt.n_classes)
        )

    def forward(self, img, labels):
        img_feature = self.conv_blocks(img)
        tmp = torch.cat((img_feature.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.fc(tmp)
        class_predict = self.classifier(img_feature.view(img.size(0), -1))
        return validity, class_predict


# Loss functions
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
source_name = "amazon"#"webcam"
target_name = "amazon"
print('Src: %s, Tar: %s' % (source_name, target_name))
source_data = load_data(source_name, data_dir='/data/xian/Office-31')
dataloader = DataLoader(
    source_data, batch_size=opt.batch_size, shuffle=True, num_workers=8, drop_last=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # range of the labels: 0 to n_row-1
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid_value = torch.rand(1)[0] * 0.3 + 0.85
        fake_value = torch.rand(1)[0] * 0.3
        valid = Variable(FloatTensor(batch_size, 1).fill_(valid_value), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(fake_value), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, _ = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images and classification loss
        validity_real, class_predict = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)
        d_pred_loss = nn.CrossEntropyLoss()(class_predict, labels)/3

        # Loss for fake images
        validity_fake, _ = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        print(float(d_real_loss), float(d_fake_loss), float(d_pred_loss))
        d_loss = (d_real_loss + d_fake_loss + d_pred_loss) / 3

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)