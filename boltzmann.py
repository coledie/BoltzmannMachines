"""
Restricted boltzman machine playing with MNIST dataset.

Made with copilot.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable


class RBM(nn.Module):
    """Restricted boltzman machine."""
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(784, 128) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(784))
        self.h_bias = nn.Parameter(torch.zeros(128))

    def sample_from_p(self, p):
        """Sample from probability."""
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))

    def v_to_h(self, v):
        """Visible to hidden."""
        p_h = F.sigmoid(F.linear(v, self.W.t(), self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        """Hidden to visible."""
        p_v = F.sigmoid(F.linear(h, self.W, self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def forward(self, v):
        """Forward pass."""
        with torch.no_grad():
            pre_h1, h1 = self.v_to_h(v)
        pre_v2, v2 = self.h_to_v(h1)
        return pre_h1, h1, pre_v2, v2


if __name__ == '__main__':
    # Hyperparameters
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.01

    # MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

    # Data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    rbm = RBM()

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(rbm.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            img = img.view(img.size(0), -1)
            pre_h1, h1, pre_v2, v2 = rbm(img)
            loss = criterion(v2, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    # Plot the real images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    images = images.view(images.size(0), -1)
    pre_h1, h1, pre_v2, v2 = rbm(images)
    images = images.view(images.size(0), 1, 28, 28)
    v2 = v2.view(v2.size(0), 1, 28, 28)

    img = make_grid(images)
    #plt.imshow(img.numpy().transpose(1, 2, 0))
    #plt.show()
    img = make_grid(v2)
    plt.imshow(img.numpy().transpose(1, 2, 0))
    plt.show()
