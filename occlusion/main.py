from __future__ import print_function
import argparse
import math

import sys
sys.path.append('..')
from linear import LinearGroupHS

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = LinearGroupHS(24*24*3, 256)
        self.linear2 = LinearGroupHS(256, 256)
        self.linear3 = LinearGroupHS(256, 1)

        self.kl_list = [self.linear1, self.linear2, self.linear3]

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))

        if self.training:
            return x.squeeze()
        else:
            return x.squeeze().mean(0)


    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, conds) in enumerate(train_loader):
        # forward pass
        data, target = data.to(device), target.to(device)
        output = model(data)

        # backward pass
        loss = F.binary_cross_entropy(output, target) + model.kl_divergence() / args.N
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # printing
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()

    correct_conds = torch.zeros(args.num_conditions)
    sum_conds = torch.zeros(args.num_conditions)

    test_loss = 0
    with torch.no_grad():
        for data, target, conds in test_loader:
            # forward pass
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.round()

            # loss
            test_loss += F.binary_cross_entropy(output, target, size_average=False).item() # sum up batch loss
            for i in range(args.num_conditions):
                correct_conds[i] += prediction[conds == i].eq(target[conds == i]).float().sum().item()
                sum_conds[i] += prediction[conds == i].shape[0]

    correct_conds = correct_conds / sum_conds
    test_loss = test_loss / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Full: {:.2f}%, Bottom: {:.2f}%, Top: {:.2f}%, None: {:.2f}%\n'.format(
        test_loss, 100 * correct_conds[0], 100 * correct_conds[1], 100 * correct_conds[2], 100 * correct_conds[3]))

    return correct_conds

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='From PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='B', help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, metavar='E', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--num-runs', type=int, default=10, metavar='NR', help='Number of runs for each conditions')
    parser.add_argument('--num-conditions', type=int, default=4, metavar='C', help='Number of conditions')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=16, metavar='N', help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_set, test_labels, test_conds = torch.load('test.pt')
    test_loader = torch.utils.data.DataLoader(TensorDataset(test_set, test_labels, test_conds), batch_size=args.batch_size, shuffle=True, **kwargs)

    iters = [256, 512, 1024, 2048, 4096, 8192]
    results = torch.zeros(args.num_runs, len(iters), args.num_conditions)

    for j in range(args.num_runs):
        for i, N in enumerate(iters):

            args.N = N
            train_set, train_labels, train_conds = torch.load('train.pt')
            train_loader = torch.utils.data.DataLoader(TensorDataset(train_set, train_labels, train_conds), batch_size=args.batch_size, shuffle=True, **kwargs)

            model = Net().to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch)
            results[j, i, :] = test(args, model, device, test_loader)
    torch.save(results, 'occlusion_results.pt')

if __name__ == '__main__':
    main()
