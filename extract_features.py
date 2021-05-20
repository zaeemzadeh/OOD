from __future__ import print_function
import numpy as np
import argparse
import time
import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import wideresnet_UoS as wrn

# Setting the name of the out-of-distribution dataset
# Tiny-ImageNet (crop):     Imagenet
# Tiny-ImageNet (resize):   Imagenet_resize
# LSUN (crop):              LSUN
# LSUN (resize):            LSUN_resize


parser = argparse.ArgumentParser(description='Pytorch Extracting Features from Neural Networks')
parser.add_argument('--path', default="WideResNet-CIFAR10", type=str, help='path to model')
parser.add_argument('--in_dataset', default="cifar10", type=str, help='in-distribution dataset')
parser.add_argument('--out_dataset', default="Imagenet_crop", type=str,
                    help='out-of-distribution dataset')
parser.add_argument('--droprate', default=0.3, type=float,help='dropout probability (default: 0.3)')
parser.add_argument('--mc', default=50, type=int,help='Number of Monte Carlo samples (default: 50)')
parser.add_argument('--no_in_dataset', dest='in_extract', action='store_false',
                    help='do not extract features for in-distribution dataset - Default=True')
parser.add_argument('--gpu', default=0, type=int, help='gpu index')
parser.set_defaults(in_extract=True)

start = time.time()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255), (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)),
])


def main():
    args = parser.parse_args()
    test(args.in_dataset, args.in_extract, args.out_dataset, args.path, args.droprate, args.mc, args.gpu)


def test(in_dataset, extract_in_features, dataName, modelPath, d_rate, MC_runs, CUDA_DEVICE):
    if in_dataset == "cifar10":
        net1 = wrn.WideResNet(28, 10, 10, d_rate)
    if in_dataset == "cifar100":
        net1 = wrn.WideResNet(28, 100, 10, d_rate)

    checkpoint = torch.load("../models/{}/model_best.pth.tar".format(modelPath))
    net1.load_state_dict(checkpoint['state_dict'])

    net1.cuda(CUDA_DEVICE)

    testsetout = torchvision.datasets.ImageFolder("../data/{}".format(dataName), transform=transform)
    testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=1,
                                                shuffle=False, num_workers=2)

    if in_dataset == "cifar10":
        testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
                                                   shuffle=False, num_workers=2)
        trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        trainloaderIn = torch.utils.data.DataLoader(trainset, batch_size=1,
                                                    shuffle=False, num_workers=2)
    if in_dataset == "cifar100":
        testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
                                                   shuffle=False, num_workers=2)
        trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform)
        trainloaderIn = torch.utils.data.DataLoader(trainset, batch_size=1,
                                                    shuffle=False, num_workers=2)

    testData(net1, CUDA_DEVICE, trainloaderIn, testloaderIn, testloaderOut, modelPath, dataName, extract_in_features,
               MC_runs)


def testData(net1, CUDA_DEVICE, trainloader, testloader_in, testloader_out, pathSave, dataName, extract_in,
             MC_runs):

    savepath = "../features/" + pathSave + "/"

    ensure_dir(savepath)
    t0 = time.time()
    net1.eval()

    if extract_in:
        print("Processing in-distribution images")
        #######################################In-distribution###########################################
        print("Processing in-distribution images: train")
        features = []
        labels = []

        for j, data in enumerate(trainloader):
            images, lab = data

            inputs = Variable(images.cuda(CUDA_DEVICE))
            labels_t = Variable(lab.cuda(CUDA_DEVICE))
            outputs, feat = net1(inputs)

            features.extend(list(feat.data.cpu().numpy()))
            labels.extend(labels_t.data.cpu())


            if j % 1000 == 0:
                print("{:4}/{:4} batches processed, {:.1f} seconds used.".format(j+1, len(trainloader), time.time()-t0))
                t0 = time.time()

        np.save(savepath+'featuresTrain_in', np.array(features))
        np.save(savepath+'labelsTrain_in', labels)
        if hasattr(net1.fc, 'bias'):
            if net1.fc.bias is not None:
                np.save(savepath + 'bias', net1.fc.bias.cpu().detach().numpy())
        t0 = time.time()

        print("Processing in-distribution images: test")
        features = []
        for j, data in enumerate(testloader_in):
            images, lab = data

            with torch.no_grad():
                inputs = Variable(images.cuda(CUDA_DEVICE))
                outputs, feat = net1(inputs)
                feat = feat[:, :, None]
                for mc in range(1, MC_runs):
                    feat = torch.cat((feat, net1(inputs)[1][:, : , None]), dim=2)
                if j == 0:
                    features = feat.cpu()
                else:
                    features = torch.cat((features, feat.cpu()), dim=0)


            if j % 1000 == 0:
                print("{:4}/{:4} batches processed, {:.1f} seconds used.".format(j+1, len(testloader_in), time.time()-t0))
                t0 = time.time()

        features = features.numpy()
        np.save(savepath+'featuresTest_in', features)
        t0 = time.time()

    print("Processing out-of-distribution images")
###################################Out-of-Distributions#####################################
    features = []
    for j, data in enumerate(testloader_out):
        images, _ = data

        with torch.no_grad():
            inputs = Variable(images.cuda(CUDA_DEVICE))
            outputs, feat = net1(inputs)
            feat = feat[:, :, None]
            for mc in range(1, MC_runs):
                feat = torch.cat((feat, net1(inputs)[1][:, :, None]), dim=2)
            if j == 0:
                features = feat.cpu()
            else:
                features = torch.cat((features, feat.cpu()), dim=0)

        if j % 1000 == 0:
            print("{:4}/{:4} batches processed, {:.1f} seconds used.".format(j+1, len(testloader_out), time.time()-t0))
            t0 = time.time()

    features = features.numpy()
    np.save(savepath+'features_out_'+dataName, features)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(directory + " was created")


if __name__ == '__main__':
    main()

















