import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import os
import math
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from opacus.validators import ModuleValidator
from opacus.accountants.utils import get_noise_multiplier
from private_vision.privacy_engine import PrivacyEngine
import opacus
from tqdm import tqdm
import warnings
import timm
import glob
import model
import json
from torch.utils.data import random_split
def prepare_results_dir(args):
    """Makedir, init tensorboard if required, save args."""
    root = os.path.join(args.result_root,
                        args.data, args.model, args.mode, str(args.eps))
    os.makedirs(root, exist_ok=True)                        
    if not args.no_tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(root)
    else:
        writer = None
    with open(os.path.join(root, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))
    return args, writer
def prepare(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    args, writer = prepare_results_dir(args)
    if args.data == 'celeba':
        class private_dataset(Dataset):

            def __init__(self, image_loc, label_loc, transform):

                label_df = pd.read_csv(label_loc)
                label_df.set_index("filename", inplace=True)
                
                # List of filenames from the label file
                label_filenames = set(label_df.index.tolist())

                # Iterate through the image directory and keep only files present in the label file
                filenames = []
                for root, dirs, files in os.walk(image_loc):
                    for file in files:
                        if file.endswith('.jpg') or file.endswith('.png'):
                            if file in label_filenames:
                                filenames.append(os.path.join(root, file))

                # Update full_filenames to only include filtered files
                self.full_filenames = filenames
                self.labels = [label_df.loc[os.path.basename(filename)].values[0] for filename in filenames]
                self.transform = transform

            def __len__(self):
                    return len(self.full_filenames)

            def __getitem__(self, idx):
                    image = Image.open(self.full_filenames[idx])
                    image = image.convert('RGB')
                    image = self.transform(image)
                    return image, self.labels[idx]

        def transformate(crop_size, re_size):
            offset_height = (218 - crop_size) // 2
            offset_width = (178 - crop_size) // 2
            def crop(x): return x[:, offset_height:offset_height +
                                  crop_size, offset_width:offset_width + crop_size]
            proc = []
            proc.append(transforms.ToTensor())
            proc.append(transforms.ToPILImage())
            proc.append(transforms.Resize((re_size, re_size)))
            proc.append(transforms.RandomHorizontalFlip())
            proc.append(transforms.ToTensor())
            return transforms.Compose(proc)
        data_dir = 'E:\DO_AN\LPG-MI\datasets\celeba_private_domain'
        label_loc = 'E:\DO_AN\LPG-MI\data_files\subset_train.csv'
        trainset = private_dataset(
            data_dir, label_loc, transformate(108, 64))
        train_size = math.ceil(0.8 * len(trainset))
        val_size = len(trainset) - train_size

        trainset ,testset = random_split(trainset , [train_size , val_size])
        print(len(trainset) , len(testset))

        
    trainloader = DataLoader(
            trainset, batch_size=args.mini_bs, shuffle = True)

    testloader = torch.utils.data.DataLoader(
                 testset, batch_size=args.mini_bs,shuffle = False)


    print('==> Building model..', args.model, '  mode ', args.mode)
    NUM_CLASSES = 10 if args.data == 'CIFAR10' else 100
    if args.model != 'vgg16_bn':
        net = timm.create_model(
        args.model, pretrained=args.pretrained, num_classes=4)
    elif args.model == 'vgg16_bn':
        net = model.VGG16_bn(1000)
        #BACKBONE_RESUME_ROOT = "E:/NCKH2023/LPG-MI/checkpoints/target_model/target_ckp/VGG_MixedGhost.tar"
        #print("Loading Backbone Checkpoint ")
        #checkpoint = torch.load(BACKBONE_RESUME_ROOT)
        net = ModuleValidator.fix(net)
        #net.load_state_dict(checkpoint['state_dict'])

    net = ModuleValidator.fix(net)
    net.to(device)
    if 'xcit' in args.model:
        for name, param in net.named_parameters():
            if 'gamma' in name or 'attn.temperature' in name:
                param.requires_grad = False

    if 'cait' in args.model:
        for name, param in net.named_parameters():
            if 'gamma_' in name:
                param.requires_grad = False

    if 'convnext' in args.model:
        for name, param in net.named_parameters():
            if '.gamma' in name or 'head.norm.' in name or 'downsample.0' in name or 'stem.1' in name:
                param.requires_grad = False

    if 'convit' in args.model:
        for name, param in net.named_parameters():
            if 'attn.gating_param' in name:
                param.requires_grad = False

    if 'beit' in args.model:
        for name, param in net.named_parameters():
            if 'gamma_' in name or 'relative_position_bias_table' in name or 'attn.qkv.weight' in name or 'attn.q_bias' in name or 'attn.v_bias' in name:
                param.requires_grad = False

    for name, param in net.named_parameters():
        if 'cls_token' in name or 'pos_embed' in name:
            param.requires_grad = False

    print('number of parameters: ', sum([p.numel() for p in net.parameters()]))

    if "ghost" in args.mode:
        criterion = nn.CrossEntropyLoss(reduction="none")
    else:
        criterion = nn.CrossEntropyLoss()
    # if args.model =='vgg16_bn':
    #     optimizer = optim.Adam(net.parameters(), lr=args.lr)
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # else :
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    n_acc_steps = args.bs // args.mini_bs

    if 'ghost' in args.mode:
        sigma = get_noise_multiplier(
            target_epsilon=args.eps,
            target_delta=1e-5,
            sample_rate=args.bs/len(trainset),
            epochs=args.epochs,
            accountant="gdp"
        )
        privacy_engine = PrivacyEngine(
            net,
            batch_size=args.bs,
            sample_size=len(trainloader.dataset),
            noise_multiplier=sigma,
            epochs=args.epochs,
            max_grad_norm=args.grad_norm,
            ghost_clipping='non' not in args.mode,
            mixed='mixed' in args.mode
        )
        privacy_engine.attach(optimizer)
    print('==> Building model..', args.model, '  mode ', ' Ghost clipping ')
    
    if args.model == 'vgg16_bn' and args.mode != 'non-DP' :
         # Training

        def train(epochs):
                    best_val_loss = float('inf')  # Initialize best validation loss

                    for epoch in range(epochs):
                        print('\nEpoch: %d' % (epoch+1))
                        net.train()
                        train_loss = 0
                        correct = 0
                        total = 0

                        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, desc='Batch')):
                            inputs, targets = inputs.to(device), targets.to(device)
                            feat, outputs = net(inputs)
                            loss = criterion(outputs, targets)

                            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                                optimizer.step(loss=loss)
                                optimizer.zero_grad()
                            else:
                                optimizer.virtual_step(loss=loss)

                            train_loss += loss.mean().item()
                            _, predicted = outputs.max(1)
                            total += targets.size(0)
                            correct += predicted.eq(targets).sum().item()

                            writer.add_scalar('training loss',
                                            train_loss,
                                            epoch * len(trainloader) + batch_idx)

                        avg_train_loss = train_loss / (batch_idx + 1)
                        train_acc = 100. * correct / total

                        print(epoch, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (avg_train_loss, train_acc, correct, total))
                        print('ε = {} '.format(args.eps))

                        # Validation phase
                        net.eval()
                        val_loss = 0
                        correct = 0
                        total = 0
                        with torch.no_grad():
                            for batch_idx, (inputs, targets) in enumerate(tqdm(testloader, desc='Validation Batch')):
                                inputs, targets = inputs.to(device), targets.to(device)
                                feat, outputs = net(inputs)
                                loss = criterion(outputs, targets)

                                val_loss += loss.mean().item()
                                _, predicted = outputs.max(1)
                                total += targets.size(0)
                                correct += predicted.eq(targets).sum().item()

                        avg_val_loss = val_loss / (batch_idx + 1)
                        val_acc = 100. * correct / total

                        print(epoch, len(testloader), 'Validation Loss: %.3f | Validation Acc: %.3f%% (%d/%d)'
                            % (avg_val_loss, val_acc, correct, total))

                        # Save model if validation loss has decreased
                        if avg_val_loss < best_val_loss:
                            print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(best_val_loss, avg_val_loss))
                            torch.save({
                                'state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'val_loss_min': avg_val_loss
                            }, 'E:/DO_AN/LPG-MI/checkpoints/target_model/{}_{}_{}.tar'.format(args.model, args.mode, args.eps))
                            best_val_loss = avg_val_loss


        return args.epochs, train
    if args.model == 'vgg16_bn' and args.mode == 'non-DP' :
        def train(epochs):
            best_val_loss = float('inf')  # Initialize best validation loss

            for epoch in range(epochs):
                print('\nEpoch: %d' % (epoch+1))
                net.train()
                train_loss = 0
                correct = 0
                total = 0

                for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, desc='Batch')):
                    inputs, targets = inputs.to(device), targets.to(device)
                    feat, outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.mean().item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    writer.add_scalar('training loss',
                                    train_loss,
                                    epoch * len(trainloader) + batch_idx)

                avg_train_loss = train_loss / (batch_idx + 1)
                train_acc = 100. * correct / total

                print(epoch, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (avg_train_loss, train_acc, correct, total))
                print('ε = {} '.format(args.eps))

                # Validation phase
                net.eval()
                val_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(tqdm(testloader, desc='Validation Batch')):
                        inputs, targets = inputs.to(device), targets.to(device)
                        feat, outputs = net(inputs)
                        loss = criterion(outputs, targets)

                        val_loss += loss.mean().item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                avg_val_loss = val_loss / (batch_idx + 1)
                val_acc = 100. * correct / total

                print(epoch, len(testloader), 'Validation Loss: %.3f | Validation Acc: %.3f%% (%d/%d)'
                    % (avg_val_loss, val_acc, correct, total))

                # Save model if validation loss has decreased
                if avg_val_loss < best_val_loss:
                    print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(best_val_loss, avg_val_loss))
                    torch.save({
                        'state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss_min': avg_val_loss
                    }, 'E:/DO_AN/LPG-MI/checkpoints/target_model/{}_{}_{}.tar'.format(args.model, args.mode, args.eps))
                    best_val_loss = avg_val_loss
        return args.epochs, train

def main(epochs, trainf, testf, args):
    for epoch in range(epochs):
        trainf(epoch)
        testf(epoch)
def main_vgg(epochs, trainf, args):
        trainf(epochs)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch DP Training')
    parser.add_argument('--lr', default=0.001,
                        type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int,
                        help='numter of epochs')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--eps', default=2, type=float, help='target epsilon')
    parser.add_argument('--grad_norm', '-gn', default=0.1,
                        type=float, help='max grad norm')
    parser.add_argument('--mode', default='ghost_mixed', help= 'ghost_mixed | non-DP | ghost | non-ghost ')
    parser.add_argument('--model', default='vgg16_bn', type=str)
    parser.add_argument('--mini_bs', type=int, default=8)
    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--data', type=str, default='celeba')
    parser.add_argument('--result_root', default="./result", type=str)
    parser.add_argument('--no_tensorboard', action='store_true', default=False,
                        help='If you dislike tensorboard, set this ``False``. default: True')
    args = parser.parse_args()
    warnings.filterwarnings("ignore")


    if args.model == 'vgg16_bn':
        epochs, trainf= prepare(args)
        main_vgg(epochs,trainf,args)
    else:
        epochs, trainf, testf = prepare(args)
        main(epochs, trainf, testf, args)
