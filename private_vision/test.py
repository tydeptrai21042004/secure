import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
import os
import math
import pandas as pd
from PIL import Image
from opacus.validators import ModuleValidator
import timm
import json
from tqdm import tqdm
import warnings
from LaplacePrivacyEngine import Laplace  # Đảm bảo rằng bạn nhập đúng lớp Laplace
from model import *

def add_laplace_noise_to_gradient(gradients, epsilon, sensitivity):
    laplace_mechanism = Laplace(epsilon=epsilon, sensitivity=sensitivity)
    noisy_gradients = torch.empty_like(gradients)
    
    # Sinh nhiễu Laplace cho từng phần tử trong gradients
    for idx in range(gradients.numel()):
        noisy_gradients.view(-1)[idx] = gradients.view(-1)[idx] + laplace_mechanism.randomise(0)
    
    return noisy_gradients

def prepare_results_dir(args):
    """Makedir, init tensorboard if required, save args."""
    root = os.path.join(args.result_root, args.data, args.model, args.mode)
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
    # Chuẩn bị dữ liệu
    print('==> Chuẩn bị dữ liệu..')
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    args, writer = prepare_results_dir(args)
    if args.data == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='/content/drive/MyDrive/DO_AN/data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='/content/drive/MyDrive/DO_AN/data', train=False, download=True, transform=transform_test)
    elif args.data == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='/data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='/data', train=False, download=True, transform=transform_test)
    elif args.data == 'celeba':
        class private_dataset(Dataset):
            def __init__(self, image_loc, label_loc, transform):
                label_df = pd.read_csv(label_loc)
                label_df.set_index("filename", inplace=True)
                # List of filenames from the label file
                label_filenames = set(label_df.index.tolist())
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
            def crop(x): return x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
            proc = []
            proc.append(transforms.ToTensor())
            proc.append(transforms.ToPILImage())
            proc.append(transforms.Resize((re_size, re_size)))
            proc.append(transforms.RandomHorizontalFlip())
            proc.append(transforms.ToTensor())
            return transforms.Compose(proc)

        data_dir = 'E:\DO_AN\LPG-MI\datasets\celeba_private_domain'
        label_loc = 'E:\DO_AN\LPG-MI\data_files\subset_train.csv'
        trainset = private_dataset(data_dir, label_loc, transformate(108, 64))
        train_size = math.ceil(0.8 * len(trainset))
        val_size = len(trainset) - train_size

        trainset, testset = random_split(trainset, [train_size, val_size])
        print(len(trainset), len(testset))

    trainloader = DataLoader(trainset, batch_size=args.mini_bs, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.mini_bs, shuffle=False)
    
    print('==> Xây dựng mô hình..', args.model)
    NUM_CLASSES = 10 if args.data == 'CIFAR10' else 100
    if args.model != 'vgg16_bn':
        net = timm.create_model(args.model, pretrained=args.pretrained, num_classes=NUM_CLASSES)
    elif args.model == 'vgg16_bn':
        net = VGG16_bn(2)
        net = ModuleValidator.fix(net)

    net = ModuleValidator.fix(net)
    net.to(device)

    # Tắt tính năng học cho một số tham số cụ thể in các mô hình khác nhau
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

    n_acc_steps = args.bs // args.mini_bs
    # if 'ghost' in args.mode:
    #     laplace_engine = Laplace(
    #             net,
    #             epsilon=args.eps,
    #             sensitivity=args.sen
    #         )
    #     laplace_engine.attach(None)  # Không sử dụng optimizer

    def update_weights(model, lr=0.001):
        """Cập nhật trọng số của mô hình dựa trên gradient hiện tại."""
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param -= lr * param.grad

    def train(epochs):
      best_val_loss = float('inf')  # Initialize best validation loss
      for epoch in range(epochs):
          print('\nEpoch: %d' % (epoch+1))
          net.train()
          train_loss = 0
          correct = 0
          total = 0

          for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), desc=f"Epoch {epoch + 1}", total=len(trainloader)):
              inputs, targets = inputs.to(device), targets.to(device)
              
              feat, outputs = net(inputs)
              loss = criterion(outputs, targets)
              
              # Nếu sử dụng reduction='none', tính trung bình mất mát
              if 'ghost' in args.mode:
                  loss = loss.mean()

              # Tính gradient
              loss.backward()

              # Thêm nhiễu Laplace vào gradient
              with torch.no_grad():
                  for param in net.parameters():
                      if param.grad is not None:
                          noisy_grad = add_laplace_noise_to_gradient(param.grad, epsilon=args.eps, sensitivity=args.sen)
                          param.grad.copy_(noisy_grad)

              # Cập nhật trọng số
              update_weights(net, lr=args.lr)

              train_loss += loss.item()
              _, predicted = outputs.max(1)
              total += targets.size(0)
              correct += predicted.eq(targets).sum().item()
              print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(trainloader)}")
              writer.add_scalar('training loss', train_loss, epoch * len(trainloader) + batch_idx)

          avg_train_loss = train_loss / (batch_idx + 1)
          train_acc = 100. * correct / total

          print(epoch, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (avg_train_loss, train_acc, correct, total))
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
                  
                  # Nếu sử dụng reduction='none', tính trung bình mất mát
                  if 'ghost' in args.mode:
                      loss = loss.mean()

                  val_loss += loss.item()
                  _, predicted = outputs.max(1)
                  total += targets.size(0)
                  correct += predicted.eq(targets).sum().item()

          avg_val_loss = val_loss / (batch_idx + 1)
          val_acc = 100. * correct / total

          print(epoch, len(testloader), 'Validation Loss: %.3f | Validation Acc: %.3f%% (%d/%d)' % (avg_val_loss, val_acc, correct, total))

          # Save model if validation loss has decreased
          if avg_val_loss < best_val_loss:
              print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(best_val_loss, avg_val_loss))
              torch.save({
                  'state_dict': net.state_dict(),
                  'val_loss_min': avg_val_loss
              }, '{}/{}/{}/{}.tar'.format(args.result_root, args.data, args.model, args.mode))
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
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--bs', default=32, type=int, help='batch size')
    parser.add_argument('--eps', default=2, type=float, help='target epsilon')
    parser.add_argument('--grad_norm', '-gn', default=0.1, type=float, help='max grad norm')
    parser.add_argument('--mode', default='ghost_mixed', help='ghost_mixed | non-private | ghost | non-ghost ')
    parser.add_argument('--model', default='vgg16_bn', type=str)
    parser.add_argument('--mini_bs', type=int, default=8)
    parser.add_argument('--pretrained', type=int, default=1)
    parser.add_argument('--data', type=str, default='celeba')
    parser.add_argument('--result_root', default="./test", type=str)
    parser.add_argument('--no_tensorboard', action='store_true', default=False, help='If you dislike tensorboard, set this ``False``. default: True')
    parser.add_argument('--sen', default=10.0, type=float, help='sensitivity parameter for the Laplace mechanism')
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    if args.model == 'vgg16_bn':
        epochs, trainf = prepare(args)
        main_vgg(epochs, trainf, args)
    else:
        epochs, trainf, testf = prepare(args)
        main(epochs, trainf, testf, args)

