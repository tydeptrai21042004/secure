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
import glob
from diffprivlib.mechanisms import Laplace
import timm
import json
from tqdm import tqdm
import time
from opacus.validators import ModuleValidator
from model import VGG16_bn  # Import lớp VGG16_bn từ model.py

def add_laplace_noise_to_gradient(gradients, epsilon, sensitivity=1.0, delta=0.0):
    laplace_mechanism = Laplace(epsilon=epsilon, sensitivity=sensitivity, delta=delta)
    noisy_gradients = [g + laplace_mechanism.randomise(0) for g in gradients.view(-1)]
    return torch.tensor(noisy_gradients, dtype=torch.float32).view(gradients.size())

def prepare_results_dir(args):
    """Tạo thư mục cho kết quả và khởi tạo tensorboard nếu cần thiết."""
    root = os.path.join(args.result_root, args.data, args.model)
    os.makedirs(root, exist_ok=True)
    if not args.no_tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(root)
    else:
        writer = None
    with open(os.path.join(root, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print(json.dumps(args.__dict__, indent=2))
    return writer

def prepare(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(f'Device: {device}')

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

    writer = prepare_results_dir(args)
    print('Arguments saved and Tensorboard initialized.')

    if args.data == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='/content/drive/MyDrive/DO_AN/data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='/content/drive/MyDrive/DO_AN/data', train=False, download=True, transform=transform_test)
    elif args.data == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='/data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='/data', train=False, download=True, transform=transform_test)
    elif args.data == 'celeba':
        print('Preparing CelebA dataset...')
        class PrivateDataset(Dataset):
            def __init__(self, image_loc, label_loc, transform):
                filenames = []
                for root, dirs, files in os.walk(image_loc):
                    for file in files:
                        if file.endswith('.jpg') == True or file.endswith('.png') == True:
                            filenames.append(file)
                self.full_filenames = glob.glob(image_loc+'*/*/*.*')

                label_df = pd.read_csv(label_loc)
                label_df.set_index("filename", inplace=True)
                self.labels = [label_df.loc[filename].values[0] for filename in filenames]
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

        data_dir = 'E:/DO_AN/LPG-MI/datasets/celeba_private_domain'
        label_loc = 'E:/DO_AN/LPG-MI/data_files/subset_train.csv'
        print(f'Loading CelebA dataset from {data_dir} and {label_loc}')
        trainset = PrivateDataset(data_dir, label_loc, transformate(108, 64))
        train_size = math.ceil(0.8 * len(trainset))
        val_size = len(trainset) - train_size
        trainset, testset = random_split(trainset, [train_size, val_size])
        print(len(trainset), len(testset))

    print('Data loaders are being created...')
    trainloader = DataLoader(trainset, batch_size=args.mini_bs, shuffle=True)
    testloader = DataLoader(testset, batch_size=100, shuffle=False)

    print('==> Xây dựng mô hình..', args.model)
    NUM_CLASSES = 10 if args.data == 'CIFAR10' else 100
    if args.model != 'vgg16_bn':
        net = timm.create_model(args.model, pretrained=args.pretrained, num_classes=NUM_CLASSES)
    elif args.model == 'vgg16_bn':
        net = VGG16_bn(NUM_CLASSES)
    net = ModuleValidator.fix(net)
    net.to(device)
    print(f'Model loaded: {args.model}')
    
    # In thông tin về số lượng tham số
    num_params = sum(p.numel() for p in net.parameters())
    print(f'Number of parameters: {num_params}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    n_acc_steps = args.bs // args.mini_bs
    print(f'Batch size: {args.bs}, Mini-batch size: {args.mini_bs}, Accumulation steps: {n_acc_steps}')

    return net, trainloader, testloader, criterion, optimizer, writer, n_acc_steps

def train(args, model, trainloader, testloader, criterion, optimizer, writer, n_acc_steps):
    device = args.device
    model.train()
    
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        print(f'\nEpoch: {epoch}')
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in tqdm(enumerate(trainloader), desc=f"Epoch {epoch+1}", total=len(trainloader)):
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # Thêm nhiễu Laplace vào gradient
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            noisy_grad = add_laplace_noise_to_gradient(param.grad, epsilon=args.eps)
                            param.grad.copy_(noisy_grad)

                # Cập nhật trọng số
                if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    optimizer.virtual_step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")

        epoch_loss = running_loss / len(trainloader.dataset)
        accuracy = 100. * correct / total

        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        if writer:
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Accuracy/train', accuracy, epoch)

    # Đánh giá mô hình
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

def main():
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình học sâu')
    parser.add_argument('--data', default='celeba', type=str, choices=['CIFAR10', 'CIFAR100', 'celeba'])
    parser.add_argument('--model', default='vgg16_bn', type=str)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--bs', default=32, type=int)
    parser.add_argument('--mini_bs', default=8, type=int)
    parser.add_argument('--eps', default=0.1, type=float)
    parser.add_argument('--result_root', default='./results', type=str)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--no_tensorboard', action='store_true')

    args = parser.parse_args()

    model, trainloader, testloader, criterion, optimizer, writer, n_acc_steps = prepare(args)
    train(args, model, trainloader, testloader, criterion, optimizer, writer, n_acc_steps)

if __name__ == '__main__':
    main()
