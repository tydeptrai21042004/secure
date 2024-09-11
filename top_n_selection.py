import os
import queue
import shutil
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from argparse import ArgumentParser
from opacus.validators import ModuleValidator
from models.classifiers import *
from tqdm import tqdm

parser = ArgumentParser(description='Reclassify the public dataset with the target model')
parser.add_argument('--model', default='DP_SGD', help='VGG16 | IR152 | FaceNet64 | DP_SGD | NON_DP')
parser.add_argument('--data_name', type=str, default='celeba', help='celeba | ffhq | facescrub')
parser.add_argument('--top_n', type=int, help='the n of top-n selection strategy.')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--save_root', type=str, default='reclassified_public_data')
parser.add_argument('--bs', type= int, default= 32, help='batch size for top n selection')
parser.add_argument('--eps', default=2, type=float, help='target epsilon')
args = parser.parse_args()

class PublicCeleba(torch.utils.data.Dataset):
    def __init__(self, file_path='E:\\DO_AN\\LPG-MI\\data_files\\celeba_ganset.txt',
                 img_root='E:\\DO_AN\\LPG-MI\\datasets\\img_align_celeba', transform=None):
        super(PublicCeleba, self).__init__()
        self.file_path = file_path
        self.img_root = img_root
        self.transform = transform
        self.images = []

        name_list, label_list = [], []

        f = open(self.file_path, "r")
        for line in f.readlines():
            img_name = line.strip()
            self.images.append(os.path.join(self.img_root, img_name))


    def __getitem__(self, index):

        img_path = self.images[index]
        
        img = Image.open(img_path)
        if self.transform != None:
            img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.images)



def top_n_selection(args, T, data_loader, device):
    """
    Top-n selection strategy.
    : args: top-n, save_path
    : T: target model
    : data_loader: dataloader of
    :return:
    """
    print("=> start inference ...")
    print(len(data_loader))
    
    all_images_prob = None
    all_images_path = None
    # get the predict confidence of each image in the public data
    with torch.no_grad():
        for i, (images, img_path) in tqdm(enumerate(data_loader)):
            bs = images.shape[0]
            images = images.to(device)
            logits = T(images)[-1]
            prob = F.softmax(logits, dim=1)  # (bs, 1000)
            prob = prob.cpu()
            if i == 0:
                all_images_prob = prob
                all_images_path = img_path
            else:
                all_images_prob = torch.cat([all_images_prob, prob], dim=0)
                all_images_path = all_images_path + img_path

    print("=> start reclassify ...")
    save_path = os.path.join(args.save_root, args.data_name, args.model + "_top" + str(args.top_n),+ "_top" + str(args.eps) )
    print(" top_n: ", args.top_n)
    print(" save_path: ", save_path)
    # top-n selection
    for class_idx in range(args.num_classes):
        bs = all_images_prob.shape[0]
        ccc = 0
        # maintain a priority queue
        q = queue.PriorityQueue()
        class_idx_prob = all_images_prob[:, class_idx]

        for j in range(bs):
            current_value = float(class_idx_prob[j])
            image_path = all_images_path[j]
            # Maintain a priority queue with confidence as the priority
            if q.qsize() < args.top_n:
                q.put([current_value, image_path])
            else:
                current_min = q.get()
                if current_value < current_min[0]:
                    q.put(current_min)
                else:
                    q.put([current_value, image_path])
# reclassify and move the images
        for m in range(q.qsize()):
            q_value = q.get()
            q_prob = round(q_value[0], 4)
            q_image_path = q_value[1]

            ori_save_path = os.path.join(save_path, str(class_idx))
            if not os.path.exists(ori_save_path):
                os.makedirs(ori_save_path)

            new_image_path = os.path.join(ori_save_path, str(ccc) + '_' + str(q_prob) + '.png')

            shutil.copy(q_image_path, new_image_path)
            ccc += 1


print(args)
print("=> load target model ...")

model_name_T = args.model
if model_name_T.startswith("VGG16"):
    T = VGG16(1000)
    path_T = './checkpoints/target_model/VGG16_88.26.tar'
    print(path_T)
if model_name_T.startswith('DP_SGD') :
    T=  VGG16_bn(1000)
    T = ModuleValidator.fix(T)
    path_T = './checkpoints/target_model/ghost_mixed.tar'
if model_name_T.startswith('NON_DP') :
    T=  VGG16_bn(1000)
    T = ModuleValidator.fix(T)
    path_T = './checkpoints/target_model/non-DP.tar'

device = "cuda:0" if torch.cuda.is_available() else "cpu"
T = torch.nn.DataParallel(T).to(device)
ckp_T = torch.load(path_T, map_location =device)
T.load_state_dict(ckp_T['state_dict'], strict=False)
T.eval()

print("=> load public dataset ...")
if args.data_name == 'celeba':
    re_size = 64
    crop_size = 108
    offset_height = (218 - crop_size) // 2
    offset_width = (178 - crop_size) // 2
    crop = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]
    celeba_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(crop),
        transforms.ToPILImage(),
        transforms.Resize((re_size, re_size)),
        transforms.ToTensor()
    ])
    data_set = PublicCeleba(file_path='E:\\DO_AN\\LPG-MI\\data_files\\celeba_ganset.txt',
                            img_root='E:\\DO_AN\\LPG-MI\\datasets\\img_align_celeba',
                            transform=celeba_transform)
    print(data_set.__getitem__(9))
    data_loader = data.DataLoader(data_set, batch_size=args.bs)
top_n_selection(args, T, data_loader, device)
