from configular import *
from easydl import *
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler
from lib.utils.randaugment import RandomAugment
from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

def listdir_nohidden(path):
    """List non-hidden items in a directory.
    Args:
         path (str): directory path.
    """
    return [f for f in os.listdir(path) if not f.startswith('.')]

def read_data(data_path, class_index):
    data_paths = []
    data_labels = []
    if args.data.dataset.name == "Office":
        domain_dir = path.join(data_path,"images")
    else:
        domain_dir = path.join(data_path)
    class_names = listdir_nohidden(domain_dir)
    class_names.sort()
    #print(class_names)
    i=0
    class_n = []
    for class_name_choose in class_names:
        if i in class_index:
            class_n.append(class_name_choose)
        i += 1

    for label, class_name in enumerate(class_n):
        class_dir = path.join(domain_dir, class_name)
        item_names = listdir_nohidden(class_dir)
        for item_name in item_names:
            item_path = path.join(class_dir, item_name)
            data_paths.append(item_path)
            data_labels.append(label)
    return data_paths, data_labels, class_n

'''
assume classes across domains are the same.
[0 1 ......................................................................... N - 1]
|----common classes --||----source private classes --||----target private classes --|
'''
a, b, c, d, e, f, j = args.data.dataset.n_share1, args.data.dataset.n_source_private1, args.data.dataset.n_share2, args.data.dataset.n_source_private2, \
                      args.data.dataset.n_share_common, args.data.dataset.n_total, args.data.dataset.n_private_common
source_class_name = []
number_class = []
source_num = 0
if len(args.data.dataset.domains) == 4 :
    source_num = 4
    g, h, l, m = args.data.dataset.n_share3, args.data.dataset.n_source_private3, args.data.dataset.n_share4, args.data.dataset.n_source_private4
    f = f - (e + b + d + h + m)
    e = int((a + c + g + l - e)/2)
    target_private_classes_num = f
    common_classes1 = [i for i in range(a - e)]
    source_private_classes1 = [i + a + c + g + l - 2 * e for i in range(b)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[0] != "S":
        if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[0] != "R":
            source_private_classes1 = [i + a + c + g + l - 2 * e + 5 for i in range(b)]
    common_classes2 = [i + a for i in range(c - 2 * e)]
    source_private_classes2 = [i + a + c + g + l - 2 * e + b for i in range(d)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[1] != "S":
        if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[1] != "R":
            source_private_classes2 = [i + a + c + g + l - 2 * e + b + 5 for i in range(d)]
    common_classes3 = [i + a + c - e for i in range(g - e)]
    source_private_classes3 = [i + a + c + g + l - 2 * e + b + d for i in range(h-j)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[2] != "S":
        if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[2] != "R":
            source_private_classes3 = [i + a + c + g + l - 2 * e + b + d + 5 for i in range(h - j)]
    common_classes4 = [i + a + c + g - e for i in range(l - e)]
    source_private_classes4 = [i + a + c + g + l - 2 * e + b + d + h for i in range(m - j)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[3] != "S":
        if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[3] != "R":
            source_private_classes4 = [i + a + c + g + l - 2 * e + b + d + h + 5 for i in range(m - j)]
    target_private_classes = [i + a + c + g + l - 2 * e + b + d + h + m for i in range(f)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.target != "S":
        if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.target != "R":
            target_private_classes = [i + a + c + g + l - 2 * e + b + d + h + m + 5 for i in range(f)]
    source_classes1 = common_classes1 + source_private_classes1
    source_classes2 = common_classes2 + source_private_classes2
    source_classes3 = common_classes3 + source_private_classes3
    source_classes4 = common_classes4 + source_private_classes4
    source_class_name.append(source_classes1)
    source_class_name.append(source_classes2)
    source_class_name.append(source_classes3)
    source_class_name.append(source_classes4)
    number_class.append(len(source_classes1))
    number_class.append(len(source_classes2))
    number_class.append(len(source_classes3))
    number_class.append(len(source_classes4))
    source_classes = common_classes1 + common_classes2 + common_classes3 + common_classes4 + source_private_classes1 + source_private_classes2 + source_private_classes3 + source_private_classes4
    target_classes = common_classes1 + common_classes2 + common_classes3 + common_classes4 + target_private_classes
    print('source_classes1 =', source_classes1, '\n', 'source_classes2 =', source_classes2, '\n', 'source_classes3 =', source_classes3, '\n', 'source_classes4 =', source_classes4
          , '\n', 'source_classes =', source_classes, '\n', 'target_classes =', target_classes)

elif len(args.data.dataset.domains) == 3:
    source_num = 3
    g, h = args.data.dataset.n_share3, args.data.dataset.n_source_private3
    f = f - (e + b + d + h)
    e = int((a + c + g - e)/2)
    target_private_classes_num = f
    common_classes1 = [i for i in range(a - e)]
    source_private_classes1 = [i + a + c + g - 2 * e for i in range(b)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[0] != "S":
        if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[0] != "R":
            source_private_classes1 = [i + a + c + g - 2 * e + 5 for i in range(b)]
    common_classes2 = [i + a for i in range(c - 2 * e)]
    source_private_classes2 = [i + a + c + g - 2 * e + b for i in range(d)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[1] != "S":
        if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[1] != "R":
            source_private_classes2 = [i + a + c + g - 2 * e + b + 5 for i in range(d)]
    common_classes3 = [i + a + c - e for i in range(g - e)]
    source_private_classes3 = [i + a + c + g - 2 * e + b + d for i in range(h-j)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[2] != "S":
        if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[2] != "R":
            source_private_classes3 = [i + a + c + g - 2 * e + b + d + 5 for i in range(h - j)]
    shared_private_classes23 = [i + a + c + g - 2 * e + b + d - j for i in range(j)]
    common_shared_classes12 = [i + a - e for i in range(e)]
    common_shared_classes23 = [i + a + c - 2 * e for i in range(e)]
    target_private_classes = [i + a + c + g - 2 * e + b + d + h for i in range(f)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.target != "S":
        if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.target != "R":
            target_private_classes = [i + a + c + g - 2 * e + b + d + h + 5 for i in range(f)]
    source_classes1 = common_classes1 + common_shared_classes12 + source_private_classes1
    source_classes2 = common_shared_classes12 + common_classes2 + common_shared_classes23 + source_private_classes2
    source_classes3 = common_shared_classes23 + common_classes3 + shared_private_classes23 + source_private_classes3
    source_classes = common_classes1 + common_shared_classes12 + common_classes2 + common_shared_classes23 + common_classes3 + source_private_classes1 + source_private_classes2 + shared_private_classes23 + source_private_classes3
    source_class_name.append(source_classes1)
    source_class_name.append(source_classes2)
    source_class_name.append(source_classes3)
    number_class.append(len(source_classes1))
    number_class.append(len(source_classes2))
    number_class.append(len(source_classes3))
    target_classes = common_classes1 + common_shared_classes12 + common_classes2 + common_shared_classes23 + common_classes3 + target_private_classes
    print('source_classes1 =', source_classes1, '\n', 'source_classes2 =', source_classes2, '\n', 'source_classes3 =', source_classes3, '\n', 'source_classes =', source_classes, '\n', 'target_classes =', target_classes)
elif len(args.data.dataset.domains) == 2:
    source_num = 2
    f = f - (e + b + d - j)
    e = a + c - e
    target_private_classes_num = f
    common_classes1 = [i for i in range(a - e)]
    source_private_classes1 = [i + a + c - e for i in range(b-j)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[0] != "S":
        if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[0] != "R":
            source_private_classes1 = [i + a + c - e + 5 for i in range(b-j)]
    common_classes2 = [i + a for i in range(c - e)]
    source_private_classes2 = [i + a + c - e + b for i in range(d-j)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[1] != "S":
        if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[1] != "R":
            source_private_classes2 = [i + a + c - e + b + 5 for i in range(d-j)]
    common_shared_classes = [i + a - e for i in range(e)]
    private_shared_classes = [i + a + c - e + b - j for i in range(j)]
    target_private_classes = [i + a + c - e + b + d - j for i in range(f)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.target != "S":
        if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.target != "R":
            target_private_classes = [i + a + c - e + b + d - j + 5 for i in range(f)]
    source_classes1 = common_classes1 + common_shared_classes + source_private_classes1 + private_shared_classes
    source_classes2 = common_shared_classes + common_classes2 + private_shared_classes + source_private_classes2
    source_classes = common_classes1 + common_shared_classes + common_classes2 + source_private_classes1 + private_shared_classes + source_private_classes2
    target_classes = common_classes1 + common_shared_classes + common_classes2 + target_private_classes
    source_class_name.append(source_classes1)
    source_class_name.append(source_classes2)
    number_class.append(len(source_classes1))
    number_class.append(len(source_classes2))
    print('source_classes1 =', source_classes1, '\n', 'source_classes2 =', source_classes2, '\n', 'source_classes =', source_classes, '\n', 'target_classes =', target_classes)
else:
    source_num = 1
    f = f - (e + b - j)
    e = a - e
    target_private_classes_num = f
    common_classes1 = [i for i in range(a - e)]
    source_private_classes1 = [i + a - e for i in range(b-j)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[0] != "S":
        if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.domains[0] != "R":
            source_private_classes1 = [i + a - e + 5 for i in range(b-j)]
    target_private_classes = [i + a - e + b - j for i in range(f)]
    if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.target != "S":
        if args.data.dataset.name == 'VisDA+ImageCLEF-DA' and args.data.dataset.target != "R":
            target_private_classes = [i + a - e + b - j + 5 for i in range(f)]
    source_classes1 = common_classes1 + source_private_classes1
    source_classes = source_classes1
    number_class.append(len(source_classes1))
    source_class_name.append(source_classes1)
    target_classes = common_classes1 + target_private_classes

    print('source_classes =', source_classes, '\n', 'target_classes =', target_classes)

class Augmentention(Dataset):
    def __init__(self, data_paths, true_labels):

        self.data_paths = data_paths
        self.labels = true_labels
        self.weak_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.75, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])])
        self.strong_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.75, 1)),
                transforms.RandomHorizontalFlip(),
                RandomAugment(3, 5),
                transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")

        each_image_w = self.weak_transform(img)
        each_image_s = self.strong_transform(img)
        labels = self.labels[index]
        return each_image_w, each_image_s, labels, index

class add_index(Dataset):
    def __init__(self, data_paths, true_labels):

        self.data_paths = data_paths
        self.labels = true_labels
        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.75, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        images = self.transform(img)
        labels = self.labels[index]
        return images, labels, index

class add_index_test(Dataset):
    def __init__(self, data_paths, true_labels):

        self.data_paths = data_paths
        self.labels = true_labels
        self.transform = transforms.Compose(
        [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        images = self.transform(img)
        labels = self.labels[index]
        return images, labels, index

target_data_paths, target_data_labels, target_class_name = read_data(target_file,  target_classes)
source1_data_paths, source1_data_labels, source1_class_name = read_data(source1_file, source_classes1)
source1_train_dataset = add_index(source1_data_paths, source1_data_labels)
source1_test_dataset = add_index_test(source1_data_paths, source1_data_labels)
target_train_dataset = Augmentention(target_data_paths, target_data_labels)
target_test_ds = add_index_test(target_data_paths, target_data_labels)

classes1 = source1_train_dataset.labels

freq1 = Counter(classes1)
class_weight1 = {x : 1.0 / freq1[x] if args.data.dataloader.class_balance else 1.0 for x in freq1}
source1_weights = [class_weight1[x] for x in source1_train_dataset.labels]
sampler1 = WeightedRandomSampler(source1_weights, len(source1_train_dataset.labels))
source_train_dl = []
source_test_dl = []

source1_train_dl = DataLoader(dataset=source1_train_dataset, batch_size=int(args.data.dataloader.batch_size),
                             sampler=sampler1, num_workers=args.data.dataloader.data_workers, drop_last=False)
source1_test_dl = DataLoader(dataset=source1_test_dataset, batch_size=int(args.data.dataloader.batch_size), shuffle=False,
                             num_workers=1, drop_last=False)
source_train_dl.append(source1_train_dl)
source_test_dl.append(source1_test_dl)

target_train_dl = DataLoader(dataset=target_train_dataset, batch_size=args.data.dataloader.batch_size,shuffle=True,
                             num_workers=args.data.dataloader.data_workers,
                             sampler=None, drop_last=False)
target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                             num_workers=1, sampler=None,
                            drop_last=False)

if len(args.data.dataset.domains) >=2:
    source2_data_paths, source2_data_labels, source2_class_name = read_data(data_path=source2_file, class_index=source_classes2)
    source2_train_dataset = add_index(source2_data_paths, source2_data_labels)
    source2_test_dataset = add_index_test(source2_data_paths, source2_data_labels)
    #print(source2_train_dataset)

    classes2 = source2_train_dataset.labels
    freq2 = Counter(classes2)
    class_weight2 = {x: 1.0 / freq2[x] if args.data.dataloader.class_balance else 1.0 for x in freq2}
    source2_weights = [class_weight2[x] for x in source2_train_dataset.labels]
    sampler2 = WeightedRandomSampler(source2_weights, len(source2_train_dataset.labels))

    source2_train_dl = DataLoader(dataset=source2_train_dataset, batch_size=int(args.data.dataloader.batch_size),
                                  sampler=sampler2, num_workers=args.data.dataloader.data_workers, drop_last=False)
    source2_test_dl = DataLoader(dataset=source2_test_dataset, batch_size=int(args.data.dataloader.batch_size), shuffle=False,
                                 num_workers=1, drop_last=False)
    source_train_dl.append(source2_train_dl)
    source_test_dl.append(source2_test_dl)

if len(args.data.dataset.domains) >= 3:
    source3_data_paths, source3_data_labels, source3_class_name = read_data(data_path=source3_file, class_index=source_classes3)
    source3_train_dataset = add_index(source3_data_paths, source3_data_labels)
    source3_test_dataset = add_index_test(source3_data_paths, source3_data_labels)

    classes3 = source3_train_dataset.labels
    freq3 = Counter(classes3)
    class_weight3 = {x: 1.0 / freq3[x] if args.data.dataloader.class_balance else 1.0 for x in freq3}
    source3_weights = [class_weight3[x] for x in source3_train_dataset.labels]
    sampler3 = WeightedRandomSampler(source3_weights, len(source3_train_dataset.labels))

    source3_train_dl = DataLoader(dataset=source3_train_dataset, batch_size=int(args.data.dataloader.batch_size),
                                  sampler=sampler3, num_workers=args.data.dataloader.data_workers, drop_last=False)
    source3_test_dl = DataLoader(dataset=source3_test_dataset, batch_size=int(args.data.dataloader.batch_size), shuffle=False,
                                 num_workers=1, drop_last=False)
    source_train_dl.append(source3_train_dl)
    source_test_dl.append(source3_test_dl)

if len(args.data.dataset.domains) >= 4:
    source4_data_paths, source4_data_labels, source4_class_name = read_data(data_path=source4_file,  class_index=source_classes4)
    source4_train_dataset = add_index(source4_data_paths, source4_data_labels)
    source4_test_dataset = add_index_test(source4_data_paths, source4_data_labels)

    classes4 = source4_train_dataset.labels
    freq4 = Counter(classes4)
    class_weight4 = {x: 1.0 / freq4[x] if args.data.dataloader.class_balance else 1.0 for x in freq4}
    source4_weights = [class_weight4[x] for x in source4_train_dataset.labels]
    sampler4 = WeightedRandomSampler(source4_weights, len(source4_train_dataset.labels))

    source4_train_dl = DataLoader(dataset=source4_train_dataset, batch_size=int(args.data.dataloader.batch_size),
                                  sampler=sampler4, num_workers=args.data.dataloader.data_workers, drop_last=False)
    source4_test_dl = DataLoader(dataset=source4_test_dataset, batch_size=int(args.data.dataloader.batch_size), shuffle=False,
                                 num_workers=1, drop_last=False)

    source_train_dl.append(source4_train_dl)
    source_test_dl.append(source4_test_dl)


    