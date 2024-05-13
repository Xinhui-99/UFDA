import numpy as np
import torch
import math
import pickle
from easydl import *
import numpy as np


def reverse_sigmoid(y):
    return torch.log(y / (1.0 - y + 1e-10) + 1e-10)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def normalize_weight(x, cut=0, expand=False):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val + 1e-10) / (max_val - min_val + 1e-10)
    if expand:
        x = x / torch.mean(x)
        # x = torch.where(x >= cut, x, torch.zeros_like(x))
    return x.detach()


def l2_norm(input, dim=1):
    norm = torch.norm(input, dim=dim, keepdim=True)
    output = torch.div(input, norm)
    return output


def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


def softmax(x, t=1.0):
    exp_x = torch.exp(x / t)
    return exp_x / torch.sum(exp_x)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(args, configs, optimizer, epoch):
    lr = configs["TrainingConfig"]["learning_rate"]
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / configs["TrainingConfig"]["total_epochs"])) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def all_accuracy(args, pre_domainnam, real_domainnam, same_class, output1, target):
    """Computes the accuracy in every predicti class"""
    with torch.no_grad():
        pre_soft, predicted = torch.max(output1.data, 1)
        output_oneht = torch.zeros(predicted.size(0), output1.size(1)).cuda().scatter_(1, predicted.view(-1, 1).cuda(),
                                                                                       1)
        pre_higher = pre_soft >= args.threthod_scores
        pre_lower = pre_soft < args.threthod_scores

        output = output_oneht
        correct_k = 0
        n = 0
        u = 0
        for i in range(len(pre_domainnam)):
            for j in range(len(real_domainnam)):
                if pre_domainnam[i] == real_domainnam[j]:
                    if u == 0:
                        correct_k = output[:, i] * target[:, j]
                        label = target[:, j]
                    else:
                        correct_k = correct_k + output[:, i] * target[:, j]
                        label = label + target[:, j]

                    n += 1
                    u += 1
        w = 0
        for j in range(len(real_domainnam)):
            if real_domainnam[j] in same_class:
                pass
            else:
                if w == 0:
                    label_oht = target[:, j]
                else:
                    label_oht = label_oht + target[:, j]
                w += 1
        m = 0
        predirect = torch.zeros(output.size(0)).float().cuda()  # current outputs

        for i in range(len(pre_domainnam)):
            if pre_domainnam[i] in same_class:
                pass
            else:
                if m == 0:
                    predirect = output[:, i]
                else:
                    predirect = predirect + output[:, i]
                m += 1
        p = 0
        unknown_label = [x for x in real_domainnam if x not in same_class]
        for i in range(len(pre_domainnam)):
            for j in range(len(real_domainnam)):
                if real_domainnam[j] in unknown_label:
                    if pre_domainnam[i] in same_class:
                        if p == 0:
                            correct_uk = (output[:, i] * target[:, j])
                        else:
                            correct_uk = correct_uk + (output[:, i] * target[:, j])
                        p += 1

        CR = (correct_k[pre_higher]).sum()
        CU = (correct_uk[pre_lower]).sum()

        correct_unk = (predirect * label_oht).sum()

        correct = (correct_k.sum() + correct_unk) / output.size(0)
        correct_all = (CR + correct_unk + CU) / output.size(0)

        return correct, correct_all


def all_real_accuracy(args, pre_domainnam, real_domainnam, same_class, output1, target):
    """Computes the accuracy in every predicti class"""
    with torch.no_grad():
        pre_soft, predicted = torch.max(output1.data, 1)
        output_oneht = torch.zeros(predicted.size(0), output1.size(1)).cuda().scatter_(1, predicted.view(-1, 1).cuda(),
                                                                                       1)
        pre_higher = pre_soft >= args.threthod_scores
        pre_lower = pre_soft < args.threthod_scores

        output = output_oneht
        n = 0
        u = 0
        for i in range(len(pre_domainnam)):
            for j in range(len(real_domainnam)):
                if pre_domainnam[i] == real_domainnam[j]:
                    if u == 0:
                        correct_k = output[:, i]
                        out_target = target[:, j]
                        acc_known = output[:, i] * target[:, j]
                    else:
                        out_target = torch.cat((out_target.cuda(), target[:, j]), 0)
                        correct_k = torch.cat((correct_k.cuda(), output[:, i]), 0)
                        acc_known = torch.cat((acc_known.cuda(), output[:, i] * target[:, j]), 0)

                    n += 1
                    u += 1
        w = 0
        for j in range(len(real_domainnam)):
            if real_domainnam[j] in same_class:
                pass
            else:
                if w == 0:
                    label_oht = target[:, j]
                    label_oht_all = target[:, j]

                else:
                    label_oht = torch.cat((label_oht.cuda(), target[:, j]), 0)
                    label_oht_all = label_oht_all + target[:, j]

                w += 1
        m = 0
        predirect = torch.zeros(output.size(0)).float().cuda()  # current outputs

        for i in range(len(pre_domainnam)):
            if pre_domainnam[i] in same_class:
                pass
            else:
                if m == 0:
                    predirect = output[:, i]
                    predirect_all = output[:, i]
                else:
                    predirect = torch.cat((predirect.cuda(), output[:, i]), 0)
                    predirect_all = predirect_all + output[:, i]

                m += 1

        p = 0
        unknown_label = [x for x in real_domainnam if x not in same_class]
        for j in range(len(real_domainnam)):
            for i in range(len(pre_domainnam)):
                if real_domainnam[j] in unknown_label:
                    if pre_domainnam[i] in same_class:
                        if p == 0:
                            correct_uk = (output[:, i] * target[:, j])
                        else:
                            correct_uk = correct_uk.cuda() + output[:, i] * target[:, j]
                        p += 1

        unknown_all = predirect_all*label_oht_all
        acc_known = acc_known.view(predicted.size(0), -1)
        out_target = out_target.view(predicted.size(0), -1)
        predirect = predirect.view(predicted.size(0), -1)
        label_oht = label_oht.view(predicted.size(0), -1)
        correct_uk = correct_uk.view(predicted.size(0), -1)
        unknown_all = unknown_all.view(predicted.size(0), -1)


        acc_k = acc_known.sum(0)
        out_tar = out_target.sum(0)
        acc_know_all = acc_k / out_tar

        acc_know = (acc_know_all.sum()) / len(acc_know_all)
        acc_unknow = unknown_all.sum() / label_oht.sum()
        acc_test = (len(same_class) * acc_know + acc_unknow) / (len(same_class) + 1)

        acc_thre_know = ((acc_known[pre_higher]).sum(0) / out_target.sum(0)).sum() / len(acc_know_all)
        acc_thre_unknow = (unknown_all.sum() + correct_uk[pre_lower].sum()) / label_oht.sum()
        acc_real_unknow = (unknown_all[pre_lower].sum() + correct_uk[pre_lower].sum()) / label_oht.sum()
        acc_real_test = (len(same_class) * acc_thre_know + acc_real_unknow) / (len(same_class) + 1)

        acc_thre_test = (len(same_class) * acc_thre_know + acc_thre_unknow) / (len(same_class) + 1)

        return acc_know, acc_unknow, acc_test, acc_thre_know, \
            acc_thre_unknow, acc_thre_test, acc_real_unknow, acc_real_test


def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples = num_samples + labels.size(0)
    return total / num_samples


def sigmoid_rampup(current, rampup_length, exp_coe=5.0):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-exp_coe * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

