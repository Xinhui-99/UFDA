from itertools import permutations, combinations
import torch
import numpy as np
from torch.autograd import Variable
import shutil

def delet_zero_row(x):
    list = []
    num = 0
    for i in range(x.size(0)):
        su = sum(x[i, :])
        if su == 0:
            list.append(i)
            num += 1
    New_x = np.delete(x.cpu(), np.s_[list], axis=0)
    return New_x

def max_value(x,y):
    if x >= y:
        return x
    else:
        return y

def otsu(out_soft):

    out_soft = out_soft.cuda()
    data_label = torch.LongTensor([])
    data_label = data_label.cuda()
    data_label = Variable(data_label)
    _, label = torch.max(out_soft.data, 1)

    data_label.resize_(out_soft.size(0)).copy_(label)
    outclass = torch.cuda.LongTensor(data_label.unsqueeze(1))
    class_one_hot_t = torch.zeros(out_soft.size(0), out_soft.size(1)).scatter_(1, outclass.cpu(), 1).cuda()

    mul_soft_t = 0
    mul_hot_t = 0
    mul_pos_t = 0
    mul_neg_t = 0

    for i in range(out_soft.size(0)):
        mul_soft_t += out_soft[i,]
        mul_pos_t += out_soft[i,] * class_one_hot_t[i, ]
        mul_neg_t += out_soft[i,] * (1 - class_one_hot_t[i, ])
        mul_hot_t += class_one_hot_t[i, ]

    mul_soft_t = torch.div(mul_soft_t, out_soft.size(0))
    mul_pos_t = torch.div(mul_pos_t, mul_hot_t + 0.001)
    mul_neg_t = torch.div(mul_neg_t, (out_soft.size(0) - mul_hot_t + 0.001))

    OTSU_all1 = torch.div(mul_hot_t * (mul_pos_t - mul_soft_t) * (mul_pos_t - mul_soft_t), out_soft.size(0))
    OTSU_all0 = torch.div(
        ((out_soft.size(0) - mul_hot_t) * (mul_neg_t - mul_soft_t) * (mul_neg_t - mul_soft_t)), out_soft.size(0))

    OTSU = (OTSU_all1 + OTSU_all0)

    return OTSU

def multi_delet_zero_row( x):
    z = []
    for j in range(len(x)):
        New_x = delet_zero_row(x[j])
        z.append(New_x)
    return z

def decentralized_training_strategy(communication_rounds, epoch_samples, batch_size, total_epochs):
    """
    Split one epoch into r rounds and perform model aggregation
    :param communication_rounds: the communication rounds in training process
    :param epoch_samples: the samples for each epoch
    :param batch_size: the batch_size for each epoch
    :param total_epochs: the total epochs for training
    :return: batch_per_epoch, total_epochs with communication rounds r
    """
    if communication_rounds >= 1:
        epoch_samples = round(epoch_samples / communication_rounds)
        total_epochs = round(total_epochs * communication_rounds)
        batch_per_epoch = round(epoch_samples / batch_size)
    elif communication_rounds in [0.2, 0.5]:
        total_epochs = round(total_epochs * communication_rounds)
        batch_per_epoch = round(epoch_samples / batch_size)
    else:
        raise NotImplementedError(
            "The communication round {} illegal, should be 0.2 or 0.5".format(communication_rounds))
    return batch_per_epoch, total_epochs

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)


