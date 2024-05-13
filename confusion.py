# -*-coding:utf-8-*-
from sklearn.metrics import confusion_matrix
import torch
import torch.utils.data
import pandas as pd
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def histo_gram(configs, args, Z_s, predicted,  source_label_set, same_class, epoch, j):
    known = 0
    unknown = 0
    k = 0
    k_un = 0
    pre = 0
    pre_un = 0
    Z_s = Z_s.cuda()
    for i in range(len(predicted)):

        Z_s0 = np.expand_dims(Z_s[i,:].cpu(), 0)
        Z_s0 = torch.tensor(Z_s0)
        pred = np.expand_dims(predicted[i].cpu(), 0)
        pred = torch.tensor(pred)
        if source_label_set[predicted[i]] in same_class:
            if k == 0:
                known = Z_s0
                pre = pred
                k += 1
            else:
                known = torch.cat((known.cuda(), Z_s0.cuda()), 0)
                pre = torch.cat((pre.cuda(), pred.cuda()))

        else:
            if k_un == 0:
                unknown = Z_s0
                pre_un = pred
                k_un += 1
            else:
                unknown = torch.cat((unknown.cuda(), Z_s0.cuda()), 0)
                pre_un = torch.cat((pre_un.cuda(), pred.cuda()))

    known = known.cpu()
    unknown = unknown.cpu()
    loss_known = -np.log(known[np.arange(len(known)), pre] + 1e-8).cuda() #F.cross_entropy(known, pre, reduce = False)
    loss_unknown = -np.log(unknown[np.arange(len(unknown)), pre_un] + 1e-8).cuda()  #F.cross_entropy(unknown, pre_un, reduce = False)
    hist_k, bins_k = np.histogram(loss_known.cpu(), bins=100)
    hist_uk, bins_uk = np.histogram(loss_unknown.cpu(), bins=100)

    fig1, ax1 = plt.subplots()
    ax1.bar(bins_k[:-1], hist_k, width=np.diff(bins_k))
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of self_entropy')
    fig1.savefig('./matrix/known_distribution_%s%d%d.png' % (configs['data']['dataset']['target'],  j, epoch), format='png')

    fig2, ax2 = plt.subplots()
    ax2.bar(bins_uk[:-1], hist_uk, width=np.diff(bins_uk))
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of self_entropy')
    fig2.savefig('./matrix/unkonwn_distribution_%s%d%d.png' % (configs['data']['dataset']['target'],  j, epoch), format='png')

def confusion_target_score(configs, args, label, predict, epoch, label_set_union, t_class_name_real):

    _, labels = torch.max(label.data, 1)
    _, predicts = torch.max(predict.data, 1)
    actual_labels = []
    predicted_labels = []

    for i in range(len(labels)):
        actual_labels.append(t_class_name_real[labels[i]])
        predicted_labels.append(label_set_union[predicts[i]])

    # 计算混淆矩阵
    confusion_matrix = np.zeros(
        (len(set(actual_labels + predicted_labels)), len(set(actual_labels + predicted_labels))))
    label_set = sorted(set(actual_labels + predicted_labels))
    for i in range(len(actual_labels)):
        if actual_labels[i] in label_set and predicted_labels[i] in label_set:
            confusion_matrix[label_set.index(actual_labels[i]), label_set.index(predicted_labels[i])] += 1

    # 可视化混淆矩阵
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap=plt.cm.Blues)

    # 添加标签和颜色条
    ax.set_xticks(np.arange(len(label_set)))
    ax.set_yticks(np.arange(len(label_set)))
    ax.set_xticklabels(label_set)
    ax.set_yticklabels(label_set)
    ax.set_ylabel('Target True label')
    ax.set_xlabel('Predicted label')
    plt.colorbar(im)
    # 在单元格中添加数量
    for i in range(len(label_set)):
        for j in range(len(label_set)):
            text = ax.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="w")

    # 设置图表属性
    ax.set_title("Confusion Matrix_target_update")
    # 保存混淆矩阵为jpg格式
    plt.savefig('./matrix/confusion_target_score_%s%d.png' % (configs['data']['dataset']['target'], epoch), format='png')
    # 显示图形
    #plt.show()

def histo_gramall(configs, args, Z_s, predicted, label_set_union, same_class, epoch, j):

    known = 0
    unknown = 0
    k = 0
    k_un = 0
    pre = 0
    pre_un = 0
    Z_s = Z_s.cuda()
    for i in range(len(predicted)):
        Z_s0 = np.expand_dims(Z_s[i,:].cpu(), 0)
        Z_s0 = torch.tensor(Z_s0)
        pred = np.expand_dims(predicted[i].cpu(), 0)
        pred = torch.tensor(pred)
        if label_set_union[predicted[i]] in same_class:
            if k == 0:
                known = Z_s0
                pre = pred
                k += 1
            else:
                known = torch.cat((known.cuda(), Z_s0.cuda()), 0)
                pre = torch.cat((pre.cuda(), pred.cuda()))
        else:
            if k_un == 0:
                unknown = Z_s0
                pre_un = pred
                k_un += 1
            else:
                unknown = torch.cat((unknown.cuda(), Z_s0.cuda()), 0)
                pre_un = torch.cat((pre_un.cuda(), pred.cuda()))

    loss_known = F.cross_entropy(known, pre, reduce = False)
    loss_unknown = F.cross_entropy(unknown, pre_un, reduce = False)
    hist_k, bins_k = np.histogram(loss_known.cpu(), bins=100)
    hist_uk, bins_uk = np.histogram(loss_unknown.cpu(), bins=100)

    fig1, ax1 = plt.subplots()
    ax1.bar(bins_k[:-1], hist_k, width=np.diff(bins_k))
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of self_entropy_all')
    fig1.savefig('./matrix/known_pre_%s%d%d.png' % (configs['data']['dataset']['target'],  j, epoch), format='png')

    fig2, ax2 = plt.subplots()
    ax2.bar(bins_uk[:-1], hist_uk, width=np.diff(bins_uk))
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of self_entropy_all')
    fig2.savefig('./matrix/unkonwn_pre_%s%d%d.png' % (configs['data']['dataset']['target'],  j, epoch), format='png')

def histo_all_known(configs, args, Z_s, predicted, label_set_union,
                      same_class, label_all, t_class_name_real, epoch, j):

    known = 0
    unknown = 0
    k = 0
    q = 0
    pre = 0
    pre_un = 0
    Z_s = Z_s.cuda()
    _, pre_label = torch.max(Z_s.data, 1)

    for i in range(len(predicted)):
        Z_s0 = np.expand_dims(Z_s[i, :].cpu(), 0)
        Z_s0 = torch.tensor(Z_s0)
        pred = np.expand_dims(pre_label[i].cpu(), 0)
        pred = torch.tensor(pred)

        if t_class_name_real[label_all[i]] in same_class:
            if k == 0:
                known = Z_s0
                pre = pred
                k += 1
            else:
                known = torch.cat((known.cuda(), Z_s0.cuda()),0)
                pre = torch.cat((pre.cuda(), pred.cuda()))

        else:
            if q == 0:
                unknown = Z_s0
                pre_un = pred
                q += 1
            else:
                unknown = torch.cat((unknown.cuda(), Z_s0.cuda()),0)
                pre_un = torch.cat((pre_un.cuda(), pred.cuda()))

    loss_known = F.cross_entropy(known, pre, reduce = False)
    loss_unknown = F.cross_entropy(unknown, pre_un, reduce = False)
    known = known.cpu()
    unknown = unknown.cpu()
    max_known, _ = torch.max(known.data, 1)
    max_unknown, _ = torch.max(unknown.data, 1)

    # Compute the histograms
    hist1, bin_edges = np.histogram(loss_known.cpu(), bins=40)
    hist2, _ = np.histogram(loss_unknown.cpu(), bins=bin_edges)

    hist3, bin_edges2 = np.histogram(max_known.cpu(), bins=40)
    hist4, _ = np.histogram(max_unknown.cpu(), bins=bin_edges2)
     # Plot the histograms
    fig, ax = plt.subplots()
    ax.bar(bin_edges[:-1], hist1,  width=np.diff(bin_edges), color='blue',  alpha=0.5,label='known')
    ax.bar(bin_edges[:-1], hist2,  width=np.diff(bin_edges), color='red', alpha=0.5,label='unknown')
    # Add a legend and axis labels
    ax.legend()
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    fig.savefig('./matrix/known_un_ce_%s%d%d.png' % (configs['data']['dataset']['target'],  j, epoch), format='png')

    fig1, ax1 = plt.subplots()
    ax1.bar(bin_edges2[:-1], hist3, width=np.diff(bin_edges2), color='blue', alpha=0.5, label='known')
    ax1.bar(bin_edges2[:-1], hist4, width=np.diff(bin_edges2), color='red', alpha=0.5, label='unknown')
    # Add a legend and axis labels
    ax1.legend()
    ax1.set_xlabel('Values')
    ax1.set_ylabel('Frequency')
    fig1.savefig('./matrix/known_un_max_%s%d%d.png' % (configs['data']['dataset']['target'], j, epoch), format='png')

    #plt.show()

def histo_gramall_target(configs, args, Z_s, predicted, label_set_union,
                      same_class, label_all, t_class_name_real, epoch, j):

    known_real = 0
    unknown_real = 0
    unknown_false = 0
    real_false = 0
    false_real = 0
    k = 0
    l = 0
    p = 0
    q = 0
    m = 0
    Z_s = Z_s.cuda()
    _, pre_label = torch.max(Z_s.data, 1)

    for i in range(len(predicted)):
        Z_s0 = np.expand_dims(Z_s[i, :].cpu(), 0)
        Z_s0 = torch.tensor(Z_s0)

        if t_class_name_real[label_all[i]] in same_class:
            if label_set_union[pre_label[i]] in same_class:
                if label_set_union[pre_label[i]] != t_class_name_real[label_all[i]]:
                    if k == 0:
                        false_real = Z_s0
                        k += 1
                    else:
                        false_real = torch.cat((false_real.cuda(), Z_s0.cuda()), 0)
                else:
                    if label_set_union[pre_label[i]] == t_class_name_real[label_all[i]]:
                        if q == 0:
                            known_real = Z_s0
                            q += 1
                        else:
                            known_real = torch.cat((known_real.cuda(), Z_s0.cuda()), 0)
            else:
                if m == 0:
                    real_false = Z_s0
                    m += 1
                else:
                    real_false = torch.cat((real_false.cuda(), Z_s0.cuda()), 0)

        else:
            if label_set_union[pre_label[i]] in same_class:
                if l == 0:
                    unknown_false = Z_s0
                    l += 1
                else:
                    unknown_false = torch.cat((unknown_false.cuda(), Z_s0.cuda()), 0)
            else:
                if p == 0:
                    unknown_real = Z_s0
                    p += 1
                else:
                    unknown_real = torch.cat((unknown_real.cuda(), Z_s0.cuda()), 0)

    known_real = known_real.cpu()
    unknown_real = unknown_real.cpu()
    loss_known, _ = torch.max(known_real.data, 1)
    hist_k, bins_k = np.histogram(loss_known.cpu(), bins=100)

    loss_unknown, _ = torch.max(unknown_real.data, 1)
    hist_unk, bins_unk = np.histogram(loss_unknown.cpu(), bins=100)

    loss_false, _ = torch.max(unknown_false.data, 1)
    hist_fal, bins_fal = np.histogram(loss_false.cpu(), bins=100)

    loss_real_false, _ = torch.max(real_false.data, 1)
    hist_reafal, bins_reafal = np.histogram(loss_real_false.cpu(), bins=100)

    loss_false_real, _ = torch.max(false_real.data, 1)
    hist_falrea, bins_falrea = np.histogram(loss_false_real.cpu(), bins=100)

    fig1, ax1 = plt.subplots()
    ax1.bar(bins_k[:-1], hist_k, width=np.diff(bins_k))
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of max_score_know')
    fig1.savefig('./matrix/known_real_%s%d%d.png' % (configs['data']['dataset']['target'],  j, epoch), format='png')

    fig2, ax2 = plt.subplots()
    ax2.bar(bins_unk[:-1], hist_unk, width=np.diff(bins_unk))
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of max_score_unk')
    fig2.savefig('./matrix/unknown_real_%s%d%d.png' % (configs['data']['dataset']['target'],  j, epoch), format='png')

    fig3, ax3 = plt.subplots()
    ax3.bar(bins_fal[:-1], hist_fal, width=np.diff(bins_fal))
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of max_score_unknow_fal')
    fig3.savefig('./matrix/unknown_fal_%s%d%d.png' % (configs['data']['dataset']['target'],  j, epoch), format='png')

    fig4, ax4 = plt.subplots()
    ax4.bar(bins_reafal[:-1], hist_reafal, width=np.diff(bins_reafal))
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of max_score_rea_fal')
    fig4.savefig('./matrix/known_real_fal_%s%d%d.png' % (configs['data']['dataset']['target'],  j, epoch), format='png')

    fig5, ax5 = plt.subplots()
    ax5.bar(bins_falrea[:-1], hist_falrea, width=np.diff(bins_falrea))
    ax5.set_xlabel('Value')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distribution of max_score_fal_rea')
    fig5.savefig('./matrix/known_fal_real_%s%d%d.png' % (configs['data']['dataset']['target'],  j, epoch), format='png')

def confusion_target(configs, args, label, predict, epoch, label_set_union, t_class_name_real):

    _, labels = torch.max(label.data, 1)
    _, predicts = torch.max(predict.data, 1)
    actual_labels = []
    predicted_labels = []

    for i in range(len(labels)):
        actual_labels.append(t_class_name_real[labels[i]])
        predicted_labels.append(label_set_union[predicts[i]])

    # 计算混淆矩阵
    confusion_matrix = np.zeros(
        (len(set(actual_labels + predicted_labels)), len(set(actual_labels + predicted_labels))))
    label_set = sorted(set(actual_labels + predicted_labels))
    for i in range(len(actual_labels)):
        if actual_labels[i] in label_set and predicted_labels[i] in label_set:
            confusion_matrix[label_set.index(actual_labels[i]), label_set.index(predicted_labels[i])] += 1

    # 可视化混淆矩阵
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap=plt.cm.Blues)

    # 添加标签和颜色条
    ax.set_xticks(np.arange(len(label_set)))
    ax.set_yticks(np.arange(len(label_set)))
    ax.set_xticklabels(label_set)
    ax.set_yticklabels(label_set)
    ax.set_ylabel('Target True label')
    ax.set_xlabel('Predicted label')
    plt.colorbar(im)

    # 在单元格中添加数量
    for i in range(len(label_set)):
        for j in range(len(label_set)):
            text = ax.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="w")

    # 设置图表属性
    ax.set_title("Confusion Matrix_target")
    # 保存混淆矩阵为jpg格式
    plt.savefig('./matrix/confusion_target_%s%d.png' % (configs['data']['dataset']['target'], epoch), format='png')
    # 显示图形
    #plt.show()

def confusion_target_all(configs, args, label, predict, epoch, label_set_union, t_class_name_real):

    _, labels = torch.max(label.data, 1)
    _, predicts = torch.max(predict.data, 1)
    actual_labels = []
    predicted_labels = []

    for i in range(len(labels)):
        actual_labels.append(t_class_name_real[labels[i]])
        predicted_labels.append(label_set_union[predicts[i]])

    # 计算混淆矩阵
    confusion_matrix = np.zeros(
        (len(set(actual_labels + predicted_labels)), len(set(actual_labels + predicted_labels))))
    label_set = sorted(set(actual_labels + predicted_labels))
    for i in range(len(actual_labels)):
        if actual_labels[i] in label_set and predicted_labels[i] in label_set:
            confusion_matrix[label_set.index(actual_labels[i]), label_set.index(predicted_labels[i])] += 1

    # 可视化混淆矩阵
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap=plt.cm.Blues)

    # 添加标签和颜色条
    ax.set_xticks(np.arange(len(label_set)))
    ax.set_yticks(np.arange(len(label_set)))
    ax.set_xticklabels(label_set)
    ax.set_yticklabels(label_set)
    ax.set_ylabel('Target True label')
    ax.set_xlabel('Predicted label')
    plt.colorbar(im)
    # 在单元格中添加数量
    for i in range(len(label_set)):
        for j in range(len(label_set)):
            text = ax.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="w")

    # 设置图表属性
    ax.set_title("Confusion Matrix_target_update")
    # 保存混淆矩阵为jpg格式
    plt.savefig('./matrix/confusion_targetall_%s%d.png' % (configs['data']['dataset']['target'], epoch), format='png')
    # 显示图形
    #plt.show()
def confusion_target_test(configs, args, label, predict, epoch, label_set_union, t_class_name_real):

    _, labels = torch.max(label.data, 1)
    _, predicts = torch.max(predict.data, 1)
    actual_labels = []
    predicted_labels = []

    for i in range(len(labels)):
        actual_labels.append(t_class_name_real[labels[i]])
        predicted_labels.append(label_set_union[predicts[i]])

    # 计算混淆矩阵
    confusion_matrix = np.zeros(
        (len(set(actual_labels + predicted_labels)), len(set(actual_labels + predicted_labels))))
    label_set = sorted(set(actual_labels + predicted_labels))
    for i in range(len(actual_labels)):
        if actual_labels[i] in label_set and predicted_labels[i] in label_set:
            confusion_matrix[label_set.index(actual_labels[i]), label_set.index(predicted_labels[i])] += 1

    # 可视化混淆矩阵
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap=plt.cm.Blues)

    # 添加标签和颜色条
    ax.set_xticks(np.arange(len(label_set)))
    ax.set_yticks(np.arange(len(label_set)))
    ax.set_xticklabels(label_set)
    ax.set_yticklabels(label_set)
    ax.set_ylabel('Target True label')
    ax.set_xlabel('Predicted label')
    plt.colorbar(im)

    # 在单元格中添加数量
    for i in range(len(label_set)):
        for j in range(len(label_set)):
            text = ax.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="w")

    # 设置图表属性
    ax.set_title("Confusion Matrix_target_test")
    # 保存混淆矩阵为jpg格式
    plt.savefig('./matrix/confusion_target_test%s%d.png' % (configs['data']['dataset']['target'], epoch), format='png')
    # 显示图形
    #plt.show()
def confusion_source(configs, args, label, predict, epoch, source_class_name, target_classes, s_number):

    _, labels = torch.max(label.data, 1)
    _, predicts = torch.max(predict.data, 1)

    actual_labels = []
    predicted_labels = []

    for i in range(len(labels)):
        actual_labels.append(target_classes[labels[i]])
        predicted_labels.append(source_class_name[predicts[i]])

    # 计算混淆矩阵
    confusion_matrix = np.zeros(
        (len(set(actual_labels + predicted_labels)), len(set(actual_labels + predicted_labels))))
    label_set = sorted(set(actual_labels + predicted_labels))
    for i in range(len(actual_labels)):
        if actual_labels[i] in label_set and predicted_labels[i] in label_set:
            confusion_matrix[label_set.index(actual_labels[i]), label_set.index(predicted_labels[i])] += 1

    # 可视化混淆矩阵
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap=plt.cm.Blues)

    # 添加标签和颜色条
    ax.set_xticks(np.arange(len(label_set)))
    ax.set_yticks(np.arange(len(label_set)))
    ax.set_xticklabels(label_set)
    ax.set_yticklabels(label_set)
    ax.set_xlabel('Source True label')
    ax.set_ylabel('Predicted label')
    plt.colorbar(im)

    # 在单元格中添加数量
    for i in range(len(label_set)):
        for j in range(len(label_set)):
            text = ax.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="w")

    # 设置图表属性
    ax.set_title("Confusion Matrix_source")
    # 保存图形为JPEG格式
    plt.savefig('./matrix/confusion_source_%s%d%d.png' % (configs['data']['dataset']['target'], s_number, epoch), format='png')
    # 显示图形
    #plt.show()
