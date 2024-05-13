import torch
import torch.nn as nn
from lib.utils.federated_utils import *
from tensorboardX import SummaryWriter
from lib.utils.utils_algo import *
import time
import torch.distributed as dist

def source_train(logger,  train_dloader_list
                 , args, model_list, classifier_list, batch_per_epoch,optimizer_list,
                 classifier_optimizer_list, epoch):

    task_criterion = nn.CrossEntropyLoss().cuda()
    for model in model_list:
        model.train()
    for classifier in classifier_list:
        classifier.train()
    # If communication rounds <1,
    # then we perform parameter aggregation after (1/communication_rounds) epochs
    # If communication rounds >=1:
    # then we extend the training epochs and use fewer samples in each epoch.

    # Train model locally on source domains
    n = 0
    for train_dloader, model, classifier, optimizer, classifier_optimizer in zip(train_dloader_list,
                                                                                 model_list,
                                                                                 classifier_list,
                                                                                 optimizer_list,
                                                                                 classifier_optimizer_list):
        n += 1

        for i, (image_s, label_s, _) in enumerate(train_dloader):

            image_s = image_s.cuda()
            label_s = label_s.long().cuda()
            # each source domain do optimize
            feature_s = model(image_s)
            output_s = classifier(feature_s)

            task_loss_s = task_criterion(output_s, label_s)  # .requires_grad_()
            with OptimizerManager(
                    [optimizer, classifier_optimizer]):
                loss = task_loss_s
                loss.backward()
    return model_list, classifier_list

def pre_train_t(logger,  test_dloader, args, model, label_set_union, num_class):

    with torch.no_grad():
        totaldt = 0
        model.eval()
        ntrain_b = len(test_dloader.dataset.labels)

        outputs_soft = torch.zeros(ntrain_b, len(label_set_union)).float()  # temporal outputs
        output = torch.zeros(ntrain_b, len(label_set_union)).float()  # current outputs
        onehot = torch.zeros(ntrain_b, num_class).float()  # current outputs

        for i,  (image, _, label0, index0) in enumerate(test_dloader):
            # if i >= batch_per_epoch: (images_w, images_s, true_labels, index)
            # break
            image_t = image.clone().detach().cuda()  # torch.tensor(image_t).cuda()
            label = label0.clone().detach().cuda()  # torch.tensor(index).cuda()
            index = index0.clone().detach().cuda()  # torch.tensor(index).cuda()

            # each source domain do optimize .cuda()
            feature_t, score_prot = model(image_t, args, eval_only=True)
            out__soft_t = torch.softmax(feature_t, dim=1)
            _, predicted = torch.max(out__soft_t.data, 1)

            out__soft_score = torch.softmax(score_prot, dim=1)
            _, predicted_score = torch.max(out__soft_score.data, 1)
            output_oneht_score = torch.zeros(predicted_score.size(0), out__soft_score.size(1)).cuda().scatter_(1, predicted_score.view(-1, 1).cuda(), 1)
            totaldt += label0.size(0)
            label_oneht = torch.zeros(label.size(0), num_class).cuda().scatter_(1, label.view(-1,1).cuda(), 1)

            if i == 0:
                output_t_score = output_oneht_score
                label_oneht_mul = label_oneht
                output_soft = out__soft_score
                ind = index

            else:
                output_t_score = torch.cat((output_t_score.cuda(), output_oneht_score), 0)
                label_oneht_mul = torch.cat((label_oneht_mul.cuda(), label_oneht), 0)
                output_soft = torch.cat((output_soft.cuda(), out__soft_score), 0)
                ind = torch.cat((ind, index), 0)

    outputs_soft[ind.cpu(), :] = output_soft.cpu()  # current outputs
    output[ind.cpu(), :] = output_t_score.cpu()  # current outputs
    onehot[ind.cpu(),:] = label_oneht_mul.cpu()  # current outputs

    return outputs_soft, output, onehot

# Validation function
def visual_source(logger, epoch, train_dloader, args,  Z_s,models, classifiers, num_class_t, numclass):

    ind = []
    ntrain_b = len(train_dloader.dataset.labels)  # len(ind_s[0].indices)
    # Testing the model2
    with torch.no_grad():
        current_domain_index = 0
        z_s = []
        m = 0
        out_soft = []

        for model, classifier in zip(models, classifiers):
            totaldt = 0
            z_s0 = torch.zeros( ntrain_b, numclass[m]).float()  # temporal outputs
            outputs_s0 = torch.zeros( ntrain_b, numclass[m]).float()   # current outputs
            label_all = (torch.zeros(ntrain_b)).to(torch.int64) # current outputs

            for i,  (image, _, label, index0) in enumerate(train_dloader):
                #if i >= batch_per_epoch:
                    #break
                image_t = image.clone().detach().cuda()  # torch.tensor(image_t).cuda()
                index = index0.clone().detach().cuda()  # torch.tensor(index).cuda()
                label_t = label.clone().detach().cpu()

                # each source domain do optimize .cuda()
                feature_t = model(image_t)
                output_t = classifier(feature_t)
                out__soft_t = torch.softmax(output_t, dim=1)
                totaldt += index.size(0)

                if i == 0:
                    out_soft = out__soft_t
                    ind = index
                    labels = label_t

                else:
                    out_soft = torch.cat((out_soft.cuda(), out__soft_t), 0)
                    ind = torch.cat((ind, index), 0)
                    labels = torch.cat((labels, label_t), 0)
            out_soft1 = out_soft[0:totaldt]
            out_soft1 = torch.tensor(out_soft1)
            out_soft1 = out_soft1.view(totaldt, -1)
            outputs_s0[ind.cpu(), :] = out_soft1.cpu()  # current outputs
            labels1 = labels[0:totaldt]
            for i in range(ntrain_b):
                label_all[ind[i]] = int(labels1[i])
            label_all_t = torch.zeros(label_all.size(0), num_class_t).cuda().scatter_(1, label_all.view(-1, 1).cuda(), 1)

            # update temporal ensemble
            Z_s[m] = args.alpha_z * Z_s[m].cpu() + (1. - args.alpha_z) * outputs_s0[ :, :]
            z_s0[:, :] = Z_s[m] * (1. / (1. - args.alpha_z ** (epoch + 1)))
            z_s.append(z_s0)
            m += 1

    return z_s, label_all_t

# Validation function
def visual_ss(logger, epoch, train_dloaders, args,  Z_ss,models, classifiers,batch_per_epoch, numclass):

    out_put = []
    ind = []

    # Testing the model2
    with torch.no_grad():
        z_ss = []
        m = 0
        for model, classifier in zip(models, classifiers):

            totaldt =0
            ntrain_b = len(train_dloaders[m].dataset.labels)
            print(ntrain_b)
            z_ss0 = torch.zeros( ntrain_b, numclass[m]).float()  # temporal outputs
            outputs_s0 = torch.zeros( ntrain_b, numclass[m]).float()   # current outputs

            for i, (image, _, index0) in enumerate(train_dloaders[m]):
                #if i >= batch_per_epoch:
                    #break
                image_t = image.clone().detach().cuda()#torch.tensor(image_t).cuda()
                index = index0.clone().detach().cuda()# torch.tensor(index).cuda()

                # each source domain do optimize .cuda()
                feature_t = model(image_t)
                output_t = classifier(feature_t)
                out__soft_t = torch.softmax(output_t, dim=1)
                totaldt +=index.size(0)

                if i == 0:
                    out_soft = out__soft_t
                    ind = index
                else:
                    out_soft = torch.cat((out_soft.cuda(), out__soft_t), 0)
                    ind = torch.cat((ind, index), 0)

            out_put.append(out_soft)
            out_soft = out_soft[0:totaldt]
            out_soft = out_soft.view(totaldt, -1)
            outputs_s0[ind.cpu(), :] = out_soft.cpu()  # current outputs

            # update temporal ensemble
            Z_ss[m] = args.alpha_z * Z_ss[m].cpu() + (1. - args.alpha_z) * outputs_s0[ :, :]
            z_ss0[:, :] = Z_ss[m] * (1. / (1. - args.alpha_z ** (epoch + 1)))
            z_ss.append(z_ss0)
            m += 1

    return  Z_ss, z_ss

def pre_test_t(logger,  test_dloader, args, model, num_class):

    with torch.no_grad():
        totaldt = 0
        model.eval()
        for i,  (image, label0, _) in enumerate(test_dloader):
            # if i >= batch_per_epoch:
            # break
            image_t = image.clone().detach().cuda()  # torch.tensor(image_t).cuda()
            label = label0.clone().detach().cuda()  # torch.tensor(index).cuda()
            # each source domain do optimize .cuda()
            feature_t , _= model(image_t, args, eval_only=True)
            out__soft_t = torch.softmax(feature_t, dim=1)
            _, predicted = torch.max(out__soft_t.data, 1)
            output_oneht = torch.zeros(predicted.size(0), out__soft_t.size(1)).cuda().scatter_(1, predicted.view(-1, 1).cuda(), 1)
            totaldt += label0.size(0)
            label_oneht = torch.zeros(label.size(0), num_class).cuda().scatter_(1, label.view(-1,1).cuda(), 1)

            if i == 0:
                output_t = output_oneht
                label_oneht_mul = label_oneht
                output_soft = out__soft_t

            else:
                output_t = torch.cat((output_t.cuda(), output_oneht), 0)
                label_oneht_mul = torch.cat((label_oneht_mul.cuda(), label_oneht), 0)
                output_soft = torch.cat((output_soft.cuda(), out__soft_t), 0)

    return output_soft, output_t, label_oneht_mul

def pre_test_s(logger,  test_dloader, args, model, classifier,num_class):

    with torch.no_grad():
        totaldt = 0
        label_oneht = 0
        output_t = 0
        model.eval()
        for i,  (image, label0, _) in enumerate(test_dloader):
            # if i >= batch_per_epoch:
            # break
            image_s = image.clone().detach().cuda()  # torch.tensor(image_t).cuda()
            label = label0.clone().detach().cuda()  # torch.tensor(index).cuda()
            # each source domain do optimize .cuda()
            feature_s = model(image_s)
            output_s = classifier(feature_s)
            out__soft_t = torch.softmax(output_s, dim=1)
            _, predicted = torch.max(out__soft_t.data, 1)
            output_oneht = torch.zeros(predicted.size(0), out__soft_t.size(1)).cuda().scatter_(1, predicted.view(-1, 1).cuda(), 1)
            totaldt += label0.size(0)
            label_one = torch.zeros(label.size(0), num_class).cuda().scatter_(1, label.view(-1,1).cuda(), 1)

            if i == 0:
                output_t = output_oneht
                label_oneht = label_one
            else:
                output_t = torch.cat((output_t.cuda(), output_oneht), 0)
                label_oneht = torch.cat((label_oneht .cuda(), label_one), 0)

    return output_t, label_oneht

def visual_st(logger, epoch, train_dloaders, args, model,batch_per_epoch, numclass , batch_size):

    ind = []
    output_st = []
    # Testing the model2

    with torch.no_grad():
        for n in range (len(train_dloaders)):
            ntrain_b = len(train_dloaders[n].dataset.labels)
            totaldt = 0
            outputs_t = torch.zeros(ntrain_b, numclass).float().cuda()  # current outputs
            model.eval()
            for i, (image, _, index0) in enumerate(train_dloaders[n]):
                # if i >= batch_per_epoch:
                # break
                image_t = image.clone().detach().cuda()#torch.tensor(image_t).cuda()
                index = index0.clone().detach().cuda()# torch.tensor(index).cuda()

                # each source domain do optimize .cuda()
                feature_t,_ = model(image_t, args, eval_only=True)
                out__soft_t = torch.softmax(feature_t, dim=1)
                totaldt += index.size(0)

                if i == 0:
                    out_soft = out__soft_t
                    ind = index
                else:
                    out_soft = torch.cat((out_soft.cuda(), out__soft_t), 0)
                    ind = torch.cat((ind, index), 0)

            out_soft = out_soft[0:totaldt]
            out_soft = out_soft.view(totaldt, -1)
            outputs_t[ind, :] = out_soft  # current outputs
            output_st.append(outputs_t)

    return  output_st

# Validation function
def visual_target(logger, epoch, train_dloader, args, model, numclass
                 ,batch_size):

    ind = []
    # Testing the model2
    with torch.no_grad():

        ntrain_b = len(train_dloader.dataset.labels)
        totaldt = 0
        outputs_t = torch.zeros(ntrain_b, numclass).float().cuda()  # current outputs
        out_soft = torch.zeros(batch_size, numclass)
        model.eval()

        for i, (image, _,  _, index0) in enumerate(train_dloader):
            #if i >= batch_per_epoch:
                #break
            image_t = image.clone().detach().cuda()  # torch.tensor(image_t).cuda()
            index = index0.clone().detach().cuda()  # torch.tensor(index).cuda()

            # each source domain do optimize .cuda()
            feature_t,_ = model(image_t, args, eval_only=True)
            out__soft_t = torch.softmax(feature_t, dim=1)
            totaldt += index.size(0)

            if i == 0:
                out_soft = out__soft_t
                ind = index

            else:
                out_soft = torch.cat((out_soft.cuda(), out__soft_t), 0)
                ind = torch.cat((ind, index), 0)

        out_soft = out_soft[0:totaldt]
        out_soft = out_soft.view(totaldt, -1)
        outputs_t[ind, :] = out_soft  # current outputs

    return  outputs_t

def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()
