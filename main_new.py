from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import random
import builtins
random.seed(1)
import torch.optim
from lib.utils.federated_utils import *
from lib.utils.utils_algo import *
import sys
from configular import *
from torch import optim
from lib.utils.utils_loss import partial_loss, SupConLoss
from model.model_t import PiCO
from model.model_s import TotalNet, Classifier
from data import *
from model.resnet_t import *
import tensorboard_logger as tb_logger
from train.train_new import *
from sklearn.mixture import GaussianMixture
import torch.backends.cudnn as cudnn
from confusion import *
import argparse
import torch.utils.data
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import yaml
import torch.multiprocessing
from lib.utils.loggings import get_logger
import torch.distributed as dist
import os

os.environ['CUDA_VISIBLE_DEVICES']='3'

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    # Creating log directory
    try:
        os.makedirs(os.path.join('log'))
        os.makedirs(os.path.join('matrix'))
    except OSError:
        pass

    file = open(r"./{}".format(argss.config))
    configs = yaml.full_load(file)
    # set the visible GPU
    cudnn.benchmark = True

    # Setting random seed
    if argss.manualSeed is None:
        argss.manualSeed = random.randint(1, 10000)
    random.seed(argss.manualSeed)
    torch.manual_seed(argss.manualSeed)
    torch.cuda.manual_seed_all(argss.manualSeed)
    if argss.dist_url == "env://" and argss.world_size == -1:
        argss.world_size = int(os.environ["WORLD_SIZE"])

    argss.distributed = argss.world_size > 1 or argss.multiprocessing_distributed
    model_path = 'ds_{ds}_lr_{lr}_loss_weight_{loss_weight}_loss_penalty_{loss_penalty}_alpha_z_{lw}_{arch}_heir_{prot_start}'.format(
                                            ds=configs["data"]["dataset"]["name"],
                                            lr=configs["TrainingConfig"]["learning_rate_begin"],
                                            loss_weight=argss.loss_weight,
                                            loss_penalty=argss.loss_penalty,
                                            lw=argss.alpha_z,
                                            arch=configs["data"]["dataset"]["target"],
                                            prot_start=argss.prot_start)

    argss.exp_dir = os.path.join(argss.exp_dir, model_path)
    if not os.path.exists(argss.exp_dir):
        os.makedirs(argss.exp_dir)
    ngpus_per_node = torch.cuda.device_count()

def main(args=argss):
    # set the dataloader list, model list, optimizer list, optimizer schedule list
    cudnn.benchmark = True
    file = open(r"./{}".format(args.config))
    configs = yaml.full_load(file)
    writer = SummaryWriter(log_dir=args.exp_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    if configs["misc"]["gpus"] is not None:
        print("Use GPU: {} for training".format(configs["misc"]["gpu_id"]))
    if args.multiprocessing_distributed and configs["misc"]["gpus"]  != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    distributed = 1 if args.distributed else 0
    if distributed == 1:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node #+ configs["misc"]["gpus"]

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    args.distributed = None #args.world_size > 1 or args.multiprocessing_distributed
    # create model
    t_logger = tb_logger.Logger(logdir=os.path.join(args.exp_dir, 'tensorboard'), flush_secs=2)

    models = []
    classifiers = []
    optimizers = []
    classifier_optimizers = []
    optimizer_schedulers = []

    # Creating data loaders
    for i in range(len(source_train_dl)):
        models.append(
            TotalNet(configs["model"]["source1_model"], args.bn_momentum,
                     pretrained=configs["model"]["pretrained"],
                     data_parallel=args.data_parallel).cuda())
        classifiers.append(
            Classifier(configs["model"]["source1_model"], args.data_parallel, classes = number_class[i]).cuda())

    # ===================optimizer
    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
    for model in models:
        optimizers.append(
            OptimWithSheduler(
                torch.optim.SGD(model.parameters(), momentum=configs["train"]["momentum"],
                                lr=configs["TrainingConfig"]["learning_rate_begin"]/10,
                                weight_decay=configs["train"]["weight_decay"],
                                nesterov=True), scheduler))
    for classifier in classifiers:
        classifier_optimizers.append(OptimWithSheduler(
         optim.SGD(classifier.parameters(), lr=configs["TrainingConfig"]["learning_rate_begin"], weight_decay=configs["train"]["weight_decay"],
                      momentum=configs["train"]["momentum"], nesterov=True), scheduler))

    # begin train
    print("Begin the {} time's training, Dataset:{}".format(args.train_time, configs["data"]["dataset"]["name"]))
    # train model
    logger = get_logger( configs["data"]["dataset"]["name"])
    batch_per_epoch, total_epochs = decentralized_training_strategy(
        communication_rounds=configs["UFDAConfig"]["communication_rounds"],
        epoch_samples=configs["TrainingConfig"]["epoch_samples"],
        batch_size=configs["data"]["dataloader"]["batch_size"],
        total_epochs=configs["TrainingConfig"]["total_epochs"])

    communication_rounds = configs["UFDAConfig"]["communication_rounds"]
    source_num = len(models[0:])
    ntrain = len(target_train_dl.dataset.labels) # len(ind_s[0].indices)
    Z_s0 = []
    Z_st0 = []
    Z_ss0 = []
    for i in range(source_num):
        ntrain_s = len(source_train_dl[i].dataset.labels)  # len(ind_s[0].indices)
        Z_st = torch.zeros(ntrain_s, len(target_classes)).float().cuda()  # intermediate values
        Z_st0.append(Z_st)
        Z_s = torch.zeros(ntrain, number_class[i]).float().cuda()  # intermediate values
        Z_s0.append(Z_s)
        Z_ss = torch.zeros(ntrain_s, number_class[i]).float().cuda()  # intermediate values
        Z_ss0.append(Z_ss)

    label_set_union = []
    same_class = []
    for j in range(source_num):  # 3
        name = source_class_name[j]
        for k in range(number_class[j]):  # 8-6-6
            if name[k] in target_classes:
                if name[k] in same_class:
                    pass
                else:
                    same_class.append(name[k])
    torch.cuda.empty_cache()
    label_set_union = pretrain(args, logger, models, classifiers, source_train_dl, target_test_dl, optimizers, Z_s0, classifier_optimizers, writer,
             source_class_name, communication_rounds, batch_per_epoch, total_epochs,target_classes, label_set_union, len(target_classes))
    torch.cuda.empty_cache()
    model = PiCO(args, configs, SupConResNet, num_class_t = len(label_set_union))
    optimizer = torch.optim.SGD(model.parameters(), configs["TrainingConfig"]["learning_rate"],
                                momentum=configs["train"]["momentum"],
                                weight_decay=configs["train"]["weight_decay"])

    optimizer_schedulers.append(
        CosineAnnealingLR(optimizer, total_epochs,
                          eta_min=configs["TrainingConfig"]["learning_end"]))

    conf_ema = torch.zeros(ntrain, len(label_set_union)).float().cuda()  # intermediate values

    conf_c = 1.0
    torch.cuda.empty_cache()
    fed_train(args, configs, logger, t_logger, communication_rounds, source_train_dl, batch_per_epoch,total_epochs,models, classifiers,
              optimizers, classifier_optimizers, optimizer, model, conf_ema, conf_c, number_class, source_class_name, ntrain, label_set_union,
              source_num, Z_s0, target_train_dl, target_test_dl, writer, Z_ss0, target_classes, len(target_classes), same_class)
    torch.cuda.empty_cache()
    # 强制结束整个程序的执行
    sys.exit()

def fed_train(args, configs, logger, t_logger, communication_rounds, train_dloaders, batch_per_epoch, total_epochs,models, classifiers,
              optimizers, classifier_optimizers, optimizer,  model, conf_ema, conf_c, numclass_source,
              source_class_name, ntrain, label_set_union,
              source_num, Z_s0, train_loaders, test_loaders, writer, Z_ss0, t_class_name_real, t_classes_number_real, same_class):

    best_all_pse = 0
    best_all_new = 0
    best_acc_test = 0

    for epoch in range(args.pre_epochs, total_epochs):
        start_upd_prot = epoch >= args.prot_start

        if args.test:
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            acc_know, acc_unknow, acc_test, acc_thre_know, acc_thre_unknow, acc_thre_test, acc_real_unknow, acc_real_test = \
                all_real_accuracy(args, label_set_union, t_class_name_real, same_class, partial_label_all0_z0,
                                  label_all_t)
            logger.info(
                ' Epoch/%d,acc_know: %4f acc_unknow: %4f acc_test: %4f acc_thre_know: %4f '
                'acc_thre_unknow: %4f acc_thre_test: %4f acc_real_unknow: %4f acc_real_test: %4f %%' % (
                    epoch, acc_know, acc_unknow, acc_test, acc_thre_know, acc_thre_unknow, acc_thre_test,
                    acc_real_unknow, acc_real_test))

        logger.info('Start Validation of epoch {}.'.format(epoch))
        logger.info('Epoch:' + str(epoch + 1) + '/' + str(total_epochs))
        models, classifiers = source_train(logger, train_dloaders, args, models,
                                           classifiers, batch_per_epoch, optimizers, classifier_optimizers, epoch)

        Z_s0, label_all_t = visual_source(logger, epoch + 1, train_loaders, args, Z_s0, models, classifiers,
                                          t_classes_number_real, numclass=numclass_source)

        partial_label_target = torch.zeros(ntrain, len(label_set_union)).float().cuda()  # intermediate values
        partial_label_target_soft = torch.zeros(ntrain, len(label_set_union)).float().cuda()
        partial_label_all = torch.zeros(ntrain, len(label_set_union)).float().cuda()
        partial_label_one = torch.zeros(ntrain, len(label_set_union)).float().cuda()
        output_newall_soft =  torch.zeros(ntrain, len(label_set_union)).float().cuda()

        out_soft_all = []
        self_ec = torch.zeros(source_num).float().cuda()
        task_criterion = nn.CrossEntropyLoss().cuda()

        for j in range(source_num):  # 3
            out_soft_t = torch.softmax(Z_s0[j], dim=1).cuda()
            soft_pre, predicted = torch.max(out_soft_t.data, 1)
            out_soft_all.append(out_soft_t)
            task_loss_t = task_criterion(Z_s0[j].cuda(), predicted.cuda())
            self_ec[j] = task_loss_t

        domain_class =  torch.zeros(source_num, len(label_set_union)).float().cuda()
        for f in range(len(label_set_union)):  # 8-6-6
            for j in range(source_num):
                for m in range(numclass_source[j]):
                    if source_class_name[j][m] == label_set_union[f]:
                        domain_class[j,f] =1.0
        add = domain_class.sum(0)
        for f in range(len(label_set_union)):  # 8-6-6
            if add[f] != 1:
                indices = [i for i in range(len(domain_class[:, f])) if domain_class[i, f] == 1]
                domain_weight = F.softmax(self_ec[indices]/0.2).cuda() #
                k = 0
                for o in indices:  # 8-6-6
                    f_inx = [i for i in range(len(source_class_name[o])) if source_class_name[o][i] == label_set_union[f]]
                    output_newall_soft[:, f] =  output_newall_soft[:, f] + domain_weight[k] * out_soft_all[o][:, f_inx[0]]
                    k += 1

        i = 0
        for f in range(len(label_set_union)):  # 8-6-6
            if add[f] == 1:
                indices = [i for i in range(len(domain_class[:, f])) if domain_class[i, f] == 1]
                for h in range(numclass_source[indices[0]]):
                    if source_class_name[indices[0]][h] == label_set_union[f]:
                        if i == 0:
                            class_s = torch.tensor(f).unsqueeze(0)
                            single = out_soft_all[indices[0]][:,h].unsqueeze(0)
                            i += 1
                        else:
                            class_s = torch.cat((class_s, torch.tensor(f).unsqueeze(0) ), 0)
                            single = torch.cat((single, out_soft_all[indices[0]][:,h].unsqueeze(0) ), 0)

        # 计算softmax值
        single_soft = single.t()
        output_newall_soft[:, class_s] = single_soft  

        for j in range(source_num):  # 3
            source_label_set = source_class_name[j]
            out_soft_t = torch.softmax(Z_s0[j], dim=1)
            soft_pre, predicted = torch.max(out_soft_t.data, 1)
            output_one = torch.ones(predicted.size(0), numclass_source[j]).cuda()
            output_oneht = torch.zeros(predicted.size(0), numclass_source[j]).cuda().scatter_(1, predicted.view(-1, 1).cuda(),1)

            for k in range(numclass_source[j]):  # 8-6-6
                for f in range(len(label_set_union)):  # 8-6-6
                    if source_label_set[k] == label_set_union[f]:
                        partial_label_target[:, f] = partial_label_target[:, f] + output_oneht[:, k]
                        partial_label_target_soft[:, f] = partial_label_target_soft[:, f] + out_soft_t[:, k].cuda()
                        partial_label_one[:, f] = partial_label_one[:, f] + output_one[:, k]

        partial_label_all0_z0 = output_newall_soft

        for i in range(partial_label_all0_z0.size(0)):
            partial_label_all0_z0[i, :] = softmax(partial_label_all0_z0[i, :], t=0.07) # partial_label_all0_z0[i, :]#

        _, label_all = torch.max(label_all_t.data, 1)
        _, predicted_all_rewei = torch.max(partial_label_all0_z0.data, 1)
        histo_all_known(configs, args, partial_label_all0_z0, predicted_all_rewei, label_set_union,same_class, label_all, t_class_name_real, epoch, 0)

        if start_upd_prot:
            output_t_score, _, target_train = pre_train_t(logger, train_loaders, args, model, label_set_union,
                                                          t_classes_number_real)
            _, Result_label = torch.max(output_t_score.data, 1)
            Result_label = Result_label.reshape(-1, 1).cpu()
            gmm = GaussianMixture(n_components=2, covariance_type='full').fit(Result_label)
            labels = gmm.predict_proba(Result_label)
            pred = labels > 0.9
            index_all = range(Result_label.size(0))
            big_score = pred.nonzero()[0]
            small_score = np.delete(index_all, big_score)
            big_score = torch.from_numpy(big_score)
            small_score = torch.from_numpy(small_score)
            small_score = small_score.cpu()
            big_score = big_score.cpu()
        else:
            small_score = []
            big_score = []

        tempY = partial_label_all0_z0.sum(dim=1).unsqueeze(1).repeat(1, partial_label_all0_z0.shape[1])
        confidence = partial_label_all0_z0.float() / tempY
        conf_ema = conf_ema.cuda()
        loss_fn = partial_loss(confidence, configs, conf_ema, conf_c, epoch)
        loss_cont_fn = SupConLoss()

        if start_upd_prot:
            conf, confc = train(train_loaders, model, loss_fn, loss_cont_fn, partial_label_target, conf, optimizer,
                         epoch, args, configs, t_logger,writer,
                         batch_per_epoch, small_score, big_score, start_upd_prot)
        else:
            conf, confc = train(train_loaders, model, loss_fn, loss_cont_fn, partial_label_target, partial_label_all0_z0, optimizer,
                         epoch, args, configs, t_logger,writer,
                         batch_per_epoch, small_score, big_score, start_upd_prot)

        loss_fn.set_conf_ema_m(epoch, args)
        label_set_new = re_pseudo(args, logger, train_dloaders, batch_per_epoch, models, classifiers,
                                  model, numclass_source, source_class_name, label_set_union, writer,
                                  source_num, Z_s0, Z_ss0, train_loaders, epoch, t_class_name_real,
                                  t_classes_number_real, same_class)
        label_set_pre = label_set_new #list(set(label_set_new) | set(label_set_pre))
        acc_pse_all_z0, acc_pse_all_all = all_accuracy(args,label_set_union, t_class_name_real, same_class, partial_label_all0_z0, label_all_t)
        acc_know, acc_unknow, acc_test, acc_thre_know, acc_thre_unknow, acc_thre_test,acc_real_unknow,acc_real_test = \
            all_real_accuracy(args,label_set_union, t_class_name_real, same_class, partial_label_all0_z0, label_all_t)
        writer.add_scalars("Train/acc_pseu", {'acc_know': acc_know, 'acc_unknow': acc_unknow,  'acc_test': acc_test,
                                              'acc_thre_know': acc_thre_know,  'acc_thre_unknow': acc_thre_unknow,  'acc_thre_test': acc_thre_test, 'acc_real_unknow': acc_real_unknow, 'acc_real_test': acc_real_test}, epoch)

        if start_upd_prot:
            acc_all_new, correct_all = all_accuracy(args,label_set_union, t_class_name_real, same_class, conf, label_all_t)
            acc_know, acc_unknow, acc_test, acc_thre_know, acc_thre_unknow, acc_thre_test,acc_real_unknow,acc_real_test = \
                all_real_accuracy(args,label_set_union, t_class_name_real, same_class, conf, label_all_t)

            if acc_all_new > best_all_new:
                best_all_new = acc_all_new
            _, conf_all = torch.max(conf.data, 1)
            histo_all_known(configs, args, conf, conf_all, label_set_union,same_class, label_all, t_class_name_real, epoch, 2)

        if acc_pse_all_z0 > best_all_pse:
            best_all_pse = acc_pse_all_z0

        output_soft,output_test, target_test = pre_test_t(logger, test_loaders, args, model, t_classes_number_real)
        logger.info(' Epoch/%d, label_set_new: %s  %%' % (epoch, label_set_pre))
        logger.info(' Epoch/%d, label_set_real: %s  %%' % (epoch, t_class_name_real))
        _, test_all = torch.max(output_test, 1)
        _, target_all = torch.max(target_test.data, 1)
        histo_all_known(configs, args, output_soft, test_all, label_set_union,same_class, target_all, t_class_name_real, epoch, 3)
        output_t_score, _, target_train = pre_train_t(logger, train_loaders, args, model, label_set_union, t_classes_number_real)
        #confusion_target_score(configs, args, target_train, output_t_score, epoch, label_set_union, t_class_name_real)

        score_real, score_real_all = all_accuracy(args, label_set_union, t_class_name_real, same_class, output_t_score, label_all_t)
        acc_know, acc_unknow, acc_test, acc_thre_know, acc_thre_unknow, acc_thre_test ,acc_real_unknow,acc_real_test= \
            all_real_accuracy(args, label_set_union, t_class_name_real, same_class, output_t_score, label_all_t)
        writer.add_scalars("Train/acc_score", {'acc_know': acc_know,
                                              'acc_unknow': acc_unknow, 'acc_test': acc_test,
                                              'acc_thre_know': acc_thre_know, 'acc_thre_unknow': acc_thre_unknow,
                                              'acc_thre_test': acc_thre_test, 'acc_real_unknow': acc_real_unknow, 'acc_real_test': acc_real_test}, epoch)

        writer.add_scalars("Train/acc", {'score_real': score_real, 'score_real_all': score_real_all}, epoch)                        
        confusion_target_score(configs, args, target_train, output_t_score, epoch, label_set_union, t_class_name_real)
        _, Result_label = torch.max(output_t_score.data, 1)
        histo_all_known(configs, args, output_t_score, Result_label, label_set_union, same_class, label_all,t_class_name_real, epoch, 5)
                        
        confusion_target_test(configs, args, target_test, output_test, epoch, label_set_union, t_class_name_real)

        acc_know, acc_unknow, acc_test, acc_thre_know, acc_thre_unknow, acc_thre_test ,acc_real_unknow,acc_real_test= \
            all_real_accuracy(args,label_set_union, t_class_name_real, same_class,  output_soft, target_test)
        writer.add_scalars("Train/acc_test", {'acc_know': acc_know,
                                              'acc_unknow': acc_unknow, 'acc_test': acc_test,
                                              'acc_thre_know': acc_thre_know, 'acc_thre_unknow': acc_thre_unknow,
                                              'acc_thre_test': acc_thre_test, 'acc_real_unknow': acc_real_unknow, 'acc_real_test': acc_real_test}, epoch)
        if acc_test > best_acc_test:
            best_acc_test = acc_test
            is_best = True
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, is_best=is_best, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
                best_file_name='{}/checkpoint_best.pth.tar'.format(args.exp_dir))

def pretrain(args, logger, models, classifiers, train_dloaders, test_loaders, optimizers, Z_s0, classifier_optimizers, writer,
             source_class_name, communication_rounds,batch_per_epoch, total_epochs,target_classes, label_set_union, t_classes_number_real):

    for epoch in range( args.pre_epochs):
        logger.info('Start Validation of epoch {}.'.format(epoch))
        logger.info('Epoch:' + str(epoch + 1) + '/' + str(total_epochs))
        if communication_rounds in [0.2, 0.5]:
            model_aggregation_frequency = round(1 / communication_rounds)
        else:
            model_aggregation_frequency = 1
        for f in range(model_aggregation_frequency):
            models, classifiers = source_train(logger, train_dloaders, args, models,
                                                     classifiers, batch_per_epoch, optimizers,  classifier_optimizers, epoch)

        for i in range(len(models)):
            output_test, target_test = pre_test_s(logger, test_loaders, args, models[i], classifiers[i], t_classes_number_real)
            confusion_source(configs, args, target_test, output_test, epoch, source_class_name[i], target_classes, i)
        for i in range(len(models)):
            label_set_union = list(set(source_class_name[i]).union(label_set_union))

    return label_set_union

def re_pseudo(args,logger,train_dloaders,batch_per_epoch,models, classifiers,
              model, numclass,class_name,label_set_union,writer,
              source_num,Z_s0,Z_ss0, train_loaders,epoch,t_class_name, num_class_t_real, same_class):

    Z_s0, _ = visual_source(logger, epoch + 1, train_loaders, args, Z_s0,
                            models, classifiers, num_class_t_real, numclass)
    Output_t = visual_target(logger, epoch + 1, train_loaders, args,
                             model, numclass=len(label_set_union) , batch_size=configs["data"]["dataloader"]["batch_size"])  # t_classes
    output_st = visual_st(logger, epoch + 1, train_dloaders[0:], args,
                          model, batch_per_epoch,
                          numclass=len(label_set_union) , batch_size=configs["data"]["dataloader"]["batch_size"])  # t_classes
    _, Z_ss0 = visual_ss(logger, epoch + 1, train_dloaders[0:], args, Z_ss0,
                         models, classifiers, batch_per_epoch,
                         numclass=numclass)

    New_t = Output_t
    z_s0 = Z_s0

    r = len(label_set_union)
    output_conf_t, Outputt = torch.max(New_t.data, 1)
    output_oneht = torch.zeros(Outputt.size(0), r).cuda().scatter_(1, Outputt.view(-1, 1).cuda(), 1).cpu()
    pre_higher_t = output_conf_t >= 2/len(label_set_union)
    int_t = np.array(pre_higher_t.cpu())
    int_tt = int_t + 0

    note = torch.zeros(source_num, r)
    dist_max = torch.zeros(r)
    dist_note = torch.zeros(r)
    site = torch.zeros(source_num, r)
    class_label = []
# multi-source nodes
    for i in range(len(label_set_union)):  # 8
        class_label.append("unknown")
        for j in range(source_num):  # 3
            output_s = z_s0[j]
            output_conf_s, Outputs = torch.max(output_s.data, 1)
            output_onehs = torch.zeros(Outputs.size(0), numclass[j]).cuda().scatter_(1, Outputs.view(-1, 1).cuda(), 1).cpu()
            pre_higher_s = output_conf_s >= 2 / len(label_set_union)
            int_s = np.array(pre_higher_s.cpu())
            int_ss = int_s+0

            for k in range(numclass[j]):  # 8-6-6
                distance = (output_oneht[:, i] *int_tt)* (output_onehs[:, k]*int_ss)
                baseline = max_value(args.re_pseu * sum(output_onehs[pre_higher_s.cpu(), k]), args.re_pseu * sum(output_oneht[pre_higher_t.cpu(), i]))
                if sum(distance) > baseline.cpu():  #
                    if sum(distance) > dist_max[i]:
                        dist_max[i] = sum(distance)
                        dist_note[i] = j + 1
                        if label_set_union[i] == class_name[j][k]:
                            class_label[i] = class_name[j][k]
                    note[j, i] = k + 1
                    site[j, i] = sum(distance)

    New_st = output_st
    z_ss0 = Z_ss0
    site = torch.zeros(source_num, r)
    index_site = torch.zeros(source_num, r)
#target node
    for j in range(source_num):  # 3
        output_conf_ss, Outputss = torch.max(z_ss0[j].data, 1)
        output_onehss = torch.zeros(Outputss.size(0), numclass[j]).cuda().scatter_(1, Outputss.view(-1, 1).cuda(), 1).cpu()
        output_conf_tt, Outputt = torch.max(New_st[j].data, 1)
        output_onehst = torch.zeros(Outputt.size(0), r).cuda().scatter_(1, Outputt.view(-1, 1).cuda(), 1).cpu()
        dis_max = torch.zeros(numclass[j])
        pre_higher_tt = output_conf_tt >= 2 / len(label_set_union)
        int_tt = np.array(pre_higher_tt.cpu())
        int_tt0 = int_tt + 0
        pre_higher_ss = output_conf_ss >= 2 / len(label_set_union)
        int_ss = np.array(pre_higher_ss.cpu())
        int_ss0 = int_ss + 0

        for i in range(len(label_set_union) ):  # 8
            for k in range(numclass[j]):  # 8-6-6
                distance = (output_onehst[:, i] *int_tt0)* (output_onehss[:, k]*int_ss0)
                baseline = max_value(args.re_pseu * sum(output_onehss[pre_higher_ss.cpu(), k]), args.re_pseu * sum(output_onehst[pre_higher_tt.cpu(), i]))
                if sum(distance) > baseline.cpu():  # 0.1 * ntrain
                    if sum(distance) > dis_max[k]:
                        dis_max[k] = sum(distance)
                        site[j, i] = sum(distance)
                        index_site[j, i] = k

    max = torch.zeros(r)
    class_s = []
    for i in range(len(label_set_union) ):  # 8
        class_s.append("unknown")
        for j in range(source_num):  # 3
            if site[j, i] > max[i]:
                if label_set_union[i] == class_name[j][int(index_site[j, i])]:
                    class_s[i] = class_name[j][int(index_site[j, i])]
                    max[i] = site[j, i]

    num = 0
    for i in range(len(label_set_union) ):  # 8
        if class_s[i] in t_class_name:
            num += 1

    for i in range(len(label_set_union)):  # 8
        if  class_s[i] != class_label[i]:
            class_s[i] = "unknown"

    return class_s

def train(train_loader, model, loss_fn, loss_cont_fn,
          label_Y, confidence, optimizer, epoch, args, configs,  tb_logger, writer,batch_per_epoch, small_score, big_score
          , start_upd_prot=True):

    batch_time = AverageMeter('Time', ':1.2f')
    data_time = AverageMeter('Data', ':1.2f')
    acc_cls = AverageMeter('Acc@Cls', ':2.2f')
    acc_proto = AverageMeter('Acc@Proto', ':2.2f')
    loss_cls_log = AverageMeter('Loss@Cls', ':2.2f')
    loss_cont_log = AverageMeter('Loss@Cont', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, acc_cls, acc_proto, loss_cls_log, loss_cont_log],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode

    model.train().cuda()
    _, predicted_all = torch.max(confidence.data, 1)  # confidence
    confidence = confidence.cpu()
    self_ec = -np.log(confidence[np.arange(len(confidence)), predicted_all.cpu()] + 1e-8).cuda()
    selfec = self_ec.reshape(-1,1).cpu()
    gmm = GaussianMixture(n_components=3, covariance_type='full').fit(selfec)

    # 预测数据的类别
    labels = gmm.predict(selfec)

    # 获取每个类别的索引
    small_loss = np.where(labels == 0)[0]
    mid_loss = np.where(labels == 1)[0]
    big_loss = np.where(labels == 2)[0]

    labels = gmm.predict_proba(selfec)
    pred = labels > 0.90

    index_all = range(self_ec.size(0))
    small_loss = pred.nonzero()[0]
    big_loss = np.delete(index_all, small_loss)
    small_loss = torch.from_numpy(small_loss)
    big_loss = torch.from_numpy(big_loss)
    small_loss = small_loss.cpu()
    big_loss = big_loss.cpu()
    conf = []

    end = time.time()
    for i, (images_w, images_s, true_labels, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        X_w, X_s, Y, index = images_w.cuda(), images_s.cuda(), label_Y[index].cuda(), index.cuda()
        # for showing training accuracy and will not be used when training
        Y_true = true_labels.long().detach().cuda()
        small_ind = np.intersect1d(index.cpu(), small_loss)
        big_ind = np.intersect1d(index.cpu(), big_loss)

        # 生成一个列表，表示array1中是否存在和list2相等的元素
        result_small = np.in1d(index.cpu(), small_ind)
        idx_small = np.where(result_small == True)
        result_big = np.in1d( index.cpu(), big_ind)
        idx_big = np.where(result_big == True)

        if Y_true.size(0) != configs["data"]["dataloader"]["batch_size"]:
            break
        cls_out, features_cont, pseudo_target_cont, score_prot, dis = model(X_w, X_s, Y, args)
        batch_size = cls_out.shape[0]
        pseudo_target_cont = pseudo_target_cont.contiguous().view(-1, 1)

        if start_upd_prot:
            conf = loss_fn.confidence_update(temp_un_conf=score_prot, batch_index=index, batchY=Y,  false = small_score, real = big_score)
        else:
            conf = confidence
            # warm up ended

        if start_upd_prot:
            mask = torch.eq(pseudo_target_cont[:batch_size], pseudo_target_cont.T).float().cuda() #None #
            # get positive set by contrasting predicted labels
        else:
            mask = None #
        # contrastive loss
        loss_cont = args.loss_weight * loss_cont_fn(features=features_cont, mask=mask, batch_size=batch_size)
        # classification loss 8256*128 mask 32*8256
        ss = torch.tensor(idx_small[0]).size(0)
        bs = torch.tensor(idx_big[0]).size(0)

        if  ss == 0:
            loss_cls = 0
        else:
            loss_cls, confc = loss_fn(cls_out, index) #loss_fn(cls_out[idx_small,:], index[idx_small]) #
        prior = torch.ones( bs, cls_out.shape[1]) / cls_out.shape[1]

        if bs == 0:
            penalty = 0
        else:
            penalty = args.loss_penalty * F.mse_loss(cls_out[idx_big], prior.cuda()) #  args.loss_penalty * F.mse_loss(cls_out, prior.cuda()) #
        loss = loss_cls + penalty + loss_cont

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        '''writer.add_scalars("Train/loss", {'loss_cls': loss_cls}, epoch*batch_size+i)
        writer.add_scalars("Train/loss", {'loss_cont': loss_cont}, epoch*batch_size+i)
        writer.add_scalars("Train/loss", {'loss_penalty': penalty}, epoch*batch_size+i)
        writer.add_scalars("Train/num", {'loss_small': ss}, epoch*batch_size+i)
        writer.add_scalars("Train/num", {'loss_big': bs}, epoch*batch_size+i)
        writer.add_scalars("Train/num", {'all': ss+bs}, epoch*batch_size+i)'''

        if epoch ==  configs["TrainingConfig"]["total_epochs"]-2:
            config_file = args.configs
            args = yaml.load(open(config_file), Loader=yaml.FullLoader)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)

    return conf, confc

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file_name)

if __name__ == "__main__":
    main()
