import torch.optim
from configular import *
from train.train_new import *


def re_pseudo(args,logger,train_dloaders,batch_per_epoch,models, classifiers,
              model, numclass,class_name,label_set_union,writer,
              source_num,Z_s0,Z_ss0, train_loaders,epoch,t_class_name, num_class_t_real, same_class):

    Z_s0, label_all_t = visual_source(logger, epoch + 1, train_loaders, args, Z_s0,
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
    output_oneht = torch.zeros(Outputt.size(0), r).cuda().scatter_(1, Outputt.view(-1, 1).cuda(), 1)
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
            output_onehs = torch.zeros(Outputs.size(0), numclass[j]).cuda().scatter_(1, Outputs.view(-1, 1).cuda(), 1)
            pre_higher_s = output_conf_s >= 2 / len(label_set_union)
            int_s = np.array(pre_higher_s.cpu())
            int_ss = int_s+0

            for k in range(numclass[j]):  # 8-6-6
                distance = (output_oneht[:, i] *int_tt)* (output_onehs[:, k]*int_ss)
                baseline = max_value(args.re_pseu * sum(output_onehs[pre_higher_s, k]), args.re_pseu * sum(output_oneht[pre_higher_t, i]))
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
        output_onehss = torch.zeros(Outputss.size(0), numclass[j]).cuda().scatter_(1, Outputss.view(-1, 1).cuda(), 1)
        output_conf_tt, Outputt = torch.max(New_st[j].data, 1)
        output_onehst = torch.zeros(Outputt.size(0), r).cuda().scatter_(1, Outputt.view(-1, 1).cuda(), 1)
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
                baseline = max_value(args.re_pseu * sum(output_onehss[pre_higher_ss, k]), args.re_pseu * sum(output_onehst[pre_higher_tt, i]))
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

    logger.info(' Epoch/%d,t_class_name_single_s: %s  %%' % (epoch, class_s))
    logger.info(' Epoch/%d,label_set_union: %s  %%' % (epoch, label_set_union))

    num = 0
    for i in range(len(label_set_union) ):  # 8
        if class_s[i] in t_class_name:
            num += 1
    writer.add_scalars("Train/source", {'Labelset_source': num / len(same_class) }, epoch)
    writer.add_scalars("Train/pseu", {'Labelset_source': num / len(class_s) }, epoch)

    logger.info(' Epoch/%d,t_class_name_source: %d  %%' % (epoch, num))
    for i in range(len(label_set_union) ):  # 8
        if class_label[i] != class_s[i]:
            if dist_max[i] > max[i]:
                class_s[i] = class_label[i]

    for i in range(len(label_set_union)):
        for j in range(len(label_set_union)):
            if j !=i :
                if class_s[j] == class_s[i] :
                    if dist_max[j] > dist_max[i]:
                        class_s[i] = "unknown"

    num = 0
    for i in range(len(label_set_union) ):  # 8
        if class_s[i] in t_class_name:
            num += 1
    writer.add_scalars("Train/source", {'Labelset_double': num / len(same_class) }, epoch)
    writer.add_scalars("Train/pseu", {'Labelset_double': num / len(class_s) }, epoch)

    logger.info(' Epoch/%d,Labelset_double: %d  %%' % (epoch, num))
    num = 0
    for i in range(len(label_set_union) ):  # 8
        if class_label[i]  in t_class_name:
            num += 1
    writer.add_scalars("Train/source", {'Labelset_target': num / len(same_class) }, epoch)
    writer.add_scalars("Train/pseu", {'Labelset_target': num / len(class_s)}, epoch)

    logger.info(' Epoch/%d,class_label_target: %d  %%' % (epoch, num))
    logger.info(' Epoch/%d,class_real_num: %d  %%' % (epoch,  len(t_class_name) ))
    logger.info(' Epoch/%d,t_class_name_double: %s  %%' % (epoch, class_s))
    logger.info(' Epoch/%d,class_real: %s  %%' % (epoch,  t_class_name))
    logger.info(' Epoch/%d,t  class_label_single_t:: %s %%' % (epoch, class_label))

    return class_s