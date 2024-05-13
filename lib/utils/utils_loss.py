import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import numpy as np

class partial_loss(nn.Module):
    def __init__(self, confidence, configs, conf_ema, conf_c, epoch, conf_ema_m=0.99):
        super().__init__()
        self.confidence = confidence
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m
        self.conf_ema = conf_ema
        self.conf_c = conf_c
        self.configs = configs
        self.epoch = epoch

    def set_conf_ema_m(self, epoch, args):
        start = args.conf_ema_range_start
        end = args.conf_ema_range_end
        self.conf_ema_m = 1. * epoch / (self.configs["TrainingConfig"]["total_epochs"]) * (end - start) + start

    def forward(self, outputs, index):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :]
        average_loss = - (final_outputs.sum(dim=1)).mean()
        return average_loss , self.confidence

    def confidence_update(self, temp_un_conf, batch_index, batchY, false, real):
        with torch.no_grad():
            result_real = np.in1d(batch_index.cpu(), real)
            idx_real = np.where(result_real == True)
            result_false = np.in1d(batch_index.cpu(), false)
            idx_false = np.where(result_false == True)

            false_pseu = torch.ones(temp_un_conf.shape[0], temp_un_conf.shape[1]) / temp_un_conf.shape[1]
            false_pseu = false_pseu.cuda()
            _, prot_pred = (temp_un_conf).max(dim=1)
            real_pseudo = F.one_hot(prot_pred, batchY.shape[1]).float().cuda().detach()

            self.conf_ema[batch_index, :] = self.conf_ema_m * self.conf_ema[batch_index, :] \
                                                      + (1 - self.conf_ema_m) * real_pseudo # pseudo_label #
            self.conf_c = self.conf_ema_m * self.conf_c
            self.confidence[batch_index, :] = self.conf_c * self.init_conf[batch_index, :] \
                                              + self.conf_ema[batch_index, :]

        return self.confidence

def gmm_partial(x):
    # x=x.cpu().detach().numpy()
    val = x.cpu().detach().numpy()

    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    # gmm.fit(list(set(val.cpu())))
    gmm.fit(val)
    prob = gmm.predict_proba(val)
    prob = prob[:, gmm.means_.argmax()]
    return prob


class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning:
        https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device)
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss