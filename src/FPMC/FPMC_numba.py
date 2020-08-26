# import math
# import numpy as np

from numba import jit
from .utils import *
from .FPMC import FPMC as FPMC_basic


class FPMC_numba(FPMC_basic):
    def __init__(self, user_list, item_list, n_factor, learn_rate, regular, neg_batch_size, std=0.01, **kwargs):
        super(FPMC_numba, self).__init__(user_list, item_list, n_factor, learn_rate, regular, neg_batch_size, std)

    def evaluation(self, data_list_3):
        np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)

        acc, mrr = evaluation_jit \
            (data_list_3[0], data_list_3[1], data_list_3[2], self.VUI_m_VIU, self.VIL_m_VLI)

        return acc, mrr

    def learn_epoch(self, data_list_3):
        VUI, VIU, VLI, VIL = learn_epoch_jit(data_list_3[0], data_list_3[1], data_list_3[2], self.neg_batch_size,
                                             np.array(list(self.item_set)), self.VUI, self.VIU, self.VLI, self.VIL,
                                             self.learn_rate, self.regular)

        self.VUI = VUI
        self.VIU = VIU
        self.VLI = VLI
        self.VIL = VIL

    def learnSBPR_FPMC(self, tr_data, te_data=None, n_epoch=10, eval_per_epoch=False, ret_in_score=False,
                       verbose=False):
        mrr_in, acc_in, mrr_out, acc_out = 0, 0, 0, 0
        tr_list_3 = data_to_list_3(tr_data)
        te_list_3 = None
        if te_data is not None:
            te_list_3 = data_to_list_3(te_data)

        for epoch in range(n_epoch):
            self.learn_epoch(tr_list_3)

            if eval_per_epoch:
                acc_in, mrr_in = self.evaluation(tr_list_3)
                if te_data is not None:
                    acc_out, mrr_out = self.evaluation(te_list_3)
                    if verbose:
                        print('\tIn sample:\t acc = %.4f\t mrr = %.4f' % (acc_in, mrr_in))
                        print('\tOut sample:\t acc = %.4f\t mrr = %.4f' % (acc_out, mrr_out))
                else:
                    if verbose:
                        print('\tIn sample:%.4f\t%.4f' % (acc_in, mrr_in))
            else:
                if verbose:
                    print('epoch %d done' % epoch)

        if not eval_per_epoch:
            acc_in, mrr_in = self.evaluation(tr_list_3)
            if te_data is not None:
                acc_out, mrr_out = self.evaluation(te_list_3)
                if verbose:
                    print('\tIn sample:\t acc = %.4f\t mrr = %.4f' % (acc_in, mrr_in))
                    print('\tOut sample:\t acc = %.4f\t mrr = %.4f' % (acc_out, mrr_out))
            else:
                if verbose:
                    print('\tIn sample:%.4f\t%.4f' % (acc_in, mrr_in))

        if ret_in_score:
            return acc_in, mrr_in, acc_out, mrr_out
        else:
            return acc_out, mrr_out

    def folding_in(self, data, n_epoch):
        u, i, b_tm1 = data.values
        if u > self.VUI.shape[0]:
            self.VUI = np.vstack((self.VUI, np.zeros((u-self.VUI.shape[0], self.n_factor))))
        if u > self.VUI.shape[0]:
            self.VUI = np.vstack((self.VUI, np.zeros((u-self.VUI.shape[0], self.n_factor))))


        for _ in range(n_epoch):
            VUI_u, VIL_i = folding_in_jit(u, i, b_tm1, np.array(list(self.item_set - set([i]))),
                                          self.VUI, self.VIU, self.VLI, self.VIL,
                                          self.learn_rate, self.regular)

            self.VUI[u] = VUI_u
            self.VIL[i] = VIL_i

        self.VUI_m_VIU[u, i] = self.VUI[u] @ self.VIU[i].T
        self.VIL_m_VLI[i, b_tm1] = self.VIL[i] @ self.VLI[b_tm1].T


@jit(nopython=True)
def compute_x_jit(u, i, b_tm1, VUI, VIU, VLI, VIL):
    acc_val = 0.0
    for l in b_tm1:
        acc_val += np.dot(VIL[i], VLI[l])
    return np.dot(VUI[u], VIU[i]) + (acc_val / len(b_tm1))


@jit(nopython=False)
def learn_epoch_jit(u_list, i_list, b_tm1_list, neg_batch_size, item_set, VUI, VIU, VLI, VIL, learn_rate, regular):
    for iter_idx in range(len(u_list)):
        d_idx = np.random.randint(0, len(u_list))
        u = u_list[d_idx]
        i = i_list[d_idx]
        b_tm1 = b_tm1_list[d_idx][b_tm1_list[d_idx] != -1]
        # b_tm1 = [x for x in b_tm1 if x != i]
        # print(i, b_tm1)
        if len(b_tm1) == 0:
            continue

        j_list = np.random.choice(item_set, size=neg_batch_size, replace=False)

        z1 = compute_x_jit(u, i, b_tm1, VUI, VIU, VLI, VIL)
        for j in j_list:
            z2 = compute_x_jit(u, j, b_tm1, VUI, VIU, VLI, VIL)
            delta = 1 - sigmoid_jit(z1 - z2)

            VUI_update = learn_rate * (delta * (VIU[i] - VIU[j]) - regular * VUI[u])
            VIUi_update = learn_rate * (delta * VUI[u] - regular * VIU[i])
            VIUj_update = learn_rate * (-delta * VUI[u] - regular * VIU[j])

            VUI[u] += VUI_update
            VIU[i] += VIUi_update
            VIU[j] += VIUj_update

            eta = np.zeros(VLI.shape[1])
            for l in b_tm1:
                eta += VLI[l]
            eta = eta / len(b_tm1)

            VILi_update = learn_rate * (delta * eta - regular * VIL[i])
            VILj_update = learn_rate * (-delta * eta - regular * VIL[j])
            VLI_updates = np.zeros((len(b_tm1), VLI.shape[1]))
            for idx, l in enumerate(b_tm1):
                VLI_updates[idx] = learn_rate * ((delta * (VIL[i] - VIL[j]) / len(b_tm1)) - regular * VLI[l])

            VIL[i] += VILi_update
            VIL[j] += VILj_update
            for idx, l in enumerate(b_tm1):
                VLI[l] += VLI_updates[idx]

    return VUI, VIU, VLI, VIL


@jit(nopython=False)
def folding_in_jit(u, i, b_tm1, item_set, VUI, VIU, VLI, VIL, learn_rate, regular):
    j = np.random.choice(item_set, size=1, replace=False)[0]

    z1 = compute_x_jit(u, i, b_tm1, VUI, VIU, VLI, VIL)
    z2 = compute_x_jit(u, j, b_tm1, VUI, VIU, VLI, VIL)
    delta = 1 - sigmoid_jit(z1 - z2)

    VUI[u] += learn_rate * (delta * (VIU[i] - VIU[j]) - regular * VUI[u])

    eta = np.zeros(VLI.shape[1])
    for l in b_tm1:
        eta += VLI[l]
    eta = eta / len(b_tm1)

    VIL[i] += learn_rate * (delta * eta - regular * VIL[i])
    return VUI[u], VIL[i]


@jit(nopython=True)
def sigmoid_jit(x):
    if x >= 0:
        return math.exp(-np.logaddexp(0, -x))
    else:
        return math.exp(x - np.logaddexp(x, 0))


@jit(nopython=True)
def compute_x_batch_jit(u, b_tm1, VUI_m_VIU, VIL_m_VLI):
    former = VUI_m_VIU[u]
    latter = np.zeros(VIL_m_VLI.shape[0])
    for idx in range(VIL_m_VLI.shape[0]):
        for l in b_tm1:
            latter[idx] += VIL_m_VLI[idx, l]
    latter = latter / len(b_tm1)

    return former + latter


@jit(nopython=True)
def evaluation_jit(u_list, i_list, b_tm1_list, VUI_m_VIU, VIL_m_VLI):
    correct_count = 0
    acc_rr = 0
    for d_idx in range(len(u_list)):
        u = u_list[d_idx]
        i = i_list[d_idx]
        b_tm1 = b_tm1_list[d_idx][b_tm1_list[d_idx] != -1]
        scores = compute_x_batch_jit(u, b_tm1, VUI_m_VIU, VIL_m_VLI)

        if i == scores.argmax():
            correct_count += 1

        rank = len(np.where(scores > scores[i])[0]) + 1
        rr = 1.0 / rank
        acc_rr += rr

    acc = correct_count / len(u_list)
    mrr = acc_rr / len(u_list)
    return acc, mrr
