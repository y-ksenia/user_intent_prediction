import pickle
import random

import numpy as np
from .utils import sigmoid


class FPMC:
    def __init__(self, user_list, item_list, n_factor, learn_rate, regular, neg_batch_size, std=0.01):
        self.user_set = set(user_list)
        self.item_set = set(item_list)

        self.n_user = max(self.user_set) + 1
        self.n_item = max(self.item_set) + 1

        self.n_factor = n_factor
        self.learn_rate = learn_rate
        self.regular = regular
        self.neg_batch_size = neg_batch_size
        self.std = std
        self.params = {'n_factor': n_factor, 'learn_rate': learn_rate, 'regular': regular,
                       'neg_batch_size': neg_batch_size, 'std': std}

        self.VUI = np.random.normal(0, self.std, size=(self.n_user, self.n_factor))
        self.VIU = np.random.normal(0, self.std, size=(self.n_item, self.n_factor))
        self.VIL = np.random.normal(0, self.std, size=(self.n_item, self.n_factor))
        self.VLI = np.random.normal(0, self.std, size=(self.n_item, self.n_factor))
        self.VUI_m_VIU = np.dot(self.VUI, self.VIU.T)
        self.VIL_m_VLI = np.dot(self.VIL, self.VLI.T)

    @staticmethod
    def dump(fpmcObj, fname):
        pickle.dump(fpmcObj, open(fname, 'wb'))

    @staticmethod
    def load(fname):
        return pickle.load(open(fname, 'rb'))

    def compute_x(self, u, i, b_tm1):
        acc_val = 0.0
        for l in b_tm1:
            acc_val += np.dot(self.VIL[i], self.VLI[l])
        return np.dot(self.VUI[u], self.VIU[i]) + (acc_val / len(b_tm1))

    def compute_x_batch(self, u, b_tm1):
        former = self.VUI_m_VIU[u]
        latter = np.mean(self.VIL_m_VLI[:, b_tm1], axis=1).T
        return former + latter

    def evaluation(self, df):

        self.VUI_m_VIU = np.dot(self.VUI, self.VIU.T)
        self.VIL_m_VLI = np.dot(self.VIL, self.VLI.T)

        data_list = df.values

        correct_count = 0
        rr_list = []
        for (u, i, b_tm1) in data_list:
            scores = self.compute_x_batch(u, b_tm1)

            if i == scores.argmax():
                correct_count += 1

            rank = len(np.where(scores > scores[i])[0]) + 1
            rr = 1.0 / rank
            rr_list.append(rr)

        try:
            acc = correct_count / len(rr_list)
            mrr = (sum(rr_list) / len(rr_list))
            return acc, mrr
        except:
            return 0.0, 0.0

    def learn_epoch(self, train_df):
        tr_data = train_df.values
        for iter_idx in range(len(tr_data)):

            u, i, b_tm1 = random.choice(tr_data)

            exclu_set = self.item_set - set([i])

            j_list = random.sample(exclu_set, self.neg_batch_size)

            z1 = self.compute_x(u, i, b_tm1)
            for j in j_list:
                z2 = self.compute_x(u, j, b_tm1)
                delta = 1 - sigmoid(z1 - z2)

                self.VUI[u] += self.learn_rate * (delta * (self.VIU[i] - self.VIU[j]) - self.regular * self.VUI[u])
                self.VIU[i] += self.learn_rate * (delta * self.VUI[u] - self.regular * self.VIU[i])
                self.VIU[j] += self.learn_rate * (-delta * self.VUI[u] - self.regular * self.VIU[j])

                eta = np.mean(self.VLI[b_tm1], axis=0)

                self.VIL[i] += self.learn_rate * (delta * eta - self.regular * self.VIL[i])
                self.VIL[j] += self.learn_rate * (-delta * eta - self.regular * self.VIL[j])
                self.VLI[b_tm1] += self.learn_rate * \
                                   ((delta * (self.VIL[i] - self.VIL[j]) / len(b_tm1)) - self.regular * self.VLI[b_tm1])

    def folding_in(self, row, n_epoch):
        row = row.values

        u, i, b_tm1 = row

        exclu_set = self.item_set - set([i])

        for _ in range(n_epoch):
            j_list = random.sample(exclu_set, self.neg_batch_size)

            z1 = self.compute_x(u, i, b_tm1)
            for j in j_list:
                z2 = self.compute_x(u, j, b_tm1)
                delta = 1 - sigmoid(z1 - z2)

                self.VUI[u] += self.learn_rate * (delta * (self.VIU[i] - self.VIU[j]) - self.regular * self.VUI[u])

                eta = np.mean(self.VLI[b_tm1], axis=0)

                self.VIL[i] += self.learn_rate * (delta * eta - self.regular * self.VIL[i])
                self.VIL[j] += self.learn_rate * (-delta * eta - self.regular * self.VIL[j])

        self.VUI_m_VIU = np.dot(self.VUI, self.VIU.T)
        self.VIL_m_VLI = np.dot(self.VIL, self.VLI.T)

    def learnSBPR_FPMC(self, tr_data, te_data=None, n_epoch=10, eval_per_epoch=False, verbose=False):
        acc_out, mrr_out = 0., 0.
        for epoch in range(n_epoch):
            self.learn_epoch(tr_data)

            if eval_per_epoch:
                acc_in, mrr_in = self.evaluation(tr_data)
                if te_data is not None:
                    acc_out, mrr_out = self.evaluation(te_data)
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
            acc_in, mrr_in = self.evaluation(tr_data)
            if te_data is not None:
                acc_out, mrr_out = self.evaluation(te_data)
                if verbose:
                    print('\tIn sample:\t acc = %.4f\t mrr = %.4f' % (acc_in, mrr_in))
                    print('\tOut sample:\t acc = %.4f\t mrr = %.4f' % (acc_out, mrr_out))
            else:
                if verbose:
                    print('\tIn sample:%.4f\t%.4f' % (acc_in, mrr_in))

        return acc_out, mrr_out
