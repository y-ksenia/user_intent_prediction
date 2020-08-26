import FPMC


def tuning(user_list, app_list, data, factor, train_ratio, n_epoch,
           lr_list=[0.0001, 0.001], reg_list=[0.0001, 0.001], n_neg_list=[10, 20], std_list=[0.01, 0.1],
           parallel=True):
    split_idx = int(len(data) * train_ratio)
    best_mrr, best_acc = 0, 0
    best_model = None
    model = None
    for lr in lr_list:
        for alpha in reg_list:
            for sigma in std_list:
                for n_neg in n_neg_list:
                    if parallel:
                        model = FPMC.FPMC_numba(user_list=user_list, item_list=app_list,
                                                n_factor=factor, learn_rate=lr, regular=alpha, neg_batch_size=n_neg,
                                                std=sigma)
                    else:
                        model = FPMC.FPMC(user_list=user_list, item_list=app_list,
                                          n_factor=factor, learn_rate=lr, regular=alpha, neg_batch_size=n_neg,
                                          std=sigma)
                    print(model.params)
                    acc_out, mrr_out = model.learnSBPR_FPMC(data[:split_idx], data[split_idx:], n_epoch)
                    if mrr_out > best_mrr and acc_out > best_acc:
                        best_mrr = mrr_out
                        best_acc = acc_out
                        best_model = model
                        print(f'\t\t mrr = {best_mrr}; acc = {best_acc}')
    if best_model:
        return best_model
    else:
        return model
