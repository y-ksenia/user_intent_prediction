import numpy as np


def find_rank(row):
    """

    :param row: rod of pd.DataFrame
    :return: reciprocal rank
    """
    return 1 / (np.where(np.array(row['preds']) == row['cur_app_idx'])[0][0] + 1)


def accuracy_precision_recall(trues, preds, K=10):
    I = len(trues)
    assert I == len(preds)
    accuracyK, precisionK, recallK, f1K = np.zeros([I, K]), np.zeros([I, K]), np.zeros([I, K]), np.zeros([I, K])
    for i in range(I):
        for k in range(K):
            intersect_size = len(set(trues[i][:k + 1]).intersection(set(preds[i][:k + 1])))
            union_size = len(set(trues[i][:k + 1]).union(set(preds[i][:k + 1])))
            relevant_size = len(trues[i][:k + 1])
            retrieved_size = len(preds[i][:k + 1])
            accuracyK[i, k] = intersect_size / union_size
            precisionK[i, k] = 0 if retrieved_size == 0 else intersect_size / retrieved_size
            recallK[i, k] = 0 if relevant_size == 0 else intersect_size / relevant_size
            f1K[i, k] = 0 if precisionK[i, k] + recallK[i, k] == 0 \
                else 2 * precisionK[i, k] * recallK[i, k] / (precisionK[i, k] + recallK[i, k])
    return accuracyK, precisionK, recallK, f1K
