# -*- coding:utf-8 -*-

def compute_eer(scores, labels):
    if isinstance(scores, list) is False:
        scores = list(scores)
    if isinstance(labels, list) is False:
        labels = list(labels)

    target_scores = []
    nontarget_scores = []

    for item in zip(scores, labels):
        if item[1] == 1:
            target_scores.append(item[0])
        else:
            nontarget_scores.append(item[0])

    target_size = len(target_scores)
    nontarget_size = len(nontarget_scores)
    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)

    target_position = 0
    for i in range(target_size-1):
        target_position = i
        nontarget_n = nontarget_size * float(target_position) / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            break
    th = target_scores[target_position]
    eer = target_position * 1.0 / target_size
    return eer, th

