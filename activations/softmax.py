import numpy as np

def softmax(feat):
    exp_feat = np.exp(feat)
    exp_feat_denom = np.sum(exp_feat, axis=-1)
    scores = exp_feat / np.expand_dims(exp_feat_denom, axis=-1)
    return scores