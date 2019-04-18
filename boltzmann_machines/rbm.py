import numpy as np
import tensorflow as tf

from base_rbm import BaseRBM
from layers import BernoulliLayer


class BernoulliRBM(BaseRBM): # BernoulliRBM inherits the base class "BaseRBM"
    """RBM with Bernoulli both visible and hidden units."""
    def __init__(self, model_path='b_rbm_model/', *args, **kwargs):
        super(BernoulliRBM, self).__init__(v_layer_cls=BernoulliLayer,
                                           h_layer_cls=BernoulliLayer,
                                           model_path=model_path, *args, **kwargs)

    def _free_energy(self, v):
        with tf.name_scope('free_energy'):
            T1 = -tf.einsum('ij,j->i', v, self._vb)
            T2 = -tf.reduce_sum(tf.nn.softplus(self._propup(v) + self._hb), axis=1)
            fe = tf.reduce_mean(T1 + T2, axis=0)
        return fe

# the MultinomialRBM and GaussianRBM were removed by Ziwei on Apr. 8th

def logit_mean(X):
    p = np.mean(X, axis=0)
    p = np.clip(p, 1e-7, 1. - 1e-7)
    q = np.log(p / (1. - p))
    return q


