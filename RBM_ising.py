# goal of this code: 
# 1. Build an Ising model
# 2. Run Gibbs/MC/MCMC sampling on it and generate enough data points
# 3. 

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

#import env
from boltzmann_machines.rbm import BernoulliRBM, logit_mean
from boltzmann_machines.rng import RNG
from boltzmann_machines.dataset import load_mnist

import tensorflow as tf

import matplotlib.pyplot as plt

#tf.enable_eager_execution()

'''
def make_rbm(X_train, X_val, args):
    print ("\nTraining model ...\n\n")
    rbm = BernoulliRBM(n_visible=784,
                           n_hidden=args.n_hidden,
                           W_init=args.w_init,
                           vb_init=logit_mean(X_train) if args.vb_init else 0.,
                           hb_init=args.hb_init,
                           n_gibbs_steps=args.n_gibbs_steps,
                           learning_rate=args.lr,
                           momentum=np.geomspace(0.5, 0.9, 8),
                           max_epoch=args.epochs,
                           batch_size=args.batch_size,
                           l2=args.l2,
                           sample_v_states=args.sample_v_states,
                           sample_h_states=True,
                           dropout=args.dropout,
                           sparsity_target=args.sparsity_target,
                           sparsity_cost=args.sparsity_cost,
                           sparsity_damping=args.sparsity_damping,
                           metrics_config=dict(
                               msre=True,
                               pll=True,
                               feg=True,
                               train_metrics_every_iter=1000,
                               val_metrics_every_epoch=2,
                               feg_every_epoch=4,
                               n_batches_for_feg=50,
                           ),
                           verbose=True,
                           display_filters=30,
                           display_hidden_activations=24,
                           v_shape=(28, 28),
                           random_seed=args.random_seed,
                           dtype=args.dtype,
                           tf_saver_params=dict(max_to_keep=1),
                           model_path=args.model_dirpath)
    rbm.fit(X_train, X_val)
    return rbm
'''

#def main():
if __name__ == '__main__':
    # training settings
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general/data
    parser.add_argument('--gpu', type=str, default='0', metavar='ID',
                        help="ID of the GPU to train on (or '' to train on CPU)")
    parser.add_argument('--n_train', type=int, default=1000, metavar='N',
                        help='number of training examples')
    parser.add_argument('--n_val', type=int, default=20, metavar='N',
                        help='number of validation examples')
    parser.add_argument('--data_path', type=str, 
                        default=os.path.expanduser('~') + '/Desktop/RBM_learning/boltzmann-machines-master/data/mnist/', 
                        metavar='PATH', help='directory for storing augmented data etc.')

    # RBM related
    parser.add_argument('--n_hidden', type=int, default=392, metavar='N',
                        help='number of hidden units')
    parser.add_argument('--w_init', type=float, default=0.01, metavar='STD', 
                        help='initialize weights from zero-centered Gaussian with this standard deviation')
    #parser.add_argument('--vb_init', action='store_false',
    #                    help='initialize visible biases as logit of mean values of features' + \
    #                         ', otherwise (if enabled) zero init')
    parser.add_argument('--vb_init', type=float, default=0., 
                        help='initial visible bias')
    parser.add_argument('--hb_init', type=float, default=0., metavar='HB',
                        help='initial hidden bias')
    parser.add_argument('--n_gibbs_steps', type=int, default=1, metavar='N', nargs='+',
                        help='number of Gibbs updates per weights update or sequence of such (per epoch)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR', nargs='+',
                        help='learning rate or sequence of such (per epoch)')
    parser.add_argument('--epochs', type=int, default=220, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=10, metavar='B',
                        help='input batch size for training')
    parser.add_argument('--l2', type=float, default=0.0, metavar='L2', # changed l2 from 1e-5 to 0
                        help='L2 weight decay coefficient')
    parser.add_argument('--sample_v_states', action='store_true',
                        help='sample visible states, otherwise use probabilities w/o sampling')
    parser.add_argument('--dropout', type=float, metavar='P',
                        help='probability of visible units being on')
    parser.add_argument('--sparsity_target', type=float, default=0.1, metavar='T',
                        help='desired probability of hidden activation')
    parser.add_argument('--sparsity_cost', type=float, default=1e-5, metavar='C',
                        help='controls the amount of sparsity penalty')
    parser.add_argument('--sparsity_damping', type=float, default=0.9, metavar='D',
                        help='decay rate for hidden activations probs')
    parser.add_argument('--random_seed', type=int, default=830, metavar='N',
                        help="random seed for model training")
    parser.add_argument('--dtype', type=str, default='float32', metavar='T',
                        help="datatype precision to use")
    parser.add_argument('--model_dirpath', type=str, default='models/rbm_ising/', metavar='DIRPATH',
                        help='directory path to save the model')

    # MLP related
    parser.add_argument('--mlp-no-init', action='store_true',
                        help='if enabled, use random initialization')
    parser.add_argument('--mlp-l2', type=float, default=1e-5, metavar='L2',
                        help='L2 weight decay coefficient')
    parser.add_argument('--mlp-lrm', type=float, default=(0.1, 1.), metavar='LRM', nargs='+',
                        help='learning rate multipliers of 1e-3')
    parser.add_argument('--mlp-epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--mlp-val-metric', type=str, default='val_acc', metavar='S',
                        help="metric on validation set to perform early stopping, {'val_acc', 'val_loss'}")
    parser.add_argument('--mlp-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--mlp-save-prefix', type=str, default='../data/rbm_', metavar='PREFIX',
                        help='prefix to save MLP predictions and targets')

    args = parser.parse_args()
    args.epochs = 200
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if len(args.mlp_lrm) == 1:
        args.mlp_lrm *= 2

    # prepare data (load + scale + split)
    print ("\nPreparing data ...\n\n")
    
    '''
    # MNIST data
    X, _ = load_mnist(mode='train', path=args.data_path)
    X /= 255.
    '''
    # Ising data
    import pickle
    def load_obj(obj_name, variable_name):
        v_path = obj_name + '/'+ variable_name + '.pkl'
        with open(v_path, 'rb') as f:
            return pickle.load(f)
    obj_name = './spin_config'
    S_config = load_obj(obj_name, 'S_config')
    T        = load_obj(obj_name, 'T'       )
    T_ind = 7
    X = S_config[:, :, T_ind].T
    T_ = T[T_ind]
    #'''

    #divide data into training and validation sets
    RNG(seed=42).shuffle(X)
    n_train = min(len(X), args.n_train)
    n_val = min(len(X), args.n_val)
    X_train = X[:n_train]
    X_val = X[-n_val:]





    # train and save the RBM model
    #rbm = make_rbm(X_train, X_val, args)
    print ("\nTraining model ...\n\n")
    rbm = BernoulliRBM(n_visible=784,
                           n_hidden=args.n_hidden,
                           W_init=args.w_init,
                           vb_init=logit_mean(X_train) if args.vb_init else 0.,
                           hb_init=args.hb_init,
                           n_gibbs_steps=args.n_gibbs_steps,
                           learning_rate=args.lr,
                           momentum=np.geomspace(0.5, 0.9, 8),
                           max_epoch=args.epochs,
                           batch_size=args.batch_size,
                           l2=args.l2,
                           sample_v_states=args.sample_v_states,
                           sample_h_states=True,
                           dropout=args.dropout,
                           sparsity_target=args.sparsity_target,
                           sparsity_cost=args.sparsity_cost,
                           sparsity_damping=args.sparsity_damping,
                           metrics_config=dict(
                               msre=True,
                               pll=True,
                               feg=True,
                               train_metrics_every_iter=10, # used to be 1000
                               val_metrics_every_epoch=2,
                               feg_every_epoch=4,
                               n_batches_for_feg=50,
                           ),
                           verbose=True,
                           display_filters=30,
                           display_hidden_activations=24,
                           v_shape=(28, 28),
                           random_seed=args.random_seed,
                           dtype=args.dtype,
                           tf_saver_params=dict(max_to_keep=1),
                           model_path=args.model_dirpath)
    
    #rbm._make_constants()
    #rbm._make_placeholders()
    #rbm._make_filters()
    #rbm._make_vars()

    rbm.fit(X_train, X_val)

    
    # load test data
    #X_test, y_test = load_mnist(mode='test', path=args.data_path)
    #X_test /= 255.
    
   
    # see the weights
    W, hb, vb = None, None, None
    if not args.mlp_no_init:
        weights = rbm.get_tf_params(scope='weights')
        W = weights['W']
        hb = weights['hb']
        vb = weights['vb']

#if __name__ == '__main__':
#    main()



