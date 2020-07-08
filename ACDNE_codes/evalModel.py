# -*- coding: utf-8 -*-
"""
Created on Mon May 27 19:18:27 2019

@author: Shen xiao

Please cite our paper as:
"Xiao Shen, Quanyu Dai, Fu-lai Chung, Wei Lu, and Kup-Sze Choi. Adversarial Deep Network Embedding for Cross-Network Node Classification. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), pages 2991-2999, 2020."

"""

import numpy as np
import tensorflow as tf
import utils
from scipy.sparse import vstack
from functools import partial
import scipy.io
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from ACDNE_model import ACDNE




def train_and_evaluate(input_data, config, random_state=0):
    
    ###get input data
    PPMI_s=input_data['PPMI_S']
    PPMI_t=input_data['PPMI_T']
    X_s=input_data['attrb_S']
    X_t=input_data['attrb_T']
    X_n_s= input_data['attrb_nei_S']
    X_n_t=input_data['attrb_nei_T']
    Y_s=input_data['label_S']
    Y_t=input_data['label_T']    
    Y_t_o=np.zeros(np.shape(Y_t)) #observable label matrix of target network, all zeros
    
    X_s_new=lil_matrix(np.concatenate((lil_matrix.toarray(X_s), X_n_s),axis=1))    
    X_t_new=lil_matrix(np.concatenate((lil_matrix.toarray(X_t), X_n_t),axis=1))                    
    n_input = X_s.shape[1]
    num_class = Y_s.shape[1] 
    num_nodes_S=X_s.shape[0]
    num_nodes_T=X_t.shape[0]
    

    
    ###model config
    clf_type = config['clf_type'] 
    dropout = config['dropout'] 
    num_epoch = config['num_epoch'] 
    batch_size = config['batch_size']
    n_hidden = config['n_hidden'] 
    n_emb = config['n_emb'] 
    l2_w = config['l2_w'] 
    net_pro_w = config['net_pro_w'] 
    emb_filename = config['emb_filename'] 
    lr_ini = config['lr_ini'] 


    whole_xs_xt_stt = utils.csr_2_sparse_tensor_tuple(vstack([X_s, X_t])) 
    whole_xs_xt_stt_nei = utils.csr_2_sparse_tensor_tuple(vstack([X_n_s, X_n_t]))
                
    with tf.Graph().as_default():
        # Set random seed
        tf.set_random_seed(random_state)
        np.random.seed(random_state)
        
        model = ACDNE(n_input, n_hidden, n_emb, num_class, clf_type, l2_w, net_pro_w, batch_size)
               
        with tf.Session() as sess:
            # Random initialize
            sess.run(tf.global_variables_initializer())


            for cEpoch in range(num_epoch):             
                S_batches = utils.batch_generator([X_s_new,Y_s], int(batch_size / 2), shuffle=True)
                T_batches = utils.batch_generator([X_t_new,Y_t_o], int(batch_size / 2), shuffle=True) 
                
                num_batch=round(max(num_nodes_S/(batch_size/2),num_nodes_T/(batch_size/2)))
                
                # Adaptation param and learning rate schedule as described in the DANN paper 
                p=float(cEpoch) / (num_epoch)
                lr=lr_ini / (1. + 10 * p)**0.75           
                grl_lambda =2. / (1. + np.exp(-10. * p)) - 1 #gradually change from 0 to 1
                                
                ##in each epoch, train all the mini batches
                for cBatch in range(num_batch):
                    ### each batch, half nodes from source network, and half nodes from target network
                    xs_ys_batch, shuffle_index_s = next(S_batches)
                    xs_batch=xs_ys_batch[0]
                    ys_batch =xs_ys_batch[1]
                    
                    xt_yt_batch, shuffle_index_t = next(T_batches)
                    xt_batch=xt_yt_batch[0]
                    yt_batch =xt_yt_batch[1]  
                    
                    x_batch = vstack([xs_batch, xt_batch])
                    batch_csr=x_batch.tocsr()
                    xb=utils.csr_2_sparse_tensor_tuple(batch_csr[:,0:n_input])
                    xb_nei=utils.csr_2_sparse_tensor_tuple(batch_csr[:,-n_input:])        
                    yb = np.vstack([ys_batch, yt_batch])
        
                    mask_L=np.array(np.sum(yb, axis=1)>0, dtype=np.float)#1 if the node is with observed label, 0 if the node is without label    
                    domain_label = np.vstack([np.tile([1., 0.], [batch_size // 2, 1]),np.tile([0., 1.], [batch_size // 2, 1])]) #[1,0] for source, [0,1] for target

                    ##topological proximity matrix between nodes in each mini-batch
                    a_s, a_t=utils.batchPPMI(batch_size,shuffle_index_s,shuffle_index_t,PPMI_s,PPMI_t)
                    
                    _ ,tloss= sess.run([model.train_op,model.total_loss], feed_dict={model.X: xb, model.X_nei:xb_nei, model.y_true: yb, model.d_label: domain_label, model.A_s: a_s, model.A_t: a_t, model.mask:mask_L, model.learning_rate: lr, model.Ada_lambda:grl_lambda, model.dropout:dropout})

                    

                    
                '''Compute evaluation on test data by the end of each epoch'''                
                pred_prob_xs_xt= sess.run(model.pred_prob, feed_dict={model.X:whole_xs_xt_stt, model.X_nei:whole_xs_xt_stt_nei, model.Ada_lambda:1.0, model.dropout:0.})                    
                pred_prob_xs=pred_prob_xs_xt[0:num_nodes_S,:]
                pred_prob_xt=pred_prob_xs_xt[-num_nodes_T:,:]
         
                print ('epoch: ', cEpoch+1) 
                F1_s=utils.f1_scores(pred_prob_xs,Y_s)
                print('Source micro-F1: %f, macro-F1: %f' %(F1_s[0],F1_s[1]))            
                F1_t=utils.f1_scores(pred_prob_xt,Y_t)
                print('Target testing micro-F1: %f, macro-F1: %f' %(F1_t[0],F1_t[1]))
                         
              
         
        
            ''' save final evaluation on test data by the end of all epoches'''
            micro=float(F1_t[0])
            macro=float(F1_t[1])  
        
                
            
            ##save embedding features
##            emb= sess.run(model.emb, feed_dict={model.X: whole_xs_xt_stt, model.X_nei:whole_xs_xt_stt_nei, model.Ada_lambda:1.0, model.dropout:0.})
##            hs=emb[0:num_nodes_S,:]
##            ht=emb[-num_nodes_T:,:]
##            print(np.shape(hs))
##            print(np.shape(ht))    
##            scipy.io.savemat(emb_filename+'_emb.mat', {'rep_S':hs, 'rep_T':ht})

            
    
    return micro,macro




