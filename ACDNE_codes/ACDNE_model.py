"""
Created on Mon May 27 19:18:27 2019

@author: Shen xiao

Please cite our paper as:
"Xiao Shen, Quanyu Dai, Fu-lai Chung, Wei Lu, and Kup-Sze Choi. Adversarial Deep Network Embedding for Cross-Network Node Classification. In Proceedings of AAAI Conference on Artificial Intelligence (AAAI), pages 2991-2999, 2020."

"""


import numpy as np
import tensorflow as tf
import utils
from flip_gradient import flip_gradient




class ACDNE(object):
    def __init__(self, n_input, n_hidden, n_emb, num_class, clf_type, l2_w, net_pro_w, batch_size):
        

        self.X = tf.sparse_placeholder(dtype=tf.float32) #each node's own attributes       
        self.X_nei = tf.sparse_placeholder(dtype=tf.float32) #each node's weighted neighbors' attributes
        self.y_true = tf.placeholder(dtype=tf.float32)
        self.d_label = tf.placeholder(dtype=tf.float32) #domain label, source network [1 0] or target network [0 1]
        self.Ada_lambda = tf.placeholder(dtype=tf.float32) #grl_lambda Gradient reversal scaler   
        self.dropout = tf.placeholder(tf.float32)
        self.A_s=tf.sparse_placeholder(dtype=tf.float32) #network proximity matrix of source network
        self.A_t=tf.sparse_placeholder(dtype=tf.float32) #network proximity matrix of target network
        self.mask= tf.placeholder(dtype=tf.float32) #check a node is with observable label (1) or not (0)
        self.learning_rate = tf.placeholder(dtype=tf.float32) 
        


        with tf.name_scope('Network_Embedding'):     
            ##feature exactor 1
            h1_self = utils.fc_layer(self.X, n_input, n_hidden[0], layer_name='hidden1_self', input_type='sparse',drop=self.dropout)
            h2_self = utils.fc_layer(h1_self, n_hidden[0], n_hidden[1], layer_name='hidden2_self')
             
            ##feature exactor 2
            h1_nei = utils.fc_layer(self.X_nei, n_input, n_hidden[0], layer_name='hidden1_nei', input_type='sparse',drop=self.dropout)    
            h2_nei = utils.fc_layer(h1_nei, n_hidden[0], n_hidden[1], layer_name='hidden2_nei')
            
            ##concatenation layer, final embedding vector representation
            self.emb = utils.fc_layer(tf.concat([h2_self, h2_nei], 1), n_hidden[-1]*2, n_emb, layer_name='concat')    
           
            
            ##pairwise constraint
            emb_s=tf.slice(self.emb, [0, 0], [int(batch_size / 2), -1])
            emb_t=tf.slice(self.emb, [int(batch_size / 2), 0], [int(batch_size / 2), -1])            
            #L2 distance between source nodes
            r_s = tf.reduce_sum(emb_s*emb_s, 1)
            r_s = tf.reshape(r_s, [-1, 1])
            Dis_s = r_s - 2*tf.matmul(emb_s, tf.transpose(emb_s)) + tf.transpose(r_s)
            net_pro_loss_s= tf.reduce_mean(tf.sparse.reduce_sum(self.A_s.__mul__(Dis_s), axis=1))
           
            #L2 distance between target nodes
            r_t = tf.reduce_sum(emb_t*emb_t, 1)
            r_t = tf.reshape(r_t, [-1, 1])
            Dis_t = r_t - 2*tf.matmul(emb_t, tf.transpose(emb_t)) + tf.transpose(r_t)                               
            net_pro_loss_t= tf.reduce_mean(tf.sparse.reduce_sum(self.A_t.__mul__(Dis_t), axis=1))
            
            self.net_pro_loss=net_pro_w * (net_pro_loss_s + net_pro_loss_t)
            
            
        with tf.name_scope('Node_Classifier'):
            ##node classification
            W_clf = tf.Variable(tf.truncated_normal([n_emb, num_class], stddev=1. / tf.sqrt(n_emb/2.)), name='clf_weight')
            b_clf = tf.Variable(tf.constant(0.1, shape=[num_class]), name='clf_bias')               
            pred_logit = tf.matmul(self.emb, W_clf) + b_clf 
            
            if clf_type == 'multi-class':
            ### multi-class, softmax output
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_logit, labels=self.y_true)       
                loss = loss * self.mask  #count loss only based on labeled nodes    
                self.clf_loss =tf.reduce_sum(loss)/tf.reduce_sum(self.mask)
                self.pred_prob =tf.nn.softmax(pred_logit)
                
            elif clf_type == 'multi-label':
            ### multi-label, sigmod output
                loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred_logit, labels=self.y_true)        
                loss = loss * self.mask[:,None] #count loss only based on labeled nodes, each column mutiply by mask
                self.clf_loss= tf.reduce_sum(loss)/tf.reduce_sum(self.mask)  
                self.pred_prob =tf.sigmoid(pred_logit)
            
        
        
            
            
        with tf.name_scope('Domain_Discriminator'):
            h_grl = flip_gradient(self.emb, self.Ada_lambda)               
            ##MLP for domain classification
            h_dann_1 = utils.fc_layer(h_grl, n_emb, 128, layer_name='dann_fc_1')  
            h_dann_2 = utils.fc_layer(h_dann_1, 128, 128, layer_name='dann_fc_2') 
            W_domain = tf.Variable(tf.truncated_normal([128, 2], stddev=1. / tf.sqrt(128 / 2.)), name='dann_weight')
            b_domain = tf.Variable(tf.constant(0.1, shape=[2]), name='dann_bias')
            d_logit = tf.matmul(h_dann_2, W_domain) + b_domain
            self.d_softmax = tf.nn.softmax(d_logit)
            self.domain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=d_logit, labels=self.d_label))

                
        all_variables = tf.trainable_variables()
        self.l2_loss =   l2_w * tf.add_n([tf.nn.l2_loss(v) for v in all_variables if 'bias' not in v.name])
        
        self.total_loss = self.net_pro_loss + self.clf_loss + self.domain_loss + self.l2_loss 
        
        self.train_op = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.total_loss)


