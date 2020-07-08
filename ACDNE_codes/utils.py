import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.io as sio
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import scipy
from sklearn.metrics import f1_score
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import normalize


def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return shuffle_index, [d[shuffle_index] for d in data]

def csr_2_sparse_tensor_tuple(csr_matrix):
    if not isinstance(csr_matrix, scipy.sparse.lil_matrix):
        csr_matrix = lil_matrix(csr_matrix)    
    coo_matrix = csr_matrix.tocoo()
    indices = np.transpose(np.vstack((coo_matrix.row, coo_matrix.col)))
    values = coo_matrix.data
    shape = csr_matrix.shape
    return indices, values, shape



def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        shuffle_index, data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                shuffle_index,data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data],shuffle_index[start:end]
        


        


def fc_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, input_type='dense', drop=0.0):
    with tf.name_scope(layer_name):
        weight = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1. / tf.sqrt(input_dim / 2.)), name='weight')
        bias = tf.Variable(tf.constant(0.1, shape=[output_dim]), name='bias')
        if input_type == 'sparse':
            activations = act(tf.sparse_tensor_dense_matmul(input_tensor, weight) + bias)
        else:
            activations = act(tf.matmul(input_tensor, weight) + bias)
        

        activations = tf.nn.dropout(activations, rate=drop)
            
        return activations





def load_network(file):

    net = sio.loadmat(file)
    X, A, Y = net['attrb'], net['network'], net['group']
    if not isinstance(X, scipy.sparse.lil_matrix):
        X = lil_matrix(X)
    
    return A, X, Y




def MyScaleSimMat(W):
    '''L1 row norm of a matrix'''   
    rowsum = np.array(np.sum(W, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    W = r_mat_inv.dot(W)
    return W


def AggTranProbMat(G, step):
    '''aggregated K-step transition probality'''
    G=MyScaleSimMat(G)
    G=csc_matrix.toarray(G)
    A_k = G
    A=G
    for k in np.arange(2,step+1):
        A_k=np.matmul(A_k,G)
        A=A+A_k/k
    
    return A
    

def ComputePPMI(A):
    '''compute PPMI, given aggregated K-step transition probality matrix as input'''
    np.fill_diagonal(A, 0)
    A = MyScaleSimMat(A)
    (p, q) = np.shape(A)
    col = np.sum(A, axis=0)
    col[col==0]=1    
    PPMI = np.log( (float(p)*A)/col[None,:]) 
    IdxNan = np.isnan(PPMI)
    PPMI[IdxNan] = 0
    PPMI[PPMI<0]=0 
    
    return PPMI
    
    

def batchPPMI(batch_size,shuffle_index_s,shuffle_index_t,PPMI_s,PPMI_t):
    '''return the PPMI matrix between nodes in each batch'''

    ##proximity matrix between source network nodes in each mini-batch
    a_s=np.zeros((int(batch_size / 2),int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            if ii != jj:
                a_s[ii,jj]=PPMI_s[shuffle_index_s[ii],shuffle_index_s[jj]]
    
    ##proximity matrix between target network nodes in each mini-batch
    a_t=np.zeros((int(batch_size / 2),int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            if ii != jj:
                a_t[ii,jj]=PPMI_t[shuffle_index_t[ii],shuffle_index_t[jj]] 
                
    return csr_2_sparse_tensor_tuple(MyScaleSimMat(a_s)),csr_2_sparse_tensor_tuple(MyScaleSimMat(a_t))



def feature_compression(features, dim=200):
    """Preprcessing of features"""
    features = features.toarray()    
    feat = lil_matrix(PCA(n_components=dim, random_state=0).fit_transform(features))
    return feat



def f1_scores(y_pred,y_true):


    def predict(y_true, y_pred):
        top_k_list = np.array(np.sum(y_true, 1), np.int32)
        predictions = []
        for i in range(y_true.shape[0]):
            pred_i = np.zeros(y_true.shape[1])
            pred_i[np.argsort(y_pred[i, :])[-top_k_list[i]:]] = 1
            predictions.append(np.reshape(pred_i, (1, -1)))
        predictions = np.concatenate(predictions, axis=0)
        
        return np.array(predictions, np.int32)

    results = {}
    predictions = predict(y_true, y_pred)
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)


    return results["micro"], results["macro"]
