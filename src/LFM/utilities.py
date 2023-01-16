import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

floatf = tf.float32

tffloat = lambda x: tf.constant(x, dtype=floatf)

def pshape(array): print(str(array)+'.shape', array.shape)

def expHighLowFunc_01(high, low, x):
    return low * tf.exp( x * tf.math.log( high/low ) ) 

def exp_start_end_x_func(start, end, x):
    return start * tf.exp( x * tf.math.log( end/start ) ) 

#########################################################
# Probability Utilities
#########################################################

#### diag cov

# We define the function used to create samples using the reparametrization trick
def reparameterize( mean, logvar):
    eps = tf.random.normal(shape=mean.shape, dtype=tf.float32)
    z = eps * tf.exp(logvar * .5) + mean
    tf.debugging.assert_all_finite(z, 'z reparam not finite')
    return z   


log2pi = tf.math.log( tf.constant(2* np.pi, dtype=tf.float32))
def log_normal_pdf(sample, mean, logvar, raxis=1):
    return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)



#### low rank cov

def reparameterize_low_rank( mean, logvar, corr, lr_k ):
    
    mean   = tf.reshape(mean,   [-1,1])
    logvar = tf.reshape(logvar, [-1,1])
    corr   = tf.reshape(corr,   [-1,lr_k])

    eps      = tf.random.normal(shape=mean.shape, dtype=tf.float32)
    eps_corr = tf.random.normal(shape=[corr.shape[1],1], dtype=tf.float32)
    z = tf.linalg.matmul(corr, eps_corr) + eps * tf.exp(logvar * .5) + mean
    tf.debugging.assert_all_finite(z, 'z reparam not finite')
    return z   


def log_normal_pdf_low_rank( sample, mean, logvar, corr, lr_k ):
    
    sample = tf.reshape(sample,   [-1,1])
    mean   = tf.reshape(mean,   [-1,1])
    logvar = tf.reshape(logvar, [-1,1])
    corr   = tf.reshape(corr,   [-1,lr_k])

    n = mean.shape[0]
    
    lam = tf.reshape(tf.exp(logvar), [-1])

    lam_inv_corr = tf.einsum("i,ij->ij", 1/lam, corr)
    
    mat = tf.eye(lr_k) + tf.linalg.matmul(corr, lam_inv_corr, transpose_a=True)

    cholMat = tf.linalg.cholesky(mat)

    tf.debugging.assert_all_finite(cholMat, 'cholMat')

    a         = tf.reshape(sample, [-1,1]) - tf.reshape(mean, [-1,1])
    lam_inv_a = tf.einsum("i,i->i",(1/lam), a[:,0])[:,None]

    tmp = tf.linalg.matmul(corr, lam_inv_a, transpose_a=True)
    # tmp_new = tf.linalg.solve(mat, tmp)
    tf.debugging.assert_all_finite(tmp, 'tmp')

    tmp_new = tf.linalg.cholesky_solve(cholMat, tmp)

    tf.debugging.assert_all_finite(tmp_new, 'tmp_new not finite')

    sol = lam_inv_a - tf.linalg.matmul(lam_inv_corr, tmp_new)

    likelihood = tf.linalg.matmul(a, sol,transpose_a=True)
    
    # log_det = tf.math.log(det_low_rank(lam, corr))
    log_det =  2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(cholMat))) + tf.reduce_sum(tf.math.log(lam))
    
    res = -0.5*(n * log2pi + log_det + likelihood)

    return res


#########################################################


#########################################################
# Plotting Utilities
#########################################################

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    'axes.labelsize':   16,
    'axes.titlesize':   16,
    'xtick.labelsize' : 16,
    'ytick.labelsize' : 16
          })
# latex font definition
plt.rc('text', usetex=True)
plt.rc('font', **{'family':'serif','serif':['Computer Modern Roman']})
from matplotlib.transforms import Bbox

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)
    
#########################################################
