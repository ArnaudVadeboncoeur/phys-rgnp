
from .utilities import *
import tensorflow as tf


class FFLayer(tf.keras.layers.Layer):

  def __init__(self, freqs, trainable=False ):
      super(FFLayer, self).__init__()
      self.units = freqs.shape[0] * 2
      self.pi    = tf.cast(np.pi, tf.float32) 
      self.freqs = freqs
      self.trainable = trainable

  def build(self, input_shape):  # Create the state of the layer (weights)
    # f_init = tf.random_normal_initializer()
    # self.f = tf.Variable(
    #                     initial_value=f_init(shape=(input_shape[-1], int(self.units/2)),
    #                                         dtype='float32'),
    #                     trainable=True)
    # self.f = tf.Variable( initial_value=self.freqDist.sample(sample_shape=(input_shape[-1], int(self.units/2))),
    #                       trainable=True )

    self.f = tf.Variable( initial_value=tf.reshape( self.freqs, [input_shape[-1], -1] ), trainable=self.trainable )
    # self.f = tf.Variable( initial_value=tf.reshape( self.freqs, [input_shape[-1], int(self.units/2)] ), trainable=True )

  def call(self, inputs):  # Defines the computation from inputs to outputs
      return tf.concat([tf.sin( 2. * self.pi * tf.matmul(inputs, self.f)), tf.cos( 2. * self.pi * tf.matmul(inputs, self.f))], -1)

      

class KIPLayer(tf.keras.layers.Layer):

    def __init__(self, k, ell, trainable=True):
        super(KIPLayer, self).__init__()
        # self.X_out = X_out # [num out pts, dim]
        self.ell   = tf.Variable(ell, trainable=trainable)
        self.k     = k.stretch(self.ell) # form kernel with particular lengthscale

    def kernel_mat(self, X, Y):
        return self.k(X,Y).mat
    
    def call(self, X_in, y_in, X_out):

        # X_in = tf.reshape(X_in, [-1,X_out.shape[1]])
        # y_in = tf.reshape(y_in, [-1,X_out.shape[1]])

        # get kernel mat between X_out and X_in:
        K = self.kernel_mat(X_out, X_in)
        # compute interpolation
        num   = tf.matmul(K, y_in)
        denom = tf.reduce_sum(K, axis=1, keepdims=True) + 1e-6
        val   = num / denom
        return val
        

class GICNet( tf.keras.Model ):

    def __init__(self, grid_kipr, kip_res, dim_branch_in_y, dim_branch_P, dimX  ):
        super(GICNet, self).__init__( )

        self.grid_kipr      = grid_kipr
        self.kip_res        = kip_res
        self.dim_branch_in_y = dim_branch_in_y
        self.dim_branch_P    = dim_branch_P
        self.dimX           = dimX
        

    def branchNetCall_conv(self, x_input):

        #x_input (field - dims 0 to n, coords - dims 0 to n)

        for layer in self.P_kLayers:
            x_input_y = tf.transpose(x_input[:, -self.dim_branch_in_y:], [0,1] )

            x_input_y = layer( x_input_y )

        # x_input = tf.concat([x_input[:, :-self.dim_trunc_in_y], tf.transpose(x_input_y, [0,1]) ], 1) -0

        x_input_x = x_input[:, :-self.dim_branch_in_y]

        # kernel interp layer
        # numX           = x_input.shape[1] - self.dim_trunc_P -0
        # x_input_concat = tf.zeros(shape=[self.grid_kipr.shape[0], 1]) -0
        x_input_concat = self.grid_kipr

        i = 0
        for layer in self.kipLayers:
            #split out x_input, y_input (coded for scalar field)
            # kipReturn      = layer(x_input[:, :-self.dim_trunc_P], x_input[:, numX+i:numX+i+1], self.grid_kipr) -0
            kipReturn      = layer(x_input_x, x_input_y[:, i:i+1], self.grid_kipr)

            x_input_concat = tf.concat([x_input_concat, kipReturn],1)
            i += 1

        # x_input = x_input_concat[:, 1:] -0
        x_input = x_input_concat

        x_input = tf.reshape( x_input, shape=tf.concat([[1], self.kip_res, [self.dim_branch_P + self.dimX]], 0) )

        for layer in self.convLayers:
            x_input = layer( x_input )

        x_input = tf.reshape(x_input, [1, -1])
        
        return x_input

    
    def truncNetCall(self, x_coord):

        #x_coord is points by dims of domain

        for layer in self.FFMlayer:
            x_coord = layer(x_coord)

        for layer in self.preTrunclayers:
            x_coord = layer(x_coord)

        return x_coord

    #input [batch, loc, s0]
    # @tf.function
    def call(self, x_input, x_coord):
        
        x_input = self.branchNetCall_conv(x_input)

        x_coord = self.truncNetCall(x_coord)

        x_input  = tf.tile(x_input, [x_coord.shape[0], 1])

        xxi = tf.concat([x_coord, x_input], 1)

        for layer in self.DeepOlayers:
            xxi = layer(xxi)
        
        return xxi
