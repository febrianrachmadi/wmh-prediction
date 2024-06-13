import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, BatchNormalization
from tensorflow.keras.layers import Activation, UpSampling2D, Multiply, Concatenate

def handle_block_names(stage, cols, type_block='decoder', type_act='relu'):
    conv_name = '{}_stage{}-{}_conv'.format(type_block, stage, cols)
    bn_name = '{}_stage{}-{}_bn'.format(type_block, stage, cols)
    act_name = '{}_stage{}-{}_{}'.format(type_block, stage, cols, type_act)
    return conv_name, bn_name, act_name


def handle_att_names(stage, cols, type_block='decoder', type_act='relu'):
    conv_name = '{}_stage{}-{}_conv'.format(type_block, stage, cols)
    bn_name = '{}_stage{}-{}_bn'.format(type_block, stage, cols)
    act_name = '{}_stage{}-{}_relu'.format(type_block, stage, cols)
    up_name = '{}_stage{}-{}_upat'.format(type_block, stage, cols)
    add_name = '{}_stage{}-{}_add'.format(type_block, stage, cols)
    sigmoid_name = '{}_stage{}-{}_sigmoid'.format(type_block, stage, cols)
    mul_name = '{}_stage{}-{}_mul'.format(type_block, stage, cols)
    merge_name = 'merge_{}-{}'.format(stage, cols)

    return conv_name, bn_name, act_name, up_name, merge_name, add_name, sigmoid_name, mul_name


def conv_relu(filters, kernel_size, use_batchnorm=False, conv_name='conv',
              bn_name='bn', act_name='relu', act_function='relu'):
  
    def layer(input_tensor):
        x = Conv2D(filters, kernel_size, padding='same', name=conv_name) (input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name) (x)
        x = Activation(act_function, name=act_name) (x)

        return x
    return layer

def conv_block(filters, stage, cols, kernel_size=3, use_batchnorm=True,
               amount=3, type_act='relu', type_block='encoder'):
    
    def layer(x):
        act_function = tf.identity if type_act == 'identity' else type_act
        conv_name, bn_name, act_name = handle_block_names(stage, cols, type_block=type_block, type_act=type_act)
        for i in range(amount):
            temp = '_'+str(i+1)
            x = conv_relu(filters, kernel_size=kernel_size, use_batchnorm=use_batchnorm, 
                          conv_name=conv_name+temp, bn_name=bn_name+temp,
                          act_name=act_name+temp, act_function=act_function) (x)
        return x
    return layer

def attention_block(filters, skip, stage, cols, upsample_rate=(2,2)):

    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name,_, add_name, sigmoid_name, mul_name = handle_att_names(stage, cols, type_block='attention')

        x_up = conv_relu(filters, kernel_size=3, conv_name=conv_name+'_before', bn_name=bn_name+'_before', act_name=relu_name+'_before') (input_tensor)

        x1 = Conv2D(filters, kernel_size=1, padding='same',
                    name=conv_name+'_skip') (skip)
        x1 = BatchNormalization(name=bn_name+'1') (x1)
        x2 = Conv2D(filters, kernel_size=1, padding='same',
                    name=conv_name+'_up') (x_up)
        x2 = BatchNormalization(name=bn_name+'2') (x2)
        
        x = Add(name=add_name) ([x1,x2])
        x = Activation('relu', name=relu_name) (x)
        x = Conv2D(1, kernel_size=1, padding='same', name=conv_name) (x)
        x = BatchNormalization(name=bn_name+'3') (x)
        x = Activation('sigmoid', name=sigmoid_name) (x)
        x = Multiply(name=mul_name) ([skip,x])

        return x
    return layer

def concatenation_block(filters, skip, stage, cols, upsample_rate=(2,2)):

    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name, merge_name, add_name, sigmoid_name, mul_name = handle_att_names(stage, cols, type_block='concatenation')

        x_up = conv_relu(filters, kernel_size=3, conv_name=conv_name+'_l1', bn_name=bn_name+'_l1', act_name=relu_name+'_l1') (input_tensor)
        x_up = conv_relu(filters, kernel_size=3, conv_name=conv_name+'_l2', bn_name=bn_name+'_l2', act_name=relu_name+'_l2') (x_up)

        x = Concatenate(name=merge_name) ([skip, x_up])
        
        return x
    return layer

def z_mu_sigma(filters, stage, cols, use_batchnorm=True, type_block='z'):
    def layer(x):
        mu = conv_block(filters, stage, cols, use_batchnorm=use_batchnorm, amount=1,
                        kernel_size=1, type_act='identity', type_block='mu') (x)
        sigma = conv_block(filters, stage, cols, use_batchnorm=use_batchnorm, amount=1,
                           kernel_size=1, type_act='softplus', type_block='sigma') (x)
                           
        z = Multiply(name='z_stage{}-{}_mul'.format(stage,cols)) ([sigma,
                                                                   tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)])
        z = Add(name='z_stage{}-{}_add'.format(stage,cols)) ([mu, z])
        return z, mu, sigma
    return layer

def increase_resolution(filters, stage, cols, times):
    def layer(x):
        for i in range(times):
            x = UpSampling2D(name='z_post_stage{}-{}_up_{}'.format(stage, cols, str(times+1))) (x)
            x = conv_block(filters, stage, cols, amount=1, type_block='z_post') (x)
        return x
    return layer