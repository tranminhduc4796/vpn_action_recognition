import os

from keras import regularizers
from i3d_inception import Inception_Inflated3d, conv3d_bn
from keras.layers import Dense, Flatten, Dropout, Reshape, Input, RepeatVector, Permute
from keras.layers import AveragePooling3D, Lambda
from keras import backend as K
from keras_dgl.layers import MultiGraphCNN
from keras.models import Model
import keras
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import normalize


os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"


def inflate_dense_spatial(x):
    a = RepeatVector(2 * 1024)(x)
    a = Permute((2, 1), input_shape=(49, 2 * 1024))(a)
    a = Reshape((2, 7, 7, 1024))(a)
    return a


def inflate_dense_temporal(x):
    a = RepeatVector(49 * 1024)(x)
    a = Permute((2, 1), input_shape=(2, 49 * 1024))(a)
    a = Reshape((2, 7, 7, 1024))(a)
    return a


def attention_reg(weight_mat):
    return 0.00001 * K.square((1 - K.sum(weight_mat)))


def embedding_loss(x, y):
    return normalize(x / normalize(x) - y / normalize(y))

class i3d_modified:
    def __init__(self, weights='rgb_imagenet_and_kinetics'):
        self.model = Inception_Inflated3d(include_top=True, weights=weights)

    def i3d_flattened(self, num_classes=60):
        i3d = Model(inputs=self.model.input, outputs=self.model.get_layer(index=-4).output)
        x = conv3d_bn(i3d.output, num_classes, 1, 1, 1, padding='same', use_bias=True, use_activation_fn=False,
                      use_bn=False, name='Conv3d_6a_1x1')
        num_frames_remaining = int(x.shape[1])
        x = Flatten()(x)
        predictions = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                            activity_regularizer=regularizers.l1(0.01))(x)
        new_model = Model(inputs=i3d.input, outputs=predictions)

        # for layer in i3d.layers:
        #    layer.trainable = False

        return new_model


def GCNN_skeleton_t16(num_nodes, num_features, graph_conv_filters_shape1, graph_conv_filters_shape2, num_filters,
                      n_neuron, n_dropout, timesteps):
    print('Build GCNN')
    X_input_t1 = Input(shape=(num_nodes, num_features))
    X_input_t2 = Input(shape=(num_nodes, num_features))
    X_input_t3 = Input(shape=(num_nodes, num_features))
    X_input_t4 = Input(shape=(num_nodes, num_features))
    X_input_t5 = Input(shape=(num_nodes, num_features))
    X_input_t6 = Input(shape=(num_nodes, num_features))
    X_input_t7 = Input(shape=(num_nodes, num_features))
    X_input_t8 = Input(shape=(num_nodes, num_features))
    X_input_t9 = Input(shape=(num_nodes, num_features))
    X_input_t10 = Input(shape=(num_nodes, num_features))
    X_input_t11 = Input(shape=(num_nodes, num_features))
    X_input_t12 = Input(shape=(num_nodes, num_features))
    X_input_t13 = Input(shape=(num_nodes, num_features))
    X_input_t14 = Input(shape=(num_nodes, num_features))
    X_input_t15 = Input(shape=(num_nodes, num_features))
    X_input_t16 = Input(shape=(num_nodes, num_features))

    X_input = Input(shape=(timesteps, num_nodes, num_features))

    graph_conv_filters_input = Input(shape=(graph_conv_filters_shape1, graph_conv_filters_shape2))

    output_t1 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t1, graph_conv_filters_input])
    output_t1 = Dropout(n_dropout)(output_t1)

    output_t2 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t2, graph_conv_filters_input])
    output_t2 = Dropout(n_dropout)(output_t2)

    output1 = Lambda(lambda x: K.expand_dims(x, axis=1))(
        output_t1)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
    output2 = Lambda(lambda x: K.expand_dims(x, axis=1))(output_t2)

    output_t3 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t3, graph_conv_filters_input])
    output_t3 = Dropout(n_dropout)(output_t3)

    output_t4 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t4, graph_conv_filters_input])
    output_t4 = Dropout(n_dropout)(output_t4)

    output3 = Lambda(lambda x: K.expand_dims(x, axis=1))(
        output_t3)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
    output4 = Lambda(lambda x: K.expand_dims(x, axis=1))(output_t4)

    output_t5 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t5, graph_conv_filters_input])
    output_t5 = Dropout(n_dropout)(output_t5)

    output_t6 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t6, graph_conv_filters_input])
    output_t6 = Dropout(n_dropout)(output_t6)

    output5 = Lambda(lambda x: K.expand_dims(x, axis=1))(
        output_t5)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
    output6 = Lambda(lambda x: K.expand_dims(x, axis=1))(output_t6)

    output_t7 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t7, graph_conv_filters_input])
    output_t7 = Dropout(n_dropout)(output_t7)

    output_t8 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t8, graph_conv_filters_input])
    output_t8 = Dropout(n_dropout)(output_t8)

    output7 = Lambda(lambda x: K.expand_dims(x, axis=1))(
        output_t7)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
    output8 = Lambda(lambda x: K.expand_dims(x, axis=1))(output_t8)

    output_t9 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t9, graph_conv_filters_input])
    output_t9 = Dropout(n_dropout)(output_t9)

    output_t10 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t10, graph_conv_filters_input])
    output_t10 = Dropout(n_dropout)(output_t10)

    output9 = Lambda(lambda x: K.expand_dims(x, axis=1))(
        output_t9)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
    output10 = Lambda(lambda x: K.expand_dims(x, axis=1))(output_t10)

    output_t11 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t11, graph_conv_filters_input])
    output_t11 = Dropout(n_dropout)(output_t11)

    output_t12 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t12, graph_conv_filters_input])
    output_t12 = Dropout(n_dropout)(output_t12)

    output11 = Lambda(lambda x: K.expand_dims(x, axis=1))(
        output_t11)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
    output12 = Lambda(lambda x: K.expand_dims(x, axis=1))(output_t12)

    output_t13 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t13, graph_conv_filters_input])
    output_t13 = Dropout(n_dropout)(output_t13)

    output_t14 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t14, graph_conv_filters_input])
    output_t14 = Dropout(n_dropout)(output_t14)

    output13 = Lambda(lambda x: K.expand_dims(x, axis=1))(
        output_t13)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
    output14 = Lambda(lambda x: K.expand_dims(x, axis=1))(output_t14)

    output_t15 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t15, graph_conv_filters_input])
    output_t15 = Dropout(n_dropout)(output_t15)

    output_t16 = MultiGraphCNN(n_neuron, num_filters, activation='elu')([X_input_t16, graph_conv_filters_input])
    output_t16 = Dropout(n_dropout)(output_t16)

    output15 = Lambda(lambda x: K.expand_dims(x, axis=1))(
        output_t15)  # adding a node invariant layer to make sure output does not depends upon the node order in a graph.
    output16 = Lambda(lambda x: K.expand_dims(x, axis=1))(output_t16)

    output = keras.layers.Concatenate(axis=1)(
        [output1, output2, output3, output4, output5, output6, output7, output8, output9,
         output10, output11, output12, output13, output14, output15, output16])
    output = keras.layers.Concatenate()([output, X_input])
    out = BatchNormalization()(output)
    out = Conv2D(64, (3, 3), activation='relu', padding='same')(out)
    # out = MaxPooling2D(pool_size=(2,2))(out)
    out = BatchNormalization()(out)
    out = Conv2D(64, (3, 3), activation='relu', padding='same')(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = BatchNormalization()(out)
    out = Conv2D(128, (3, 3), activation='relu', padding='same')(out)
    out = MaxPooling2D(pool_size=(2, 2))(out)
    out = BatchNormalization()(out)
    out = Dropout(0.5)(out)
    out = Flatten()(out)
    out_new = Dense(256, activation='relu')(out)
    out_new = Dense(128, activation='relu')(out_new)
    out_new = Dropout(n_dropout, name='gcnn_out')(out_new)
    # output_final = Dense(num_classes, activation='softmax')(out_new)
    model = Model(inputs=[X_input_t1, X_input_t2, X_input_t3, X_input_t4, X_input_t5, X_input_t6, X_input_t7,
                          X_input_t8, X_input_t9, X_input_t10, X_input_t11, X_input_t12, X_input_t13, X_input_t14,
                          X_input_t15, X_input_t16, X_input, graph_conv_filters_input], outputs=out_new)
    return model


def embed_model_spatio_temporal_gcnn(n_neuron, timesteps, num_nodes, num_features,
                                     graph_conv_filters_shape1, graph_conv_filters_shape2,
                                     num_filters, num_classes, n_dropout, protocol):
    i3d = i3d_modified(weights='rgb_imagenet_and_kinetics')
    model_branch = i3d.i3d_flattened(num_classes=num_classes)
    '''
    if protocol == 'CS': # to be replaced with the values in the yaml file
       model_branch.load_weights('/data/stars/user/sdas/PhD_work/STA_appearance/NTU_CS/i3d/weights_ntu_aug_4/epoch_7.hdf5')
    else:
       model_branch.load_weights('/data/stars/user/sdas/PhD_work/CVPR20/NTU_120/I3D/weights_ntu_set_i3d_full_body/epoch_12.hdf5')
    '''
    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9)
    model_branch.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print('Build model...')
    model_inputs = []
    model_gcnn = GCNN_skeleton_t16(num_nodes, num_features, graph_conv_filters_shape1,
                                   graph_conv_filters_shape2, num_filters,
                                   n_neuron, n_dropout, timesteps)

    z1 = Dense(256, activation='tanh', name='z1_layer', trainable=True)(model_gcnn.get_layer('gcnn_out').output)
    z2 = Dense(128, activation='tanh', name='z2_layer', trainable=True)(model_gcnn.get_layer('gcnn_out').output)

    fc_main_spatial = Dense(49, activity_regularizer=attention_reg,
                            kernel_initializer=keras.initializers.GlorotNormal(),
                            bias_initializer='zeros',
                            activation='sigmoid', trainable=True, name='dense_spatial')(z1)
    fc_main_temporal = Dense(2, activity_regularizer=attention_reg,
                             kernel_initializer=keras.initializers.GlorotNormal(),
                             bias_initializer='zeros',
                             activation='softmax', trainable=True, name='dense_temporal')(z2)
    atten_mask_spatial = keras.layers.core.Lambda(inflate_dense_spatial,
                                                  output_shape=(2, 7, 7, 1024))(fc_main_spatial)
    atten_mask_temporal = keras.layers.core.Lambda(inflate_dense_temporal,
                                                   output_shape=(2, 7, 7, 1024))(fc_main_temporal)
    atten_mask = keras.layers.Multiply()([atten_mask_spatial, atten_mask_temporal])

    for layer in model_branch.layers:
        layer.trainable = True

    for layer in model_gcnn.layers:
        layer.trainable = True

    for i in model_gcnn.input:
        model_inputs.append(i)
    model_inputs.append(model_branch.input)

    flatten_video = Flatten(name='flatten_video')(model_branch.get_layer('Mixed_5c').output)
    embed_video = Dense(256,
                        activation='sigmoid',
                        trainable=True,
                        kernel_constraint=keras.constraints.UnitNorm(axis=0),
                        name='dense_video')(flatten_video)
    embed_skeleton = Dense(256,
                           activation='sigmoid',
                           trainable=True,
                           kernel_constraint=keras.constraints.UnitNorm(axis=0),
                           name='dense_skeleton')(fc_main_spatial)

    embed_output = Lambda(lambda x: embedding_loss(x[0], x[1]), output_shape=lambda inp_shp: (inp_shp[0][0], 1),
                          name='embed_output')([embed_video, embed_skeleton])

    multiplied_features = keras.layers.Multiply()([atten_mask, model_branch.get_layer('Mixed_5c').output])
    added_features = keras.layers.Add()([multiplied_features, model_branch.get_layer('Mixed_5c').output])

    x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool' + 'second')(
        added_features)
    x = Dropout(n_dropout)(x)

    x = conv3d_bn(x, num_classes, 1, 1, 1, padding='same', use_bias=True, use_activation_fn=False, use_bn=False,
                  name='Conv3d_6a_1x1' + 'second')

    x = Flatten(name='flatten' + 'second')(x)
    predictions = Dense(num_classes, activation='softmax', name='action_output')(x)
    model = Model(inputs=model_inputs, outputs=[predictions, embed_output], name='spatial_temporal_attention')

    return model
