# ------------------------------------------
# Model train file for VPN
# Created By Srijan Das and Saurav Sharma
# ------------------------------------------

from pathlib import Path

from keras.callbacks import ReduceLROnPlateau, CSVLogger, Callback, EarlyStopping, ModelCheckpoint

# from multiprocessing import cpu_count
from NTU_gcnn_Loader import *
from embedding_gcnn_attention_model import *
from compute_adjacency import *
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def set_seed(value):
    np.random.seed(value)


class CustomModelCheckpoint(Callback):
    def __init__(self, model_parallel, path):
        super(CustomModelCheckpoint, self).__init__()
        self.save_model = model_parallel
        self.path = path
        self.nb_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.nb_epoch += 1
        if self.nb_epoch % 10 == 0:
            self.save_model.save(self.path + str(self.nb_epoch) + '.hdf5')


def get_adjacency_mat_for_pose(args):
    if args.num_nodes == 14:
        a_mat = compute_open_pose_adjacency(args.dataset, args.alpha, args.beta)
    else:
        a_mat = compute_adjacency(args.dataset, args.alpha, args.beta)

    a_mat = np.repeat(a_mat, args.batch_size, axis=0)
    a_mat = np.reshape(a_mat, [args.batch_size, a_mat.shape[1], a_mat.shape[1]])
    graph_conv_filters = preprocess_adj_tensor_with_identity(a_mat, args.sym_norm)
    return graph_conv_filters


def trainer(args):
    # set seed first
    set_seed(8)

    # define undirected graph for Input poses
    graph_conv_filters = get_adjacency_mat_for_pose(args)

    # create vpn model
    model = embed_model_spatio_temporal_gcnn(args.n_neuron, args.timesteps, args.num_nodes, args.num_features,
                                             graph_conv_filters.shape[1], graph_conv_filters.shape[2],
                                             args.num_filters, args.num_classes, args.n_dropout, args.protocol)

    # define loss and weightage to different loss components
    losses = {
        "action_output": "categorical_crossentropy",
        "embed_output": "mean_squared_error",
    }
    loss_weights = {"action_output": args.action_wt, "embed_output": args.embed_wt}

    # define optimizer
    optimizer = keras.optimizers.SGD(lr=args.lr, momentum=args.momentum)

    model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer, metrics=['accuracy'])

    # define data generators
    train_generator = DataGenerator(args.paths, graph_conv_filters, args.timesteps, args.train_ds_name,
                                    args.num_classes, args.stack_size, batch_size=args.batch_size,
                                    num_features=args.num_features, num_nodes=args.num_nodes)
    val_generator = DataGenerator(args.paths, graph_conv_filters, args.timesteps, args.test_ds_name, args.num_classes,
                                  args.stack_size, batch_size=args.batch_size,
                                  num_features=args.num_features, num_nodes=args.num_nodes)
    # model loggers
    csv_logger = CSVLogger('_'.join([args.model_name, args.dataset, '.csv']))
    reduce_lr = ReduceLROnPlateau(monitor=args.monitor, factor=args.factor, patience=args.patience)
    early_stopping = EarlyStopping(min_delta=1e-3, patience=10)
    # create folder to save model checkpoints if not already exists
    Path(os.path.join(args.weights_loc + args.model_name)).mkdir(parents=True, exist_ok=True)
    # model_checkpoint = CustomModelCheckpoint(model, os.path.join(args.weights_loc + args.model_name, 'epoch_'))
    model_checkpoint = ModelCheckpoint(filepath=os.path.join(args.weights_loc, args.model_name + '.hdf5'),
                                       save_best_only=True,
                                       monitor='val_action_output_accuracy',
                                       mode='max')

    print(f'Training for {args.dataset} dataset starts!')
    model.fit_generator(generator=train_generator,
                        validation_data=val_generator,
                        use_multiprocessing=args.multi_proc,
                        epochs=args.epochs,
                        callbacks=[csv_logger, model_checkpoint, reduce_lr, early_stopping],
                        # workers=cpu_count() - 2
                        )
    print(f'Training for {args.dataset} dataset is complete!')
