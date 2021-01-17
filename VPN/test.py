from keras.callbacks import CSVLogger, Callback, ModelCheckpoint

from NTU_gcnn_Loader import *
from embedding_gcnn_attention_model import *
from compute_adjacency import *
from utils import *
from keras.models import load_model


os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

model_name = 'vpn'
protocol = 'cv'
num_classes = 6
batch_size = 4
stack_size = 16
n_neuron = 64
n_dropout = 0.3
timesteps = 16
seed = 8
np.random.seed(seed)

losses = {
    "action_output": "categorical_crossentropy",
    "embed_output": "mean_squared_error",
}
lossWeights = {"action_output": 0.9, "embed_output": 0.1}

alpha = 5
beta = 2
dataset_name = 'NTU'
num_features = 2
num_nodes = 14

A = compute_open_pose_adjacency(dataset_name, alpha, beta)
A = np.repeat(A, batch_size, axis=0)
A = np.reshape(A, [batch_size, A.shape[1], A.shape[1]])
SYM_NORM = True
num_filters = 2
graph_conv_filters = preprocess_adj_tensor_with_identity(A, SYM_NORM)

model_path = 'best'
model = load_model(model_path)

paths = {
    'skeleton': '../skeleton/vpn_2d_npy/',
    'cnn': '../video/',
    'split_path': '../splits/'
}

test = 'validation'

val_generator = DataGenerator(paths, graph_conv_filters, timesteps, test, num_classes, stack_size,
                              batch_size=batch_size, num_nodes=num_nodes, num_features=num_features)

model.evaluate(val_generator, return_dict=True)
