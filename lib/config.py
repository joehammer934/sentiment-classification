#data path
WORD_VECTOR_PATH = "./word_vector/yiche_data"
VOCABULARY_PATH = "./data/vocabulary_data/vocab"
ORGINAL_PATH = "./data/original_data/"
TOCKEN_PATN = "./data/tocken_data/"
EMBEDDING_DIM = 256

choose_car_path = "./data/tocken_data/choose_car"
buy_car_path = "./data/tocken_data/buy_car"
no_car_path = "./data/tocken_data/no_car"

###########################################
#neural network paramters
vocabulary_size = 112300 + 2
BATCH_SIZE = 64
keep_prob = 1.0
class_num = 3
##用于防止梯度消失和梯度爆炸
max_grad_norm = 0.5

max_steps = 3
hidden_neural_size = 256
hidden_layer_num = 3
learning_rate = 0.01
###########################################
#data_paramters
MAX_SEQUENCE_LENGTH = 50
VALID_SPLIT = 0.2

###########################################
#checkpoint_path
checkpoint_path = "./model/"

###########################################
#builder path
export_path_base = "./builder/"
version = 1
sever_model_path = export_path_base + "100"