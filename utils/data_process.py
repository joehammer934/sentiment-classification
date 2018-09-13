from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
from lib import config


def read_data(choose_path,buy_path,no_path):
    text = []
    label = []
    text_length = []
    with open(choose_path) as f:
        for line in f:
            text.append([int(i) for i in line.strip().split()])
            label.append([1,0,0])
            text_length.append(len(line.strip().split()))

    with open(buy_path) as f:
        for line in f:
            text.append([int(i) for i in line.strip().split()])
            label.append([0,1,0])
            text_length.append(len(line.strip().split()))

    with open(no_path) as f:
        for line in f:
            text.append([int(i) for i in line.strip().split()])
            label.append([0,0,1])
            text_length.append(len(line.strip().split()))

    padding_text = pad_sequences(text, maxlen=config.MAX_SEQUENCE_LENGTH)
    return padding_text, text, label, text_length

def data_split(choose_path,buy_path,no_path):
    """
    对数据进行切分，分为训练集和测试集
    :param padding_text: padding之后的文本
    :param labels: 数据的label
    :return:
    """
    padding_text, text, labels, text_length = read_data(choose_path,buy_path,no_path)
    # 打乱数据集的顺序
    indeices = np.arange(len(padding_text))
    np.random.shuffle(indeices)
    padding_text= padding_text[indeices]
    labels = np.array(labels)[indeices]

    num_validation_samples = int(len(padding_text)*config.VALID_SPLIT)
    # 切割数据
    x_train = padding_text[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_test = padding_text[-num_validation_samples:]
    y_test = labels[-num_validation_samples:]

    return x_train,y_train,x_test,y_test


def generate_batch(data,label):
    batch_size = config.BATCH_SIZE
    index = [i for i in range(len(data))]
    batch_index = random.sample(index, batch_size)

    batch_data = []
    batch_label = []
    for i in batch_index:
        batch_data.append(data[i])
        batch_label.append(label[i])
    return batch_data,batch_label