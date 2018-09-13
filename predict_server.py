import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from utils import data_utils,data_process
from lib import config
import model_utils.online_lstm_model


vocb, rev_vocb = data_utils.initialize_vocabulary(config.VOCABULARY_PATH)
embedding_matrix, word_list = data_utils.load_pretained_vector(config.WORD_VECTOR_PATH)
test_sentence = ["我","讨厌","这","辆","车"]
test_token_sentence = [[vocb[i.encode('utf-8')] for i in test_sentence]]
padding_sentence = pad_sequences(test_token_sentence, maxlen=config.MAX_SEQUENCE_LENGTH)

with tf.Session() as sess:
    signature_key = "serving_default"
    input_sentence = "input_sentence"
    # word_embedding = "word_embedding"
    predict_classification = "predict_classification"
    predict_scores = "predict_scores"

    meta_graph_def = tf.saved_model.loader.load(sess,
                                                [tf.saved_model.tag_constants.SERVING],
                                                config.sever_model_path)
    #从meta_graph_def中取出SignatureDef对象
    signature = meta_graph_def.signature_def

    #从signature中找出具体输入输出的tensor name
    input_sentence_tensor_name = signature[signature_key].inputs[input_sentence].name
    # word_embedding_tensor_name = signature[signature_key].inputs[word_embedding].name
    predict_classification_tensor_name = signature[signature_key].outputs[predict_classification].name
    predict_scores_tensor_name = signature[signature_key].outputs[predict_scores].name

    #获取tensor 并inference
    input_sentence_inference = sess.graph.get_tensor_by_name(input_sentence_tensor_name)
    # word_embedding_inference = sess.graph.get_tensor_by_name(word_embedding_tensor_name)
    predict_classification_inference = sess.graph.get_tensor_by_name(predict_classification_tensor_name)
    predict_scores_inference = sess.graph.get_tensor_by_name(predict_scores_tensor_name)

    #预测
    classification, scores = sess.run([predict_classification_inference, predict_scores_inference],
                                      feed_dict={
                                          input_sentence_inference: padding_sentence,
                                          # word_embedding_inference: embedding_matrix
                                      })
    print("classification is:",classification)
    print("score:",scores)

