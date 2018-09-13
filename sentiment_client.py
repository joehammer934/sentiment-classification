from tensorflow_serving_client.protos import predict_pb2, prediction_service_pb2
from grpc.beta import implementations
import tensorflow as tf
import time

from keras.preprocessing.sequence import pad_sequences
from utils import data_utils,data_process
from lib import config

#文件读取和处理
vocb, rev_vocb = data_utils.initialize_vocabulary(config.VOCABULARY_PATH)
test_sentence_ = ["我", "讨厌", "这", "车"]
test_token_sentence = [[vocb.get(i.encode('utf-8'), 1) for i in test_sentence_]]
#将多条数据放到一个request中：
for i in range(128):
    test_token_sentence.append([vocb.get(i.encode('utf-8'), 1) for i in test_sentence_])
padding_sentence = pad_sequences(test_token_sentence, maxlen=config.MAX_SEQUENCE_LENGTH)
#计时
start_time = time.time()
#建立连接
IP = "your ip address"
port = 8000#replace this with your server port 
channel = implementations.insecure_channel(IP, port)
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
request = predict_pb2.PredictRequest()
#这里由保存和运行时定义，第一个启动tensorflow serving时配置的model_name，第二个是保存模型时的方法名
request.model_spec.name = "sentiment_classification"
request.model_spec.signature_name = "serving_default"
#入参参照入参定义
request.inputs["input_sentence"].ParseFromString(tf.contrib.util.make_tensor_proto(padding_sentence,
                                                                                   dtype=tf.int64).SerializeToString())
#第二个参数是最大等待时间，因为这里是block模式访问的
response = stub.Predict(request, 10.0)
results = {}
for key in response.outputs:
    tensor_proto = response.outputs[key]
    nd_array = tf.contrib.util.make_ndarray(tensor_proto)
    results[key] = nd_array
print("cost %ss to predict: " % (time.time() - start_time))
print("predict label is:",results["predict_classification"])
# print(results["predict_scores"])