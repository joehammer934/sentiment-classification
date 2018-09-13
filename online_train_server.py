import os,time
import tensorflow as tf

from model_utils import online_lstm_model
from utils import data_utils,data_process
from lib import config
#载入词向量，生成字典
embedding_matrix, word_list = data_utils.load_pretained_vector(config.WORD_VECTOR_PATH)
data_utils.create_vocabulary(config.VOCABULARY_PATH, word_list)
#数据预处理
for file_name in os.listdir(config.ORGINAL_PATH):
    data_utils.data_to_token_ids(config.ORGINAL_PATH + file_name, config.TOCKEN_PATN + file_name,
                                   config.VOCABULARY_PATH)
vocabulary_size = len(word_list) + 2
#获取训练数据
x_train, y_train, x_test, y_test = data_process.data_split(choose_path=config.choose_car_path,
                                                           buy_path=config.buy_car_path,
                                                           no_path=config.no_car_path)

with tf.Graph().as_default():
    #build graph
    model = online_lstm_model.RNN_Model(vocabulary_size, config.BATCH_SIZE, embedding_matrix)
    logits = model.logits
    loss  = model.loss
    cost = model.cost
    acu = model.accuracy
    prediction = model.prediction
    train_op = model.train_op

    saver = tf.train.Saver()
    #GPU设置
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        count = 0
        init = tf.global_variables_initializer()
        sess.run(init)
        while count<config.max_steps:
            start_time = time.time()
            #生成训练batch
            batch_x_train, batch_y_train = data_process.generate_batch(x_train, y_train)
            input_feed = {
                model.input_data: batch_x_train,
                model.target: batch_y_train
            }
            #进行训练
            train_loss, train_cost, train_acu, _ = sess.run(fetches=[loss, cost, acu, train_op],
                                                            feed_dict=input_feed)

            count += 1
            if count % 100 == 0:
                #每100轮，验证一次
                sum_valid_cost = 0
                sum_valid_acu = 0
                for i in range(len(x_test)//64):
                    batch_x_valid, batch_y_valid = data_process.generate_batch(x_test, y_test)
                    valid_feed = {
                        model.input_data: batch_x_valid,
                        model.target: batch_y_valid
                    }
                    valid_loss, valid_cost, valid_acu = sess.run(fetches=[loss, cost, acu],
                                                                feed_dict=valid_feed)
                    sum_valid_cost += valid_cost
                    sum_valid_acu += valid_acu
                print("current step: %f, train cost: %f, train accuracy: %f, cost_time: %f"%(count, train_cost, train_acu, time.time()-start_time))
                print("valid cost: %f, valid accuracy: %f"%(sum_valid_cost/(len(x_test)//64), sum_valid_acu/(len(x_test)//64)))
        #将模型保存为可用于线上服务的文件（一个.pb文件，一个variables文件夹）
        export_path_base = config.export_path_base
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(count)))
        print('Exporting trained model to', export_path)

        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # 建立签名映射
        """
        build_tensor_info：建立一个基于提供的参数构造的TensorInfo protocol buffer，
        输入：tensorflow graph中的tensor；
        输出：基于提供的参数（tensor）构建的包含TensorInfo的protocol buffer
        """
        input_sentence = tf.saved_model.utils.build_tensor_info(model.input_data)
        classification_outputs_classes = tf.saved_model.utils.build_tensor_info(model.prediction)
        classification_outputs_scores = tf.saved_model.utils.build_tensor_info(model.logits)

        """
        signature_constants：SavedModel保存和恢复操作的签名常量。
        
        如果使用默认的tensorflow_model_server部署模型，
        这里的method_name必须为signature_constants中CLASSIFY,PREDICT,REGRESS的一种
        """
        #定义模型的输入输出，建立调用接口与tensor签名之间的映射
        classification_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={
                    "input_sentence":
                        input_sentence
                },
                outputs={
                    "predict_classification":
                        classification_outputs_classes,
                    "predict_scores":
                        classification_outputs_scores
                },
                method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

        """
        tf.group : 创建一个将多个操作分组的操作，返回一个可以执行所有输入的操作
        """
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        """
        add_meta_graph_and_variables：建立一个Saver来保存session中的变量，
                                      输出对应的原图的定义，这个函数假设保存的变量已经被初始化；
                                      对于一个SavedModelBuilder，这个API必须被调用一次来保存meta graph；
                                      对于后面添加的图结构，可以使用函数 add_meta_graph()来进行添加
        """
        #建立模型名称与模型签名之间的映射
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            #保存模型的方法名，与客户端的request.model_spec.signature_name对应
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    classification_signature},
            legacy_init_op=legacy_init_op)

        builder.save()
        print("Build Done")
