import tensorflow as tf
import numpy as np
from lib import config

class RNN_Model(object):

    def __init__(self,
                 vocabulary_size,
                 batch_size,
                 embedding_matrix,
                 x_length=None,
                 is_training=True):

        self.X_length=x_length

        self.keep_prob=config.keep_prob
        #self.batch_size=tf.Variable(0,dtype=tf.int32,trainable=False)
        self.batch_size=batch_size

        num_step=config.MAX_SEQUENCE_LENGTH
        self.input_data=tf.placeholder(tf.int64,[None,num_step], name="input_data")
        self.target = tf.placeholder(tf.int64,[None,config.class_num], name="label")
        #self.mask_x = tf.placeholder(tf.float64,[num_step,None])

        class_num=config.class_num
        hidden_neural_size=config.hidden_neural_size
        vocabulary_size=vocabulary_size
        embed_dim=config.EMBEDDING_DIM
        hidden_layer_num=config.hidden_layer_num
        #self.new_batch_size = tf.placeholder(tf.int32,shape=[],name="new_batch_size")
        #self._batch_size_update = tf.assign(self.batch_size,self.new_batch_size)

        #build LSTM network
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_neural_size,forget_bias=0.0,state_is_tuple=True)
        if self.keep_prob<1:
            lstm_cell =  tf.contrib.rnn.DropoutWrapper(
                lstm_cell,output_keep_prob=self.keep_prob
            )

        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell]*hidden_layer_num,
                                           state_is_tuple=True)
        """
        #双向LSTM
        cell = tf.contrib.rnn.static_bidirectional_rnn([lstm_cell]*hidden_layer_num,
                                                               state_is_tuple=True)
        #动态LSTM
        cell = tf.nn.dynamic_rnn(
            cell  = lstm_cell,
            dtype = tf.float64,
            sequence_length = self.X_length
        )
        """
        self._initial_state = cell.zero_state(tf.shape(self.input_data)[0],dtype=tf.float64)

        #构建网络
        #embedding layer
        with tf.name_scope("embedding_layer"):
            #embedding = tf.get_variable("embedding",[vocabulary_size,embed_dim],dtype=tf.float64)
            self.embedding = tf.Variable(tf.convert_to_tensor(embedding_matrix),
                                         name="embedding_matrix",
                                         trainable=False,
                                         dtype=tf.float64,
                                         expected_shape=[vocabulary_size,embed_dim])
            inputs=tf.nn.embedding_lookup(self.embedding,self.input_data)

        if self.keep_prob<1:
            inputs = tf.nn.dropout(inputs,self.keep_prob)

        #out_put=[]
        state=self._initial_state
        with tf.variable_scope("LSTM_layer"):
            for time_step in range(num_step):
                if time_step>0: tf.get_variable_scope().reuse_variables()
                (cell_output, state)=cell(inputs[:,time_step,:], state)
                #out_put.append(cell_output)
                out_put = cell_output
        #out_put=out_put*self.mask_x[:,:,None]

        #with tf.name_scope("mean_pooling_layer"):
        #    out_put=tf.reduce_sum(out_put,0)/(tf.reduce_sum(self.mask_x,0)[:,None])

        with tf.name_scope("Softmax_layer_and_output"):
            softmax_w = tf.get_variable("softmax_w",[hidden_neural_size,class_num],dtype=tf.float64)
            softmax_b = tf.get_variable("softmax_b",[class_num],dtype=tf.float64)
            self.logits = tf.nn.sigmoid(tf.matmul(out_put,softmax_w)+softmax_b)


        with tf.name_scope("loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.target)
            self.cost = tf.reduce_mean(self.loss)

        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.logits,1)
            correct_prediction = tf.equal(self.prediction,tf.argmax(self.target,1))
            self.correct_num=tf.reduce_sum(tf.cast(correct_prediction,tf.float64))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float64),name="accuracy")

        #add summary
        loss_summary = tf.summary.scalar("loss",self.cost)
        #add summary
        accuracy_summary=tf.summary.scalar("accuracy_summary",self.accuracy)

        if not is_training:
            return

        self.globle_step = tf.Variable(0,name="globle_step",trainable=False)
        self.lr = tf.Variable(0.0,trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                      config.max_grad_norm)


        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in zip(grads, tvars):
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.grad_summaries_merged = tf.summary.merge(grad_summaries)

        self.summary =tf.summary.merge([loss_summary,accuracy_summary,self.grad_summaries_merged])

        #optimizer = tf.train.GradientDescentOptimizer(self.loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        # optimizer = tf.train.GradientDescentOptimizer(0.5)

        #optimizer.apply_gradients(zip(grads, tvars))
        self.train_op=optimizer.apply_gradients(zip(grads, tvars))

        #self.new_lr = tf.placeholder(tf.float64,shape=[],name="new_learning_rate")
        #self._lr_update = tf.assign(self.lr,self.new_lr)

    #def assign_new_lr(self,session,lr_value):
    #    session.run(self._lr_update,feed_dict={self.new_lr:lr_value})

    #def assign_new_batch_size(self,session,batch_size_value):
    #    session.run(self._batch_size_update,feed_dict={self.new_batch_size:batch_size_value})