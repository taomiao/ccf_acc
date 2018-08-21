import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import numpy as np
import os
import pandas as pd
from sklearn.externals import joblib


class DeepFM:
    def __init__(self, feature_size, field_size, batch_size,
                 embedding_size=16, dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=1,
                 learning_rate=0.5,
                 optimizer_type="adagrad",
                 batch_norm=0,
                 batch_norm_decay=0.995,
                 verbose=False,
                 random_seed=818,
                 use_fm=True,
                 use_deep=True,
                 loss_type="logloss",
                 eval_metric=roc_auc_score,
                 l2_reg=0.0,
                 greater_is_better=True,
                 train_dir=None,
                 validation_dir=None,
                 test_dir=None,
                 TEST_dir=None,
                 test_y_file=None,
                 TEST_y_file=None,
                 save_file_name=None):

        self.feature_size = feature_size  # denote as M, size of the feature dictionary
        self.field_size = field_size  # denote as F, size of the feature fields
        self.embedding_size = embedding_size  # denote as K, size of the feature embedding
        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []

        self.train_file_dir = train_dir
        self.validation_file_dir = validation_dir
        self.test_file_dir = test_dir
        self.TEST_file_dir = TEST_dir
        self.epoch_num = 1
        self.shadow_init = False

        self.shadow_iter = 0
        self.shadow_eval = []

        #   self.shadow_TEST = []
        self.shadow_test = []

        #   self.save_test = test_y_file
        #      self.save_TEST = TEST_y_file

        self.train_file_list = os.listdir(self.train_file_dir)
        self.train_file_list.sort()
        self.train_size = len(self.train_file_list)
        self.train_file_list = list(map(lambda x: os.path.join(self.train_file_dir, x), self.train_file_list))

        self.validation_file_list = os.listdir(self.validation_file_dir)
        self.validation_file_list.sort()
        self.validation_size = len(self.validation_file_list)
        self.validation_file_list = list(
            map(lambda x: os.path.join(self.validation_file_dir, x), self.validation_file_list))

        self.test_file_list = os.listdir(self.test_file_dir)
        self.test_file_list.sort()
        self.test_size = len(self.test_file_list)
        self.test_file_list = list(map(lambda x: os.path.join(self.test_file_dir, x), self.test_file_list))

        self.save_list = []
        #      self.TEST_file_list = os.listdir(self.TEST_file_dir)
        #      self.TEST_file_list.sort()
        #     self.TEST_size = len(self.TEST_file_list)
        #      self.TEST_file_list = list(map(lambda x: os.path.join(self.TEST_file_dir, x), self.TEST_file_list))
        #       self.save_file_name = save_file_name
        self.saver = None
        self.params = []
        self._init_graph()


    def init_norm(self, input, output):
        return self.glorot_norm(input, output)

    def he_norm(self, input, output):
        return np.square(2.0 / input)

    def glorot_norm(self, input, output):
        return np.sqrt(2.0 / (input + output))

    def he_uniform(self, input, output):
        return np.square(6.0 / input)

    def glorot_uniform(self, input, output):
        return np.sqrt(6.0 / (input + output))

    def auc_loss(self, label, out, gamma, p):
        label = tf.reshape(label, [-1, 1])
        out = tf.reshape(out, [-1, 1])
        pos_idx = tf.where(tf.equal(label, 1.0))[:, 0]
        neg_idx = tf.where(tf.equal(label, 0.0))[:, 0]

        pos_out = tf.gather(out, pos_idx)
        neg_out = tf.gather(out, neg_idx)

        loss = tf.reshape(
            -(tf.matmul(pos_out, tf.ones([1, tf.shape(neg_idx)[0]]))
              - tf.matmul(tf.ones([tf.shape(pos_out)[0], 1]), tf.reshape(neg_out, [1, -1]))
              - gamma),
            [-1, 1]
        )
        loss = tf.where(tf.greater(loss, 0), loss, tf.zeros([tf.shape(neg_idx)[0] * tf.shape(pos_idx)[0], 1]))
        loss = tf.pow(loss, p)
        loss = tf.reduce_mean(loss)
        return loss

    def auc_exp_loss(self, label, out):
        label = tf.reshape(label, [-1, 1]) * 2 - 1
        out = tf.reshape(out, [-1, 1]) * 2 - 1
        pos_idx = tf.where(tf.equal(label, 1.0))[:, 0]
        neg_idx = tf.where(tf.equal(label, -1.0))[:, 0]

        pos_out = tf.gather(out, pos_idx)
        neg_out = tf.gather(out, neg_idx)

        nt = tf.reshape(
            tf.matmul(tf.ones([tf.shape(pos_out)[0], 1]), tf.reshape(neg_out, [1, -1]))
            - tf.matmul(pos_out, tf.ones([1, tf.shape(neg_idx)[0]])),
            [-1, 1]
        )
        exp_nt = tf.exp(nt)
        loss = tf.reduce_mean(exp_nt)
        return loss

    def auc_log_loss(self, label, out):
        label = tf.reshape(label, [-1, 1]) * 2 - 1
        out = tf.reshape(out, [-1, 1]) * 2 - 1
        pos_idx = tf.where(tf.equal(label, 1.0))[:, 0]
        neg_idx = tf.where(tf.equal(label, -1.0))[:, 0]

        pos_out = tf.gather(out, pos_idx)
        neg_out = tf.gather(out, neg_idx)

        nt = tf.reshape(
            tf.matmul(tf.ones([tf.shape(pos_out)[0], 1]), tf.reshape(neg_out, [1, -1]))
            - tf.matmul(pos_out, tf.ones([1, tf.shape(neg_idx)[0]])),
            [-1, 1]
        )
        exp_nt = tf.exp(nt)
        ln_1_exp_nt = tf.log(1 + exp_nt)
        loss = tf.reduce_mean(ln_1_exp_nt)
        return loss

    def op_auc(self, label, out):
        label = tf.reshape(label, [-1, 1]) * 2 - 1
        out = tf.reshape(out, [-1, 1]) * 2 - 1
        pos_idx = tf.where(tf.equal(label, 1.0))[:, 0]
        neg_idx = tf.where(tf.equal(label, -1.0))[:, 0]

        pos_out = tf.gather(out, pos_idx)
        neg_out = tf.gather(out, neg_idx)

        loss = tf.reshape(
            1
            - (tf.matmul(pos_out, tf.ones([1, tf.shape(neg_idx)[0]]))
               - tf.matmul(tf.ones([tf.shape(pos_out)[0], 1]), tf.reshape(neg_out, [1, -1]))
               ),
            [-1, 1]
        )
        loss = tf.where(tf.greater(loss, 0), loss, tf.zeros([tf.shape(neg_idx)[0] * tf.shape(pos_idx)[0], 1]))
        loss = tf.pow(loss, 2)
        loss = tf.reduce_mean(loss)
        # loss = tf.tanh(loss)
        return loss

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            #            self.INPUT_INIT()
            tf.set_random_seed(self.random_seed)

            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")

            self.X = tf.placeholder(tf.int32, shape=[None, 40], name="x")
            self.feat = tf.placeholder(tf.int32, shape=[None, 20], name="feat")
            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")

            self.train_phase = tf.placeholder(tf.bool, name="train_phase")
            self.pre_train_phase = tf.placeholder(tf.bool, name="pre_train_phase")

            self.weights = self._initialize_weights()

            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"], self.X)
            self.embeddings2 = tf.nn.embedding_lookup(self.weights["feature_embeddings2"], self.feat)
            self.embeddings = tf.concat([self.embeddings, self.embeddings2], axis=1)
            self.embeddings = tf.reshape(self.embeddings,
                                         [-1, self.field_size, self.field_size, self.embedding_size])  # None *F*F *K

            # ---- FM part------

            # ---- first order ----


            # --- second order ----
            # sum square
            self.summed_features_emb_t = tf.transpose(self.embeddings, [0, 2, 1, 3])  # None * F*F*K
            self.summed_features_full = tf.multiply(self.summed_features_emb_t, self.embeddings)  # None*F*F*K

            self.ones = tf.ones_like(self.summed_features_full)
            self.op = tf.contrib.linalg.LinearOperatorLowerTriangular(tf.transpose(self.ones, [0, 3, 1, 2]))
            self.upper_tri_mask = tf.less(tf.transpose(self.op.to_dense(), [0, 2, 3, 1]), self.ones)

            self.ffm_embs_out = tf.boolean_mask(self.summed_features_full, self.upper_tri_mask)
            self.y_second_order = tf.reshape(self.ffm_embs_out,
                                             [-1, self.field_size * (self.field_size - 1) // 2 * self.embedding_size])
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])

            input_size = self.field_size * (self.field_size - 1) // 2 * self.embedding_size

            # --- Deep part ---
            self.y_deep = self.y_second_order  # None*(F*(F-1)/2*k)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" % i)
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[i + 1])

            concat_input = self.y_deep

            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])

            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.out = tf.clip_by_value(self.out, 1e-10, 1)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "auc_loss":
                self.out = tf.nn.sigmoid(self.out)
                self.out = tf.clip_by_value(self.out, 1e-10, 1)
                self.loss = self.auc_loss(self.label, self.out, gamma=0.7, p=2)  # gamma:0.1 - 0.7,p:2,3
            elif self.loss_type == "op_auc":
                self.out = tf.nn.sigmoid(self.out)
                self.out = tf.clip_by_value(self.out, 1e-10, 1)
                self.loss = self.op_auc(self.label, self.out)
            elif self.loss_type == "auc_exp_loss":
                self.out = tf.nn.sigmoid(self.out)
                self.out = tf.clip_by_value(self.out, 1e-10, 1)
                self.loss = self.auc_exp_loss(self.label, self.out)
            elif self.loss_type == "auc_log_loss":
                self.out = tf.nn.sigmoid(self.out)
                self.out = tf.clip_by_value(self.out, 1e-10, 1)
                self.loss = self.auc_log_loss(self.label, self.out)
            self.reg = 0.0
            # l2 regularization on weights
            if self.l2_reg > 0:
                self.reg += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
                if self.use_deep:
                    for i in range(len(self.deep_layers)):
                        self.reg += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d" % i])
            self.loss = self.loss + self.reg
            # optimizer
            if self.optimizer_type == 'adadelta':
                self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate, rho=0.95, epsilon=1e-8).minimize(
                    self.loss)
            elif self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)
            elif self.optimizer_type == 'ftrl':
                self.optimizer = tf.train.FtrlOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            # init
            self.saver = tf.train.Saver(max_to_keep=3)

            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def get_mid_feature(self):
        y = []
        Y = []
        for k, filename in enumerate(self.test_file_list):
            data = joblib.load(filename)
            X = data["x"]
            label = data["y"]
            feat = data["feat"]
            feed_dict = {
                self.X: X,
                self.label: label,
                self.feat: feat,
                self.dropout_keep_fm: self.dropout_fm,
                self.dropout_keep_deep: self.dropout_deep,
                self.train_phase: False,
            }
            for i in label:
                y.append(i[0])
            mid_feature = self.sess.run(self.y_second_order, feed_dict=feed_dict)
            test_res = np.array(mid_feature)
            np.savetxt('split_mid_feature_{}.csv'.format(k), test_res, delimiter=',')

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def _initialize_weights(self):
        weights = dict()
        s_weights = dict()
        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_size, (self.field_size) * self.embedding_size], 0.0, 0.1),
            name="feature_embeddings")  # feature_size * K
        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1

        weights["feature_embeddings2"] = tf.Variable(
            tf.random_normal([10000, (self.field_size) * self.embedding_size], 0.0, 0.1),
            name="feature_embeddings2")  # feature_size * K
        weights["feature_bias2"] = tf.Variable(
            tf.random_uniform([10000, 1], 0.0, 1.0), name="feature_bias2")  # feature_size * 1

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size * (self.field_size - 1) // 2 * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                        dtype=np.float32)  # 1 * layers[0]

        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size ))
        weights["concat_projection"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)


        input_size = 300
        norm_init = self.init_norm(input_size, 1)
        weights["concat_projection"] = tf.Variable(
            np.random.normal(loc=0, scale=norm_init, size=(input_size, 1)),
            dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)
        print(self.deep_layers[-1])
        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def fit_on_batch(self, X, Y, feat):
        feed_dict = {
            self.X: X,
            self.label: Y,
            self.feat: feat,
            self.dropout_keep_fm: self.dropout_fm,
            self.dropout_keep_deep: self.dropout_deep,
            self.train_phase: True,
        }

        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        # print(loss)
        return loss

    def fit(self):
        self.shadow_init = False
        for k, filename in enumerate(self.train_file_list):
            data = joblib.load(filename)
            x = data["x"]
            y = data["y"]
            feat = data["feat"]
            loss = self.fit_on_batch(X=x, Y=y, feat=feat)
            if k % 200 == 125:
                print("loss:", loss)
                sc = self.evaluate()
                #print(self.sess.run(self.weights))
                if sc > 0.8:
                    self.predict()
                    # print(self.shadow_iter)
                    # ckpt_path = './ckpt/1_2'
                    # self.saver.save(self.sess, ckpt_path + "-"+"%.4f"%(sc))
                    # print("BATCH "+str(k)+" FINISH")
        sc = self.evaluate()
        self.predict()

        self.saver.save(self.sess, "./model/test_model")
        for weight_name in self.weights:
            print("weight name:" + weight_name)
            with self.sess.as_default():
                param = np.array(self.weights[weight_name].eval())
                if param.size < 2:
                    with open("./params/" + weight_name + ".csv", "w") as f:
                        f.write(str(float(param)))
                else:
                    np.savetxt('./params/' + weight_name + '.csv', param, delimiter=',')
        self.sess.close()

    def evaluate(self):
        y = []
        Y = []
        for k, filename in enumerate(self.validation_file_list):
            data = joblib.load(filename)
            X = data["x"]
            label = data["y"]
            feat = data["feat"]
            feed_dict = {
                self.X: X,
                self.label: label,
                self.feat: feat,
                self.dropout_keep_fm: self.dropout_fm,
                self.dropout_keep_deep: self.dropout_deep,
                self.train_phase: False,
            }
            for i in label:
                y.append(i[0])
            y_pred = self.sess.run(self.out, feed_dict=feed_dict)
            for i in y_pred:
                Y.append(i[0])
                # print(y)
                # print(Y)
        score = self.eval_metric(y, Y)
        print("auc", score)
        return score

    def _predict(self):
        self.predict_test()
        self.shadow_iter = self.shadow_iter + 1
        return

    def predict(self):
        self.predict_test()
        # self.predict_TEST()
        self.shadow_iter = self.shadow_iter + 1
        return

    def predict_test(self):
        y = []
        Y = []
        for k, filename in enumerate(self.test_file_list):
            data = joblib.load(filename)
            X = data["x"]
            label = data["y"]
            feat = data["feat"]
            feed_dict = {
                self.X: X,
                self.label: label,
                self.feat: feat,
                self.dropout_keep_fm: self.dropout_fm,
                self.dropout_keep_deep: self.dropout_deep,
                self.train_phase: False,
            }
            for i in label:
                y.append(i[0])
            y_pred = self.sess.run(self.out, feed_dict=feed_dict)
            for i in y_pred:
                Y.append(i[0])
        test_res = np.array(Y)
        np.savetxt('result2.csv',test_res,fmt='%1.6f',delimiter=',')
        #if self.shadow_iter == 0:
        #    self.shadow_test = Y
        #else:
        #    for i in range(len(self.shadow_test)):
        #        self.shadow_test[i] = self.shadow_test[i] + Y[i]
        #print(self.eval_metric(y, np.array(self.shadow_test) / (self.shadow_iter + 1)))
        return


if __name__ == "__main__":
    feature_size = 12461
    fieldNum = 40 + 20
    model = DeepFM(
        feature_size=feature_size,
        field_size=fieldNum,
        batch_size=1024,  #
        epoch=1,
        embedding_size=30,
        learning_rate=0.003,
        optimizer_type="adam",
        dropout_fm=[1.0, 1.0],
        deep_layers=[300, 300, 300],
        dropout_deep=[1, 1, 1, 1, 1, 1],
        batch_norm=0,
        batch_norm_decay=0.995,
        verbose=True,
        random_seed=157,
        use_fm=True,
        use_deep=True,
        loss_type="logloss",
        l2_reg=0,

        train_dir='./train_npz',
        validation_dir='./validation_npz',
        test_dir='./TEST_npz',
    )
    #model.fit()
    model.saver.restore(model.sess,'./model/test_model')
    #model.predict_test()
    model.get_mid_feature()
