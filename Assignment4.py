from six.moves import cPickle as pickle
import numpy as np
import tensorflow as tf
import time
import math

#TF MAIN-CLASS

class BaseTensorFlow:
    def __init__(self):
        self.batch_size = 128
        self.NUM_CLASSES = 10
        self.starting_learning_rate = 0.01
        self.train_dir = './'
        self.num_steps = 5001
        self.IMAGE_PIXELS = 784
        self.SEED=66478
        self.lamb=5e-4

    def model(self, images, input_size, output_size, isEval=None):
        raise Exception('Error', 'Not implemented')

    def loadData(self):
        pickle_file = 'notMNIST.pickle'

        with open(pickle_file, 'rb') as f:
          save = pickle.load(f)
          self.train_dataset = save['train_dataset']
          self.train_labels = save['train_labels']
          self.valid_dataset = save['valid_dataset']
          self.valid_labels = save['valid_labels']
          self.test_dataset = save['test_dataset']
          self.test_labels = save['test_labels']
          del save  # hint to help gc free up memory

          self.train_dataset = self.train_dataset.reshape((-1, self.IMAGE_PIXELS)).astype(np.float32)
          self.valid_dataset = self.valid_dataset.reshape((-1, self.IMAGE_PIXELS)).astype(np.float32)
          self.test_dataset = self.test_dataset.reshape((-1, self.IMAGE_PIXELS)).astype(np.float32)

          print('Training set', self.train_dataset.shape, self.train_labels.shape)
          print('Validation set', self.valid_dataset.shape, self.valid_labels.shape)
          print('Test set', self.test_dataset.shape, self.test_labels.shape)
          print('\n\n')

    def restrictData(self, tamano=6000):
        np.random.seed(self.SEED)
        selection=np.random.choice(self.train_dataset.shape[0], tamano, replace=False)
        self.train_dataset=self.train_dataset[selection, :]
        self.train_labels=self.train_labels[selection]
        print('Restricted Training set', self.train_dataset.shape, self.train_labels.shape)
        print('\n\n')

    def loss_function(self,logits, labels):
        labels = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, self.batch_size), 1)
        concated = tf.concat(1, [indices, labels])
        onehot_labels = tf.sparse_to_dense(
              concated, tf.pack([self.batch_size, self.NUM_CLASSES]), 1.0, 0.0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                          onehot_labels,
                                                          name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss

    def training(self,loss):
        train_size = self.train_dataset.shape[0]
        tf.scalar_summary(loss.op.name, loss)
        global_step = tf.Variable(0)
        learning_rate=self.starting_learning_rate
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def evaluation(self,logits, labels):
        correct = tf.nn.in_top_k(logits, labels, 1)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def preapare_placeholder_inputs(self):
        self.images_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size,self.IMAGE_PIXELS))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size))

    def fill_feed_dict(self, dataset, labels, step):
        selection=np.random.choice(dataset.shape[0], self.batch_size, replace=False)
        images_feed = dataset[selection, :]
        labels_feed = labels[selection]
        feed_dict = {
            self.images_placeholder: images_feed,
            self.labels_placeholder: labels_feed,
        }
        return feed_dict

    def do_eval(self,sess,
            eval_correct,
            dataset,
            labels):
        true_count = 0
        steps_per_epoch = labels.shape[0] // self.batch_size
        num_examples = steps_per_epoch * self.batch_size
        for step in range(steps_per_epoch):
            feed_dict = self.fill_feed_dict(dataset, labels, step)
            true_count += sess.run(eval_correct, feed_dict=feed_dict)

        precision = 1.0*true_count / num_examples
        print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (num_examples, true_count, precision))

    def run_training(self,sess, eval_correct, train_op, loss):

        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(self.train_dir, graph=sess.graph)
        saver = tf.train.Saver()

        feed_dict = self.fill_feed_dict(self.train_dataset, self.train_labels, 0)

        for step in range(self.num_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss],
                                       feed_dict=feed_dict)

            feed_dict = self.fill_feed_dict(self.train_dataset, self.train_labels, step+1)

            duration = time.time() - start_time
            if step % 1000 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            if (step + 1) % 5000 == 0 or (step + 1) == self.num_steps:
                saver.save(sess, self.train_dir, global_step=step)
                print('Training Data Eval:')
                self.do_eval(sess, eval_correct,
                    feed_dict[self.images_placeholder], feed_dict[self.labels_placeholder])
                print('Validation Data Eval:')
                self.do_eval(sess, eval_correct, self.valid_dataset, self.valid_labels)

    def process(self):
        with tf.Graph().as_default():
            self.preapare_placeholder_inputs()

            logits_train, regularizer = self.model(self.images_placeholder,
                                    self.IMAGE_PIXELS, self.NUM_CLASSES)
            loss = self.loss_function(logits_train, self.labels_placeholder)
            loss += self.lamb * regularizer
            train_op = self.training(loss)

            logits_eval = self.model(self.images_placeholder,
                                     self.IMAGE_PIXELS, self.NUM_CLASSES, isEval=True)
            eval_correct = self.evaluation(logits_eval, self.labels_placeholder)

            with tf.Session() as sess:
                init = tf.initialize_all_variables()
                sess.run(init)

                self.run_training(sess, eval_correct, train_op, loss)
                print('Test Data Eval:')
                self.do_eval(sess,
                    eval_correct,
                    self.test_dataset, self.test_labels)

########### Convolutions ##########

class Convolutions(BaseTensorFlow):
    def __init__(self):
        BaseTensorFlow.__init__(self)
        self.lamb=5e-4
        self.num_channels=1
        self.batch_size = 16
        self.patch_size = 5
        self.depth = 16
        self.num_hidden = 64
        self.image_size=28
        self.number=self.image_size // 4 * self.image_size // 4 * self.depth

    def loadData(self):
        pickle_file = 'notMNIST.pickle'

        with open(pickle_file, 'rb') as f:
          save = pickle.load(f)
          self.train_dataset = save['train_dataset']
          self.train_labels = save['train_labels']
          self.valid_dataset = save['valid_dataset']
          self.valid_labels = save['valid_labels']
          self.test_dataset = save['test_dataset']
          self.test_labels = save['test_labels']
          del save  # hint to help gc free up memory

          self.train_dataset = self.train_dataset.reshape((-1, self.image_size,
                                    self.image_size, self.num_channels)).astype(np.float32)
          self.valid_dataset = self.valid_dataset.reshape((-1, self.image_size,
                                    self.image_size, self.num_channels)).astype(np.float32)
          self.test_dataset = self.test_dataset.reshape((-1, self.image_size,
                                    self.image_size, self.num_channels)).astype(np.float32)

          print('Training set', self.train_dataset.shape, self.train_labels.shape)
          print('Validation set', self.valid_dataset.shape, self.valid_labels.shape)
          print('Test set', self.test_dataset.shape, self.test_labels.shape)
          print('\n\n')
        
    def preapare_placeholder_inputs(self):
        self.images_placeholder = tf.placeholder(tf.float32, 
             shape=(self.batch_size,self.image_size, self.image_size, self.num_channels))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size))
    
    def model(self, images, input_size, output_size, isEval=None):
        #Declaring variables
        with tf.variable_scope('softmax_linear', reuse=isEval):
            weights_h1 = tf.get_variable("weights_h1", 
                        [self.patch_size, self.patch_size, self.num_channels, self.depth],
                        initializer=tf.random_normal_initializer(0.0, 0.1,seed=self.SEED))
            weights_h2 = tf.get_variable("weights_h2", 
                        [self.patch_size, self.patch_size, self.depth, self.depth],
                        initializer=tf.random_normal_initializer(0.0, 0.1,seed=self.SEED))
            weights_h3 = tf.get_variable("weights_h3", 
                        [self.number, self.num_hidden],
                        initializer=tf.random_normal_initializer(0.0, 0.1,seed=self.SEED))
            weights_out = tf.get_variable("weights_out", 
                        [self.num_hidden, self.NUM_CLASSES],
                        initializer=tf.random_normal_initializer(0.0, 0.1,seed=self.SEED))

            biases_h1 = tf.get_variable("biases_h1", [self.depth],
                initializer=tf.constant_initializer(0.0))
            biases_h2 = tf.get_variable("biases_h2", [self.depth],
                initializer=tf.constant_initializer(0.0))
            biases_h3 = tf.get_variable("biases_h3", [self.num_hidden],
                initializer=tf.constant_initializer(0.0))
            biases_out = tf.get_variable("biases_out", [self.NUM_CLASSES],
                initializer=tf.constant_initializer(0.0))

            #Constructing Variables and Evaluating logits
            #print(images.get_shape().as_list())
            #print(weights_h1.get_shape().as_list())

            layer_1 = tf.nn.relu(
                tf.add(tf.nn.conv2d(images, weights_h1, [1, 2, 2, 1], padding='SAME'), biases_h1))
            layer_2 = tf.nn.relu(
                tf.add(tf.nn.conv2d(layer_1, weights_h2, [1, 2, 2, 1], padding='SAME'), biases_h2))
            shape = layer_2.get_shape().as_list()
            reshape = tf.reshape(layer_2, [shape[0], shape[1] * shape[2] * shape[3]])
            layer_3 = tf.nn.relu(tf.add(tf.matmul(reshape, weights_h3), biases_h3))
            logits=tf.add(tf.matmul(layer_3, weights_out),biases_out)

            reg_linear = (tf.nn.l2_loss(weights_out)+tf.nn.l2_loss(weights_h3)+
                          tf.nn.l2_loss(weights_h2)+tf.nn.l2_loss(weights_h1))             
            regularizers = 0

            if isEval:
                return logits
            else:
                return (logits, regularizers)

########### Convolutions 2 ##########

class Convolutions1(Convolutions):
    def __init__(self):
        Convolutions.__init__(self)

    def model(self, images, input_size, output_size, isEval=None):
        #Declaring variables
        with tf.variable_scope('softmax_linear', reuse=isEval):
            weights_h1 = tf.get_variable("weights_h1", 
                        [self.patch_size, self.patch_size, self.num_channels, self.depth],
                        initializer=tf.random_normal_initializer(0.0, 0.1,seed=self.SEED))
            weights_h2 = tf.get_variable("weights_h2", 
                        [self.patch_size, self.patch_size, self.depth, self.depth],
                        initializer=tf.random_normal_initializer(0.0, 0.1,seed=self.SEED))
            weights_h3 = tf.get_variable("weights_h3", 
                        [self.number, self.num_hidden],
                        initializer=tf.random_normal_initializer(0.0, 0.1,seed=self.SEED))
            weights_out = tf.get_variable("weights_out", 
                        [self.num_hidden, self.NUM_CLASSES],
                        initializer=tf.random_normal_initializer(0.0, 0.1,seed=self.SEED))
            
            
            biases_h1 = tf.get_variable("biases_h1", [self.depth],
                initializer=tf.constant_initializer(0.0))
            biases_h2 = tf.get_variable("biases_h2", [self.depth],
                initializer=tf.constant_initializer(0.0))
            biases_h3 = tf.get_variable("biases_h3", [self.num_hidden],
                initializer=tf.constant_initializer(0.0))
            biases_out = tf.get_variable("biases_out", [self.NUM_CLASSES],
                initializer=tf.constant_initializer(0.0))
            
            #Constructing Variables and Evaluating logits
            layer_1 = tf.nn.relu(
                tf.add(tf.nn.conv2d(images, weights_h1, [1, 1, 1, 1], padding='SAME'), biases_h1))
            pool_1 = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            layer_2 = tf.nn.relu(
                tf.add(tf.nn.conv2d(pool_1, weights_h2, [1, 1, 1, 1], padding='SAME'), biases_h2))
            pool_2 = tf.nn.max_pool(layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            
            shape = pool_2.get_shape().as_list()
            reshape = tf.reshape(pool_2, [shape[0], shape[1] * shape[2] * shape[3]])
            layer_3 = tf.nn.relu(tf.add(tf.matmul(reshape, weights_h3), biases_h3))
            logits=tf.add(tf.matmul(layer_3, weights_out),biases_out)

            if isEval:
                return logits
            else:
                regularizers=(tf.nn.l2_loss(weights_out)+tf.nn.l2_loss(weights_h3)+
                              tf.nn.l2_loss(weights_h2)+tf.nn.l2_loss(weights_h1))
                return (logits, regularizers)
            
########### Convolutions 2 ##########

class Convolutions2(Convolutions1):
    def __init__(self):
        Convolutions1.__init__(self)
        self.num_steps=15001
        self.keep_prob=0.75

    def training(self,loss):
        train_size = self.train_dataset.shape[0]
        tf.scalar_summary(loss.op.name, loss)
        global_step = tf.Variable(0)
        learning_rate=tf.train.exponential_decay(0.3, global_step, 3500, 0.86, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

    def model(self, images, input_size, output_size, isEval=None):
        #Declaring variables
        with tf.variable_scope('softmax_linear', reuse=isEval):
            weights_h1 = tf.get_variable("weights_h1", 
                        [self.patch_size, self.patch_size, self.num_channels, self.depth],
                        initializer=tf.random_normal_initializer(0.0, 0.1,seed=self.SEED))
            weights_h2 = tf.get_variable("weights_h2", 
                        [self.patch_size, self.patch_size, self.depth, self.depth],
                        initializer=tf.random_normal_initializer(0.0, 0.1,seed=self.SEED))
            weights_h3 = tf.get_variable("weights_h3", 
                        [self.number, self.num_hidden],
                        initializer=tf.random_normal_initializer(0.0, 0.1,seed=self.SEED))
            weights_out = tf.get_variable("weights_out", 
                        [self.num_hidden, self.NUM_CLASSES],
                        initializer=tf.random_normal_initializer(0.0, 0.1,seed=self.SEED))

            biases_h1 = tf.get_variable("biases_h1", [self.depth],
                initializer=tf.constant_initializer(0.0))
            biases_h2 = tf.get_variable("biases_h2", [self.depth],
                initializer=tf.constant_initializer(0.0))
            biases_h3 = tf.get_variable("biases_h3", [self.num_hidden],
                initializer=tf.constant_initializer(0.0))
            biases_out = tf.get_variable("biases_out", [self.NUM_CLASSES],
                initializer=tf.constant_initializer(0.0))

            #Constructing Variables and Evaluating logits
            if isEval:
                layer_1 = tf.nn.relu(
                    tf.add(tf.nn.conv2d(images, weights_h1, [1, 1, 1, 1], padding='SAME'), biases_h1))
                pool_1 = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                                        padding='SAME')

                layer_2 = tf.nn.relu(
                    tf.add(tf.nn.conv2d(pool_1, weights_h2, [1, 1, 1, 1], padding='SAME'), biases_h2))
                pool_2 = tf.nn.max_pool(layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                                        padding='SAME')

                shape = pool_2.get_shape().as_list()
                reshape = tf.reshape(pool_2, [shape[0], shape[1] * shape[2] * shape[3]])
                layer_3 = tf.nn.relu(tf.add(tf.matmul(reshape, weights_h3), biases_h3))
                logits=tf.add(tf.matmul(layer_3, weights_out),biases_out)
                
                return logits
            else:
                layer_1 = tf.nn.relu(
                    tf.add(tf.nn.conv2d(images, weights_h1, [1, 1, 1, 1], padding='SAME'), biases_h1))
                layer_1=tf.nn.dropout(layer_1, self.keep_prob)
                pool_1 = tf.nn.max_pool(layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                                        padding='SAME')

                layer_2 = tf.nn.relu(
                    tf.add(tf.nn.conv2d(pool_1, weights_h2, [1, 1, 1, 1], padding='SAME'), biases_h2))
                layer_2=tf.nn.dropout(layer_2, self.keep_prob)
                pool_2 = tf.nn.max_pool(layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                                        padding='SAME')

                shape = pool_2.get_shape().as_list()
                reshape = tf.reshape(pool_2, [shape[0], shape[1] * shape[2] * shape[3]])
                layer_3 = tf.nn.relu(tf.add(tf.matmul(reshape, weights_h3), biases_h3))
                layer_3=tf.nn.dropout(layer_3, self.keep_prob)
                logits=tf.add(tf.matmul(layer_3, weights_out),biases_out)
                
                regularizers=(tf.nn.l2_loss(weights_out)+tf.nn.l2_loss(weights_h3)+
                              tf.nn.l2_loss(weights_h2)+tf.nn.l2_loss(weights_h1))
                return (logits, regularizers)
