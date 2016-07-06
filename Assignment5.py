# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
#%matplotlib inline
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
import time
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

########### BaseWord2Vec  ##########

class BaseWord2Vec:
    def __init__(self):
        self.filename = 'text8.zip'
        self.expected_bytes =  31344016
        self.url = 'http://mattmahoney.net/dc/'
        self.vocabulary_size = 50000
        self.SEED = 66478

        self.batch_size = 128
        self.embedding_size = 128 # Dimension of the embedding vector.
        self.skip_window = 1 # How many words to consider left and right.
        self.num_skips = 2 # How many times to reuse an input to generate a label.
        # We pick a random validation set to sample nearest neighbors. here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        self.valid_size = 16 # Random set of words to evaluate similarity on.
        self.valid_window = 100 # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.array(random.sample(range(self.valid_window),
                                                     self.valid_size))
        self.num_sampled = 64 # Number of negative examples to sample.
        self.num_steps = 400001
        self.top_k = 8 # Number of nearest neighbors
        self.data_index = 0 # Data index is used to randomly sample batch  ??

    def maybe_download(self):
        ''' Checks path and downloads text.8 '''
        if not os.path.exists(self.filename):
            self.filename,_ =urlretrieve(self.url+self.filename, self.filename)
        statinfo = os.stat(self.filename)
        if statinfo.st_size == self.expected_bytes:
            print('Found and verified %s' % self.filename)
        else:
            print(statinfo.st_size)
            raise Exception('Failed')

    def read_data(self):
        ''' Reads zip file '''
        with zipfile.ZipFile(self.filename) as f:
            data = tf.compat.as_str(f.read(f.namelist()[0])).split()
        return data

    def subsample(self, words, t = 1e-3):
        count = collections.Counter(words)
        N = len(words)

        # Calculate the frequency of each unique word and create a map from words to word indices.
        freq = np.full((len(count),), 1.0 / N)
        dictionary = dict()
        idx = 0
        for word in count:
            dictionary[word] = idx
            freq[idx] *= count[word]
            idx = idx + 1

        # Calculate the probability of a word being discarded.
        # Use the formula given in 'Distributed Representations of Words and Phrases and their Compositionality'
        # Set the probability to 0 if a word's frequency is less than the threshold.
        p = np.subtract(1.0, np.sqrt(np.divide(t, freq)))
        p[freq < t] = 0.0

        sampled_words = []
        rs = np.random.random_sample(N)
        for i in range(N):
            word = words[i]
            idx = dictionary[word]
            if rs[i] < p[idx]:
                continue
            sampled_words.append(word)
        self.words=sampled_words

    def build_dataset(self, words):
        '''
        Build the dictionary and replace rare words with UNK token
        The reverse dictionary has the labels for data
        '''
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(self.vocabulary_size - 1))
        dictionary = dict()

        for word, _ in count:
            dictionary[word] = len(dictionary)

        data = list()
        unk_count = 0

        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count = unk_count + 1
            data.append(index)
        count[0][1] = unk_count
        self.reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        self.data=data
        self.count=count
        self.dictionary=dictionary

    def wrap_data(self):
        self.maybe_download()
        words = self.read_data()
        print(len(words))
        #words = self.subsample(words)
        self.build_dataset(words)
        print('Most common words (+UNK)', self.count[:5])
        print('Sample data', self.data[:10])
        print('\n')
        del words

    def preapare_placeholder_inputs(self):
        '''Declares datasets X and Y care the types'''
        self.dataset_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size,1))
        self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

    def generate_batch(self, step):
        raise Exception('Error', 'Not implemented')

    def model(self, images, input_size, output_size, isEval=None):
        raise Exception('Error', 'Not implemented')

    def training(self,loss):
        ''' Optimizer.
        Note: The optimizer will optimize the softmax_weights AND the embeddings.
        This is because the embeddings are defined as a variable quantity and the
        optimizer's `minimize` method will by default modify all variable quantities
        that contribute to the tensor it is passed.
        See docs on `tf.train.Optimizer.minimize()` for more details.
        '''
        train_op = tf.train.AdagradOptimizer(1.0).minimize(loss)
        return train_op

    def similarity(self, embeddings, dataset):
        '''Calculates cosine distance in the valid_set '''
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                  self.valid_dataset)
        similarity =  tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
        return similarity, normalized_embeddings, embeddings

    def do_eval(self,sim):
        print('\n Masure Similarity')
        for i in range(self.valid_size):
            valid_word = self.reverse_dictionary[self.valid_examples[i]]
            nearest = (-sim[i, :]).argsort()[1:(self.top_k+1)] #(self.top_k+1)
            log = 'Nearest to %s:' % valid_word
            for k in range(self.top_k):
                close_word = self.reverse_dictionary[nearest[k]]
                log = '%s %s,' % (log, close_word)
            print(log)
        print('\n')

    def run_training(self, sess, similarity, train_op,loss):
        '''Here is the iteration, every step new stochastic? (i think not
        because the batch needs to preserve the order of the words to generate
        the contexts of the words) minibatch, the
        batch_generator constructs depending on skip-gram or cbow '''
        feed_dict=self.generate_batch(0)
        average_loss = 0
        for step in range(self.num_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss],
                feed_dict=feed_dict)
            feed_dict = self.generate_batch(step+1)
            duration = time.time() - start_time
            average_loss +=loss_value
            if (step % 4000) == 0 and step>0 :
                average_loss = average_loss/4000
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0
            if (step % 20000) == 0 and step>0:
                self.do_eval(similarity.eval())

    def process(self):
        ''' In this wrapper
            1) Declare Tensors
            2) Initiate a Session, then the variables
            3) Run training with the Tensors, feeding the batch
            4) Evaluating every certain steps
        '''
        with tf.Graph().as_default():
            self.preapare_placeholder_inputs()

            loss = self.model(self.dataset_placeholder, self.labels_placeholder)

            train_op = self.training(loss)

            similarity, normalized_embeddings, embeddings = self.model(self.dataset_placeholder,
                                                    self.labels_placeholder, isEval=True)

            with tf.Session() as sess:
                init = tf.initialize_all_variables()
                sess.run(init)

                self.run_training(sess, similarity, train_op, loss)
                print('Finished Training')
                self.final_embeddings = embeddings.eval()
                self.final_normalized_embeddings = normalized_embeddings.eval()

    def visualization(self):
        num_points = 100

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        two_d_embeddings = tsne.fit_transform(self.final_embeddings[1:num_points+1, :])

        labels = [ self.reverse_dictionary[i] for i in range(1, num_points+1) ]
        assert self.final_embeddings.shape[0] >= len(labels), 'More labels than embeddings'
        pylab.figure(figsize=(15,15))  # in inches
        for i, label in enumerate(labels):
            x, y = self.final_embeddings[i,:]
            pylab.scatter(x, y)
            pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
        pylab.show()

    def get_question_answer(self, word1, word2, word3):
        idx1 = self.dictionary[word1]
        idx2 = self.dictionary[word2]
        idx3 = self.dictionary[word3]
        summed = self.final_embeddings[idx2,:] - self.final_embeddings[idx1,:]+ self.final_embeddings[idx3,:]
        neg_scaled_sim = -np.dot(self.final_normalized_embeddings, summed)

        ap = neg_scaled_sim.argpartition(3)
        # argpartition() does not sort the partition, so it must be sorted separately.
        ap = ap[neg_scaled_sim[ap].argsort()]
        for idx in ap[0:3]:
            if idx != idx1 and idx != idx2 and idx != idx3:
                return print('%s - %s + %s = %s' % (word1, word2, word3,self.reverse_dictionary[idx]) )
        return print('%s - %s + %s = %s' % (word1, word2, word3, reverse_dictionary[ap[3]]) )

########### Skip-Gram  ##########

class SkipGram(BaseWord2Vec):
    def __init__(self):
        BaseWord2Vec.__init__(self)

    def generate_batch(self, step):
        '''
        This function operates with data (labeled words)
        Uses batch_size
        num_skips (words skipped within window)
        skip_window (size of the sides of the window generates span)
        Function to generate a training batch for the skip-gram model.
        '''
        #global data_index
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        batch_labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1 # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)

        for _ in range(span):
            ''' it does not randomize embedding it needs the order '''
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)

        for i in range(self.batch_size // self.num_skips):
            target = self.skip_window  # target label at the center of the buffer
            targets_to_avoid = [ self.skip_window ]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    ''' protects in case of repeated word? '''
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * self.num_skips + j] = buffer[self.skip_window]
                batch_labels[i * self.num_skips + j, 0] = buffer[target]
            buffer.append(self.data[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.data)

        feed_dict = {self.dataset_placeholder : batch,
                     self.labels_placeholder : batch_labels}
        return feed_dict

    def model(self, dataset, labels, isEval=None):

        with tf.variable_scope('softmax_linear', reuse=isEval):
            embeddings = tf.get_variable("embeddings", [self.vocabulary_size, self.embedding_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0, seed=self.SEED))
            weights = tf.get_variable("weights", [self.vocabulary_size, self.embedding_size],
                initializer=tf.truncated_normal_initializer(0.0, 1.0 / math.sqrt(float(self.embedding_size)),
                          seed=self.SEED))
            biases = tf.get_variable("biases", [self.vocabulary_size],
                initializer=tf.constant_initializer(0.0))

            # Look up embeddings for inputs.
            embed = tf.nn.embedding_lookup(embeddings, dataset)
            # Compute the softmax loss, using a sample of the negative labels each time.
            loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights, biases, embed,
                               labels, self.num_sampled, self.vocabulary_size))

            similarity, normalized_embeddings, embeddings = self.similarity(embeddings, dataset)

            if isEval == None:
                return loss
            if isEval == True:
                return similarity, normalized_embeddings, embeddings

########### CBOW  ##########

class CBOW(BaseWord2Vec):
    def __init__(self):
        BaseWord2Vec.__init__(self)
        self.batch_size = 128
        self.embedding_size = 128 # Dimension of the embedding vector.
        self.context_window = 4 # How many words to consider left and right
        # We pick a random validation set to sample nearest neighbors. here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        self.valid_size = 16 # Random set of words to evaluate similarity on.
        self.valid_window = 100 # Only pick dev samples in the head of the distribution.
        self.valid_examples = np.array(random.sample(range(self.valid_window),
                                                     self.valid_size))
        self.num_sampled = 64 # Number of negative examples to sample.
        self.cbowbatch_size = self.batch_size * self.context_window #
        self.data_index = 0 # Maximum

    def preapare_placeholder_inputs(self):
        '''Declares datasets X and Y care the types'''
        self.dataset_placeholder = tf.placeholder(tf.int32, shape=(self.cbowbatch_size))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size,1))
        self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

    def generate_batch(self, step):
        #the context for a word should always have even number of elements on each side
        assert (self.context_window + 1) % 2 == 1
        # should fail if we cannot fit exactly N contexts in one batch
        assert self.cbowbatch_size % (self.context_window) == 0

        batch = np.ndarray(shape=(self.cbowbatch_size), dtype=np.int32)
        batch_labels = np.ndarray(shape=(self.cbowbatch_size //
                                         self.context_window, 1), dtype=np.int32)
        # 0,1,2,3 4,5,6,7
        mid_element = self.context_window // 2

        for i in range(self.batch_size):
            buffer = [None] * self.context_window
            for j in range(self.context_window + 1):
                idx = (self.data_index + j) % len(self.data)
                mid_element_idx = (self.data_index + mid_element) % len(self.data)

                if idx < mid_element_idx:
                    buffer[j] = self.data[idx]
                elif idx > mid_element_idx:
                    buffer[j-1] = self.data[idx]
                else:
                    batch_labels[i,0] = self.data[mid_element_idx]

            random.shuffle(buffer)
            for j in range(self.context_window):
                batch[i*self.context_window + j] = buffer[j]
            self.data_index = (self.data_index + 1) % len(self.data)
            # This line protects the loop, in case that the iterations surpass
            # the words available in the data, so it resets the data_index
            if self.data_index + self.context_window == len(self.data):
                self.data_index=0

        feed_dict = {self.dataset_placeholder : batch,
                     self.labels_placeholder : batch_labels}
        return feed_dict

    def model(self, dataset, labels, isEval=None):

        with tf.variable_scope('softmax_linear', reuse=isEval):
            embeddings = tf.get_variable("embeddings", [self.vocabulary_size, self.embedding_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0, seed=self.SEED))
            segments = tf.constant([x // self.context_window for x in
                range(self.cbowbatch_size)])
            weights = tf.get_variable("weights", [self.vocabulary_size, self.embedding_size],
                initializer=tf.truncated_normal_initializer(0.0, 1.0 / math.sqrt(float(self.embedding_size)),
                          seed=self.SEED))
            biases = tf.get_variable("biases", [self.vocabulary_size],
                initializer=tf.constant_initializer(0.0))

            # Look up embeddings for inputs.
            embed = tf.nn.embedding_lookup(embeddings, dataset)
            compressed_embeddings = tf.segment_sum(embed, segments) # merging couple of embeded words into one input
            # Compute the softmax loss, using a sample of the negative labels each time.
            loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights, biases,
                                           compressed_embeddings,
                               labels, self.num_sampled, self.vocabulary_size))

            similarity, normalized_embeddings, embeddings = self.similarity(embeddings, dataset)

            if isEval == None:
                return loss
            if isEval == True:
                return similarity, normalized_embeddings, embeddings

########### IMPLEMENTATION  ##########

#modelo = SkipGram()
modelo = CBOW()
modelo.wrap_data()
modelo.process()
modelo.get_question_answer('wife','husband','his')
modelo.visualization()


#print('data:', [modelo.reverse_dictionary[di] for di in modelo.data[:8]])
#
#for num_skips, skip_window in [(2, 1), (4, 2)]:
#    data_index = 0
#    dictionary = modelo.generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
#    batch = dictionary['self.dataset_placeholder']
#    labels = dictionary['self.labels_placeholder']
#    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
#    print('    batch:', [modelo.reverse_dictionary[bi] for bi in batch])
#    print('    labels:', [modelo.reverse_dictionary[li] for li in labels.reshape(8)])





