import tensorflow as tf
#import numpy as np

class PoSTagger(object):
	"""
	A simple PoS tagger implementation in Tensorflow.
	Uses an embedding layer followed by a fully connected layer with ReLU and a softmax layer.
	"""
	def __init__(self, num_classes, vocab_size, embedding_size, past_words): # sequence_length, filter_sizes, num_filters, l2_reg_lambda=0.0
		# Minibatch placeholders for input and output
		# The word indices of the window
		self.input_x = tf.placeholder(tf.int32, [None, past_words+1], name="input_x")
		# The target pos-tags
		self.input_y = tf.placeholder(tf.int64, [None	], name="input_y")

		with tf.device('/gpu:0'):

			# Embedding layer
			with tf.name_scope("embedding"):
				# TODO Create an embedding matrix
				embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))

			# Fully connected layer with ReLU
			with tf.name_scope("model"):
				# TODO Create feature vector
				features = tf.Variable(tf.zeros([embedding_size]))

				# TODO send feature vector through hidden layer
				hidden_units= tf.constant(1)
				weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, hidden1_units], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))), name='weights')
				biases = tf.Variable(tf.zeros([hidden_units]), name='biases')
				hidden = tf.nn.relu(tf.matmul(weights, features) + biases)

				# TODO Compute softmax logits
				self.logits = tf.matmul(hidden, weights) + biases

				# TODO Compute the mean loss using tf.nn.sparse_softmax_cross_entropy_with_logits
				self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=logits, name='xentropy')

			# Calculate accuracy
			with tf.name_scope("accuracy"):
				# TODO compute the average accuracy over the batch (remember tf.argmax and tf.equal)
				self.predictions = tf.equal(tf.argmax(input_y,1), tf.argmax(logits,1))
				self.accuracy = tf.reduce_mean(tf.cast(self.predictions, tf.float32))
