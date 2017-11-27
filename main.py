import model
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 32
IMAGE_CHANNELS = 3

image = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
label = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHNNELS])
learning_rate = tf.placeholder(tf.float32, [])
net = model.resnet(image, 1)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=net))
opt = tf.MomentumOptimizer(learning_rate, 0.9)
train_op = opt.minimize(cross_entropy)
correct_pre = tf.equal(tf.argmax(net, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pre, dtype=tf.float32))
saver = tf.train.saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
checkpoint = tf.train.latest_checkpoint('.')
if checkpoint is not None:
    print('Restoring from checkpoint ' + checkpoint)
    saver.restore(sess, ckeckpoint)
else:
    print('Could not find the checkpoint to restore')



sess.close()