import numpy as np
import os
from cnn import *
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
from sklearn.model_selection import train_test_split
import xlwt
import tensorflow as tf


#  Hyperparameters
tf.compat.v1.flags.DEFINE_integer("head_size", 30, "the length of headline matrix")
tf.compat.v1.flags.DEFINE_integer("body_size", 100, "the length of body matrix")
tf.compat.v1.flags.DEFINE_integer("image_size", 20, "the length of image matrix")

tf.compat.v1.flags.DEFINE_float("num_filters", 128, "the number of filters)")
tf.compat.v1.flags.DEFINE_float("final_len", 32, "the output length of fully connected layer)")

# Training parameters
tf.compat.v1.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.compat.v1.flags.DEFINE_integer("num_epochs", 60, "Number of training epochs (Default: 100)")
tf.compat.v1.flags.DEFINE_integer("num_batchs", 200, "Number of batchs per fold (Default: 100)")
tf.compat.v1.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularization lambda (Default: 3.0)")
tf.compat.v1.flags.DEFINE_float("learning_rate", 5e-4, "Which learning rate to start with.")
tf.compat.v1.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (Default: 0.5)")

# Misc Parameters
tf.compat.v1.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.compat.v1.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.compat.v1.flags.FLAGS
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{} = {}".format(attr.upper(), value._value))


def train(x_train_head, x_train_body, x_train_image, y_train, modelfolder, boarddir):

    with tf.Graph().as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.compat.v1.Session(config=session_conf)

        with sess.as_default():
            layer = Graph(head_size=FLAGS.head_size,
                          body_size=FLAGS.body_size,
                          image_size=FLAGS.image_size,
                          input_len=32,
                          embedding_len=300,
                          l2_reg_lambda=FLAGS.l2_reg_lambda,
                          lr=FLAGS.learning_rate,
                          num_filters=FLAGS.num_filters,
                          final_len=FLAGS.final_len)

            # Initialize all variables
            sess.run(tf.compat.v1.global_variables_initializer())
            writer1 = tf.compat.v1.summary.FileWriter(boarddir + '/plot_1', sess.graph)

            print("Train...\n")
            length = int(float(len(y_train)) / 5)

            for epoch in range(FLAGS.num_epochs):
                print(' ----------- epoch No.' + str(epoch + 1) + ' ----------- ')
                for i in range(5):  # 5-fold
                    print('fold No.', str(i + 1))
                    x_dev_head = x_train_head[i * length: (i + 1) * length]
                    x_t_head = np.concatenate((x_train_head[:i * length], x_train_head[(i + 1) * length:]), axis=0)

                    x_dev_body = x_train_body[i * length: (i + 1) * length]
                    x_t_body = np.concatenate((x_train_body[:i * length], x_train_body[(i + 1) * length:]), axis=0)

                    x_dev_image = x_train_image[i * length: (i + 1) * length]
                    x_t_image = np.concatenate((x_train_image[:i * length], x_train_image[(i + 1) * length:]), axis=0)

                    y_dev = y_train[i * length: (i + 1) * length]
                    y_t = np.concatenate((y_train[:i * length], y_train[(i + 1) * length:]), axis=0)

                    for batch in range(FLAGS.num_batchs):

                        batch_indices = np.random.choice(np.arange(y_t.shape[0]), FLAGS.batch_size)
                        x_batch_head = x_t_head[batch_indices]
                        x_batch_body = x_t_body[batch_indices]
                        x_batch_image = x_t_image[batch_indices]
                        y_batch = y_t[batch_indices]

                        _, train_accuracy, trainloss = \
                            sess.run([layer.train_op, layer.accuracy, layer.loss], feed_dict={layer.input_headline_: x_batch_head,
                                                                                              layer.input_body_: x_batch_body,
                                                                                              layer.input_image_: x_batch_image,
                                                                                              layer.input_y: y_batch,
                                                                                              layer.dropout_keep_prob: FLAGS.dropout_keep_prob,
                                                                                              layer.batch_size: FLAGS.batch_size})

                        if (batch + 1) % 50 == 0:
                            print(" step %d, training accuracy: clickbait %g,  loss %g" % ((batch + 1), train_accuracy, trainloss))

                    # training set
                    summary1 = sess.run(layer.merged, feed_dict={layer.input_headline_: x_batch_head,
                                                                 layer.input_body_: x_batch_body,
                                                                 layer.input_image_: x_batch_image,
                                                                 layer.input_y: y_batch,
                                                                 layer.dropout_keep_prob: FLAGS.dropout_keep_prob,
                                                                 layer.batch_size: FLAGS.batch_size})


                    # draw training set
                    writer1.add_summary(summary1, epoch * 5 + i)

                    # validation set
                    dev_accuracy_fake, dev_loss = sess.run(
                        [layer.accuracy, layer.loss], feed_dict={layer.input_headline_: x_dev_head,
                                                                               layer.input_body_: x_dev_body,
                                                                               layer.input_image_: x_dev_image,
                                                                               layer.input_y: y_dev,
                                                                               layer.dropout_keep_prob: 1.0,
                                                                               layer.batch_size: len(y_dev)})

                    print("validation accuracy: clickbait: %g, loss: %g" % (dev_accuracy_fake, dev_loss))

                if (epoch + 1) % 1 == 0:
                    saver = tf.compat.v1.train.Saver()
                    saver.save(sess, modelfolder + str(epoch+1))




def main(_):
    print('===============================================')
    print('load vectors and labels ... ')

    x_head = np.load('～/pf_embedding/case_headline.npy')
    x_body = np.load('～/pf_embedding/case_body.npy')
    x_image = np.load('～/pf_embedding/case_image.npy')
    y = np.load('～/pf_embedding/case_y_fn.npy')

    print('split training set and test set ... ')
    x_head_train, x_head_test, y_train, y_test = train_test_split(x_head, y, test_size=0.2, random_state=4)
    x_body_train, x_body_test, x_image_train, x_image_test = train_test_split(x_body, x_image, test_size=0.2, random_state=4)

    modelfolder = './ckp-pre-e-r-1-0_ds_case/iteration'
    log = 'save-pre-e-r-1-0_ds_case/logs'

    train(x_head_train, x_body_train, x_image_train, y_train, modelfolder, log)


if __name__ == '__main__':
    tf.compat.v1.app.run()
