import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import xlwt

def test(x_test_head, x_test_body, x_test_image,  y_test, model):

    checkpoint_file = model

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.compat.v1.Session(config=session_conf)

        with sess.as_default():
            # Initialize all variables
            sess.run(tf.compat.v1.global_variables_initializer())

            saver = tf.compat.v1.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            predictions = graph.get_operation_by_name("loss/predictions").outputs[0]
            accuracy = graph.get_operation_by_name("loss/accuracy").outputs[0]
            input_head = graph.get_operation_by_name("input_headline").outputs[0]
            input_body = graph.get_operation_by_name("input_body").outputs[0]
            input_image = graph.get_operation_by_name("input_image").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]

            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            batch_size = graph.get_operation_by_name("batch_size").outputs[0]


            all_predictions_fake, acc_fake  = \
                sess.run([predictions, accuracy], feed_dict={input_head: x_test_head,
                                                            input_body: x_test_body,
                                                            input_image: x_test_image,
                                                            input_y: y_test,
                                                            dropout_keep_prob: 1.0,
                                                            batch_size: len(y_test)})

            predictionss_click = tf.convert_to_tensor(all_predictions_fake)
            actuals_click = tf.argmax(y_test, 1)

            #  ----------------------------------------------------------------

            # for clickbait detection
            actuals = actuals_click
            predictionss = predictionss_click

            ones_like_actuals = tf.ones_like(actuals)
            zeros_like_actuals = tf.zeros_like(actuals)
            ones_like_predictions = tf.ones_like(predictionss)
            zeros_like_predictions = tf.zeros_like(predictionss)

            tp_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, ones_like_actuals),
                        tf.equal(predictionss, ones_like_predictions)
                    ),
                    "float"
                )
            )

            tn_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, zeros_like_actuals),
                        tf.equal(predictionss, zeros_like_predictions)
                    ),
                    "float"
                )
            )

            fp_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, zeros_like_actuals),
                        tf.equal(predictionss, ones_like_predictions)
                    ),
                    "float"
                )
            )

            fn_op = tf.reduce_sum(
                tf.cast(
                    tf.logical_and(
                        tf.equal(actuals, ones_like_actuals),
                        tf.equal(predictionss, zeros_like_predictions)
                    ),
                    "float"
                )
            )

            tp, tn, fp, fn = sess.run([tp_op, tn_op, fp_op, fn_op])


            tpr = float(tp) / (float(tp) + float(fn))

            accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))
            print('Clickbait: ')
            print('ACC. = ' + str(accuracy))

            precision = float(tp) / (float(tp) + float(fp))
            print('precision = ' + str(precision))

            recall = tpr
            print('recall = ' + str(recall))

            f1_score = (2 * (precision * recall)) / (precision + recall)
            print('f1_score = ' + str(f1_score))

            print('tp:' + str(tp))
            print("tn: " + str(tn))
            print('fp:' + str(fp))
            print("fn: " + str(fn))

            return [accuracy, precision, recall, f1_score, tp, tn, fp, fn, all_predictions_fake]


if __name__ == '__main__':
    print('===============================================')
    print('load vectors and labels ... ')

    x_head = np.load('~/pf_embedding/case_headline.npy')
    x_body = np.load('~/pf_embedding/case_body.npy')
    x_image = np.load('~/pf_embedding/case_image.npy')
    y = np.load('~/pf_embedding/case_y_fn.npy')

    outdir = '~/pf_embedding/'
    with open(outdir + 'case_keys.txt', 'r') as f:
        key_list = f.readlines()

    key_list = [key[:-1] for key in key_list]
    key_list = np.array(key_list)

    print('split training set and test set ... ')
    x_head_train, x_head_test, y_train, y_test = train_test_split(x_head, y, test_size=0.2, random_state=4)
    x_body_train, x_body_test, x_image_train, x_image_test = train_test_split(x_body, x_image, test_size=0.2,
                                                                              random_state=4)

    key_train, key_test, y_train, y_test = train_test_split(key_list, y, test_size=0.2, random_state=4)

    modelfolder = './ckp-pre-e-r-1-0_ds_case/iteration'

    print('===============================================')
    print('test......')

    wb = xlwt.Workbook()
    sheet2 = wb.add_sheet('case', cell_overwrite_ok=True)
    sheet2.write(0, 0, 'news id')
    sheet2.write(0, 1, 'ground truth')
    sheet2.write(0, 2, 'predicted label')

    ''' TO FIND OUT THE CKP WITH THE BEST PERFORMANCE '''

    ckp_index = 3
    model = modelfolder + str(ckp_index)
    print('test model : ' + model)
    acc, pre, rec, f1, tp, tn, fp, fn, all_predictions_fake = test(x_head_test, x_body_test, x_image_test, y_test, model)
    for i in range(len(all_predictions_fake)):
        sheet2.write(i+1, 0, str(key_test[i]))
        if y_test[i][0] == 1:
            sheet2.write(i+1, 1, "fake")
        else:
            sheet2.write(i + 1, 1, "real")

        if all_predictions_fake[i] == 0:
            sheet2.write(i + 1, 2, "fake")
        else:
            sheet2.write(i + 1, 2, "real")

    wb.save('./performance/' + 'performance-pre-e-r-1-0_ds_case_label.xls')











