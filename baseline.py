import argparse
import os

import numpy as np
import tensorflow as tf

import lib


def inference(x, num_classes):
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = tf.reshape(x, [-1, 196])

    with tf.variable_scope(None, 'layer'):
        w = tf.get_variable('w', [196, 256],
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable('b', [256],
                            initializer=tf.zeros_initializer())
        x = tf.matmul(x, w) + b
        x = tf.nn.relu(x)

    with tf.variable_scope(None, 'layer'):
        w = tf.get_variable('w', [256, 128],
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable('b', [128],
                            initializer=tf.zeros_initializer())
        x = tf.matmul(x, w) + b
        x = tf.nn.relu(x)

    with tf.variable_scope(None, 'layer'):
        w = tf.get_variable('w', [128, 64],
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable('b', [64],
                            initializer=tf.zeros_initializer())
        x = tf.matmul(x, w) + b
        x = tf.nn.relu(x)

    with tf.variable_scope(None, 'layer'):
        w = tf.get_variable('w', [64, num_classes],
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable('b', [num_classes],
                            initializer=tf.zeros_initializer())
        x = tf.matmul(x, w) + b

    return x


def loss(logits, labels):
    with tf.name_scope('loss'):
        loss_pred = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        loss_pred = tf.reduce_mean(loss_pred)
        return loss_pred


def train(loss, learning_rate, global_step):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # Add update ops as training dependency (e.g. for batch_norm)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(loss=loss, global_step=global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'fashionmnist'], default='mnist')
    parser.add_argument('--iterations', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--val-interval', type=int, default=5000)
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--workdir', required=True)
    FLAGS = parser.parse_args()

    wdir = lib.create_unique_dir(FLAGS.workdir)
    os.chdir(wdir)

    logger = lib.get_logger()
    logger.info(wdir)
    logger.info(FLAGS)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        tf.set_random_seed(FLAGS.seed)

        # Global variables
        global_step_op = tf.train.get_or_create_global_step()
        is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

        # Dataset loading and switching logic
        if FLAGS.dataset == 'mnist':
            input_data = lib.Mnist()
        elif FLAGS.dataset == 'fashionmnist':
            input_data = lib.FashionMnist()
        dataset_trn = input_data.input('train', FLAGS.batch_size)
        dataset_val = input_data.input('val', FLAGS.batch_size)
        dataset_test = input_data.input('test', FLAGS.batch_size)

        iter_trn = dataset_trn.make_one_shot_iterator()
        iter_val = dataset_val.make_initializable_iterator()
        iter_test = dataset_val.make_initializable_iterator()

        handle = tf.placeholder(tf.string, shape=[], name='dataset_handle')
        handle_trn = sess.run(iter_trn.string_handle())
        handle_val = sess.run(iter_val.string_handle())
        handle_test = sess.run(iter_test.string_handle())

        iterator = tf.data.Iterator.from_string_handle(handle, dataset_trn.output_types, dataset_trn.output_shapes)
        features_op, labels_op = iterator.get_next()

        # Inference, loss and training
        learning_rate_op = 1e-3
        with tf.device('/device:GPU:0'):
            logits_op = inference(features_op, num_classes=input_data.num_classes)
            loss_op = loss(logits_op, labels_op)
            train_op = train(loss_op, learning_rate_op, global_step_op)

        # Diagnostics
        acc_op, acc_update_op, acc_reset_op = lib.acc(logits_op, labels_op)
        tf.summary.scalar('learning_rate', learning_rate_op)
        tf.summary.scalar('loss', loss_op)
        tf.summary.scalar('accuracy', acc_op)
        summary_op = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)
        summary_writer = tf.summary.FileWriter('tf', sess.graph)

        logger.info("Trainable model parameters: {}".format(lib.get_trainable_var_count()))
        logger.info("Starting training")
        h_val_acc = []
        for batch in range(FLAGS.iterations):
            step_np, loss_np, acc_np, _ = sess.run([global_step_op, loss_op, acc_update_op, train_op],
                                                   feed_dict={is_training: True, handle: handle_trn})

            # Log metrics
            if (batch + 1) % FLAGS.log_interval == 0:
                logger.debug("Batch {}: loss={:.4f}, acc={:.4f}".format(batch, loss_np, acc_np))
                summary = sess.run(summary_op, feed_dict={is_training: False, handle: handle_trn})
                summary_writer.add_summary(summary, step_np)

            # Validate model
            if (batch + 1) % FLAGS.val_interval == 0:
                sess.run(acc_reset_op)
                sess.run(iter_val.initializer)
                iterations = int(np.ceil(len(input_data.val[0]) / FLAGS.batch_size))
                for val_batch in range(iterations):
                    sess.run(acc_update_op, feed_dict={is_training: False, handle: handle_val})
                val_acc_np = sess.run(acc_op)
                h_val_acc.append([batch, val_acc_np])
                logger.info("val_acc: {:.4f}".format(val_acc_np))
                sess.run(acc_reset_op)

        h_val_acc = np.array(h_val_acc)
        np.savez('h_val_acc.npz', batch=h_val_acc[:, 0], sigma_mean=h_val_acc[:, 1])

        # Save model
        saver.save(sess, 'tf/model.chk')
        logger.info("Finished training")

        if FLAGS.test:
            sess.run(acc_reset_op)
            sess.run(iter_test.initializer)
            iterations = int(np.ceil(len(input_data.test[0]) / FLAGS.batch_size))
            for test_batch in range(iterations):
                sess.run(acc_update_op, feed_dict={handle: handle_test})

            test_acc_np = sess.run(acc_op)
            logger.info("test_acc: {:.4f}".format(test_acc_np))
            np.savez('test_acc.npz', test_acc=test_acc_np)
