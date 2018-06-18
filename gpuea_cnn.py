import argparse
import os
import copy

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import lib


class Seeder:
    def __init__(self, initial_seed):
        self.curr_seed = initial_seed

    def __call__(self):
        seed = self.curr_seed
        self.curr_seed += 1
        return seed


class Population:
    """
    Population is a container that holds references to all population variables.
    """
    def __init__(self, seeder, flags, num_classes, maxloss=1000.):
        self.lmbda = flags.lmbda
        self.seeder = seeder

        # Network parameters
        self.var_glorot_uniform('w0', [self.lmbda, 5, 5, 1, 16])
        self.var_glorot_uniform('w1', [self.lmbda, 5, 5, 16, 32])
        self.var_glorot_uniform('w2', [self.lmbda, 4*4*32, num_classes])
        self.var_zero('b0', [self.lmbda, 1, 1, 1, 16])
        self.var_zero('b1', [self.lmbda, 1, 1, 1, 32])
        self.var_zero('b2', [self.lmbda, 1, num_classes])

        # Self-adaptive parameters
        self.var('exploit_rate', tf.constant(flags.exploit_rate, shape=[self.lmbda]))
        self.var('sigma', tf.constant(flags.sigma, shape=[self.lmbda]))

        # Exponential moving average of loss
        self.loss_ema = tf.get_variable('loss_ema', initializer=tf.constant(maxloss, shape=[self.lmbda]), trainable=False)
        self.fitness = -self.loss_ema

        # Order contains indices of loss_ema from smallest to largest, i.e. the best individual is indexed by order[0]
        self.order = tf.nn.top_k(self.fitness, k=self.lmbda, sorted=True)[1]

    def var_zero(self, name, shape):
        self.var(name, tf.zeros(shape))

    def var_glorot_uniform(self, name, shape):
        if len(shape) == 3:
            # Param tensor of linear layer
            fan_in = shape[2]
            fan_out = shape[1]
        elif len(shape) == 5:
            # Param tensor of conv layer
            receptive_field = shape[1] * shape[2]
            fan_in = shape[3] * receptive_field
            fan_out = shape[4] * receptive_field

        limit = tf.sqrt(6 / (fan_in + fan_out))
        self.var(name, tf.random_uniform(shape, -limit, limit, seed=self.seeder()))

    def var(self, name, init):
        var = tf.get_variable(name, initializer=init, trainable=False)
        setattr(self, name, var)

    def apply(self, fn):
        result = copy.copy(self)
        result.w0 = fn(self.w0)
        result.w1 = fn(self.w1)
        result.w2 = fn(self.w2)
        result.b0 = fn(self.b0)
        result.b1 = fn(self.b1)
        result.b2 = fn(self.b2)
        return result


def weighted_mean(var, repro_indices, weight=0.5):
    weight = expand_like(weight, var)
    return weight * tf.gather(var, repro_indices[0]) + (1 - weight) * tf.gather(var, repro_indices[1])


def expand_like(to_expand, target):
    return expand(to_expand, target.shape)


def expand(to_expand, target_shape):
    for _ in range(len(to_expand.shape), len(target_shape)):
        to_expand = tf.expand_dims(to_expand, -1)
    return to_expand


def crossover_random_uniform(pop, repro_indices, seeder):
    def xfn(var):
        # Create crossover masks that decide which part of which individual is kept
        mask = tf.cast(tf.random_uniform(var.shape, seed=seeder()) < expand_like(pop.exploit_rate, var), tf.float32)
        return mask * tf.gather(var, repro_indices[0]) + (1 - mask) * tf.gather(var, repro_indices[1])

    off = pop.apply(xfn)
    off.exploit_rate = xfn(off.exploit_rate)
    off.sigma = xfn(off.sigma)
    off.loss_ema = weighted_mean(pop.loss_ema, repro_indices, weight=pop.exploit_rate)
    return off


def crossover_arithmetic(pop, repro_indices):
    off = pop.apply(lambda var: weighted_mean(var, repro_indices, weight=pop.exploit_rate))
    off.exploit_rate = weighted_mean(off.exploit_rate, repro_indices, weight=pop.exploit_rate)
    off.sigma = weighted_mean(off.sigma, repro_indices, weight=pop.exploit_rate)
    off.loss_ema = weighted_mean(pop.loss_ema, repro_indices, weight=pop.exploit_rate)
    return off


def crossover_rows(pop, repro_indices, seeder):
    def xfn(var):
        # Create crossover mask that decides which rows of which individual are kept
        shape = var.shape[0:2]
        mask = tf.cast(tf.random_uniform(shape, seed=seeder()) < expand(pop.exploit_rate, shape), tf.float32)
        mask = expand_like(mask, var)
        return mask * tf.gather(var, repro_indices[0]) + (1 - mask) * tf.gather(var, repro_indices[1])

    off = pop.apply(xfn)
    off.exploit_rate = xfn(off.exploit_rate)
    mask = tf.cast(tf.random_uniform(off.sigma.shape, seed=seeder()) < pop.exploit_rate, tf.float32)
    off.sigma = mask * tf.gather(off.sigma, repro_indices[0]) + (1 - mask) * tf.gather(off.sigma, repro_indices[1])
    off.loss_ema = weighted_mean(pop.loss_ema, repro_indices, weight=pop.exploit_rate)
    return off


def crossover_none(pop, repro_indices):
    def xfn(var):
        return tf.gather(var, repro_indices[0])
    off = pop.apply(xfn)
    off.exploit_rate = xfn(off.exploit_rate)
    off.sigma = xfn(off.sigma)
    off.loss_ema = xfn(off.loss_ema)
    return off


def mutation_random_normal(off, offset, seeder):
    def mfn(var):
        shape = var.shape.as_list()
        zero_shape = [offset] + shape[1:]
        normal_shape = [shape[0] - offset] + shape[1:]
        zero = tf.zeros(zero_shape)
        noise = tf.random_normal(normal_shape, 0, 1, seed=seeder()) * expand_like(off.sigma[offset:], var)
        noise = tf.concat([zero, noise], 0)
        return var + noise

    return off.apply(mfn)


def evaluate(off, features, labels, batch_size):
    num_classes = labels.shape[1]

    # Repeat input data so that its size matches the population
    x = tf.tile(tf.expand_dims(features, 0), [off.lmbda, 1, 1, 1, 1])
    y = tf.tile(tf.expand_dims(labels, 0), [off.lmbda, 1, 1])

    # Shapes of data
    # x: [lmbda, batch_size, height, width, channels]
    # y: [lmbda, batch_size, label]
    # Shapes of convolution parameters
    # off.w0: [lmbda, filter_height, filter_width, in_channels, out_channels]
    # off.b0: [lmbda, 1, 1, 1, out_channels]
    # Shapes of dense parameters
    # off.w2: [lmbda, in_units, out_units]
    # off.b2: [lmbda, 1, out_units]

    slices = []
    for i in range(off.lmbda):
        z = tf.nn.conv2d(x[i], off.w0[i], strides=[1, 1, 1, 1], padding='VALID')
        slices.append(z)
    x = tf.stack(slices, 0)
    x = x + off.b0
    x = tf.nn.relu(x)

    x = tf.reshape(x, [off.lmbda * batch_size, 24, 24, 16])
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    x = tf.reshape(x, [off.lmbda, batch_size, 12, 12, 16])

    slices = []
    for i in range(off.lmbda):
        z = tf.nn.conv2d(x[i], off.w1[i], strides=[1, 1, 1, 1], padding='VALID')
        slices.append(z)
    x = tf.stack(slices, 0)
    x = x + off.b1
    x = tf.nn.relu(x)

    x = tf.reshape(x, [off.lmbda * batch_size, 8, 8, 32])
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    x = tf.reshape(x, [off.lmbda, batch_size, 4*4*32])

    x = tf.einsum('pmn,pbm->pbn', off.w2, x)
    x = x + off.b2

    assert x.shape == [off.lmbda, batch_size, num_classes]

    # Calculate loss (batched over population)
    logits = tf.reshape(x, [off.lmbda * batch_size, num_classes])
    labels = tf.reshape(y, [off.lmbda * batch_size, num_classes])
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    loss = tf.reshape(loss, [off.lmbda, batch_size])

    return tf.reduce_mean(loss, axis=1)


def select_elite(pop, k):
    # Get elite and make them mate themselves, i.e. stay unchanged
    elite = pop.order[:k]
    elite = tf.expand_dims(elite, 0)
    elite = tf.tile(elite, [2, 1])
    return elite


def select_exprank(seeder, pop, k, alpha):
    # Create probabilities for exponential rank selection
    n = pop.lmbda
    c = alpha ** (1 / n)
    p = np.log((c - 1) / (c ** n - 1) * c ** (n - np.arange(1, n + 1)))
    p = np.expand_dims(p, 0)
    p = np.tile(p, [2, 1])
    # Draw ranked parent indices according to p
    rank_indices = tf.multinomial(p, k, seed=seeder(), output_dtype=tf.int32)
    # Reverse rank indices because they are in reverse order wrt the pop.order
    rank_indices = tf.reverse(rank_indices, [1])
    # Get ranked parent pairs
    ranked = tf.gather(pop.order, rank_indices)
    return ranked


def select_truncation(seeder, pop, k, proportion):
    proportion = int(proportion * pop.lmbda)
    # Create vector of equal probabilities for `proportion` different choices
    p = tf.constant(1, shape=[2, proportion], dtype=tf.float32)
    trunc_indices = tf.multinomial(p, k, seed=seeder(), output_dtype=tf.int32)
    trunc = tf.gather(pop.order[:proportion], trunc_indices)
    return trunc


def order_selection_indices_by_fitness(pop, selection_indices):
    fitness = tf.gather(pop.fitness, selection_indices)
    mask = tf.cast(fitness[0] > fitness[1], tf.int32)
    return tf.stack([mask * selection_indices[0] + (1 - mask) * selection_indices[1],
                     mask * selection_indices[1] + (1 - mask) * selection_indices[0]], 0)


def evolutionary_algorithm(features, labels, num_classes, flags):
    seeder = Seeder(flags.seed)
    step = tf.get_variable('step', initializer=tf.constant(0), trainable=False)

    count_elite = int(round(flags.lmbda * flags.elite_p))
    count_xo = int(round(flags.lmbda * flags.xo_p))
    count_mut = int(round(flags.lmbda * flags.mut_p))
    assert count_elite + count_xo + count_mut == flags.lmbda

    with tf.device('/device:GPU:0'):
        # Create population of size mu
        with tf.variable_scope('population'):
            pop = Population(seeder, flags, num_classes)
        # Select parents
        with tf.name_scope('selection'):
            # Select (possibly zero) elites from the population. They are paired with themselves, i.e. not
            # changed during reproduction.
            elite = select_elite(pop, k=count_elite)
            # Select the remaining parent pairs using the specified method
            if flags.select_exprank is not None:
                other = select_exprank(seeder,
                                       pop,
                                       k=count_xo + count_mut,
                                       alpha=flags.select_exprank)
            elif flags.select_trunc is not None:
                other = select_truncation(seeder,
                                          pop,
                                          k=count_xo + count_mut,
                                          proportion=flags.select_trunc)

            other_xo = other[:, :count_xo]
            # Mut individuals mate with themselves and are therefore unchanged by the XO operators
            other_mut = other[0, count_xo:]
            other_mut = tf.tile(tf.expand_dims(other_mut, 0), [2, 1])

            # Concatenate and sort
            selected_indices = tf.concat([elite, other_xo, other_mut], axis=1)
            selected_indices = order_selection_indices_by_fitness(pop, selected_indices)
            assert selected_indices.shape == [2, flags.lmbda]
        # Create offspring from selected_indices
        with tf.name_scope('offspring'):
            if flags.crossover == 'uniform':
                off = crossover_random_uniform(pop, selected_indices, seeder)
            elif flags.crossover == 'rowwise':
                off = crossover_rows(pop, selected_indices, seeder)
            elif flags.crossover == 'arithmetic':
                off = crossover_arithmetic(pop, selected_indices)
            elif flags.crossover == 'none':
                off = crossover_none(pop, selected_indices)

            off = mutation_random_normal(off, count_elite + count_xo, seeder)

        # Create ops that control hyperparameters
        with tf.name_scope('parameter_control'):
            if flags.sigma_expdecay is not None:
                expdecay = tf.train.exponential_decay(flags.sigma, global_step=step, decay_steps=flags.sigma_expdecay, decay_rate=0.99)
                off.sigma = tf.tile(tf.reshape(expdecay, [1]), [flags.lmbda])
            elif flags.sigma_cyclic is not None:
                # Parameters are: lower bound, period length
                center = (flags.sigma + flags.sigma_cyclic[0]) / 2
                width = flags.sigma - center
                cyclic = center + tf.cos(tf.cast(step, tf.float32) * 2 * np.pi / flags.sigma_cyclic[1]) * width
                off.sigma = tf.tile(tf.reshape(cyclic, [1]), [flags.lmbda])
            elif flags.sigma_selfadaptive is not None:
                off.sigma = off.sigma * tf.exp(flags.sigma_selfadaptive * tf.random_normal(off.sigma.shape))

            if flags.exploit_rate_selfadaptive is not None:
                off.exploit_rate = tf.minimum(off.exploit_rate * tf.exp(flags.exploit_rate_selfadaptive * tf.random_normal(off.exploit_rate.shape)), 1)

        # Evaluate offspring
        with tf.name_scope('evaluation'):
            loss = evaluate(off, features, labels, flags.batch_size)
            loss_ema = (1 - flags.alpha) * off.loss_ema + flags.alpha * loss

        # Create op that overwrites current population
        with tf.name_scope('assign'):
            assign_ops = tf.group(tf.assign(pop.w0, off.w0),
                                  tf.assign(pop.w1, off.w1),
                                  tf.assign(pop.w2, off.w2),
                                  tf.assign(pop.b0, off.b0),
                                  tf.assign(pop.b1, off.b1),
                                  tf.assign(pop.b2, off.b2),
                                  tf.assign(pop.exploit_rate, off.exploit_rate),
                                  tf.assign(pop.sigma, off.sigma),
                                  tf.assign(pop.loss_ema, loss_ema))

        # Prepare access of best individual
        best_index = pop.order[0]
        with tf.name_scope('validation'):
            # Extract best individual's weights
            w0 = pop.w0[best_index]
            w1 = pop.w1[best_index]
            w2 = pop.w2[best_index]
            b0 = pop.b0[best_index]
            b1 = pop.b1[best_index]
            b2 = pop.b2[best_index]
            # Validation
            x = tf.nn.conv2d(features, w0, strides=[1, 1, 1, 1], padding='VALID')
            x = x + b0
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            x = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='VALID')
            x = x + b1
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            x = tf.reshape(x, [flags.batch_size, -1])
            x = tf.einsum('mn,bm->bn', w2, x)

            val_logits = x + b2

        # Save statistics
        loss_mean, loss_var = tf.nn.moments(pop.loss_ema, axes=[0])
        tf.summary.scalar('loss_ema_mean', loss_mean)
        tf.summary.scalar('loss_ema_var', loss_var)

        er_mean, er_var = tf.nn.moments(pop.exploit_rate, axes=[0])
        tf.summary.scalar('exploit_mean', er_mean)
        tf.summary.scalar('exploit_var', er_var)

        sigma_mean, sigma_var = tf.nn.moments(pop.sigma, axes=[0])
        tf.summary.scalar('sigma_mean', sigma_mean)
        tf.summary.scalar('sigma_var', sigma_var)

    return tf.group(assign_ops, tf.assign_add(step, 1), name='run_generation'), val_logits, sigma_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'fashionmnist'], default='mnist')
    parser.add_argument('--iterations', type=int, default=30000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lmbda', type=int, default=16000)
    parser.add_argument('--sigma', type=float, default=1e-3)
    sigma_group = parser.add_mutually_exclusive_group()
    sigma_group.add_argument('--sigma-selfadaptive', type=float)  # tau
    sigma_group.add_argument('--sigma-expdecay', type=float)  # decay_steps
    sigma_group.add_argument('--sigma-cyclic', type=float, nargs='+')  # [lower bound, period]
    parser.add_argument('--crossover', choices=['uniform', 'arithmetic', 'rowwise', 'none'], default='uniform')
    parser.add_argument('--exploit-rate', type=float, default=0.5)
    parser.add_argument('--exploit-rate-selfadaptive', type=float)  # tau
    select_group = parser.add_mutually_exclusive_group(required=True)
    select_group.add_argument('--select-exprank', type=float)
    select_group.add_argument('--select-trunc', type=float)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--elite-p', type=float, default=0)
    parser.add_argument('--xo-p', type=float, default=0.5)
    parser.add_argument('--mut-p', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--val-interval', type=int, default=1000)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--profile', action='store_true')
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
        # Dataset loading and switching logic
        with tf.name_scope('data_pipeline'):
            if FLAGS.dataset == 'mnist':
                input_data = lib.Mnist()
            elif FLAGS.dataset == 'fashionmnist':
                input_data = lib.FashionMnist()

            dataset_trn = input_data.input('train', FLAGS.batch_size)
            dataset_val = input_data.input('val', FLAGS.batch_size)

            iter_trn = dataset_trn.make_one_shot_iterator()
            iter_val = dataset_val.make_initializable_iterator()

            handle = tf.placeholder(tf.string, shape=[], name='dataset_handle')
            handle_trn = sess.run(iter_trn.string_handle())
            handle_val = sess.run(iter_val.string_handle())

            iterator = tf.data.Iterator.from_string_handle(handle, dataset_trn.output_types, dataset_trn.output_shapes)
            features_op, labels_op = iterator.get_next()

        # Inference, loss and training
        run_generation_op, logits_op, sigma_mean_op = evolutionary_algorithm(features_op, labels_op,
                                                                             num_classes=input_data.num_classes,
                                                                             flags=FLAGS)

        # Diagnostics
        acc_op, acc_update_op, acc_reset_op = lib.acc(logits_op, labels_op)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1)

        summary_train_op = tf.summary.merge_all()
        summary_val_op = tf.summary.scalar('val_acc', acc_op)
        summary_writer = tf.summary.FileWriter('tf', sess.graph)

        if FLAGS.profile:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            options = None
            run_metadata = None

        logger.info("Trainable model parameters: {}".format(lib.get_trainable_var_count()))
        logger.info("Starting training")
        h_val_acc = []
        h_sigma_mean = []
        for batch in range(FLAGS.iterations):
            sess.run(run_generation_op, feed_dict={handle: handle_trn}, options=options, run_metadata=run_metadata)

            if FLAGS.profile:
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('profiling-{}.json'.format(batch), 'w') as f:
                    f.write(chrome_trace)

            # Log metrics
            if (batch + 1) % FLAGS.log_interval == 0:
                summary, sigma_mean_np = sess.run([summary_train_op, sigma_mean_op], feed_dict={handle: handle_trn})
                summary_writer.add_summary(summary, batch)
                h_sigma_mean.append([batch, sigma_mean_np])

            # Validate model
            if (batch + 1) % FLAGS.val_interval == 0:
                sess.run(acc_reset_op)
                sess.run(iter_val.initializer)
                iterations = len(input_data.val[0]) // FLAGS.batch_size
                for val_batch in range(iterations):
                    sess.run(acc_update_op, feed_dict={handle: handle_val})
                summary = sess.run(summary_val_op, feed_dict={handle: handle_val})
                summary_writer.add_summary(summary, batch)

                val_acc_np = sess.run(acc_op)
                h_val_acc.append([batch, val_acc_np])
                logger.info("val_acc: {:.4f}".format(val_acc_np))

                sess.run(acc_reset_op)

        h_val_acc = np.array(h_val_acc)
        h_sigma_mean = np.array(h_sigma_mean)
        np.savez('h_val_acc.npz', batch=h_val_acc[:, 0], sigma_mean=h_val_acc[:, 1])
        np.savez('h_sigma_mean.npz', batch=h_sigma_mean[:, 0], val_acc=h_sigma_mean[:, 1])

        # Save model
        logger.debug("Saving model")
        saver.save(sess, 'tf/model.chk')
        logger.info("Finished")
