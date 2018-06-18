import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


REPETITIONS = 15


def get_files_in(dir, suffix):
    return [os.path.join(dp, f)
            for dp, dn, fn in os.walk(os.path.expanduser(dir))
            for f in fn
            if f.endswith(suffix)]


def get_average(npzs, key):
    batch = None
    val_acc = []
    for npz in npzs:
        data = np.load(npz)
        assert batch is None or np.array_equal(batch, data['batch'])
        batch = data['batch']
        val_acc.append(data[key])
    val_acc_mean = np.mean(val_acc, 0)
    val_acc_std = np.std(val_acc, 0)
    return batch, val_acc_mean, val_acc_std


def get_final_val_accs(npzs):
    val_accs = []
    for npz in npzs:
        data = np.load(npz)
        val_acc = data['sigma_mean'][-1]
        val_accs.append(val_acc)
    return val_accs


def is_statistically_greater(npzs_a, npzs_b):
    a = get_final_val_accs(npzs_a)
    b = get_final_val_accs(npzs_b)
    u, p = mannwhitneyu(a, b, alternative='greater')
    return u, p


def get_filesets(root, suffix, filter=None):
    filesets = dict()
    subdirs = [name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
    subdirs = sorted(subdirs)
    for name in subdirs:
        if filter is None or filter in name:
            filesets[name] = get_files_in(os.path.join(root, name), suffix)
    return filesets


def plot_lmbda_batch_alpha():
    def fs(lmbda, batch, alpha):
        return get_files_in(f'lmbda-batch-alpha/lmbda{lmbda:04d}-batch{batch:03d}-alpha{alpha:.2f}', 'h_val_acc.npz')

    batch_sizes = [8, 16, 64, 256, 512]
    fig, axarr = plt.subplots(3, 3, figsize=(7, 7), sharex=True, sharey=True)
    for i, lmbda in enumerate([100, 1000, 2000]):
        for j, alpha in enumerate([0.2, 0.5, 1.0]):
            fig_data = []
            for batch in batch_sizes:
                fs_data = []
                for npz in fs(lmbda, batch, alpha):
                    data = np.load(npz)
                    fs_data.append(data['sigma_mean'][-1])

                if (lmbda, batch, alpha) != (1000, 8, 0.2):
                    assert len(fs_data) == REPETITIONS, (lmbda, batch, alpha)
                fig_data.append(fs_data)

            axarr[i, j].set_title(f"$\\lambda = {lmbda:d}, \\alpha = {alpha:.2f}$")
            axarr[i, j].boxplot(fig_data, labels=batch_sizes)

    axarr[1, 0].set_ylabel("Validation accuracy")
    axarr[2, 1].set_xlabel("Batch size")
    fig.tight_layout()
    plt.show()


def plot_lmbda_batch_trunc():
    def fs(lmbda, batch, trunc):
        return get_files_in(f'lmbda-batch-trunc/lmbda{lmbda:04d}-batch{batch:03d}-trunc{trunc:.2f}', 'h_val_acc.npz')

    trunc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    fig, axarr = plt.subplots(2, 2, figsize=(7, 5), sharex=True, sharey='row')
    for i, batch in enumerate([8, 512]):
        for j, lmbda in enumerate([100, 1000]):
            fig_data = []
            for tr in trunc:
                fs_data = []
                for npz in fs(lmbda, batch, tr):
                    data = np.load(npz)
                    fs_data.append(data['sigma_mean'][-1])
                assert len(fs_data) == REPETITIONS
                fig_data.append(fs_data)

            axarr[i, j].set_title(f"batch size = {batch:d}, $\\lambda = {lmbda:d}$")
            axarr[i, j].boxplot(fig_data, labels=trunc)

            median = np.median(fig_data, axis=1)
            print("Median:", median)
            print("rho0.1 -> rho0.2 abs: {:.4f}".format(median[1] - median[0]))
            print("rho0.1 -> rho0.2 rel: {:.4f}".format((median[1] - median[0]) / median[0]))

    fig.text(0.5, 0.02, 'Selection proportion $\\rho$', ha='center')
    fig.text(0.01, 0.5, 'Validation accuracy', va='center', rotation='vertical')
    fig.tight_layout(pad=2, h_pad=1.08, w_pad=1.08)
    plt.show()


def plot_xo_mut():
    def fs(xo, xo_p, mut):
        return get_files_in(f'xo-mut/{xo}{xo_p:.2f}-{mut}', 'h_val_acc.npz')

    xo_ops = ['uniform', 'arithmetic']
    levels = [0, 0.5, 0.75]
    mut_adaptations = ['const', 'expdecay', 'selfadaptive']
    fig, axarr = plt.subplots(3, 2, figsize=(7, 7), sharex='col', sharey='row')
    for i, mut in enumerate(mut_adaptations):
        for j, xo in enumerate(xo_ops):
            fig_data = []
            medians = []
            for xo_p in levels:
                fs_data = []
                sub_xo = [xo] if xo_p != 0 else xo_ops
                for sxo in sub_xo:
                    for npz in fs(sxo, xo_p, mut):
                        data = np.load(npz)
                        fs_data.append(data['sigma_mean'][-1])
                if xo_p == 0:
                    assert len(fs_data) == 2 * REPETITIONS, (xo, xo_p, mut)
                else:
                    assert len(fs_data) == REPETITIONS, (xo, xo_p, mut)
                fig_data.append(fs_data)
                medians.append(np.median(fs_data))

            if mut == 'const':
                title = 'constant $\\sigma$'
            elif mut == 'expdecay':
                title = 'exp. decaying $\\sigma$'
            elif mut == 'selfadaptive':
                title = 'self-adaptive $\\sigma$'
            axarr[i, j].set_title(title)
            labels = [f"{xo}\n$p_C={xo_p}$" for xo_p in levels]
            labels[0] = 'none\n$p_C=0$'
            axarr[i, j].boxplot(fig_data, labels=labels)

            print(title)
            print(labels)
            print(medians)

    axarr[1, 0].set_ylabel("Validation accuracy")
    fig.tight_layout()
    plt.show()


def plot_sigma():
    def fs(xo, xo_p, mut):
        return get_files_in(f'xo-mut/{xo}{xo_p:.2f}-{mut}', 'h_sigma_mean.npz')

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    for i, xo in enumerate(['uniform', 'arithmetic']):
        for xo_p in [0.5, 0.75]:
            if i == 0:
                cmap = plt.get_cmap('Blues')
                color = cmap((0.5 + xo_p) / 1.25)
            else:
                cmap = plt.get_cmap('Reds')
                color = cmap((0.5 + xo_p) / 1.25)

            batch, val_acc_mean, val_acc_std = get_average(fs(xo, xo_p, 'selfadaptive'), 'val_acc')
            ax.plot(batch, val_acc_mean, label=f"{xo}, $p_C={xo_p:.2f}$", c=color)
            ax.fill_between(batch, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std, color=color, alpha=0.2)

    fileset = fs('uniform', 0, 'selfadaptive') + fs('arithmetic', 0, 'selfadaptive')
    batch, val_acc_mean, val_acc_std = get_average(fileset, 'val_acc')
    ax.plot(batch, val_acc_mean, label="none, $p_C=0$", c='orange')
    ax.fill_between(batch, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std, color='orange', alpha=0.2)

    ax.set_xlim(0, 50000)
    ax.set_ylim(0, 0.005)
    ax.set_ylabel("Mean self-adaptive $\\sigma$")
    ax.set_xlabel("Iterations")
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_final():
    FILESETS = get_filesets('final', 'test_acc.npz')

    test_acc = []
    labels = []
    for name, fileset in FILESETS.items():
        fs_data = []
        for file in fileset:
            data = float(np.load(file)['test_acc'])
            fs_data.append(data)
        assert len(fs_data) == REPETITIONS
        test_acc.append(fs_data)
        if name == 'ea':
            labels.append("EA")
        elif name == 'sgd':
            labels.append("Adam")

    print(np.median(test_acc, axis=1))
    print(np.min(test_acc, axis=1))
    print(np.max(test_acc, axis=1))

    plt.figure(figsize=(4, 2.5))
    plt.ylabel("Test accuracy")
    plt.boxplot(test_acc, labels=labels)
    plt.tight_layout()
    plt.show()


def plot_any():
    FILESETS = get_filesets('lmbda-batch-alpha', 'h_val_acc.npz')

    for name, fileset in FILESETS.items():
        batch, val_acc_mean, val_acc_std = get_average(fileset, 'sigma_mean')
        try:
            print(name, val_acc_mean[-1])
        except:
            continue
        plt.plot(batch, val_acc_mean, label=name)
        plt.fill_between(batch, val_acc_mean-val_acc_std, val_acc_mean+val_acc_std, alpha=0.2)
    plt.legend()
    plt.show()


def stats_tests():
    def a_gt_b(path_a, path_b):
        a = get_files_in(path_a, 'h_val_acc.npz')
        b = get_files_in(path_b, 'h_val_acc.npz')
        print('{:50.50}'.format(path_a), '>', '{:50.50}'.format(path_b), is_statistically_greater(a, b))

    # p_C
    a_gt_b('xo-mut/uniform0.50-const',
           'xo-mut/uniform0.00-const')
    a_gt_b('xo-mut/uniform0.50-expdecay',
           'xo-mut/uniform0.00-expdecay')
    a_gt_b('xo-mut/uniform0.50-selfadaptive',
           'xo-mut/uniform0.00-selfadaptive')

    a_gt_b('xo-mut/arithmetic0.50-const',
           'xo-mut/arithmetic0.00-const')
    a_gt_b('xo-mut/arithmetic0.50-expdecay',
           'xo-mut/arithmetic0.00-expdecay')
    a_gt_b('xo-mut/arithmetic0.50-selfadaptive',
           'xo-mut/arithmetic0.00-selfadaptive')

    # Adaptation scheme
    a_gt_b('xo-mut/uniform0.75-const',
           'xo-mut/uniform0.50-const')
    a_gt_b('xo-mut/uniform0.75-const',
           'xo-mut/uniform0.75-expdecay')
    a_gt_b('xo-mut/uniform0.75-const',
           'xo-mut/uniform0.75-selfadaptive')
    a_gt_b('xo-mut/uniform0.75-const',
           'xo-mut/uniform0.50-expdecay')
    a_gt_b('xo-mut/uniform0.75-const',
           'xo-mut/uniform0.50-selfadaptive')

    a_gt_b('xo-mut/arithmetic0.50-selfadaptive',
           'xo-mut/arithmetic0.75-selfadaptive')
    a_gt_b('xo-mut/arithmetic0.50-selfadaptive',
           'xo-mut/arithmetic0.50-const')
    a_gt_b('xo-mut/arithmetic0.50-selfadaptive',
           'xo-mut/arithmetic0.50-expdecay')
    a_gt_b('xo-mut/arithmetic0.50-selfadaptive',
           'xo-mut/arithmetic0.75-const')
    a_gt_b('xo-mut/arithmetic0.50-selfadaptive',
           'xo-mut/arithmetic0.75-expdecay')


os.chdir('../results')
plot_lmbda_batch_alpha()
