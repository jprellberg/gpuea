import subprocess
import sys


def get_batch_iter_alpha_calls():
    for lmbda in [100, 1000, 2000]:
        for batch, iter in [(8, 100000), (16, 100000), (64, 100000), (256, 50000), (512, 50000)]:
            for alpha in [0.2, 0.5, 1.0]:
                yield ['--workdir', f'results/lmbda-batch-alpha/lmbda{lmbda:04d}-batch{batch:03d}-alpha{alpha:.2f}',
                       '--crossover', 'uniform',
                       '--elite-p', '0.05',
                       '--xo-p', '0.50',
                       '--mut-p', '0.45',
                       '--sigma', '0.001',
                       '--select-trunc', '0.50',
                       '--alpha', f'{alpha:.2f}',
                       '--lmbda', f'{lmbda:d}',
                       '--iterations', f'{iter:d}',
                       '--batch-size', f'{batch:d}']


def get_trunc_calls():
    for lmbda in [100, 1000]:
        for batch, iter in [(8, 100000), (512, 50000)]:
            for trunc_p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                yield ['--workdir', f'results/lmbda-batch-trunc/lmbda{lmbda:04d}-batch{batch:03d}-trunc{trunc_p:.2f}',
                       '--crossover', 'uniform',
                       '--elite-p', '0.05',
                       '--xo-p', '0.50',
                       '--mut-p', '0.45',
                       '--sigma', '0.001',
                       '--select-trunc', f'{trunc_p:.2f}',
                       '--alpha', '1.0',
                       '--lmbda', f'{lmbda:d}',
                       '--iterations', f'{iter:d}',
                       '--batch-size', f'{batch:d}']


def get_xo_mut_calls():
    for xo in ['arithmetic', 'uniform']:
        for xo_p in [0, 0.5, 0.75]:
            for mut in [[], ['--sigma-expdecay', '100'], ['--sigma-selfadaptive', '0.01']]:
                if not mut:
                    mut_type = 'const'
                elif 'expdecay' in mut[0]:
                    mut_type = 'expdecay'
                elif 'selfadaptive' in mut[0]:
                    mut_type = 'selfadaptive'
                yield ['--workdir', f'results/xo-mut/{xo}{xo_p:.2f}-{mut_type}',
                       '--crossover', xo,
                       '--elite-p', '0.05',
                       '--xo-p', f'{xo_p:.2f}',
                       '--mut-p', f'{0.95 - xo_p:.2f}',
                       '--sigma', '0.001',
                       '--select-trunc', '0.50',
                       '--alpha', '1.0',
                       '--lmbda', '1000',
                       '--iterations', '50000',
                       '--batch-size', '512'] + mut


def get_final_calls():
    yield ['--workdir', f'results/final/ea',
           '--crossover', 'uniform',
           '--elite-p', '0.05',
           '--xo-p', '0.75',
           '--mut-p', '0.20',
           '--sigma', '0.001',
           '--select-trunc', '0.50',
           '--alpha', '1.0',
           '--lmbda', '2000',
           '--iterations', '50000',
           '--batch-size', '1024',
           '--test']


def get_calls():
    ea_calls = []
    ea_calls += get_batch_iter_alpha_calls()
    ea_calls += get_trunc_calls()
    ea_calls += get_xo_mut_calls()
    ea_calls += get_final_calls()
    for call in ea_calls:
        for seed in range(0, 15):
            yield ['python3', '-u', 'gpuea.py', '--seed', f'{seed:d}'] + call

    for seed in range(0, 15):
        yield ['python3', '-u', 'baseline.py',
               '--seed', f'{seed:d}',
               '--workdir', 'results/final/sgd',
               '--iterations', '50000',
               '--batch-size', '256',
               '--test']


def launch_call(id):
    call = list(get_calls())[id]
    print("Launching:", ' '.join(call))
    subprocess.run(call)


def list_calls():
    calls = get_calls()
    for i, call in enumerate(calls):
        print(i, ' '.join(call), sep='\t')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        id = int(sys.argv[1])
        launch_call(id)
    else:
        print("Expected task ID as argument")
        list_calls()
