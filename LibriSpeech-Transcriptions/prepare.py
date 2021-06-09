import yaml
from pathlib import Path
from progressbar import ProgressBar
import random
import os


DATASETS = ['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500']

def modify(dataset, dictionary, extend_range, replace_test, replace_length, out_file):
    bar = ProgressBar(max_value=len(dataset))
    bar.start()
    for i, line in enumerate(dataset):
        bar.update(i)
        for word in line.split():
            m_word = ''

            for c in word:
                    m_word += c * extend_range()
                
            m_word = list(m_word)

            for i in range(len(m_word)):
                if replace_test():
                    rl = replace_length()
                    m_word[i:i+rl] = [random.choice(list(dictionary))] * rl

            out_file.write(''.join(m_word) + ' ')
        out_file.write('\n')
    bar.finish()

def load_dataset(name):
    dataset = list(line.strip() for line in open(name + '.txt', 'r', encoding='utf8'))
    dictionary = set()
    for line in dataset:
        dictionary |= set(line)
    dictionary.remove(' ')
    return dataset, dictionary

def process(config_file_name):
    random.seed(290956)
    config = None
    with open(Path('conf') / config_file_name) as config_file:
        config = yaml.full_load(config_file)
    outPath = Path.resolve(Path(config_file_name) / '..' / config_file_name[:-5])

    if not os.path.exists(outPath):
        os.makedirs(outPath)

    extend_range = lambda: random.choice(config['extend_length'])
    replace_test = lambda: random.uniform(0, 1) < config['replace_prob']
    replace_length = lambda: random.choice(config['replace_length'])

    if not os.path.exists('train-full-960.txt'):
        print("Preparing train-full-960...", flush=True)

        with open('train-full-960.txt', 'w', encoding='utf8') as out:
            for name in ['train-clean-100', 'train-clean-360', 'train-other-500']:
                for line in open(name + '.txt', 'r', encoding='utf8'):
                    out.write(line)
    
    print("Preparing dataset modifications", flush=True)
    for dataset_name in DATASETS:
        dataset, dictionary = load_dataset(dataset_name)
        if not os.path.exists(outPath / (dataset_name + '.txt')):
            print(f"Processing dataset {dataset_name}", flush=True)
            with open(outPath / (dataset_name + '.txt'), 'w', encoding='utf8') as out_file:
                modify(dataset, dictionary, extend_range, replace_test, replace_length, out_file)

    if not os.path.exists(outPath / 'train-full-960.txt'):
        with open(outPath / 'train-full-960.txt', 'w', encoding='utf8') as out:
            for name in ['train-clean-100', 'train-clean-360', 'train-other-500']:
                for line in open(outPath / (name + '.txt'), 'r', encoding='utf8'):
                    out.write(line)

def main():
    for file in os.listdir('./conf'):
        if file.endswith('.yaml'):
            print(f"Processing {file}", flush=True)
            process(file)

if __name__ == '__main__':
    main()