import torch

import os
import json
import colorama
import argparse
import prefixed
import importlib

import dataclasses


from model import transformer_network, transformer_network_config, transformer_block_config
from train import train, train_config, hyperparameter_config



parser = argparse.ArgumentParser()

parser.add_argument('--device', type = str, default = 'cuda')
parser.add_argument('--dataset', type = str, default = 'tiny_stories')


args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')


    # load the dataset module
    dataset_module = importlib.import_module(f'custom_datasets.tiny_stories')


    # load the configs from the json files
    traincfg, modelcfg, hparams = None, None, None
    with open('config/traincfg.json', 'r') as f:
        traincfg = train_config(**json.load(f))
    with open('config/hparams.json', 'r') as f:
        hparams = hyperparameter_config(**json.load(f))



    dataset, tokenizer = dataset_module.get_dataset_and_tokenizer(traincfg.sequence_length)


    with open('config/modelcfg.json', 'r') as f:
        # set the model's vocab size to the dataset's vocab size
        modelcfg_dict = json.load(f)
        block_configs = [transformer_block_config(**block) for block in modelcfg_dict.pop('block_configs')]
        modelcfg = transformer_network_config(**modelcfg_dict, block_configs = block_configs, vocab_size = tokenizer.get_vocab_size())


    # create the path to log the info and dump the configs as jsons
    train_run_path_created = (traincfg.train_run_path != None)
    train_run_index = 0
    while not train_run_path_created:
        if os.path.exists('trains/train-' + str(train_run_index)):
            train_run_index += 1
        else:
            os.makedirs('trains/train-' + str(train_run_index))
            traincfg.train_run_path = 'trains/train-' + str(train_run_index)

            os.makedirs(traincfg.train_run_path  + '/models')
            os.makedirs(traincfg.train_run_path  + '/stats')

            with open(traincfg.train_run_path  + '/settings.json', 'w') as f:
                f.write(json.dumps(dataclasses.asdict(traincfg)))
            with open(traincfg.train_run_path  + '/hyperparameters.json', 'w') as f:
                f.write(json.dumps(dataclasses.asdict(hparams)))
            with open(traincfg.train_run_path  + '/model.json', 'w') as f:
                f.write(json.dumps(dataclasses.asdict(modelcfg)))

            train_run_path_created = True




    


    model = transformer_network(modelcfg).to(args.device)


    print(colorama.Fore.BLUE)
    print('parameters:', f'{prefixed.Float(sum(p.numel() for p in model.parameters())):.2h}', 'parameters')
    print(colorama.Style.RESET_ALL, end='')



    train(
        traincfg,
        hparams,
        model,
        dataset,
        tokenizer,
        args.device
    )