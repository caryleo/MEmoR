import argparse
import collections
from parse_config import ConfigParser
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
import torch
from utils.util import create_model, create_dataloader, create_trainer

def main(config):
    logger = config.get_logger('train')

    # fix random seeds for reproducibility
    SEED = 125
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    # setup data_loader instances
    data_loader = create_dataloader(config)
    
    # print(config["visual"]["concept_size"])
    valid_data_loader = data_loader.split_validation()

    test = data_loader.dataset.edge_matrix_v
    # build model architecture, then print to console
    model = create_model(config)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    trainer = create_trainer(model, criterion, metrics, logger,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader
                      )

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Emotion Reasoning in Daily Life')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    
    main(config)
