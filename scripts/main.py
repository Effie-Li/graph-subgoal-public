import os
import glob
os.environ["WANDB_SILENT"] = 'true'
import yaml
import argparse
import dill as pkl
from datetime import datetime

from data.dataset import GraphDataset, GraphDataModule
from model.model_module import GraphModelModule
from eval import model_eval

import torch
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, LearningRateMonitor
from tqdm import tqdm

api = wandb.Api()

class LitProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = tqdm(disable=True)
        return bar
    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar
    def init_test_tqdm(self):
        bar = tqdm(disable=True)
        return bar

def generate_exp_name(config):
    dataset_shortname = config['dataset']['name']
    g = config['dataset']['n_graph']
    i = config['dataset']['n_isomorph']
    split_mode = config['dataset']['split_params']['mode']
    model_description = f'%s_%s_%s_h%d' % (config['model']['type'].replace('_', ''), 
                                           config['model']['graph_input'].replace('_', ''),
                                           ''.join(config['model']['architecture'].split('.')),
                                           config['model']['n_heads'])
    t = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'{dataset_shortname}_{g}x{i}-split_{split_mode}-{model_description}-{t}'
    return exp_name

def train(config, gpu, debug=False, wandb_mode='online'):

    # setup dataset
    dataset = GraphDataset(
        fname=config['dataset']['fname'],
        n_graph=config['dataset']['n_graph'],
        n_isomorph=config['dataset']['n_isomorph'],
        n_node=config['dataset']['n_node']
    )
    config['dataset']['include_graphs'] = dataset.include_graphs
    datamodule = GraphDataModule(
        dataset=dataset, 
        batch_size=config['dataset']['batch_size'], 
        split_spec=config['dataset']['split_params'],
    )
    config['dataset']['train_idx'] = datamodule.train_idx
    config['dataset']['val_idx'] = datamodule.val_idx

    # setup model
    config['model']['n_graph'] = dataset.n_graph * dataset.n_isomorph
    config['model']['n_graph_node'] = len(dataset.input_vocab)
    config['model']['pad_index'] = dataset.pad_index
    config['model']['max_pathlen'] = config['dataset']['max_pathlen']
    model = GraphModelModule(**config['model'])

    config['checkpoint']['save_weights_only'] = True # to reduce ckpt size

    if debug:
        trainer = pl.Trainer(gpus=[gpu])
        trainer.fit(model, datamodule)
    
    else:
        # setup wandb and trainer
        wandb.login()
        wandb.init(
            project=config['wandb']['project'],
            name=generate_exp_name(config),
            dir=config['wandb']['save_dir'],
            config=config,
            mode=wandb_mode,
            resume='allow',
            settings=wandb.Settings(start_method='thread')
        )
        wandb_logger = WandbLogger(
            project=config['wandb']['project'],
            save_dir=config['wandb']['save_dir'],
            config=config
        )
        checkpoint_callback = ModelCheckpoint(**config['checkpoint'])
        lr_monitor = LearningRateMonitor(logging_interval='step')
        bar = LitProgressBar() # disable val progress bar
        trainer = pl.Trainer(
            logger=wandb_logger,
            callbacks=[checkpoint_callback, lr_monitor, bar],
            gpus=[gpu],
            **config['trainer']
        )

        trainer.model = model
        ckpt_path = f'{trainer.log_dir}{wandb.run.project}/{wandb.run.id}/checkpoints'
        trainer.save_checkpoint(f'{ckpt_path}/init.ckpt')

        # let it train, let it train, let it train
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)

        # final evaluation on each problem in validation and cache prediction
        model_eval(model, datamodule, 
                   device=f'cuda:{trainer.devices[0]}', 
                   fname=f'{trainer.log_dir}{wandb.run.project}/{wandb.run.id}/final_eval.pkl')

        wandb.finish()

def main(args):
    base_config = args.base_config
    assert base_config != ''
    config = yaml.safe_load(open(base_config))

    # override base config with command line args
    config['dataset']['n_graph'] = args.n_graph
    config['dataset']['n_isomorph'] = args.n_isomorph
    config['dataset']['batch_size'] = args.batch_size
    config['dataset']['split_params']['mode'] = args.split_mode
    config['dataset']['split_params']['train_prop'] = args.split_train_prop

    config['trainer']['max_epochs'] = args.max_epochs
    config['trainer']['val_check_interval'] = args.val_check_interval
    config['checkpoint']['save_top_k'] = 0 if args.no_ckpt else args.save_top_k
    config['checkpoint']['every_n_epochs'] = args.ckpt_every_n_epochs

    config['model']['type'] = args.model_type
    config['model']['graph_input'] = args.graph_input
    config['model']['lr'] = args.lr
    config['model']['architecture'] = args.architecture
    config['model']['n_heads'] = args.n_heads
    config['model']['embed_dim'] = args.embed_dim
    config['model']['mlp_dim'] = args.mlp_dim
    # find appropriate model output dim
    if args.model_type == 'autoreg':
        config['model']['out_dim'] = config['dataset']['n_node'] + 2 # 2-eos
    else:
        config['model']['out_dim'] = config['dataset']['n_node']

    train(config, args.gpu, args.debug, args.wandb_mode)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # meta exp settings
    parser.add_argument('--base_config', default='')
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--debug', action=argparse.BooleanOptionalAction)
    parser.add_argument('--wandb_mode', type=str, default='online')
    parser.add_argument('--no_ckpt', action=argparse.BooleanOptionalAction)
    # exp task settings
    parser.add_argument('--n_graph', type=int, default=30)
    parser.add_argument('--n_isomorph', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--split_mode', type=str)
    parser.add_argument('--split_train_prop', type=int, default=0.75)
    # trainer + checkpointing
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--save_top_k', type=int, default=1)
    parser.add_argument('--val_check_interval', type=int, default=20)
    parser.add_argument('--ckpt_every_n_epochs', type=int, default=20)
    # exp model settings
    parser.add_argument('--model_type', type=str, default='autoreg')
    parser.add_argument('--graph_input', type=str, default='graph_embed')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--architecture', type=str, default='a.f.a.f')
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--mlp_dim', type=int, default=32)

    args = parser.parse_args()
    main(args)