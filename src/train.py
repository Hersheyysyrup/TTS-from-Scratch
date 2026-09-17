import os
import argparse
import torch
import torch.nn.functional as F
import torchaudio.transforms as AT
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import set_seed
from accelerate import Accelerator
import matplotlib.pyplot as plt

from model import Tacotron2, Tacotron2Config
from dataset import TTSDataset, TTSCollator, BatchSampler, denormalize
from tokenizer import Tokenizer

def parse_args():

    parser = argparse.ArgumentParser()

    ###SETUP CONFIG###  
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--working_directory", type=str, required=True)
    parser.add_argument("--save_audio_gen", type=str, required=True)
    parser.add_argument("--path_to_train_manifest", type=str, required=True)
    parser.add_argument("--path_to_val_manifest", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    ### TRAINING CONFIG ###
    parser.add_argument("--training_epochs", type=int, default=500)
    parser.add_argument("--console_out_iters", type=int, default=5)
    parser.add_argument("--wandb_log_iters", type=int, default=5)
    parser.add_argument("--checkpoint_epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_eps", type=float, default=1e-6)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--start_decay_epochs", type=int, default=None)

    ###MODEL CONFIG###
    parser.add_argument("--character_embed_dim", type=int, default=512)
    parser.add_argument("--encoder_kernel_size", type=int, default=5)
    parser.add_argument("--encoder_n_convolutions", type=int, default=3)
    parser.add_argument("--encoder_embed_dim", type=int, default=512)
    parser.add_argument("--encoder_dropout_p", type=float, default=0.5)
    parser.add_argument("--decoder_rnn_embed_dim", type=int, default=1024)
    parser.add_argument("--decoder_dropout_p", type=float, default=0.1)
    parser.add_argument("--decoder_prenet_dim", type=int, default=256)
    parser.add_argument("--decoder_prenet_depth", type=int, default=2)
    parser.add_argument("--decoder_prenet_dropout_p", type=float, default=0.5)
    parser.add_argument("--decoder_postnet_num_convs", type=int, default=5)
    parser.add_argument("--decoder_postnet_n_filters", type=int, default=512)
    parser.add_argument("--decoder_postnet_kernel_size", type=int, default=5)
    parser.add_argument("--decoder_postnet_dropout_p", type=float, default=0.5)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--attention_dropout_p", type=float, default=0.1)
    parser.add_argument("--attention_location_n_filters", type=int, default=32)
    parser.add_argument("--attention_location_kernel_size", type=int, default=31)

    ### DATASET CONFIG ###
    parser.add_argument("--sampling_rate", type=int, default=22050)
    parser.add_argument("--num_mels", type=int, default=80)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--hop_size", type=int, default=256)
    parser.add_argument("--min_db", type=float, default=-100.0)
    parser.add_argument("--max_scaled_abs", type=float, default=1.0)
    parser.add_argument("--fmin", type=int, default=0)
    parser.add_argument("--fmax", type=int, default=8000)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction)

    return parser.parse_args()

### Parser Arguments ###
args = parse_args()

### Set Speed ###
if args.seed is not None:
    set_seed(args.seed)

###Init Accelerator ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator (project_dir = path_to_experiment,
                           log_with = "wandb" if args.log_wandb else None)

if args.log_wandb:
    accelerator.init_trackers(
        project_name= args.experiment_name, init_kwargs= {"wandb": {"name": args.run_name}}
    )

accelerator.print(args)

### Create paths for gen saves ###
if accelerator.is_main_process:
    os.makedirs(args.save_audio_gen , exist_ok=True)

###Load Tokenizer ###
tokenizer = Tokenizer()

### LOAD MODEL ###
config = Tacotron2Config(
    num_mels=args.num_mels,
    num_chars=tokenizer.vocab_size,
    character_embed_dim=args.character_embed_dim,
    pad_token_id=tokenizer.pad_token_id,
    encoder_kernel_size=args.encoder_kernel_size,
    encoder_n_convolutions=args.encoder_n_convolutions,
    encoder_embed_dim=args.encoder_embed_dim,
    encoder_dropout_p=args.encoder_dropout_p,
    decoder_embed_dim=args.decoder_rnn_embed_dim,
    decoder_dropout_p=args.decoder_dropout_p,
    decoder_prenet_dim=args.decoder_prenet_dim,
    decoder_prenet_depth=args.decoder_prenet_depth,
    decoder_prenet_dropout_p=args.decoder_prenet_dropout_p,
    decoder_postnet_num_convs=args.decoder_postnet_num_convs,
    decoder_postnet_n_filters=args.decoder_postnet_n_filters,
    decoder_postnet_kernel_size=args.decoder_postnet_kernel_size,
    decoder_postnet_dropout_p=args.decoder_postnet_dropout_p,
    attention_dim=args.attention_dim,
    attention_dropout_p=args.attention_dropout_p,
    attention_location_n_filters=args.attention_location_n_filters,
    attention_location_kernel_size=args.attention_location_kernel_size,
)

model = Tacotron2(config)
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
accelerator.print(f"Total Trainable Parameters: {total_trainable_params}")

optimizer = torch.optim.Adam(model.parameters(),
                             lr = args.learning_rate,
                             weight_decay= args.weight_decay,
                             eps = args.adam_eps)

### Load Dataset ###
trainset = TTSDataset(args.path_to_train_manifest,
                      sample_rate = args.sample_rate,
                      n_fft = args.n_fft,
                      window_size = args.window_size,
                      hop_size = args.hop_size,
                      fmin = args.fmin,
                      fmax = args.fmax,
                      num_mels = args.num_mels,
                      min_db = args.min_db,
                      max_scaled_abs = args.max_scaled_abs)

testset = TTSDataset(args.path_to_val_manifest,
                    sample_rate = args.sample_rate,
                    n_fft = args.n_fft,
                    window_size = args.window_size,
                    hop_size = args.hop_size,
                    fmin = args.fmin,
                    fmax = args.fmax,
                    num_mels = args.num_mels,
                    min_db = args.min_db,
                    max_scaled_abs = args.max_scaled_abs)

collator = TTSCollator()

train_sampler = BatchSampler(trainset,
                             batch_size = args.batch_size,
                             drop_last = accelerator.num_processes > 1)

train_loader = DataLoader(trainset,
                          batch_sampler= train_sampler,
                          num_workers= args.num_workers,
                          collate_fn = collator)

test_loader = DataLoader(trainset,
                         batch_size = args.batch_size,
                         num_workers= args.num_workers,
                         collate_fn= collator)

### Prepare Everything ###
model, optimizer, train_loader, test_loader = accelerator.prepare(
    model, optimizer, train_loader, test_loader
)

using_scheduler = False
if args.start_decay_epochs is not None:
    accelerator.print("Using LR Scheduler!!")
    using_scheduler = True
    init_lr = args.learning_rate
    min_lr = args.min_learning_rate
    decay_epochs = args.training_epochs - args.start_decay_epochs
    decay_gamma = (min_lr / init_lr) ** (1 / decay_epochs)

    def lr_lambda(epoch):
        if epoch < args.start_decay_epochs:
            return 1.0
        else:
            return decay_gamma ** (epoch - args.start_decay_epochs)

### Load Checkpoint ### 
"""if we want to train for multiple days this creates a checkpoint to continue working from there"""
if args.resume_from_checkpoint is not None:

    path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)

    with accelerator.main_process_first():
        accelerator.load_state(path_to_checkpoint)
    
    completed_epochs = int(args.resume_from_checkpoint.split("_")[-1]) + 1
    completed_steps = completed_epochs * len(train_loader)
    accelerator.print(f"Resuming from Epoch: {completed_epochs}")

    if using_scheduler:
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=completed_epochs-1)

else:
    completed_epochs = 0
    completed_steps = 0

    if using_scheduler:
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        
