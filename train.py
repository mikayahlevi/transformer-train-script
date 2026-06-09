import torch
import datasets

import time
import math
import colorama
import os

from dataclasses import dataclass
from typing import Callable, Any, cast

from pipeline import pipeline_protocol

@dataclass
class train_config:
    sequence_length: int
    batch_size: int


    total_steps: int

    update_interval: int

    schedule_interval: int



@dataclass
class hyperparameter_config:
    lr_warmup_steps: int
    lr_decay_steps: int

    start_lr: float
    peak_lr: float
    end_lr: float

    betas: tuple[float, float]

    weight_decay: float


def configure_optimizer(model, hyperparameters):
    wte_weight = model.wte.weight

    final_ln_weight = model.final_ln.weight

    first_ln_weights = [block.first_ln.weight for block in model.blocks]
    second_ln_weights = [block.second_ln.weight for block in model.blocks]

    mlp_up_weights = [block.mlp[0].weight for block in model.blocks]
    mlp_down_weights = [block.mlp[2].weight for block in model.blocks]

    query_layer_weights = [block.attention.query_layer.weight for block in model.blocks]
    key_layer_weights = [block.attention.key_layer.weight for block in model.blocks]
    value_layer_weights = [block.attention.value_layer.weight for block in model.blocks]
    attention_down_weights = [block.attention.attention_down.weight for block in model.blocks]

    optim_groups = [
        {
            'params': [wte_weight] + [final_ln_weight] + first_ln_weights + second_ln_weights,
            'weight_decay': 0.0
        },
        {
            'params': mlp_up_weights + mlp_down_weights + query_layer_weights + key_layer_weights + value_layer_weights + attention_down_weights,
            'weight_decay': hyperparameters.weight_decay
        }
    ]

    return torch.optim.AdamW(optim_groups, lr = 1.0, betas = hyperparameters.betas)



def format_info(info):
    tmp = ''
    for i, (key, value) in enumerate(info.items()):
        if i == 0:
            tmp += str(value).rjust(6) + ' ' + key
        else:
            tmp += '      ' + str(value).rjust(6) + ' ' + key

    return tmp


class accumulating_metric:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float):
        self.sum += value
        self.count += 1

    def get_average_and_reset(self) -> float:
        avg = self.sum / self.count if self.count > 0 else 0.0
        self.sum = 0.0
        self.count = 0

        return avg



def train(
    settings: train_config,
    hyperparameters: hyperparameter_config,

    model,

    checkpoint,

    dataset: datasets.DatasetDict,

    tokenizer,

    pipeline: pipeline_protocol[Any, Any],

    log_metric: Callable[[int, str, float], None],

    loss_log_interval: int,
    eval_interval: int,
    metric_print_interval: int,
    checkpoint_save_interval: int,

    checkpoint_save_path: str,

    device
):
    print(colorama.Fore.GREEN)
    print('starting training')
    print(colorama.Style.RESET_ALL, end='')


    train_dataloader, val_dataloader = pipeline.get_dataloaders(dataset, batch_size = settings.batch_size, shuffle = True, pin_memory = (device == 'cuda'))
    train_dataloader_iter = iter(train_dataloader)

    mask_value = -100

    criterion = torch.nn.CrossEntropyLoss(reduction = 'mean', ignore_index = mask_value)


    def lr_lambda(sched_step: int) -> float:
        train_step = sched_step * settings.schedule_interval
        decay_step = train_step - hyperparameters.lr_warmup_steps

        if train_step < hyperparameters.lr_warmup_steps:
            scale = train_step / hyperparameters.lr_warmup_steps
            return hyperparameters.start_lr + scale * (hyperparameters.peak_lr - hyperparameters.start_lr)
        elif decay_step < hyperparameters.lr_decay_steps:
            # interpolates between 1 and 0
            scale = 0.5 * (1 + math.cos(math.pi * decay_step / hyperparameters.lr_decay_steps))
            return hyperparameters.end_lr + scale * (hyperparameters.peak_lr - hyperparameters.end_lr)
        else:
            return hyperparameters.end_lr

    optimizer = configure_optimizer(model, hyperparameters)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda = lr_lambda
    )


    start_step = 0
    if checkpoint is not None:
        # note the model state dict is already loaded
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_step = checkpoint['step']

        print(colorama.Fore.YELLOW)
        print('resuming training from checkpoint at step', start_step)
        print(colorama.Style.RESET_ALL, end='')

        # advance dataloader to the correct position within the current epoch
        for _ in range(start_step % len(train_dataloader)):
            next(train_dataloader_iter)



    logged_tr_loss_avg = accumulating_metric()
    display_tr_loss_sum = accumulating_metric()
    last_val_loss = 'n/a'


    start = time.time()
    for step in range(start_step, settings.total_steps):
        model.train()

        ids = next(train_dataloader_iter)['ids']

        inputs, labels = pipeline.get_training_pairs(ids, tokenizer,  mask_value)
        inputs, labels = inputs.to(device), labels.to(device)

        logits = model(inputs)

        # flatten batch and sequence dimensions into one dimension for computing the loss
        loss = criterion(logits.flatten(-3, -2), labels.flatten(-2, -1))


        logged_tr_loss_avg.update(loss.item())
        display_tr_loss_sum.update(loss.item())


        (loss / settings.update_interval).backward()

        # note that we use step + 1 since we are measuring the completed steps and we have already executed a single step

        if (step + 1) % len(train_dataloader) == 0:
            # reset the dataloader iterator at the end of each epoch
            train_dataloader_iter = iter(train_dataloader)

        if (step + 1) % settings.update_interval == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()


        if (step + 1) % settings.schedule_interval == 0:
            scheduler.step()

            log_metric(step, 'lr', float(scheduler.get_last_lr()[0]))


        if (step + 1) % loss_log_interval == 0:
            log_metric(step, 'train_loss', logged_tr_loss_avg.get_average_and_reset())


        if (step + 1) % eval_interval == 0:
            val_loss_acc = accumulating_metric()

            model.eval()
            with torch.inference_mode():
                for val_item in iter(val_dataloader):
                    val_inputs, val_labels = pipeline.get_training_pairs(val_item['ids'], tokenizer, mask_value)
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                    val_logits = model(val_inputs)

                    val_loss = criterion(val_logits.flatten(-3, -2), val_labels.flatten(-2, -1))

                    val_loss_acc.update(val_loss.item())

            model.train()

            last_val_loss = val_loss_acc.get_average_and_reset()

            log_metric(step, 'val_loss', last_val_loss)


        if (step + 1) % metric_print_interval == 0:
            print_info = {
                'time elapsed': time.strftime('%H:%M:%S', time.gmtime(time.time() - start)),
                'done': '{:.2f}'.format(100 * (step + 1) / settings.total_steps) + '%',
                'steps': step + 1,
                'epochs': (step + 1) // len(train_dataloader),
            }


            print_info['avg train loss'] = '{:.4f}'.format(display_tr_loss_sum.get_average_and_reset())

            print_info['last val loss'] = '{:.4f}'.format(last_val_loss) if last_val_loss != 'n/a' else 'n/a'

            print_info['current lr'] = '{:.6f}'.format(scheduler.get_last_lr()[0])

            # print the info
            print(colorama.Fore.YELLOW, end='')
            print(format_info(print_info))
            print(colorama.Style.RESET_ALL, end='')



        if (step + 1) % checkpoint_save_interval == 0:
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_save_path, 'checkpoint-' + 'step-' + str(step + 1) + '.pt'))

    print(colorama.Fore.GREEN)
    print('training finished:', settings.total_steps, 'steps completed')
    print(colorama.Style.RESET_ALL, end='')
