import torch
import datasets

import time
import colorama
import os


from dataclasses import dataclass
import dataclasses
import contextlib
from typing import Callable, Any, cast

from pipeline import pipeline_protocol

@dataclass
class train_config:
    sequence_length: int
    batch_size: int


    total_steps: int = 5000


    update_interval: int = 1

    schedule_interval: int = 20



@dataclass
class hyperparameter_config:
    lr_warmup_steps: int
    lr_decay_steps: int

    start_lr: float
    peak_lr: float
    end_lr: float

    betas: tuple[float, float]

    weight_decay: float


def get_params_with_names(named_parameters, names: list[str]):
    filtered_parameters = []
    for n, p in named_parameters:
        for name in names:
            if name in n:
                filtered_parameters.append(p)
                break
    return filtered_parameters


def remove_params_with_names(named_parameters, names: list[str]):
    filtered_parameters = []
    for n, p in named_parameters:
        name_in_n = False
        for name in names:
            if name in n:
                name_in_n = True
        if name_in_n == False:
            filtered_parameters.append(p)
    return filtered_parameters


def configure_optimizer(model, hyperparameters):
    nodecay_param_names = ['ln.weight', 'wte.weight']

    optim_groups = [
        {
            'params': remove_params_with_names(model.named_parameters(), nodecay_param_names),
            'weight_decay': hyperparameters.weight_decay
        },
        {
            'params': get_params_with_names(model.named_parameters(), nodecay_param_names),
            'weight_decay': 0.0
        }
    ]

    return torch.optim.AdamW(optim_groups, lr = hyperparameters.peak_lr, betas = hyperparameters.betas)



def format_info(info):
    tmp = ''
    for i, (key, value) in enumerate(info.items()):
        if i == 0:
            tmp += str(value).rjust(6) + ' ' + key
        else:
            tmp += '      ' + str(value).rjust(6) + ' ' + key

    return tmp



def train(
    settings: train_config,
    hyperparameters: hyperparameter_config,

    model,

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


    non_contributing_tokens: list[int] = pipeline.get_non_contributing_tokens(tokenizer)

    if len(non_contributing_tokens) > 1:
        raise NotImplementedError('multiple non-contributing tokens not yet supported')
    elif non_contributing_tokens:
        criterion = torch.nn.CrossEntropyLoss(reduction = 'mean', ignore_index = non_contributing_tokens[0])
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction = 'mean')


    optimizer = configure_optimizer(model, hyperparameters)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = hyperparameters.start_lr / hyperparameters.peak_lr, end_factor = 1.0, total_iters = hyperparameters.lr_warmup_steps // settings.schedule_interval),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = hyperparameters.lr_decay_steps // settings.schedule_interval, eta_min = hyperparameters.end_lr)
        ]
    )



    logged_train_loss_sum = 0.0
    displayed_train_loss_sum = 0.0
    displayed_last_val_loss = 'n/a'


    start = time.time()
    for step in range(settings.total_steps):
        model.train()

        ids = next(iter(train_dataloader))['ids'].to(device)

        inputs, labels = pipeline.get_training_pairs(ids)

        logits = model(inputs)

        # flatten batch and sequence dimensions into one dimension for computing the loss
        loss = criterion(logits.flatten(-3, -2), labels.flatten(-2, -1))


        logged_train_loss_sum += loss.item()
        displayed_train_loss_sum += loss.item()


        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if (step + 1) % settings.update_interval == 0:
            optimizer.step()
            optimizer.zero_grad()


        if (step + 1) % settings.schedule_interval == 0:
            scheduler.step()

            log_metric(step, 'lr', float(scheduler.get_last_lr()[0]))


        if (step + 1) % loss_log_interval == 0:
            train_loss_average = logged_train_loss_sum / loss_log_interval
            logged_train_loss_sum = 0.0

            log_metric(step, 'train_loss', train_loss_average)


        if (step + 1) % eval_interval == 0:
            val_loss_sum = 0.0

            model.eval()
            with torch.inference_mode():
                for val_item in iter(val_dataloader):
                    val_ids = val_item['ids'].to(device)
                    val_inputs, val_labels = val_ids[..., :-1], val_ids[..., 1:]

                    val_logits = model(val_inputs)

                    val_loss = criterion(val_logits.flatten(-3, -2), val_labels.flatten(-2, -1))

                    val_loss_sum += val_loss.item()

            model.train()

            val_loss_avg = val_loss_sum / len(val_dataloader)

            displayed_last_val_loss = val_loss_avg

            log_metric(step, 'val_loss', val_loss_avg)


        if (step + 1) % metric_print_interval == 0:
            print_info = {
                'time elapsed': time.strftime('%H:%M:%S', time.gmtime(time.time() - start)),
                'done': '{:.2f}'.format(100 * (step + 1) / settings.total_steps) + '%',
                'steps': step + 1,
                'epochs': (step + 1) // len(train_dataloader),
            }


            print_info['avg train loss'] = '{:.4f}'.format(displayed_train_loss_sum / metric_print_interval)
            displayed_train_loss_sum = 0

            print_info['last val loss'] = '{:.4f}'.format(displayed_last_val_loss) if displayed_last_val_loss != 'n/a' else 'n/a'

            print_info['current lr'] = '{:.6f}'.format(scheduler.get_last_lr()[0])

            # print the info
            print(colorama.Fore.YELLOW, end='')
            print(format_info(print_info))
            print(colorama.Style.RESET_ALL, end='')



        if (step + 1) % checkpoint_save_interval == 0:
            torch.save(model, os.path.join(checkpoint_save_path, 'checkpoint-' + 'step-' + str(step + 1) + '.pt'))

    print(colorama.Fore.GREEN)
    print('training finished:', settings.total_steps, 'steps completed')
    print(colorama.Style.RESET_ALL, end='')
