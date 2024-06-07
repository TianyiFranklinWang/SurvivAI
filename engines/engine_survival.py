import time
from collections import OrderedDict
from contextlib import suppress

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored
from timm import utils
from timm.models import model_parameters


def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        _logger,
        device=torch.device('cuda'),
        lr_scheduler=None,
        saver=None,
        amp_autocast=suppress,
        loss_scaler=None,
        model_ema=None,
):
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    for batch_idx, (visual_features, omics_features, labels, event_times, censorships) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        visual_features = [visual_feature.to(device) for visual_feature in visual_features]
        omics_features = [omics_feature.to(device) for omics_feature in omics_features]
        labels = [label.to(device) for label in labels]
        event_times = [event_time.to(device) for event_time in event_times]
        censorships = [censorship.to(device) for censorship in censorships]

        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            loss = 0
            with amp_autocast():
                for visual_feature, omics_feature, label, event_time, censorship in zip(visual_features, omics_features,
                                                                                        labels, event_times,
                                                                                        censorships):
                    output = model(visual_feature, omics_feature)
                    loss += loss_fn(output, label, event_time, censorship)
            loss /= len(visual_features)
            if accum_steps > 1:
                loss /= accum_steps
            return loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if has_no_sync and not need_update:
            with model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)

        if not args.distributed:
            losses_m.update(loss.item() * accum_steps, len(visual_features))
        update_sample_count += len(visual_features)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model, step=num_updates)

        if args.synchronize_step and device.type == 'cuda':
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item() * accum_steps, len(visual_features))
                update_sample_count *= args.world_size

            if utils.is_primary(args):
                _logger.info(
                    f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                    f'({100. * update_idx / (updates_per_epoch - 1):>3.0f}%)]  '
                    f'Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  '
                    f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                    f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                    f'LR: {lr:.3e}  '
                    f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                )

        if saver is not None and args.recovery_interval and (
                (update_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(
        model,
        loader,
        loss_fn,
        args,
        _logger,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        log_suffix=''
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (visual_features, omics_features, labels, event_times, censorships) in enumerate(loader):
            last_batch = batch_idx == last_idx

            visual_features = [visual_feature.to(device) for visual_feature in visual_features]
            omics_features = [omics_feature.to(device) for omics_feature in omics_features]
            labels = [label.to(device) for label in labels]
            event_times = [event_time.to(device) for event_time in event_times]
            censorships = [censorship.to(device) for censorship in censorships]

            with amp_autocast():
                for visual_feature, omics_feature, label, event_time, censorship in zip(visual_features, omics_features,
                                                                                        labels, event_times,
                                                                                        censorships):
                    output = model(visual_feature, omics_feature)

                    if isinstance(output, (tuple, list)):
                        output = output[0]

                    loss = loss_fn(output, label, event_time, censorship)

                    hazards = torch.sigmoid(output)
                    survival = torch.cumprod(1 - hazards, dim=1)
                    risk = -torch.sum(survival, dim=1)
                    all_risk_scores.append(risk.detach().cpu().numpy())
                    all_censorships.append(censorship.detach().cpu().numpy())
                    all_event_times.append(event_time.detach().cpu().numpy())

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == 'cuda':
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), len(visual_features))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                )

    c_index = concordance_index_censored(
        (1 - np.concatenate(all_censorships)).astype(bool),
        np.concatenate(all_event_times),
        np.concatenate(all_risk_scores),
        tied_tol=1e-8
    )[0]
    if args.distributed:
        c_index = utils.reduce_tensor(torch.tensor(c_index), args.world_size).item()

    metrics = OrderedDict([('loss', losses_m.avg), ('c-index', c_index)])

    return metrics
