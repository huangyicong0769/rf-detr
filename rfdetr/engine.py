# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import random

import torch
import torch.nn.functional as F

import rfdetr.util.misc as utils
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.datasets.coco import compute_multi_scale_scales

try:
    from torch.amp import autocast, GradScaler
    DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    DEPRECATED_AMP = True
from typing import DefaultDict, List, Callable
from rfdetr.util.misc import NestedTensor
import numpy as np


def _classification_metrics(all_logits, all_labels, thresholds=(0.5,)):
    """Compute common multi-label classification metrics.

    all_logits: [N, C] tensor
    all_labels: [N, C] tensor of {0,1}
    """
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).float()
    probs = logits.sigmoid()
    num_classes = probs.shape[1]

    def prf(thr):
        preds = (probs >= thr).float()
        tp = (preds * labels).sum(dim=0)
        fp = (preds * (1 - labels)).sum(dim=0)
        fn = ((1 - preds) * labels).sum(dim=0)

        precision_c = tp / (tp + fp + 1e-9)
        recall_c = tp / (tp + fn + 1e-9)

        macro_p = precision_c.mean().item()
        macro_r = recall_c.mean().item()
        macro_f1 = (2 * macro_p * macro_r) / (macro_p + macro_r + 1e-9)

        tp_micro = tp.sum()
        fp_micro = fp.sum()
        fn_micro = fn.sum()
        micro_p = (tp_micro / (tp_micro + fp_micro + 1e-9)).item()
        micro_r = (tp_micro / (tp_micro + fn_micro + 1e-9)).item()
        micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r + 1e-9)
        return {
            f'macro_precision@{thr}': macro_p,
            f'macro_recall@{thr}': macro_r,
            f'macro_f1@{thr}': macro_f1,
            f'micro_precision@{thr}': micro_p,
            f'micro_recall@{thr}': micro_r,
            f'micro_f1@{thr}': micro_f1,
            f'class_error@{thr}': (1.0 - micro_f1) * 100.0,
        }

    def average_precision():
        aps = []
        for c in range(num_classes):
            scores = probs[:, c]
            targets = labels[:, c]
            pos_total = targets.sum()
            if pos_total == 0:
                aps.append(float("nan"))
                continue
            # sort desc by score
            sorted_scores, idx = torch.sort(scores, descending=True)
            sorted_targets = targets[idx]
            tp = sorted_targets
            fp = 1 - sorted_targets
            tp_cum = torch.cumsum(tp, dim=0)
            fp_cum = torch.cumsum(fp, dim=0)
            precision = tp_cum / (tp_cum + fp_cum + 1e-9)
            recall = tp_cum / (pos_total + 1e-9)
            # prepend (0,1) for integration stability
            precision = torch.cat([torch.tensor([1.0], device=precision.device), precision])
            recall = torch.cat([torch.tensor([0.0], device=recall.device), recall])
            ap = torch.trapz(precision, recall).item()
            aps.append(ap)
        valid_aps = [a for a in aps if not math.isnan(a)]
        mAP = float(np.mean(valid_aps)) if len(valid_aps) > 0 else 0.0
        return mAP, aps

    def auroc():
        aucs = []
        for c in range(num_classes):
            scores = probs[:, c]
            targets = labels[:, c]
            pos = targets.sum()
            neg = (1 - targets).sum()
            if pos == 0 or neg == 0:
                aucs.append(float("nan"))
                continue
            sorted_scores, idx = torch.sort(scores, descending=True)
            sorted_targets = targets[idx]
            tps = torch.cumsum(sorted_targets, 0)
            fps = torch.cumsum(1 - sorted_targets, 0)
            tpr = tps / (pos + 1e-9)
            fpr = fps / (neg + 1e-9)
            tpr = torch.cat([torch.tensor([0.0], device=tpr.device), tpr])
            fpr = torch.cat([torch.tensor([0.0], device=fpr.device), fpr])
            auc = torch.trapz(tpr, fpr).item()
            aucs.append(auc)
        valid_aucs = [a for a in aucs if not math.isnan(a)]
        macro_auc = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else 0.0
        return macro_auc, aucs

    metrics = {}
    for thr in thresholds:
        metrics.update(prf(thr))
    mAP, per_class_ap = average_precision()
    macro_auc, per_class_auc = auroc()
    metrics['mAP'] = mAP
    metrics['macro_auc'] = macro_auc
    metrics['per_class_ap'] = per_class_ap
    metrics['per_class_auc'] = per_class_auc
    return metrics

def get_autocast_args(args):
    if DEPRECATED_AMP:
        return {'enabled': args.amp, 'dtype': torch.bfloat16}
    else:
        return {'device_type': 'cuda', 'enabled': args.amp, 'dtype': torch.bfloat16}


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = {},
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
    callbacks: DefaultDict[str, List[Callable]] = None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10
    start_steps = epoch * num_training_steps_per_epoch

    print("Grad accum steps: ", args.grad_accum_steps)
    print("Total batch size: ", batch_size * utils.get_world_size())

    # Add gradient scaler for AMP
    if DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler('cuda', enabled=args.amp)

    optimizer.zero_grad()
    assert batch_size % args.grad_accum_steps == 0
    sub_batch_size = batch_size // args.grad_accum_steps
    print("LENGTH OF DATA LOADER:", len(data_loader))
    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        it = start_steps + data_iter_step
        callback_dict = {
            "step": it,
            "model": model,
            "epoch": epoch,
        }
        for callback in callbacks["on_train_batch_start"]:
            callback(callback_dict)
        if "dp" in schedules:
            if args.distributed:
                model.module.update_drop_path(
                    schedules["dp"][it], vit_encoder_num_layers
                )
            else:
                model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][it])
            else:
                model.update_dropout(schedules["do"][it])

        if args.multi_scale and not args.do_random_resize_via_padding:
            scales = compute_multi_scale_scales(args.resolution, args.expanded_scales, args.patch_size, args.num_windows)
            random.seed(it)
            scale = random.choice(scales)
            with torch.inference_mode():
                samples.tensors = F.interpolate(samples.tensors, size=scale, mode='bilinear', align_corners=False)
                samples.mask = F.interpolate(samples.mask.unsqueeze(1).float(), size=scale, mode='nearest').squeeze(1).bool()

        for i in range(args.grad_accum_steps):
            start_idx = i * sub_batch_size
            final_idx = start_idx + sub_batch_size
            new_samples_tensors = samples.tensors[start_idx:final_idx]
            new_samples = NestedTensor(new_samples_tensors, samples.mask[start_idx:final_idx])
            new_samples = new_samples.to(device)
            new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets[start_idx:final_idx]]

            with autocast(**get_autocast_args(args)):
                outputs = model(new_samples, new_targets)
                loss_dict = criterion(outputs, new_targets)
                weight_dict = criterion.weight_dict
                losses = sum(
                    (1 / args.grad_accum_steps) * loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict
                )


            scaler.scale(losses).backward()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k:  v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(loss_dict_reduced)
            raise ValueError("Loss is {}, stopping training".format(loss_value))

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
        if ema_m is not None:
            if epoch >= 0:
                ema_m.update(model)
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def coco_extended_metrics(coco_eval):
    """
    Safe version: ignores the –1 sentinel entries so precision/F1 never explode.
    """

    iou_thrs, rec_thrs = coco_eval.params.iouThrs, coco_eval.params.recThrs
    # np.argwhere returns an array; pick the first match index safely
    iou50_idx_arr = np.flatnonzero(np.isclose(iou_thrs, 0.50))
    iou50_idx, area_idx, maxdet_idx = (int(iou50_idx_arr[0]), 0, 2)

    P = coco_eval.eval["precision"]
    S = coco_eval.eval["scores"]

    prec_raw = P[iou50_idx, :, :, area_idx, maxdet_idx]

    prec = prec_raw.copy().astype(float)
    prec[prec < 0] = np.nan

    f1_cls   = 2 * prec * rec_thrs[:, None] / (prec + rec_thrs[:, None])
    f1_macro = np.nanmean(f1_cls, axis=1)

    best_j   = int(f1_macro.argmax())

    macro_precision = float(np.nanmean(prec[best_j]))
    macro_recall    = float(rec_thrs[best_j])
    macro_f1        = float(f1_macro[best_j])

    score_vec = S[iou50_idx, best_j, :, area_idx, maxdet_idx].astype(float)
    score_vec[prec_raw[best_j] < 0] = np.nan
    score_thr = float(np.nanmean(score_vec))

    map_50_95, map_50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])

    per_class = []
    cat_ids = coco_eval.params.catIds
    cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
    for k, cid in enumerate(cat_ids):
        p_slice = P[:, :, k, area_idx, maxdet_idx]
        valid   = p_slice > -1
        ap_50_95 = float(p_slice[valid].mean()) if valid.any() else float("nan")
        ap_50    = float(p_slice[iou50_idx][p_slice[iou50_idx] > -1].mean()) if (p_slice[iou50_idx] > -1).any() else float("nan")

        pc = float(prec[best_j, k]) if prec_raw[best_j, k] > -1 else float("nan")
        rc = macro_recall

        #Doing to this to filter out dataset class
        if np.isnan(ap_50_95) or np.isnan(ap_50) or np.isnan(pc) or np.isnan(rc):
            continue

        per_class.append({
            "class"      : cat_id_to_name[int(cid)],
            "map@50:95"  : ap_50_95,
            "map@50"     : ap_50,
            "precision"  : pc,
            "recall"     : rc,
        })

    per_class.append({
        "class"     : "all",
        "map@50:95" : map_50_95,
        "map@50"    : map_50,
        "precision" : macro_precision,
        "recall"    : macro_recall,
    })

    return {
        "class_map": per_class,
        "map"      : map_50,
        "precision": macro_precision,
        "recall"   : macro_recall
    }

def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = ("bbox",) if not args.segmentation_head else ("bbox", "segm")
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        # Add autocast for evaluation
        with autocast(**get_autocast_args(args)):
            outputs = model(samples)

        if args.fp16_eval:
            for key in outputs.keys():
                if key == "enc_outputs":
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][
                                sub_key
                            ].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results_all = postprocess(outputs, orig_target_sizes)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results_all)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        results_json = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])
        stats["results_json"] = results_json
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

        if "segm" in iou_types:
            results_json = coco_extended_metrics(coco_evaluator.coco_eval["segm"])
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return stats, coco_evaluator


def train_one_epoch_cls(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args=None,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    callbacks: DefaultDict[str, List[Callable]] | None = None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = f"Epoch (cls): [{epoch}]"

    model.train()
    criterion.train()

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        with autocast(**get_autocast_args(args)):
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['loss_ce']

        # micro-F1 based class error (@0.5) to mirror detection's class_error reporting
        with torch.no_grad():
            probs = outputs['logits'].sigmoid()
            labels = torch.stack([t['labels_multi'] for t in targets], dim=0).float()
            preds = (probs >= 0.5).float()
            tp = (preds * labels).sum()
            fp = (preds * (1 - labels)).sum()
            fn = ((1 - preds) * labels).sum()
            micro_p = tp / (tp + fp + 1e-9)
            micro_r = tp / (tp + fn + 1e-9)
            micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r + 1e-9)
            class_error = (1.0 - micro_f1) * 100.0

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        if ema_m is not None:
            ema_m.update(model)

        metric_logger.update(loss=loss.item(), **{k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(class_error=class_error.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats["train_loss"] = stats.get("loss")
    stats["epoch"] = epoch
    stats["flavor"] = "train"
    if callbacks:
        for callback in callbacks["on_fit_epoch_end"]:
            callback(stats)
    return stats


@torch.no_grad()
def evaluate_cls(model, criterion, data_loader, device, args=None, callbacks: DefaultDict[str, List[Callable]] | None = None, epoch: int | None = None, flavor: str = "base"):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test (cls):"

    all_logits, all_labels = [], []

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        with autocast(**get_autocast_args(args)):
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)

        with torch.no_grad():
            probs = outputs['logits'].sigmoid()
            labels = torch.stack([t['labels_multi'] for t in targets], dim=0).float()
            preds = (probs >= 0.5).float()
            tp = (preds * labels).sum()
            fp = (preds * (1 - labels)).sum()
            fn = ((1 - preds) * labels).sum()
            micro_p = tp / (tp + fp + 1e-9)
            micro_r = tp / (tp + fn + 1e-9)
            micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r + 1e-9)
            class_error = (1.0 - micro_f1) * 100.0

        metric_logger.update(loss=loss_dict['loss_ce'].item(), class_error=class_error.item())

        all_logits.append(outputs['logits'].detach().cpu())
        all_labels.append(torch.stack([t['labels_multi'].cpu() for t in targets], dim=0))

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {"epoch": epoch if epoch is not None else -1}
    stats.update({k: meter.global_avg for k, meter in metric_logger.meters.items()})
    stats["test_loss"] = stats.get("loss")
    stats["flavor"] = flavor
    metrics = _classification_metrics(all_logits, all_labels, thresholds=(0.5,))
    stats.update(metrics)

    # Build results_json to mirror detection outputs
    class_names = getattr(args, "class_names", None) or [str(i) for i in range(all_logits[0].shape[1])]
    per_class_ap = metrics.get("per_class_ap", [])
    per_class_auc = metrics.get("per_class_auc", [])
    class_map = []
    for idx, name in enumerate(class_names):
        ap = per_class_ap[idx] if idx < len(per_class_ap) else float("nan")
        auc = per_class_auc[idx] if idx < len(per_class_auc) else float("nan")
        class_map.append({
            "class": name,
            "ap": ap,
            "auc": auc,
        })
    class_map.append({
        "class": "all",
        "map": metrics.get("mAP", 0.0),
        "macro_auc": metrics.get("macro_auc", 0.0),
        "macro_precision@0.5": metrics.get("macro_precision@0.5", 0.0),
        "macro_recall@0.5": metrics.get("macro_recall@0.5", 0.0),
        "macro_f1@0.5": metrics.get("macro_f1@0.5", 0.0),
        "micro_precision@0.5": metrics.get("micro_precision@0.5", 0.0),
        "micro_recall@0.5": metrics.get("micro_recall@0.5", 0.0),
        "micro_f1@0.5": metrics.get("micro_f1@0.5", 0.0),
    })

    stats["results_json"] = {
        "class_map": class_map,
        "map": metrics.get("mAP", 0.0),
        "macro_auc": metrics.get("macro_auc", 0.0),
        "macro_precision@0.5": metrics.get("macro_precision@0.5", 0.0),
        "macro_recall@0.5": metrics.get("macro_recall@0.5", 0.0),
        "macro_f1@0.5": metrics.get("macro_f1@0.5", 0.0),
        "micro_precision@0.5": metrics.get("micro_precision@0.5", 0.0),
        "micro_recall@0.5": metrics.get("micro_recall@0.5", 0.0),
        "micro_f1@0.5": metrics.get("micro_f1@0.5", 0.0),
    }
    if callbacks:
        for callback in callbacks["on_fit_epoch_end"]:
            callback(stats)
    return stats
