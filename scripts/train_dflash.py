#!/usr/bin/env python3
# coding=utf-8
"""DFlash Training Script."""

import argparse
import contextlib
import json
import logging
import os
import shutil
import time
import warnings
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from accelerate.utils import set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from datasets import Dataset, concatenate_datasets, load_dataset
from specforge.args import SGLangBackendArgs, TrackerArgs
from specforge.core.dflash import OnlineDFlashModel
from specforge.data import build_eagle3_dataset, prepare_dp_dataloaders
from specforge.distributed import destroy_distributed, get_dp_group, init_distributed
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.dflash_target_model import (
    DFlashTargetModel,
    get_dflash_target_model,
)
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.tracker import create_tracker
from specforge.utils import (
    get_last_checkpoint,
    print_on_rank0,
    print_with_rank,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DFlash Draft Model")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument(
        "--target-model-backend",
        type=str,
        default="hf",
        choices=["sglang", "hf"],
        help="Backend for target model: 'sglang' (service) or 'hf' (local)",
    )
    model_group.add_argument("--draft-config-path", type=str, default=None)
    model_group.add_argument("--block-size", type=int, default=16)
    model_group.add_argument("--num-draft-layers", type=int, default=1)
    model_group.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="MASK token ID. If not provided, auto-detect from tokenizer.",
    )
    model_group.add_argument(
        "--attention-backend",
        type=str,
        default="flex_attention",
        choices=["eager", "sdpa", "flex_attention"],
        help="Attention backend for draft model.",
    )
    model_group.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    model_group.add_argument(
        "--num-anchors",
        type=int,
        default=512,
        help="Number of anchor positions per sequence",
    )
    model_group.add_argument(
        "--loss-decay-gamma",
        type=float,
        default=None,
        help="Gamma for exponential loss decay weighting (paper Eq.4). "
        "Suggested: 7 for block_size=16, 5 for 10, 4 for 8. None disables.",
    )

    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument("--train-data-path", type=str, nargs="+", required=True)
    dataset_group.add_argument("--eval-data-path", type=str, default=None)
    dataset_group.add_argument("--chat-template", type=str, default="qwen")
    dataset_group.add_argument("--is-preformatted", action="store_true")
    dataset_group.add_argument("--dataloader-num-workers", type=int, default=8)
    dataset_group.add_argument(
        "--build-dataset-num-proc",
        type=int,
        default=int(os.environ.get("SPECFORGE_DATA_NUM_PROC", 8)),
    )

    training_group = parser.add_argument_group("training")
    training_group.add_argument("--num-epochs", type=int, default=6)
    training_group.add_argument("--batch-size", type=int, default=1)
    training_group.add_argument("--learning-rate", type=float, default=6e-4)
    training_group.add_argument("--max-length", type=int, default=3072)
    training_group.add_argument("--warmup-ratio", type=float, default=0.04)
    training_group.add_argument("--max-grad-norm", type=float, default=1.0)
    training_group.add_argument("--accumulation-steps", type=int, default=1)
    training_group.add_argument("--seed", type=int, default=42)
    training_group.add_argument("--resume", action="store_true")
    training_group.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Directory of the checkpoint to resume training from",
    )
    training_group.add_argument(
        "--reset-scheduler",
        action="store_true",
        help="When loading from --ckpt-dir, load optimizer Adam state (momentum) "
        "but reset the LR scheduler for the new total_steps. "
        "Useful for Phase 2 fine-tuning where you want to continue "
        "optimizer momentum but start a fresh LR schedule.",
    )

    output_group = parser.add_argument_group("output")
    output_group.add_argument("--output-dir", type=str, required=True)
    output_group.add_argument("--cache-dir", type=str, default="./cache")
    output_group.add_argument("--log-interval", type=int, default=50)
    output_group.add_argument("--eval-interval", type=int, default=1000)
    output_group.add_argument("--save-interval", type=int, default=1000)

    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="The size of the tensor parallel for the target model",
    )

    tracker_group = parser.add_argument_group("tracker")
    TrackerArgs.add_args(tracker_group)

    dist_group = parser.add_argument_group("distributed")
    dist_group.add_argument("--dist-timeout", type=int, default=30)

    # SGLang specific args
    sglang_group = parser.add_argument_group("sglang backend")
    SGLangBackendArgs.add_args(sglang_group)

    return parser.parse_args()


def build_models(args) -> Tuple[DFlashTargetModel, DFlashDraftModel]:
    """Build target model (backend wrapper) and draft model."""
    print_on_rank0(
        f"Loading target model from {args.target_model_path} using {args.target_model_backend} backend"
    )

    target_model_kwargs = {}
    if args.target_model_backend == "sglang":
        target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()

    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend=args.target_model_backend,
        torch_dtype=torch.bfloat16,
        device="cuda" if args.target_model_backend == "hf" else None,
        trust_remote_code=args.trust_remote_code,
        **target_model_kwargs,
    )

    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(args.draft_config_path)
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
    else:
        target_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config.num_hidden_layers = args.num_draft_layers
        draft_config.block_size = args.block_size
        draft_config.num_target_layers = target_config.num_hidden_layers
        print_on_rank0("Auto-generated draft config from target model")

    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}

    draft_config._attn_implementation = args.attention_backend
    print_on_rank0(f"Using attention backend: {args.attention_backend}")

    draft_model = DFlashDraftModel(draft_config).cuda().to(torch.bfloat16)

    target_model.set_capture_layers(draft_model.target_layer_ids)

    print_on_rank0(
        f"Draft config: block_size={draft_config.block_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"num_target_layers={draft_config.num_target_layers}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )

    return target_model, draft_model


def parse_data_path_with_fraction(raw_path: str) -> Tuple[str, float]:
    """Parse a data path that may include a sampling fraction.

    Supports the format: /path/to/data.jsonl::0.1
    where 0.1 means use 10% of the dataset. Defaults to 1.0 (use all data).
    """
    if "::" in raw_path:
        path, fraction_str = raw_path.rsplit("::", 1)
        fraction = float(fraction_str)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"Sampling fraction must be in (0, 1], got {fraction} for {path}"
            )
        return path, fraction
    return raw_path, 1.0


def load_dataset_from_dir(path: str) -> Dataset:
    """Load a dataset directory by reading state.json _data_files for fast arrow loading.

    Falls back to load_dataset() if no state.json is found.
    """
    state_json_path = os.path.join(path, "state.json")
    if os.path.exists(state_json_path):
        with open(state_json_path, "r") as f:
            state = json.load(f)
        data_files = state.get("_data_files", [])
        if data_files:
            arrow_paths = [
                os.path.join(path, entry["filename"])
                for entry in data_files
                if os.path.exists(os.path.join(path, entry["filename"]))
            ]
            if arrow_paths:
                datasets = [Dataset.from_file(p) for p in arrow_paths]
                ds = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
                print_on_rank0(
                    f"Loaded {len(arrow_paths)} arrow file(s) from {path} via state.json"
                )
                return ds
        print_on_rank0(
            f"state.json found in {path} but no valid _data_files, falling back to load_dataset"
        )

    loaded = load_dataset(path)
    if "train" in loaded:
        return loaded["train"]
    return next(iter(loaded.values()))


def build_dataloader(args, tokenizer) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train and eval dataloaders."""
    import hashlib

    cache_params_string = (
        f"{','.join(sorted(args.train_data_path))}-"
        f"{args.max_length}-"
        f"{args.chat_template}-"
        f"{args.target_model_path}"
    )
    cache_key = hashlib.md5(cache_params_string.encode()).hexdigest()

    # Parse paths and optional sampling fractions (e.g. "data.jsonl::0.1")
    parsed_paths = [parse_data_path_with_fraction(p) for p in args.train_data_path]

    arrow_file_entries = [(p, f) for p, f in parsed_paths if p.endswith(".arrow")]
    arrow_dir_entries = [(p, f) for p, f in parsed_paths if os.path.isdir(p)]
    json_entries = [
        (p, f) for p, f in parsed_paths
        if not p.endswith(".arrow") and not os.path.isdir(p)
    ]

    raw_datasets = []

    for path, fraction in arrow_file_entries:
        ds = Dataset.from_file(path)
        if fraction < 1.0:
            original_len = len(ds)
            n_samples = max(1, int(original_len * fraction))
            ds = ds.shuffle(seed=args.seed).select(range(n_samples))
            print_on_rank0(f"Sampled {n_samples}/{original_len} ({fraction:.0%}) from {path}")
        raw_datasets.append(ds)

    for path, fraction in arrow_dir_entries:
        ds = load_dataset_from_dir(path)
        if fraction < 1.0:
            original_len = len(ds)
            n_samples = max(1, int(original_len * fraction))
            ds = ds.shuffle(seed=args.seed).select(range(n_samples))
            print_on_rank0(f"Sampled {n_samples}/{original_len} ({fraction:.0%}) from {path}")
        raw_datasets.append(ds)

    for path, fraction in json_entries:
        ds = load_dataset("json", data_files=path)["train"]
        if fraction < 1.0:
            original_len = len(ds)
            n_samples = max(1, int(original_len * fraction))
            ds = ds.shuffle(seed=args.seed).select(range(n_samples))
            print_on_rank0(f"Sampled {n_samples}/{original_len} ({fraction:.0%}) from {path}")
        raw_datasets.append(ds)

    # Preprocess each dataset separately (they may have different schemas),
    # then concatenate the processed results.
    processed_datasets = []
    for i, ds in enumerate(raw_datasets):
        ds_cache_key = f"{cache_key}_part{i}" if cache_key else None
        ds_cache_dir = os.path.join(args.cache_dir, "processed_dataset") if cache_key else None
        processed = build_eagle3_dataset(
            dataset=ds,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
            cache_dir=ds_cache_dir,
            cache_key=ds_cache_key,
            num_proc=args.build_dataset_num_proc,
        )
        print_on_rank0(f"Preprocessed dataset {i}: {len(processed)} samples")
        processed_datasets.append(processed)

    train_eagle3_dataset = concatenate_datasets(processed_datasets)

    min_loss_tokens = 2 * args.block_size
    original_size = len(train_eagle3_dataset)
    train_eagle3_dataset = train_eagle3_dataset.filter(
        lambda x: x["loss_mask"].sum() >= min_loss_tokens
    )
    print_on_rank0(
        f"Filtered train dataset: {original_size} -> {len(train_eagle3_dataset)} samples"
    )

    train_dataloader = prepare_dp_dataloaders(
        train_eagle3_dataset,
        args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        process_group=get_dp_group(),
    )

    eval_dataloader = None
    if args.eval_data_path:
        eval_path = args.eval_data_path
        if eval_path.endswith(".arrow"):
            eval_dataset = Dataset.from_file(eval_path)
        elif os.path.isdir(eval_path):
            eval_dataset = load_dataset_from_dir(eval_path)
        else:
            eval_dataset = load_dataset("json", data_files=eval_path)["train"]
        eval_eagle3_dataset = build_eagle3_dataset(
            dataset=eval_dataset,
            tokenizer=tokenizer,
            chat_template=args.chat_template,
            max_length=args.max_length,
            is_preformatted=args.is_preformatted,
        )
        eval_original_size = len(eval_eagle3_dataset)
        eval_eagle3_dataset = eval_eagle3_dataset.filter(
            lambda x: x["loss_mask"].sum() >= min_loss_tokens
        )
        print_on_rank0(
            f"Filtered eval dataset: {eval_original_size} -> {len(eval_eagle3_dataset)} samples"
        )
        eval_dataloader = prepare_dp_dataloaders(
            eval_eagle3_dataset,
            args.batch_size,
            num_workers=args.dataloader_num_workers,
            shuffle=False,
            process_group=get_dp_group(),
        )

    return train_dataloader, eval_dataloader


def save_checkpoint(args, epoch, step, dflash_model, draft_model, optimizer, total_micro_steps=0):
    """Save checkpoint."""
    save_dir = os.path.join(args.output_dir, f"epoch_{epoch}_step_{step}")
    if dist.get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()

    with FSDP.state_dict_type(dflash_model, StateDictType.FULL_STATE_DICT):
        state_dict = dflash_model.state_dict()
        draft_state_dict = {
            k.replace("draft_model.", ""): v
            for k, v in state_dict.items()
            if "draft_model." in k
        }

        if dist.get_rank() == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": step,
                    "total_micro_steps": total_micro_steps,
                    "args": args,
                    **optimizer.state_dict(),
                },
                os.path.join(save_dir, "training_state.pt"),
            )

            draft_model.save_pretrained(save_dir, state_dict=draft_state_dict)

            modeling_src = os.path.join(
                os.path.dirname(__file__),
                "..",
                "specforge",
                "modeling",
                "draft",
                "dflash.py",
            )
            modeling_dst = os.path.join(save_dir, "dflash.py")
            if os.path.exists(modeling_src):
                shutil.copy(modeling_src, modeling_dst)

            print_on_rank0(f"Saved checkpoint to {save_dir}")

    dist.barrier()


def record_metrics(
    args,
    loss: float,
    accuracy: float,
    global_step: int,
    total_steps: int,
    tracker,
    optimizer,
    mode: str = "train",
    acceptance_length: float = None,
) -> None:
    logdict = {}

    if mode == "train" and optimizer is not None:
        logdict["train/lr"] = optimizer.get_learning_rate()

    logdict[f"{mode}/loss"] = loss
    logdict[f"{mode}/accuracy"] = accuracy
    if acceptance_length is not None:
        logdict[f"{mode}/acceptance_length"] = acceptance_length

    accept_str = f", AccLen: {acceptance_length:.2f}" if acceptance_length is not None else ""
    print_on_rank0(
        f"{mode.capitalize()} - Step {global_step}/{total_steps}, Loss: {loss:.4f}, Acc: {accuracy:.4f}{accept_str}"
    )

    tracker.log(logdict, step=global_step)


def main():

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger().setLevel(logging.INFO)
    warnings.filterwarnings(
        "ignore",
        "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed",
    )

    args = parse_args()
    set_seed(args.seed)

    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    print_with_rank("Initialized distributed")

    target_model, draft_model = build_models(args)

    draft_model_last_checkpoint = None
    if args.ckpt_dir is not None:
        if os.path.isdir(args.ckpt_dir):
            draft_model_last_checkpoint = args.ckpt_dir
            print_on_rank0(f"Using checkpoint: {draft_model_last_checkpoint}")
        else:
            raise ValueError(
                f"Provided ckpt dir {args.ckpt_dir} is not a valid directory."
            )

    if args.resume and os.path.isdir(args.output_dir):
        draft_model_last_checkpoint = get_last_checkpoint(
            args.output_dir, prefix=r"epoch_\d+_step"
        )
        print_on_rank0(f"Last checkpoint detected: {draft_model_last_checkpoint}")

    resume_state = None
    if draft_model_last_checkpoint:
        loaded_model = DFlashDraftModel.from_pretrained(
            draft_model_last_checkpoint, torch_dtype=torch.bfloat16
        )
        draft_model.load_state_dict(loaded_model.state_dict())
        del loaded_model
        print_on_rank0("Loaded draft model weights from checkpoint")

        training_state_path = os.path.join(
            draft_model_last_checkpoint, "training_state.pt"
        )
        if os.path.exists(training_state_path):
            resume_state = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
            print_on_rank0(
                f"Will resume from epoch {resume_state['epoch']}, "
                f"step {resume_state['global_step']}"
            )

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)

    if args.mask_token_id is not None:
        mask_token_id = args.mask_token_id
    elif tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    print_on_rank0(f"Using mask_token_id: {mask_token_id}")

    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids
    print_on_rank0(f"dflash_config: {draft_model.config.dflash_config}")

    train_dataloader, eval_dataloader = build_dataloader(args, tokenizer)

    total_steps = (args.num_epochs * len(train_dataloader)) // args.accumulation_steps
    print_on_rank0(f"Total training steps: {total_steps}")

    print_on_rank0("Loading target embeddings and head...")
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key="model.embed_tokens.weight",  # Adjust if Qwen/Llama differs
        lm_head_key="lm_head.weight",
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )

    dflash_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
    )

    dflash_model = FSDP(
        dflash_model,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        ),
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
    )
    print_with_rank("Initialized FSDP")

    optimizer = BF16Optimizer(
        draft_model,
        lr=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )

    start_epoch = 0
    global_step = 0
    total_micro_steps = 0
    if resume_state is not None:
        if args.reset_scheduler:
            # Phase 2 fine-tuning: load Adam momentum but use fresh LR schedule
            optimizer.optimizer.load_state_dict(resume_state["optimizer_state_dict"])
            print_on_rank0(
                "Loaded optimizer Adam state (momentum) from checkpoint. "
                "Scheduler reset for new training run."
            )
        else:
            optimizer.load_state_dict(resume_state)
            start_epoch = resume_state["epoch"]
            global_step = resume_state["global_step"]
            total_micro_steps = resume_state.get(
                "total_micro_steps", global_step * args.accumulation_steps
            )
            print_on_rank0(f"Restored optimizer and scheduler, lr={optimizer.get_learning_rate():.6f}")
        del resume_state

    skip_steps = max(0, total_micro_steps - start_epoch * len(train_dataloader))

    print_on_rank0(f"Initializing tracker (report_to={args.report_to})...")
    tracker = create_tracker(args, args.output_dir)
    print_on_rank0("Tracker initialized successfully.")

    last_time = time.time()
    accum_loss = 0.0  # running sum of loss across accumulation window
    accum_acc = 0.0
    accum_accept_len = 0.0
    print_on_rank0(f"Starting training from epoch {start_epoch}, step {global_step}")

    micro_step = 0
    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        # Discard any partial accumulation from previous epoch
        if micro_step != 0:
            dflash_model.zero_grad(set_to_none=True)
            micro_step = 0
            accum_loss = 0.0
            accum_acc = 0.0
            accum_accept_len = 0.0
        dflash_model.train()

        if dist.get_rank() == 0:
            progress_bar = tqdm(
                train_dataloader, desc=f"Training Epoch {epoch}", leave=True
            )
        else:
            progress_bar = train_dataloader

        for step_in_epoch, data in enumerate(progress_bar):
            if epoch == start_epoch and step_in_epoch < skip_steps:
                continue
            total_micro_steps += 1
            micro_step += 1

            input_ids = data["input_ids"].cuda()
            attention_mask = data["attention_mask"].cuda()
            loss_mask = data["loss_mask"].cuda()
            target_output = target_model.generate_dflash_data(
                input_ids, attention_mask, loss_mask
            )
            hidden_states = target_output.hidden_states.cuda()  # Ensure on GPU

            loss, accuracy, acceptance_length = dflash_model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
            )

            accum_loss += loss.detach() / args.accumulation_steps
            accum_acc += accuracy.detach() / args.accumulation_steps
            accum_accept_len += acceptance_length.detach() / args.accumulation_steps

            is_accumulating = micro_step % args.accumulation_steps != 0
            sync_context = dflash_model.no_sync if is_accumulating else contextlib.nullcontext
            with sync_context():
                (loss / args.accumulation_steps).backward()

            if dist.get_rank() == 0:
                elapsed = time.time() - last_time
                last_time = time.time()
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{accuracy.item():.4f}",
                        "accept_len": f"{acceptance_length.item():.2f}",
                        "iter_time": f"{elapsed:.2f}s",
                    }
                )

            if not is_accumulating:
                optimizer.step()
                global_step += 1
                micro_step = 0

                if global_step % args.log_interval == 0:
                    dp_group = get_dp_group()
                    dp_world_size = dist.get_world_size(dp_group)
                    loss_log = accum_loss.clone()
                    acc_log = accum_acc.clone()
                    accept_len_log = accum_accept_len.clone()
                    dist.all_reduce(loss_log, group=dp_group)
                    dist.all_reduce(acc_log, group=dp_group)
                    dist.all_reduce(accept_len_log, group=dp_group)
                    loss_log = loss_log / dp_world_size
                    acc_log = acc_log / dp_world_size
                    accept_len_log = accept_len_log / dp_world_size

                    record_metrics(
                        args,
                        loss_log.item(),
                        acc_log.item(),
                        global_step,
                        total_steps,
                        tracker,
                        optimizer,
                        mode="train",
                        acceptance_length=accept_len_log.item(),
                    )

                if (
                    eval_dataloader is not None
                    and global_step % args.eval_interval == 0
                ):
                    dflash_model.eval()
                    eval_losses = []
                    eval_accs = []
                    eval_accept_lens = []

                    for eval_data in tqdm(
                        eval_dataloader,
                        desc=f"Evaluating Epoch {epoch}",
                        disable=dist.get_rank() != 0,
                    ):
                        with torch.no_grad():
                            eval_input_ids = eval_data["input_ids"].cuda()
                            eval_attention_mask = eval_data["attention_mask"].cuda()
                            eval_loss_mask = eval_data["loss_mask"].cuda()
                            eval_target_output = target_model.generate_dflash_data(
                                eval_input_ids, eval_attention_mask, eval_loss_mask
                            )
                            eval_hidden_states = eval_target_output.hidden_states.cuda()

                            eval_loss, eval_acc, eval_accept_len = dflash_model(
                                input_ids=eval_input_ids,
                                hidden_states=eval_hidden_states,
                                loss_mask=eval_loss_mask,
                            )
                            eval_losses.append(eval_loss)
                            eval_accs.append(eval_acc)
                            eval_accept_lens.append(eval_accept_len)

                    avg_eval_loss = torch.stack(eval_losses).mean()
                    avg_eval_acc = torch.stack(eval_accs).mean()
                    avg_eval_accept_len = torch.stack(eval_accept_lens).mean()
                    dp_group = get_dp_group()
                    dp_world_size = dist.get_world_size(dp_group)
                    dist.all_reduce(avg_eval_loss, group=dp_group)
                    dist.all_reduce(avg_eval_acc, group=dp_group)
                    dist.all_reduce(avg_eval_accept_len, group=dp_group)
                    avg_eval_loss = avg_eval_loss / dp_world_size
                    avg_eval_acc = avg_eval_acc / dp_world_size
                    avg_eval_accept_len = avg_eval_accept_len / dp_world_size

                    record_metrics(
                        args,
                        avg_eval_loss.item(),
                        avg_eval_acc.item(),
                        global_step,
                        total_steps,
                        tracker,
                        optimizer=None,
                        mode="eval",
                        acceptance_length=avg_eval_accept_len.item(),
                    )

                    dflash_model.train()

                if global_step % args.save_interval == 0:
                    save_checkpoint(
                        args, epoch, global_step, dflash_model, draft_model, optimizer,
                        total_micro_steps=total_micro_steps,
                    )

                accum_loss = 0.0
                accum_acc = 0.0
                accum_accept_len = 0.0

    save_checkpoint(
        args, args.num_epochs, global_step, dflash_model, draft_model, optimizer,
        total_micro_steps=total_micro_steps,
    )

    tracker.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
