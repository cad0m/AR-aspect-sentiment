"""The training loop and small helpers shared by notebooks 03, 04, 05 and 06.
Models, losses and hyperparameters stay in the notebooks."""
import copy
import random
import time

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compact_state(model):
    """Weights on the CPU, in half precision: a BERT goes from 430 MB to 215 MB on Drive."""
    return {
        name: (value.half() if value.is_floating_point() else value).cpu()
        for name, value in model.state_dict().items()
    }


def to_device(batch):
    return {name: value.to(DEVICE) for name, value in batch.items()}


def collect_outputs(loader, compute_outputs):
    """compute_outputs(batch) gives the logits of a batch. Returns the logits of the whole loader."""
    outputs = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="predicting", leave=False):
            outputs.append(compute_outputs(to_device(batch)).float().cpu())
    return torch.cat(outputs)


def fit(model, train_loader, compute_loss, validation_score, optimizer, scheduler=None, epochs=15, patience=3, mixed_precision=False):
    """Early stopping on validation_score(model), higher is better. Returns one row per epoch."""
    use_amp = mixed_precision and DEVICE == "cuda"
    scaler = torch.amp.GradScaler(DEVICE, enabled=use_amp)
    best_score, best_epoch, best_state = -1.0, 0, None
    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            optimizer.zero_grad()
            with torch.autocast(DEVICE, enabled=use_amp):
                loss = compute_loss(model, to_device(batch))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.3f}")

        model.eval()
        score = float(validation_score(model))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_loader), "val_score": score,
                        "seconds": time.time() - start})
        print(f"epoch {epoch}: loss {history[-1]['train_loss']:.4f}, val {score:.4f}, {history[-1]['seconds']:.0f}s")

        if score > best_score:
            best_score, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            print("early stopping")
            break

    model.load_state_dict(best_state)
    model.eval()
    return history


def tokenized_rows(tokenizer, texts, second_texts=None, labels=None, max_length=128):
    """One dict per text, for transformers.DataCollatorWithPadding. second_texts: the part after [SEP]."""
    encoded = tokenizer(list(texts), None if second_texts is None else list(second_texts), truncation=True, max_length=max_length)
    rows = [dict(zip(encoded.keys(), values)) for values in zip(*encoded.values())]
    if labels is not None:
        for row, label in zip(rows, labels):
            row["labels"] = label
    return rows


def token_inputs(batch):
    return {name: value for name, value in batch.items() if name in ("input_ids", "attention_mask", "token_type_ids")}
