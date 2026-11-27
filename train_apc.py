import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from src.config import ABSAConfig, APC_LABELS, APC_LABEL2ID, APC_ID2LABEL
from src.data_pipeline import build_apc_datasets


class WeightedSequenceTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = torch.tensor([1.35, 2.10, 0.60], device=logits.device, dtype=logits.dtype)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_apc_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": macro_f1,
    }


def train_apc(cfg: ABSAConfig):
    datasets = build_apc_datasets(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.apc_model_name, use_fast=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.apc_model_name,
        num_labels=len(APC_LABELS),
        id2label=APC_ID2LABEL,
        label2id=APC_LABEL2ID,
        attn_implementation="eager",
    )

    save_path = os.path.join(cfg.output_dir, "apc_deberta_v3")
    training_args = TrainingArguments(
        output_dir=save_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.apc_batch_size,
        per_device_eval_batch_size=cfg.apc_batch_size * 2,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.apc_epochs,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_ratio,
        max_grad_norm=1.0,
        fp16=cfg.use_fp16,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = WeightedSequenceTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("dev", datasets["test"]),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_apc_metrics,
    )

    trainer.train()

    pred_output = trainer.predict(datasets["test"])
    preds = np.argmax(pred_output.predictions, axis=-1)
    labels = pred_output.label_ids

    print("\n[Phase 2 APC Test Metrics]:", pred_output.metrics)
    print("\n[Confusion Matrix (rows=true, cols=pred) | Order:", APC_LABELS, "]")
    print(confusion_matrix(labels, preds))

    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    return pred_output.metrics


if __name__ == "__main__":
    train_apc(ABSAConfig())