import os
import torch
import torch.nn as nn
import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from src.config import ABSAConfig, ATE_LABELS, ATE_LABEL2ID, ATE_ID2LABEL
from src.data_pipeline import build_ate_datasets

seqeval = evaluate.load("seqeval")


class WeightedTokenTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = torch.tensor([0.5, 2.0, 2.0], device=logits.device, dtype=logits.dtype)
        loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_ate_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_predictions = [
        [ATE_ID2LABEL[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [ATE_ID2LABEL[l] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def train_ate(cfg: ABSAConfig):
    datasets = build_ate_datasets(cfg)
    # add_prefix_space=True is required for RoBERTa fast offset alignment
    tokenizer = AutoTokenizer.from_pretrained(cfg.ate_model_name, use_fast=True, add_prefix_space=True)

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.ate_model_name,
        num_labels=len(ATE_LABELS),
        id2label=ATE_ID2LABEL,
        label2id=ATE_LABEL2ID,
        attn_implementation="eager",
    )

    save_path = os.path.join(cfg.output_dir, "ate_deberta_v3")
    training_args = TrainingArguments(
        output_dir=save_path,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.ate_batch_size,
        per_device_eval_batch_size=cfg.ate_batch_size * 2,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.ate_epochs,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_ratio,
        max_grad_norm=1.0,
        fp16=cfg.use_fp16,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = WeightedTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("dev", datasets["test"]),
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_ate_metrics,
    )

    trainer.train()
    test_metrics = trainer.evaluate(datasets["test"])
    print("\n[Phase 1 ATE Test Metrics]:", test_metrics)

    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    return test_metrics


if __name__ == "__main__":
    train_ate(ABSAConfig())