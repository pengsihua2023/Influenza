from transformers import AutoTokenizer, EsmForMaskedLM, EsmConfig, Trainer, TrainingArguments, IntervalStrategy, EarlyStoppingCallback, TrainerCallback
from transformers.utils import send_example_telemetry
from transformers.trainer_utils import EvalLoopOutput
from datasets import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from Bio import SeqIO
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
from peft import LoraConfig, get_peft_model
import torch.distributed as dist
import multiprocessing
from typing import Dict

# 移除自定义的分类头和模型类，因为我们将使用EsmForMaskedLM
# class CustomEsmClassificationHead 和 class CustomEsmForSequenceClassification 被移除

class TrainingMetricsCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch.is_integer():
            train_result = self.trainer.evaluate(eval_dataset=self.trainer.train_dataset, metric_key_prefix="train")
            eval_result = self.trainer.evaluate(eval_dataset=self.trainer.eval_dataset, metric_key_prefix="eval")
            print("\n" + "="*50)
            print(f"Epoch {int(state.epoch)} Training Metrics:")
            print(f"Training Loss: {train_result['train_loss']:.4f}")
            print(f"Training Perplexity: {np.exp(train_result['train_loss']):.4f}")
            print("\nEvaluation Metrics:")
            print(f"Evaluation Loss: {eval_result['eval_loss']:.4f}")
            print(f"Evaluation Perplexity: {np.exp(eval_result['eval_loss']):.4f}")
            print("="*50 + "\n")

class CustomTrainer(Trainer):
    def __init__(self, *args, eval_on_train=True, **kwargs):
        super().__init__(*args, processing_class=kwargs.pop("tokenizer", None), **kwargs)
        self.eval_on_train = eval_on_train

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        outputs = model(**inputs)
        loss = outputs["loss"]
        self.accelerator.backward(loss)
        return loss.detach() / self.args.gradient_accumulation_steps

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        self.model.eval()
        total_eval_loss = 0
        total_samples = 0

        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                outputs = self.model(**inputs)
                loss = outputs["loss"]
            total_eval_loss += loss.detach().float()
            total_samples += inputs["input_ids"].size(0)

        avg_loss = total_eval_loss / len(dataloader)
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss.item(),
            f"{metric_key_prefix}_perplexity": np.exp(avg_loss.item())
        }
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=total_samples)

    def on_train_end(self, args, state, control, **kwargs):
        super().on_train_end(args, state, control, **kwargs)
        print("\n" + "="*50)
        print("Training complete. Running final evaluations...")
        if self.eval_on_train:
            train_result = self.evaluate(eval_dataset=self.train_dataset, metric_key_prefix="train")
            print(f"Final Training Loss: {train_result['train_loss']:.4f}")
            print(f"Final Training Perplexity: {np.exp(train_result['train_loss']):.4f}")
        eval_result = self.evaluate(eval_dataset=self.eval_dataset, metric_key_prefix="eval")
        print(f"Final Evaluation Loss: {eval_result['eval_loss']:.4f}")
        print(f"Final Evaluation Perplexity: {np.exp(eval_result['eval_loss']):.4f}")
        print("="*50 + "\n")

# 其余辅助函数保持不变
log_dir = './logs-650M-Influenza-A'
os.makedirs(log_dir, exist_ok=True)
print(f"Log directory is set at {log_dir}")

def read_fasta_file(fasta_path):
    return [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
    return {
        "loss": loss.item(),
        "perplexity": np.exp(loss.item())
    }

send_example_telemetry("protein_language_modeling_notebook", framework="pytorch")

# 加载数据
fasta_sequences = read_fasta_file("yiwan-sequence.fasta")

# Split the sequences into train and test sets without labels
train_sequences, test_sequences = train_test_split(
    fasta_sequences, test_size=0.20, shuffle=True
)

# 准备模型和tokenizer
model_checkpoint = "facebook/esm2_t33_650M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# 对序列进行掩码处理
def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """Prepare masked tokens inputs/labels for masked language modeling."""
    inputs = inputs.clone()
    labels = inputs.clone()
    
    # 创建掩码
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    
    # 随机选择要掩码的位置
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # 只计算掩码位置的损失
    
    # 80%的时间用[MASK]替换
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
    # 10%的时间用随机token替换
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]
    
    return inputs, labels

# 准备数据集
def prepare_dataset(sequences, tokenizer):
    tokenized = tokenizer(sequences, padding=True, truncation=True, max_length=50, return_tensors="pt")
    input_ids, labels = mask_tokens(tokenized["input_ids"], tokenizer)
    return Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    })

train_dataset = prepare_dataset(train_sequences, tokenizer)
test_dataset = prepare_dataset(test_sequences, tokenizer)

# 训练参数
args = TrainingArguments(
    output_dir=f"{model_checkpoint.split('/')[-1]}-Inflenza-A",
    do_train=True,
    do_eval=True,
    eval_strategy=IntervalStrategy.EPOCH,
    save_strategy=IntervalStrategy.EPOCH,
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.05,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    deepspeed="ds_config.json",
    push_to_hub=False,
    logging_dir='./logs-Influenza-A',
    logging_strategy=IntervalStrategy.STEPS,
    logging_steps=100,
    fp16=True,
    warmup_steps=1000,
    lr_scheduler_type="cosine",
)

# 初始化模型，使用EsmForMaskedLM
config = EsmConfig.from_pretrained(model_checkpoint)
model = EsmForMaskedLM.from_pretrained(model_checkpoint, config=config)

# 定义LoRA配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense", "intermediate.dense", "output.dense"],
    lora_dropout=0.2,
    bias="none",
)

# 应用LoRA到模型
model = get_peft_model(model, lora_config)

model_name = model_checkpoint.split("/")[-1]

early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,
    early_stopping_threshold=0.0001
)

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    eval_on_train=True,
    callbacks=[early_stopping_callback]
)

# 添加训练指标回调
trainer.add_callback(TrainingMetricsCallback(trainer))

# 开始训练
try:
    trainer.train()
except Exception as e:
    print(f"Training failed: {e}")
finally:
    if dist.is_initialized():
        dist.barrier()

# 保存模型前确保所有进程都完成了训练
if dist.is_initialized():
    dist.barrier()

# 只在主进程上保存模型
if not dist.is_initialized() or dist.get_rank() == 0:
    try:
        # 保存模型和tokenizer
        model_path = f"{model_name}-Influenza-A"
        print(f"\nStarting to save model to {model_path}...")
        
        # 确保保存目录存在
        os.makedirs(model_path, exist_ok=True)
        
        # 保存模型
        print("Saving model...")
        # 合并权重
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
        # 保存合并后的模型
        model.save_pretrained(
            model_path,
            safe_serialization=True
        )
        print("Model saved successfully")
        
        print("Saving tokenizer...")
        tokenizer.save_pretrained(model_path)
        print("Tokenizer saved successfully")
        
        print("Saving training state...")
        trainer.save_state()
        print("Training state saved successfully")
        
        # 保存模型配置
        print("Saving model configuration...")
        model.config.save_pretrained(model_path)
        print("Model configuration saved successfully")
        
        # 保存训练参数
        print("Saving training arguments...")
        import json
        args_dict = vars(args).copy()
        for key in list(args_dict.keys()):
            if not isinstance(args_dict[key], (str, int, float, bool, list, dict, type(None))):
                args_dict[key] = str(args_dict[key])
        with open(os.path.join(model_path, "training_args.json"), "w") as f:
            json.dump(args_dict, f, indent=2)
        print("Training arguments saved successfully")
        
        # 检查文件是否保存成功
        saved_files = os.listdir(model_path)
        print(f"\nSaved files in {model_path}:")
        for file in saved_files:
            print(f"- {file}")
        
        print(f"\nModel and tokenizer saved successfully to {model_path}")
        
        # 打印保存的模型信息
        print("\nModel information:")
        print(f"Model type: {type(model).__name__}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        raise e

# 确保所有进程都完成了保存操作
if dist.is_initialized():
    print("\nWaiting for all processes to complete saving...")
    dist.barrier()
    print("All processes completed saving")

print("\nTraining and saving completed successfully!")

# 确保所有进程都正确退出
if dist.is_initialized():
    dist.destroy_process_group()
    print("Process group destroyed")

# 强制退出程序
import sys
sys.exit(0)