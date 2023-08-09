import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 独自設定
import bitsandbytes as bnb
import transformers
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("../")
from logging import DEBUG, basicConfig, config, getLogger

from logconfig.logconf import log_conf, logfilepath
from PathSetting import PathSetting
from setting import BASE_MODEL_NAME, PEFT_NAME
from util.ioutil import read_json

# ログ設定
basicConfig(filename=logfilepath, level=DEBUG)
config.dictConfig(log_conf)
logger = getLogger(__name__)

# Constants
EVAL_STEPS = 200
SAVE_STEPS = 200
LOGGING_STEPS = 20
NUM_TRAIN_EPOCHS = 30
LEARNING_RATE = 3e-4
SAVE_TOTAL_LIMIT = 3
CUTOFF_LEN = 256
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False, model_max_length=512)


def main():
    ps = PathSetting()
    data = read_json(ps.get_dataset_filename())
    train_dataset, val_dataset = split_dataset(data)
    model = define_model(BASE_MODEL_NAME)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=transformers.TrainingArguments(
            num_train_epochs=NUM_TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            logging_steps=LOGGING_STEPS,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_steps=SAVE_STEPS,
            output_dir=ps.dir_ft_results,
            save_total_limit=SAVE_TOTAL_LIMIT,
            push_to_hub=False,
            auto_find_batch_size=True,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False
    trainer.train()
    model.config.use_cache = True

    trainer.model.save_pretrained(PEFT_NAME)


def define_model(model_name):
    logger.info("モデル読み込み開始")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
    )
    logger.info("モデル読み込み完了")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
    )
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }


def generate_prompt(data_point):
    result = f"""### 指示:
{data_point["input"]}

### 回答:
{data_point["completion"]}
"""
    # 改行→<NL>
    result = result.replace("\n", "<NL>")
    return result


def split_dataset(data):
    train_dataset = []
    val_dataset = []

    for i in range(len(data)):
        if i % 5 == 0:
            x = tokenize(generate_prompt(data[i]), tokenizer)
            val_dataset.append(x)
        else:
            x = tokenize(generate_prompt(data[i]), tokenizer)
            train_dataset.append(x)
    return train_dataset, val_dataset


if __name__ == "__main__":
    main()
