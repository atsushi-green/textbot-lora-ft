import os
import sys
from logging import DEBUG, basicConfig, config, getLogger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import bitsandbytes as bnb
import transformers
from MyDataset import Dolly15KJa
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("../")
from logconfig.logconf import log_conf, logfilepath
from PathSetting import PathSetting
from setting import BASE_MODEL_NAME, PEFT_NAME
from util.ioutil import make_directory, read_json

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


def main():
    ps = PathSetting(PEFT_NAME)

    dataset = Dolly15KJa()
    train_dataset = dataset.dataset["train"]
    model = define_model(BASE_MODEL_NAME)
    make_directory(ps.dir_model)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        args=transformers.TrainingArguments(
            num_train_epochs=NUM_TRAIN_EPOCHS,
            learning_rate=LEARNING_RATE,
            logging_steps=LOGGING_STEPS,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_steps=SAVE_STEPS,
            output_dir=ps.dir_model,
            save_total_limit=SAVE_TOTAL_LIMIT,
            push_to_hub=False,
            auto_find_batch_size=True,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(dataset.tokenizer, mlm=False),
    )

    model.config.use_cache = False
    trainer.train()
    model.config.use_cache = True

    trainer.model.save_pretrained(ps.dir_model)


def define_model(model_name):
    logger.info("モデル読み込み開始")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    logger.info("モデル読み込み終了")

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


if __name__ == "__main__":
    main()
