import sys

sys.path.append("../")
from logging import DEBUG, basicConfig, config, getLogger

from logconfig.logconf import log_conf, logfilepath
from peft import PeftModel
from setting import BASE_MODEL_NAME, PEFT_NAME
from transformers import AutoModelForCausalLM, AutoTokenizer

basicConfig(filename=logfilepath, level=DEBUG)
config.dictConfig(log_conf)
logger = getLogger(__name__)


def main():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        load_in_8bit=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)

    model = PeftModel.from_pretrained(
        model,
        PEFT_NAME,
        # device_map="auto"
    )
    model.eval()
    while True:
        query = input()
        logger.info(f"query: {query}")
        logger.info("\n**********")
        logger.info("生成結果:")
        logger.info(generate(model, tokenizer, query))
        print(generate(model, tokenizer, query))
        logger.info("**********\n")


def generate_prompt(data_point):
    if data_point["input"]:
        result = f"""### 指示:
{data_point["instruction"]}

### 入力:
{data_point["input"]}

### 回答:
"""
    else:
        result = f"""### 指示:
{data_point["instruction"]}

### 回答:
"""

    # 改行→<NL>
    result = result.replace("\n", "<NL>")
    return result


def generate(model, tokenizer, instruction, input=None, maxTokens=256) -> str:
    prompt = generate_prompt({"instruction": instruction, "input": input})
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, add_special_tokens=False).input_ids.cuda()
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=maxTokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.75,
        top_k=40,
        no_repeat_ngram_size=2,
    )
    outputs = outputs[0].tolist()

    # 最後のEOSトークンまでデコード
    if tokenizer.eos_token_id in outputs:
        eos_list = [i for i, x in enumerate(outputs) if x == tokenizer.eos_token_id]
        decoded = tokenizer.decode(outputs[: eos_list[len(eos_list) - 1]])

        sentinel = "### 回答:"
        sentinelLoc = decoded.find(sentinel)
        if sentinelLoc >= 0:
            result = decoded[sentinelLoc + len(sentinel) :]
            return result.replace("<NL>", "\n")  # <NL>→改行
        else:
            return "Warning: Expected prompt template to be emitted.  Ignoring output."
    else:
        return "Warning: no <eos> detected ignoring output"
