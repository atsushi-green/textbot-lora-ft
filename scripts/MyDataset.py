# https://github.com/tatsu-lab/stanford_alpaca
from logging import DEBUG, basicConfig, config, getLogger

import datasets
from logconfig.logconf import log_conf, logfilepath
from PathSetting import PathSetting
from setting import TOKENIER_NAME
from transformers import AutoTokenizer, LlamaTokenizer
from util.ioutil import read_json

# ログ設定
basicConfig(filename=logfilepath, level=DEBUG)
config.dictConfig(log_conf)
logger = getLogger(__name__)


class Dolly15KJa:
    def __init__(self, data_max_length: int = 1024) -> None:
        logger.info("利用データセット: kunishou/databricks-dolly-15k-ja")
        self.PROMPT_DICT = {
            "prompt_input": (
                "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。"
                "要求を適切に満たす応答を書きなさい。\n\n"
                "### 指示:\n{instruction}\n\n### 入力:{input}\n\n### 応答:"
            ),
            "prompt_no_input": (
                "以下は、タスクを説明する指示です。" "要求を適切に満たす応答を書きなさい。\n\n" "### 指示:\n{instruction}\n\n### 応答:\n{output}"
            ),
        }
        self.data_max_length = data_max_length  # 最大2048 VRAMのサイズに合わせて変更
        self.dataset_name = "kunishou/databricks-dolly-15k-ja"
        self.dataset = datasets.load_dataset(self.dataset_name)
        self.tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1")
        self.dataset = self.dataset.map(lambda samples: self.tokenize(samples), batched=True)
        logger.info("データセット読み込み完了")

    def tokenize(self, samples):
        prompts = []

        # データセットの instruction 列と input 列と output 列を組み合わせてプロンプトを組み立てます。
        for instruction, input, output in zip(samples["instruction"], samples["input"], samples["output"]):
            # QAにinput(=LLMに与えるヒントや文脈)があるとき
            if input:
                prompt = self.PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input, output=output)
            # QAにinput(=LLMに与えるヒントや文脈)がないとき
            else:
                prompt = self.PROMPT_DICT["prompt_no_input"].format(instruction=instruction, output=output)
            prompts.append(prompt + self.tokenizer.eos_token)

        res = self.tokenizer(prompts, padding=False, truncation=True, max_length=self.data_max_length)
        return res


class my_data01:
    def __init__(
        self,
        ps: PathSetting,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIER_NAME, use_fast=False, model_max_length=512)

        data = read_json(ps.get_dataset_filename())
        train_dataset, val_dataset = self.split_dataset(data)

    def tokenize(self, prompt):
        CUTOFF_LEN = 256
        result = self.tokenizer(prompt, truncation=True, max_length=CUTOFF_LEN, padding=False, add_special_tokens=True)
        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
        }

    def generate_prompt(self, data_point):
        result = f"""### 指示:
    {data_point["input"]}

    ### 回答:
    {data_point["completion"]}
    """
        # 改行→<NL>
        result = result.replace("\n", "<NL>")
        return result

    def split_dataset(self, data):
        train_dataset = []
        val_dataset = []

        for i in range(len(data)):
            if i % 5 == 0:
                x = self.tokenize(self.generate_prompt(data[i]))
                val_dataset.append(x)
            else:
                x = self.tokenize(self.generate_prompt(data[i]))
                train_dataset.append(x)
        return train_dataset, val_dataset


if __name__ == "__main__":
    dataset = Dolly15KJa()
    print(dataset.dataset)
    print(dataset.dataset["train"])
