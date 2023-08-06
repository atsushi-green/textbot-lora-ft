import sys
from logging import DEBUG, basicConfig, config, getLogger
from typing import Dict, List

sys.path.append("../")

from logconfig.logconf import log_conf, logfilepath
from PathSetting import PathSetting
from util.ioutil import write_json

# ログ設定
basicConfig(filename=logfilepath, level=DEBUG)
config.dictConfig(log_conf)
logger = getLogger(__name__)


def main():
    ps = PathSetting()
    # 生テキストデータの読み込み
    text_lines = read_raw_data(ps)

    # データセット形式に合わせたjsonで出力するために、辞書形式に変換
    dataset = format_text(text_lines)

    # json形式でdatasetを保存
    write_json(dataset, ps.get_dataset_filename())


def read_raw_data(ps: PathSetting) -> List[str]:
    """生テキストデータを読み込む

    Args:
        ps (PathSetting): パス設定オブジェクト

    Returns:
        List[str]: _description_
    """
    text_lines = []
    # FIXME: ファイルのつなぎめがおかしいが、全体から見たら少ないので、とりあえず無視
    for filename in ps.get_data_filenames():
        with open(filename, encoding="utf8") as f:
            logger.info(f"{filename} を読み込み")
            raw_text_lines = f.readlines()
            # 空行を削除
            for line in raw_text_lines:
                if line != "\n":
                    text_lines.append(line)
    return text_lines


def format_text(text_lines: List[str]) -> List[Dict[str, str]]:
    """rinnaモデルのFTデータ仕様に合わせた
    入力と出力の文章対辞書のリストを作成する

    Args:
        text_lines (List[str]): 1文ずつ格納されたリストのデータ

    Returns:
        List[Dict[str, str]]: 1組の入出力文章が1要素のリスト
    """
    result_dict_list = []  # jsonとして出力する辞書のリスト
    input_text_list = []  # 入力になる文のリスト
    output_text_list = []  # 出力になる文章

    num_input_sentences = 5  # 入力する行数
    num_output_sentences = 5  # 出力する行数
    num_inoutsentences = num_input_sentences + num_output_sentences

    for i in range(1, len(text_lines)):  # i=0の時に空っぽになるので1から始める
        mod = i % num_inoutsentences
        # multiline_numで割り切れたら、1セット分のデータが揃ったとみなす
        if mod == 0:
            input_text = "".join(input_text_list)
            output_text = "".join(output_text_list)
            formatted = {"input": input_text, "completion": output_text}
            result_dict_list.append(formatted)
            # リセット
            input_text_list = []  # 入力になる文のリスト
            output_text_list = []  # 出力になる文章

        # 前半のnum_input_sentences個は入力文として扱う
        if mod < num_input_sentences:
            input_text_list.append(text_lines[i])
        # 後半は出力文として扱う
        else:
            output_text_list.append(text_lines[i])
    return result_dict_list


if __name__ == "__main__":
    main()
