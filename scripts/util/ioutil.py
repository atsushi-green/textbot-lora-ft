import csv
import json
import os
import sys
from logging import getLogger
from pathlib import Path
from typing import Any, List, Union

import yaml

sys.path.append("../")

logger = getLogger(__name__)


def make_directory(dirpath: Union[Path, str]) -> bool:
    """引数で受け取ったパスのディレクトリを作成し、
       作成した場合はTrueを返す。

    Args:
        dirpath (Path): 作りたいディレクトリパス

    Returns:
        bool: _新たにディレクトリを作ったらTrue
    """
    if os.path.exists(dirpath):
        return False
    else:
        # 存在しなければ作る。
        os.makedirs(dirpath, exist_ok=True)
        logger.info(f'ディレクトリ"{dirpath}"を作成')
        return True


def save_csv(list_of_list: list, savepath: Union[Path, str]) -> None:
    """二次元リストをcsv形式で保存する。

    Args:
        list_of_list (list): 保存したい二次元リスト
        savepath (Path): 保存付ファイルパス
    """
    with open(savepath, "w") as f:
        writer = csv.writer(f, lineterminator=os.linesep)
        writer.writerows(list_of_list)
    logger.info(f'ファイル"{savepath}"を保存')


def read_csv(readpath: Union[Path, str]) -> List[List[Any]]:
    with open(readpath) as f:
        reader = csv.reader(f)
        data = [e for e in reader]
    return data


def save_text(text: str, savepath: Path) -> None:
    with open(savepath, "w") as f:
        f.write(text)
    logger.info(f'ファイル"{savepath}"を保存')
    return


def read_yaml(yaml_file_path: Union[Path, str]) -> dict:
    logger.info(f'ファイル"{yaml_file_path}"を読み込み')
    with open(yaml_file_path, "r") as yml:
        return yaml.safe_load(yml)


def read_json(file_path: Union[Path, str]) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json(dict_data, file_path: Union[Path, str]):
    logger.info(f'"{file_path}"を書き込み')

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dict_data, f, indent=4)
