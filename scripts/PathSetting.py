from pathlib import Path
from typing import List


class PathSetting:
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug  # 最小のデータ数で実行確認モード
        self.dir_home = Path.cwd().parent  # scriptsの親ディレクトリ
        self.dir_raw_data = self.dir_home / "data"
        self.dir_dataset = self.dir_home / "dataset"
        self.dir_ft_results = self.dir_home / "ft_results"

    def get_data_filenames(self) -> List[Path]:
        if self.debug:
            return sorted(self.dir_raw_data.glob("*.txt"))[:1]
        else:
            return sorted(self.dir_raw_data.glob("*.txt"))

    def get_dataset_filename(self, dataset_name: str = "") -> Path:
        if dataset_name:
            return self.dir_dataset / f"{dataset_name}_dataset.json"
        else:
            return self.dir_dataset / "dataset.json"
