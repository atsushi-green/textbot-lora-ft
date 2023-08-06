import datetime
import json
import sys
from pathlib import Path

sys.path.append("../../")
from util.ioutil import make_directory

with open("logconfig/logconfig.json", "r") as f:
    log_conf = json.load(f)

# ファイル名をタイムスタンプで作成（これだけはPathSettingsにかかれない）
logfilepath = f'../logs/{datetime.datetime.now().strftime("%Y-%m-%d")}.log'
make_directory(Path("../logs"))
log_conf["handlers"]["fileHandler"]["filename"] = logfilepath
