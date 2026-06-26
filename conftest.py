"""pytest 用の共通設定。

リポジトリルートを import パスに追加し、各テストファイルでの
sys.path 操作を不要にする。既存テストの sys.path 挿入が残っていても
害はない（同じパスを二重に追加するだけ）。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
