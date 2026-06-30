"""pytest 用の共通設定。

src layout のパッケージ `sqm_ground_state` を import できるよう、
リポジトリ直下の `src/` を import パスに追加する。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
