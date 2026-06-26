# sqm-ground-state

確率過程量子化（Parisi-Wu法）で量子力学系の基底状態を数値的に求める。

## 概要

1次元量子力学系に対して Parisi-Wu 確率過程量子化（ランジュバン方程式によるシミュレーション）を実装し、基底状態のエネルギーと波動関数を数値的に求めるプロジェクト。

詳細な要件は [REQUIREMENTS.md](REQUIREMENTS.md) を参照。

## セットアップ

このプロジェクトは Python 3.14 と uv を前提にする。

```bash
uv sync
```

テストは次で実行する。

```bash
uv run pytest -q
```

coverage 付きで確認する場合は次を使う。

```bash
uv run pytest --cov=src --cov-report=term-missing
```

HTML レポートも出す場合は次を使う。

```bash
uv run pytest --cov=src --cov-report=term-missing --cov-report=html
```

coverage は `pyproject.toml` で branch coverage 有効、最低ライン 80% に設定している。

実行例も uv 経由で起動できる。

```bash
uv run python examples/harmonic.py
uv run python examples/anharmonic.py
uv run python examples/double_well.py
```

## ドキュメント

- [物理的に何を計算しているか](PHYSICS.md)
  - ユークリッド時間の経路積分
  - 確率過程量子化によるサンプリング
  - 相関関数、有効質量、波動関数ヒストグラムの意味
- [要件定義](REQUIREMENTS.md)

## 構成

コードは役割ごとに次の層に分かれている。

- ドメイン: `src/potentials.py`（ポテンシャル V・V'・解析的に分かる量の単一の源）
- 数値カーネル: `src/lattice.py`、`src/langevin.py`、`src/exact.py`、`src/observables.py`、`src/analysis.py`
- オーケストレーション: `src/experiment.py`（解析解・有限差分・Parisi-Wu の 3 者比較を実行）と `src/runner.py`（出力保存）
- 実行例: `examples/` 以下の各スクリプトは設定を渡して `run_experiment` を呼ぶだけ

`src/action.py` と `src/analytic.py` は従来の関数 API を保つ後方互換レイヤーで、
内部的には `src/potentials.py` と `src/lattice.py` に委譲する。

## プロットの見た目

プロット（`effective_mass.png` / `wavefunction.png`）は `seaborn` の `whitegrid` を優先して使い、白基調のグリッド付きスタイルで出力する。`seaborn` が使えない場合は matplotlib 側の白基調スタイルへフォールバックする。

## 日本語フォント

プロットのタイトル・凡例は**日本語**で出力する。
そのため **`Noto Sans CJK JP` が OS にインストール済みであること**を前提とする。
Python パッケージ（`pyproject.toml`）には含まれない。

### インストール例

Debian / Ubuntu / WSL:

```bash
sudo apt install fonts-noto-cjk
fc-cache -fv
```

macOS では Noto Sans CJK がシステムに同梱されていることが多い。Font Book で
「Noto Sans CJK JP」の有無を確認する。

### フォントがない場合

`examples/*.py` 実行時、数値計算と `summary.json` の保存は行われるが、
プロット生成はスキップされ、README への案内メッセージが表示される。

使用フォント名は `src/experiment.py` の `PLOT_FONT_FAMILY` で定義している。
matplotlib の font cache に未登録でも、OS に fonts-noto-cjk があれば
実行時に自動登録を試みる。

## 実行例で得られる出力

`examples/harmonic.py` は調和振動子を、`examples/anharmonic.py` は非調和振動子を、`examples/double_well.py` は二重井戸ポテンシャルを計算し、結果を `outputs/` 以下に保存する。
出力先はパラメータごとに分かれるため、複数の設定を試しても結果が混ざらない。

各 example では、可能な範囲で次の 3 種類を比較する。

1. 解析解または解析的に分かる量
2. 非 Parisi-Wu の数値計算（有限差分対角化）
3. Parisi-Wu 確率過程量子化による数値計算

代表的な出力は次の通り。

- `effective_mass.png`: ユークリッド相関関数から求めた有効質量
- `wavefunction.png`: 基底状態波動関数の確率密度の近似
- `summary.json`: 実行パラメータと主要な数値結果

実行例:

```bash
uv run python examples/harmonic.py
uv run python examples/anharmonic.py
uv run python examples/double_well.py
```
