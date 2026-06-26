"""
1 次元量子力学系のポテンシャルを表すドメインオブジェクト。

このモジュールはプロジェクト全体の「ポテンシャルの単一の真実の源」である。
1 つのポテンシャルは概念的に次をまとめて持つ。

- V(x)             ポテンシャル
- V'(x)            その導関数（ランジュバンのドリフト力に使う）
- name             出力ディレクトリ名などに使う物理系名
- display_params   出力名・summary に埋め込むパラメータ（順序つき）
- analytic(mass)   解析的に分かる量（厳密解があればその値、なければ None）

従来は V と V' が action.py の別々の関数として、解析的な量が analytic.py の
別々の辞書関数として散らばっていた。それらをこのオブジェクトに集約し、
action.py と analytic.py はここへ委譲する薄い互換レイヤーにする。

数値計算カーネル（langevin, exact, observables, analysis）はポテンシャルに
依存しない汎用処理のままにし、Potential はカーネルへ渡す callable を提供する。
"""

import numpy as np


class Potential:
    """1 次元ポテンシャルの抽象基底クラス。

    サブクラスは次を実装する。

    - ``potential(x)``   : V(x)
    - ``force(x)``       : V'(x)
    - ``display_params`` : 出力名・summary 用の順序つきパラメータ辞書
    - ``analytic(mass)`` : 解析的に分かる量の辞書

    ``potential`` と ``force`` は自身のパラメータを閉じ込めているため、
    呼び出し側は x 以外の追加パラメータを渡す必要がない。
    """

    #: 物理系名（出力ディレクトリ名に使う）。
    name = "potential"

    def potential(self, x):
        raise NotImplementedError

    def force(self, x):
        raise NotImplementedError

    @property
    def display_params(self):
        """出力名・summary に埋め込む順序つきパラメータ辞書。"""
        return {}

    def analytic(self, mass=1.0):
        """解析的に分かる量を辞書で返す。

        ``available`` が True なら ``E0`` / ``gap`` / ``x2`` に厳密値が入る。
        False のときは ``reason`` に理由が入り、系によっては古典的に分かる
        量（極小点や障壁高さなど）が追加で入る。
        """
        raise NotImplementedError


class HarmonicPotential(Potential):
    """調和振動子 V(x) = 1/2 * omega^2 * x^2。"""

    name = "harmonic"

    def __init__(self, omega=1.0):
        self.omega = omega

    def potential(self, x):
        return 0.5 * self.omega**2 * x**2

    def force(self, x):
        return self.omega**2 * x

    @property
    def display_params(self):
        return {"omega": self.omega}

    def analytic(self, mass=1.0):
        # ばね定数は omega^2（質量に依らない）。物理的な角振動数は
        # Omega = omega / sqrt(mass) になる。mass=1 では Omega = omega。
        physical_omega = self.omega / np.sqrt(mass)
        alpha = mass * physical_omega
        return {
            "available": True,
            "description": "調和振動子の厳密解",
            "E0": 0.5 * physical_omega,
            "gap": physical_omega,
            "x2": 1.0 / (2.0 * alpha),
            "density": lambda x: np.sqrt(alpha / np.pi) * np.exp(-alpha * np.asarray(x) ** 2),
            "density_prefactor": np.sqrt(alpha / np.pi),
            "density_exponent": alpha,
        }


class AnharmonicPotential(Potential):
    """非調和振動子 V(x) = 1/2 * omega^2 * x^2 + lam * x^4。"""

    name = "anharmonic"

    def __init__(self, omega=1.0, lam=1.0):
        self.omega = omega
        self.lam = lam

    def potential(self, x):
        return 0.5 * self.omega**2 * x**2 + self.lam * x**4

    def force(self, x):
        return self.omega**2 * x + 4.0 * self.lam * x**3

    @property
    def display_params(self):
        return {"omega": self.omega, "lambda": self.lam}

    def analytic(self, mass=1.0):
        return {
            "available": False,
            "reason": "一般の lambda に対する閉じた形の厳密な解析スペクトルはない",
            "E0": None,
            "gap": None,
            "x2": None,
        }


class DoubleWellPotential(Potential):
    """二重井戸ポテンシャル V(x) = lam * (x^2 - v^2)^2。"""

    name = "double_well"

    def __init__(self, lam=1.0, v=1.0):
        self.lam = lam
        self.v = v

    def potential(self, x):
        return self.lam * (x**2 - self.v**2) ** 2

    def force(self, x):
        return 4.0 * self.lam * x * (x**2 - self.v**2)

    @property
    def display_params(self):
        return {"lambda": self.lam, "v": self.v}

    def analytic(self, mass=1.0):
        # 量子スペクトルの閉じた解析解は一般にないが、極小点・障壁高さ・
        # 井戸底での局所調和振動子近似の角振動数は解析的に分かる。
        local_curvature = 8.0 * self.lam * self.v**2
        local_omega = np.sqrt(local_curvature / mass)
        return {
            "available": False,
            "reason": "二重井戸の量子スペクトルは一般に閉じた解析解を持たない",
            "E0": None,
            "gap": None,
            "x2": None,
            "classical_minima": [-self.v, self.v],
            "classical_minimum_energy": 0.0,
            "barrier_position": 0.0,
            "barrier_height": self.lam * self.v**4,
            "local_harmonic_omega": local_omega,
            "local_harmonic_E0_approx": 0.5 * local_omega,
        }


# ---------------------------------------------------------------------------
# 便利コンストラクタ（example から使う想定の小文字名）
# ---------------------------------------------------------------------------

def harmonic(omega=1.0):
    """調和振動子ポテンシャルを作る。"""
    return HarmonicPotential(omega=omega)


def anharmonic(omega=1.0, lam=1.0):
    """非調和振動子ポテンシャルを作る。"""
    return AnharmonicPotential(omega=omega, lam=lam)


def double_well(lam=1.0, v=1.0):
    """二重井戸ポテンシャルを作る。"""
    return DoubleWellPotential(lam=lam, v=v)
