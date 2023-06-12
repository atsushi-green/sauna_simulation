# 仮定
# サウナーは、そのサウナ施設が持つパラメータλのポアソン分布に従って入室する
# サウナーは、それぞれが持つパラメータ1/μの指数分布に従ってサウナを退室する

import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma, poisson
from tqdm import tqdm

# ======環境設定======
FONTNAME = "IPAexGothic"
plt.rcParams["font.family"] = FONTNAME
SYUIRO = "#F26649"
seed = 314  # 乱数シードを314に固定
np.random.seed(seed=seed)
random.seed(seed)

# ======定数======
MAX_SIMULATION_MINUTES = 10000  # シミュレーションを終了する時間(分)
MAX_SIMULATION_SECONDS = MAX_SIMULATION_MINUTES * 60
LAMBDA = 2  # 1分当たりの平均入室者数(実際は観測不可能だが、このシミュレーションでは既知とする)
ASSUMED_MU = 8  # 他のサウナー達が滞在する時間の平均値[分]
N = LAMBDA * ASSUMED_MU  # 定常状態におけるサウナ室内人数の期待値(自分が入室した時の滞在人数はNになると仮定)

# 初心者:5分で退室する
# 中級者8分で退室する
# 上級者10、12分で退室する
# 初心者20%、中級者60%、上級者20%とする。
MU_DISTRIBUTION = [5, 5, 8, 8, 8, 8, 8, 8, 10, 12]
assert np.average(MU_DISTRIBUTION) == ASSUMED_MU  # μの平均が合っていることを確認

# 自分がサウナ室内で滞在したい時間を5,8,12分のそれぞれの場合でシミュレーションする
TAU_LSIT = [5, 8, 12]

# False: 退室人数のみを数える通常バージョン
# True: 入退室人数を数えて精度を上げられるか検証するバージョン
COUNT_ENTER_EXIT_FLG = False


def main():
    # enter_count_list[t] はt秒後にサウナに入室する人数(t=0,1,..,)
    enter_count_list = np.zeros(MAX_SIMULATION_SECONDS, dtype=np.int32)
    # exit_count_list[t] はt秒後にサウナを退室する人数(t=0,1,..,)
    exit_count_list = np.zeros(MAX_SIMULATION_SECONDS, dtype=np.int32)
    # num_in_sauna_list[t] はt秒後にサウナに入っている人数(t=0,1,..,)
    num_in_sauna_list = np.zeros(MAX_SIMULATION_SECONDS, dtype=np.int32)

    # ■ step1-1. 自分以外の人の動きを(入退室)をシミュレーション
    for el_sec in range(MAX_SIMULATION_SECONDS):
        # 1週1秒のforループシミュレーション
        if judge_enter_sauna_with_poisson(LAMBDA):  # ポアソン分布でこの1秒間に入室があったかどうか判定
            enter_count_list[el_sec] += 1
            # 今入った人の滞在時間は、パラメータmuの指数分布に従う
            mu = random.choice(MU_DISTRIBUTION)
            # そして今入った人は、u秒後にサウナを退室する(uを指数分布で決める)
            u = calc_exit_time_with_exp(mu)
            # u = calc_exit_time_with_exp(ASSUMED_MU)  # こっちはより仮定が強い
            # シミュレーション開始から(el_sec + u秒後に退室したことになる)
            if el_sec + u < MAX_SIMULATION_SECONDS:
                exit_count_list[el_sec + u] += 1
                num_in_sauna_list[el_sec : el_sec + u] += 1
            else:
                num_in_sauna_list[el_sec:] += 1

    # ■ step1-2. 自分以外の人の動きを確認
    print(f"サウナ内滞在人数の平均値={np.average(num_in_sauna_list):.2f}(理論値は{N:.2f})")
    # 入室人数の分布と退室人数の分布がポアソン分布になっていることを確認
    enter_count_list_minute = round_sec2min(enter_count_list)
    exit_count_list_minute = round_sec2min(exit_count_list)

    x_list = range(np.max(enter_count_list_minute))
    theory_dist = poisson.pmf(k=x_list, mu=LAMBDA)
    draw_histgram(
        enter_count_list_minute, theory_dist, theory_ave=LAMBDA, theory_var=LAMBDA, savename="01.1分あたりの入室人数の分布.png"
    )

    x_list = range(np.max(exit_count_list_minute))
    theory_dist = poisson.pmf(k=x_list, mu=LAMBDA)
    draw_histgram(
        exit_count_list_minute, theory_dist, theory_ave=LAMBDA, theory_var=LAMBDA, savename="02.1分あたりの退室人数の分布.png"
    )
    # サウナ室内の定常状態を確認(サウナ開店後から何分程度経てば定常上たになるかも確認できる)
    # [0:h*3600:60]: 初めからh時間後までを1分間隔で
    draw_in_sauna_people(num_in_sauna_list[0 : 4 * 3600 : 60], N, savename="03.サウナ室内滞在人数の変遷_4時間.png")
    draw_in_sauna_people(num_in_sauna_list[0 : 24 * 3600 : 60], N, savename="03.サウナ室内滞在人数の変遷_24時間.png")

    # ■　step2-1. 自分が入室してから、中の人を数えて、退室するまでの時間をシミュレーション
    # サウナに行く前に、自分が滞在したい時間τを決める
    # 決めた時間τごとに、自分が入室してから退室するまでの時間を記録する辞書を作成
    result_dict: Dict[int, List[int]] = {tau: [] for tau in TAU_LSIT}

    for tau in TAU_LSIT:
        for el_sec in tqdm(range(MAX_SIMULATION_SECONDS)):
            # el_sec: 自分がサウナに入った時刻(秒)
            n = num_in_sauna_list[el_sec]  # 自分が入室した時、まずはサウナ室内の人数を数えてnを得る
            # n = N  # こっちは自分が入室する時は必ず定常状態の期待値の人数がサウナに滞在していると仮定

            # ガンマ分布のパラメータαを計算し、α人退室するまでに経過する時間(my_stay_sec)を計算する
            if COUNT_ENTER_EXIT_FLG:
                alpha = 2 * tau * n / ASSUMED_MU
                my_stay_sec = calc_my_stay_sec(el_sec, alpha, exit_count_list, enter_count_list)
            else:
                alpha = tau * n / ASSUMED_MU
                my_stay_sec = calc_my_stay_sec(el_sec, alpha, exit_count_list)

            # もし自分がサウナに入ってから退室するまでにα人数え終わらなかったら、実験結果から除く
            if my_stay_sec >= 0:
                result_dict[tau].append(my_stay_sec // 60)

    # ■ step2-2. シミュレーション結果の確認
    for tau in TAU_LSIT:
        # αとβの理論値を計算
        if COUNT_ENTER_EXIT_FLG:
            # 入室人数も数える場合2倍にする
            beta_theory = 1 / (2 * LAMBDA)
        else:
            beta_theory = 1 / LAMBDA
        alpha_theory = tau / beta_theory

        # 自分が待ちたい時間がtauの時のシミュレーション結果を取り出す
        res = result_dict[tau]
        # res = res[len(res) // 10 :]  # 前半10%は定常状態ではない可能性があるので除く
        theory_ave = alpha_theory * beta_theory
        theory_var = alpha_theory * beta_theory**2

        print(f"=====自分がtau={tau}分滞在したいときの実際の滞在時間=====")
        print(f"ガンマ分布理論値: 平均={theory_ave:.3f}, 分散={theory_var:.3f}")
        print(f"シミュレーション: 平均={np.average(res):.3f}, 分散={np.var(res):.3f}")
        print()
        # ガンマ分布の理論値を作成
        x_list = range(np.max(res))
        theory_dist = gamma.pdf(x=x_list, a=alpha_theory, scale=beta_theory)
        draw_histgram(
            res,
            theory_dist,
            theory_ave=theory_ave,
            theory_var=theory_var,
            savename=f"04.自分が入室してから退室するまでの時間の分布(分)(τ={tau}).png",
        )


def judge_enter_sauna_with_poisson(lambda_: float) -> bool:
    """ポアソン分布に従って、ある微小時間(1秒)にサウナへの入室者がいるかどうかを判定
    ポアソン分布では、微小時間に2回以上事象が発生することはないことを仮定している。

    Args:
        lambda_ (float): ポアソン分布のパラメータλ(人/分)

    Returns:
        bool: この微小時間に人が入ってくるならTrue, そうでないならFalse
    """
    return np.random.rand() < (lambda_ / 60)


def calc_exit_time_with_exp(mu: float) -> int:
    """指数分布に従い、サウナを退室するまでの時間を決める

    Args:
        mu (float): 指数分布のパラメータμ(分/回)

    Returns:
        int: 何秒後にサウナを退室するか
    """
    # 簡単のために、round(偶数丸め)で整数に丸める
    return round(np.random.exponential(scale=mu * 60, size=None))


def round_sec2min(lis_sec: np.ndarray) -> np.ndarray:
    """1要素が1秒単位のものを、60個ずつで足し合わせることで1要素を1分単位にする

    Args:
        lis_sec (np.ndarray): 1秒単位の結果

    Returns:
        np.ndarray: 1分単位の結果
    """
    lis_min = [sum(lis_sec[s * 60 : (s + 1) * 60]) for s in range(len(lis_sec) // 60 - 1)]
    return lis_min


def calc_my_stay_sec(en_sec: int, alpha: float, exit_counts: np.ndarray, enter_counts: np.ndarray = None) -> int:
    """自分の滞在時間を計算する

    Args:
        en_sec (int): 自分が入室した時刻(秒)
        alpha (float): 退室するまでに数えるべき人数
        exit_counts (np.ndarray): 1秒ごとの退室人数
        enter_counts (np.ndarray, optional): 1秒ごとの入室人数. Defaults to None.

    Returns:
        int: alpha人数えるまで自分が滞在した時間(秒)
    """
    my_count = 0  # 自分がサウナに入ってから数えた(退室)人数
    my_stay_sec = 0  # 自分がサウナに入ってから経過した時間(秒)
    # HACK: 以下のwhileループは累積和を用いて高速化できるが、ソースコードの直感的理解を優先してこのままにしている
    while en_sec + my_stay_sec < MAX_SIMULATION_SECONDS:
        my_count += exit_counts[en_sec + my_stay_sec]  # 1秒間の退室人数を数える
        if enter_counts is not None:
            # 入室も数える場合
            my_count += enter_counts[en_sec + my_stay_sec]

        if my_count >= alpha:
            # alpha人数え終わったら自分が退室する
            return my_stay_sec
        my_stay_sec += 1
    else:
        # シミュレーションの最後まで退室人数がαに達しなかったので、実験結果から除く
        return -1


def draw_histgram(x: np.ndarray, theory: np.ndarray, theory_ave: float, theory_var: float, savename: str) -> None:
    """xのヒストグラムを描画し、右上の方に位置に平均と分散を表示する。

    Args:
        x (np.ndarray): ヒストグラムの元データ
        theory (np.ndarray): 理論的な分布
        theory_ave (float): 理論的な平均値
        theory_var (float): 理論的な分散
        savename (str): 保存ファイル名
    """

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x, bins=max(x) + 1, density=True, color=SYUIRO, label="シミュレーション", range=(min(x) - 0.5, max(x) + 0.5))
    ax.plot(theory, color="black", linestyle="--", label="理論値")
    ax.set_title(savename[:-4])
    ax.set_xlabel("時間[分]")
    # ax.set_xlabel("人数")
    ax.set_ylabel("確率密度")
    # テキスト表示位置の調整
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    text_pos_x = (x_min * 3 + x_max * 7) / 10
    text_pos_y = y_max - (y_max - y_min) / 2
    ax.text(
        text_pos_x,
        text_pos_y,
        f"シミュレーション\n平均: {np.average(x):.2f}\n分散: {np.var(x):.2f}\n\n理論値\n平均: {theory_ave:.2f}\n分散: {theory_var:.2f}",
    )
    ax.legend(loc="upper right")
    plt.savefig(savename)


def draw_in_sauna_people(x: np.ndarray, ave: float, savename: str) -> None:
    """サウナ室内の滞在人数の変遷を描画する。

    Args:
        x (np.ndarray): 1分ごとのサウナ室内の滞在人数
        ave (float): 定常状態におけるサウナ室内の期待値(理論値)
        savename (str): 保存ファイル名
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, color=SYUIRO)
    x_min, x_max = ax.get_xlim()
    ax.axhline(y=ave, color="black", linestyle="--")
    # (x_min + x_max) / 6, ave + 0.5は単なる表示位置の調整
    ax.text((x_min + x_max) / 6, ave + 0.5, f"定常状態の期待値: {ave:.2f}")
    ax.set_xlabel("時間[分]")
    ax.set_ylabel("サウナ内滞在人数")
    ax.grid(True)
    plt.savefig(savename)


if __name__ == "__main__":
    main()
