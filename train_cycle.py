# ====================
# 学習サイクルの実行
# ====================

# パッケージのインポート
from dual_network import dual_network
from self_play import *
from train_network import train_network
from evaluate_network import evaluate_network
import multiprocessing as mp

if __name__ == '__main__':

    mp.set_start_method('spawn')

    # デュアルネットワークの作成
    dual_network()

    for i in range(25):
        print('Train',i,'====================')
        # セルフプレイ部
        self_play()

        # パラメータ更新部
        train_network()

        # 新パラメータ評価部
        evaluate_network()
