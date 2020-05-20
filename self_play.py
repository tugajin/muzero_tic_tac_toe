# ====================
# セルフプレイ部
# ====================

# パッケージのインポート
from game import State
from pv_mcts import pv_mcts_scores
from dual_network import DN_OUTPUT_SIZE
from datetime import datetime
from pathlib import Path
import numpy as np
import pickle
import os
import torch
from dual_network import *
from numpy.random import *

# パラメータの準備
SP_GAME_COUNT = 500 # セルフプレイを行うゲーム数（本家は25000）
SP_TEMPERATURE = 1.0 # ボルツマン分布の温度パラメータ

# 先手プレイヤーの価値
def first_player_value(ended_state):
    # 1:先手勝利, -1:先手敗北, 0:引き分け
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0

# 学習データの保存
def write_data(history):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True) # フォルダがない時は生成
    path = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
        pickle.dump(history, f)

# 1ゲームの実行
def play(model):
    # 学習データ
    history = []

    # 状態の生成
    state = State()

    while True:
        # ゲーム終了時
        if state.is_done():
            break

        # 合法手の確率分布の取得

        scores, values = pv_mcts_scores(model, state, SP_TEMPERATURE)

        # 学習データに状態と方策を追加
        policies = [0] * DN_OUTPUT_SIZE
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy

        # 行動の取得
        action = np.random.choice(state.legal_actions(), p=scores)

        # state, policy, value, 探索結果, 選ばれた手、それから先の局面
        history.append([[state.pieces, state.enemy_pieces], policies, None, values, action, None])

        # 次の状態の取得
        state = state.next(action)

    # 学習データに価値を追加
    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value

    # 最後の局面情報を取っておく
    last_state = history[-1][0]
    last_policy = [0] * DN_OUTPUT_SIZE 
    v0 = history[0][2]
    v1 = history[1][2]

    for i in range(len(history)):
        rp = []
        for inc in range(3):
            index = i+inc
            if index < len(history):
                rp.append(history[i+inc])
            else:
                v = v0 if ((i+inc) % 2) == 0 else v1 
                a = randint(9)
                rp.append([last_state,last_policy,v,v,a,None])
        history[i][5] = rp

    return history

# セルフプレイ
def self_play():
    # 学習データ
    history = []

    # ベストプレイヤーのモデルの読み込み
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    model0 = RepNet()
    model0.load_state_dict(torch.load("./model/best_r.h5"))
    model0 = model0.double()
    model0 = model0.to(device)
    model0.eval()

    model1 = DynamicsNet()
    model1.load_state_dict(torch.load("./model/best_d.h5"))
    model1 = model1.double()
    model1 = model1.to(device)
    model1.eval()

    model2 = PredictNet()
    model2.load_state_dict(torch.load("./model/best_p.h5"))
    model2 = model2.double()
    model2 = model2.to(device)
    model2.eval()
    model = (model0,model1,model2)

    # 複数回のゲームの実行
    for i in range(SP_GAME_COUNT):
        # 1ゲームの実行
        h = play(model)
        history.extend(h)

        # 出力
        print('\rSelfPlay {}/{}'.format(i+1, SP_GAME_COUNT), end='')
    print('')

    # 学習データの保存
    write_data(history)

 

# 動作確認
if __name__ == '__main__':
    self_play()
