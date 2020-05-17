# ====================
# パラメータ更新部
# ====================

# パッケージのインポート
from dual_network import *
from pathlib import Path
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import random

# パラメータの準備
RN_EPOCHS = 30 # 学習回数
#RN_BATCH_SIZE = 16 # バッチサイズ
RN_BATCH_SIZE = 128 # バッチサイズ

# 学習データの読み込み
def load_data():
    history_path = sorted(Path('./data').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)

# デュアルネットワークの学習
def train_network():
    # 学習データの読み込み
    history = load_data()
    xs, y_policies, y_values, y_deep_values, actions, y_rq = zip(*history)

    # 学習のための入力データのシェイプの変換
    file, rank, channel = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), channel, file, rank)
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)
    y_deep_values = np.array(y_deep_values)
    y_rq = np.array(y_rq)
    
    # ベストプレイヤーのモデルの読み込み
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    model0 = InitialNet()
    model0.load_state_dict(torch.load('./model/best_i.h5'))
    model0 = model0.double()
    model0 = model0.to(device)
    model0.train()

    model1 = RecurrentNet()
    model1.load_state_dict(torch.load('./model/best_r.h5'))
    model1 = model1.double()
    model1 = model1.to(device)
    model1.train()
    
    optimizer = optim.SGD(model0.parameters(),lr=0.01)
    
    print("len:" + str(len(xs)))
    
    indexs = [i for i in range(0,len(xs))]
    
    criterion_policies = nn.CrossEntropyLoss()
    criterion_values = nn.MSELoss()

    
    for i in range(0,RN_EPOCHS):
        print("epoch:" + str(i),end="")
        random.shuffle(indexs)
        sum_loss = 0.0
        sum_num = 0
        minbatch_loss = torch.tensor([0],dtype=torch.double)
        minbatch_loss = minbatch_loss.to(device)
        minbatch_num = 0
        for j in indexs:
            for k in range(3):
                if k == 0:
                    x = torch.tensor([xs[j]],dtype=torch.double)
                    yp = np.array([y_policies[j]])
                    yp = yp.argmax(axis = 1)
                    yp = torch.tensor(yp,dtype=torch.long)
                    yv = torch.tensor(y_values[j],dtype=torch.double)
                    
                    x = x.to(device)
                    outputs = model0(x)
                else:
                    
                    yp = np.array([y_rq[j][k][1]])
                    yp = yp.argmax(axis = 1)
                    yp = torch.tensor(yp,dtype=torch.long)
                    yv = torch.tensor(y_rq[j][k][2],dtype=torch.double)
                    action = y_rq[j][k-1][4]
                    
                    yp = yp.to(device)
                    yv = yv.to(device)
                    action_tensor = action_to_tensor(np.array([action]))
                    action_tensor = action_tensor.to(device)
                    outputs = model1(rq,action_tensor)
                    

                output_policy = outputs[0]
                output_value = torch.squeeze(outputs[1])
                rq = outputs[2]

                yp = yp.to(device)
                yv = yv.to(device)

                loss_policies = criterion_policies(output_policy,yp)
                loss_values = criterion_values(output_value,yv)
                loss = loss_policies + loss_values
                minbatch_loss += loss

            minbatch_num += 1 

            if minbatch_num == RN_BATCH_SIZE:
                
                optimizer.zero_grad()
                minbatch_loss.backward()
                optimizer.step()
                sum_loss += minbatch_loss.item()
                sum_num += 1
                minbatch_loss = 0
                minbatch_num = 0

        print(" avg loss " + str(sum_loss / sum_num))

    # 最新プレイヤーのモデルの保存
    torch.save(model0.state_dict(), './model/latest_i.h5')
    torch.save(model1.state_dict(), './model/latest_r.h5')

# 動作確認
if __name__ == '__main__':
    train_network()
