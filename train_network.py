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
#RN_BATCH_SIZE = 2 # バッチサイズ
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
    xs, y_policies, y_values, y_deep_values, actions, y_rp = zip(*history)

    # 学習のための入力データのシェイプの変換
    file, rank, channel = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), channel, file, rank)
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)
    y_deep_values = np.array(y_deep_values)
    y_rp = np.array(y_rp)
    
    # ベストプレイヤーのモデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    model0 = RepNet()
    model0.load_state_dict(torch.load("./model/best_r.h5"))
    model0 = model0.double()
    model0 = model0.to(device)
    model0.train()

    model1 = DynamicsNet()
    model1.load_state_dict(torch.load("./model/best_d.h5"))
    model1 = model1.double()
    model1 = model1.to(device)
    model1.train()

    model2 = PredictNet()
    model2.load_state_dict(torch.load("./model/best_p.h5"))
    model2 = model2.double()
    model2 = model2.to(device)
    model2.train()
    
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

        x_list = []
        yp_list =[]
        yv_list = []
        yrp_list = []

        for j in indexs:
            x_list.append(xs[j])
            yp_list.append(y_policies[j])
            yv_list.append(y_values[j])
            yrp_list.append(y_rp[j])

            if len(x_list) == RN_BATCH_SIZE:

                x_list = np.array(x_list)
                yp_list = np.array(yp_list)
                yv_list = np.array(yv_list)
                yrp_list = np.array(yrp_list)

                minbatch_loss = 0

                for k in range(3):
                    if k == 0:
                        x = torch.tensor(x_list,dtype=torch.double)
                        yp = yp_list.argmax(axis = 1)
                        yp = torch.tensor(yp,dtype=torch.long)
                        yv = torch.tensor(yv_list,dtype=torch.double)
                        
                        x = x.to(device)
                        rp = model0(x)
                    else:
                        yp = np.array(yrp_list[:,k,1])
                        yp = np.array([np.array(yp[i]) for i in range(len(yp))])
                        yp = yp.argmax(axis = 1)
                        yp = torch.tensor(yp,dtype=torch.long)
                        yv = torch.tensor(yrp_list[:,k,2].tolist(),dtype=torch.double)
                        action = yrp_list[:,k-1,4]
                    
                        yp = yp.to(device)
                        yv = yv.to(device)
                        action_tensor = action_to_tensor(np.array(action))
                        action_tensor = action_tensor.to(device)
                        rp = model1(rp,action_tensor)
                    
                    outputs = model2(rp)
                    output_policy = outputs[0]
                    output_value = torch.squeeze(outputs[1])

                    yp = yp.to(device)
                    yv = yv.to(device)

                    loss_policies = criterion_policies(output_policy,yp)
                    loss_values = criterion_values(output_value,yv)
                    loss = loss_policies + loss_values
                    minbatch_loss += loss

                optimizer.zero_grad()
                minbatch_loss.backward()
                optimizer.step()
                sum_loss += minbatch_loss.item()
                sum_num += 1
                minbatch_loss = 0
                minbatch_num = 0

                x_list = []
                yp_list =[]
                yv_list = []
                yrp_list = []

        print(" avg loss " + str(sum_loss / sum_num))

    # 最新プレイヤーのモデルの保存
    torch.save(model0.state_dict(), './model/latest_i.h5')
    torch.save(model1.state_dict(), './model/latest_r.h5')

# 動作確認
if __name__ == '__main__':
    train_network()
