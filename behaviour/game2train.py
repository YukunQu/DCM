import os
import pandas as pd


def game2trian_acc(subj):
    train_data_blocked = []
    data_dir = r'/mnt/data/Project/DCM/BIDS/sourcedata/sub_{}/Behaviour/fmri_task-game2-train'.format(subj)
    game2train_tmp = os.listdir(data_dir)
    game2train_list = []
    for g2t in game2train_tmp:
        if '.csv' in g2t:
            game2train_list.append(g2t)
    
    for g2t in game2train_list:
        data_path = os.path.join(data_dir, g2t)
        print(subj,g2t)
        if os.path.getsize(data_path) > 9680:
            game2data = pd.read_csv(data_path)
            train_data = game2data.dropna(subset=['train_accuracy'])
            data_len = len(train_data)
            block_num = int(data_len/32)
            for i in range(block_num):
                print('block',i)
                if i < (block_num-1):
                    block_data = train_data[i*32:(i+1)*32].copy()    
                else:
                    block_data = train_data[i*32:].copy()
                
                # 假设这个block是攻击力
                block_ap_diff = block_data['ap_diff']
                block_dp_diff = block_data['dp_diff']
                
                corrAns_ap = [1 if a < 0 else 2 for a in block_ap_diff]
                corrAns_dp = [1 if a < 0 else 2 for a in block_dp_diff]
                corrAns = block_data['correctAns'].to_list()
                
                if corrAns == corrAns_ap:
                    block_data.loc[:,'dim'] = 'ap'
                elif corrAns == corrAns_dp:
                    block_data.loc[:,'dim'] = 'dp'
                else:
                    print("The code is wrong! ")
                
                train_data_blocked.append(block_data.iloc[-1,:][['姓名','participant','date','dim','train_accuracy']])
    return train_data_blocked
    

game2train_result = [] 
subjests = [str(i).zfill(3) for i in range(27,50)]
for subj in subjests:
    sub_result = game2trian_acc(subj)
    if isinstance(sub_result, list):
        game2train_result.extend(sub_result)
        


i = 1
save_path = r'/mnt/data/Project/DCM/BIDS/derivatives/behaviour/result/training/{}_game2train_result.csv'.format(i)
while os.path.exists(save_path):
    i += 1
    save_path = r'/mnt/data/Project/DCM/BIDS/derivatives/behaviour/result/training/{}_game2train_result.csv'.format(i)

game2train_result = pd.DataFrame(game2train_result)  
game2train_result.to_csv(save_path)
