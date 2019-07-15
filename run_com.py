from torch import nn, optim
import torch
import torch.utils.data as utils
from tqdm import tqdm
from model.blocks_compact import gcn
import numpy as np
import pandas as pd


def loadInputs(idx,unitLen,device, dir='/work/qyuan/molecular_gcn/'):
    adj = None
    features = None
    adj = np.load(dir +'database/CEP/adj/'+str(idx)+'.npy')
    features = np.load(dir +'database/CEP/features/'+str(idx)+'.npy')
    adj = torch.stack([torch.from_numpy(i) for i in adj])
    adj.requires_grad_(True)
    adj = adj.to(device)
    features = torch.stack([torch.from_numpy(i) for i in features])
    features.requires_grad_(True)
    features = features.to(device)
    retOutput =(np.load(dir + 'database/CEP/pve.npy')[idx*unitLen:(idx+1)*unitLen]).astype('float')
    retOutput = torch.from_numpy(np.array(retOutput)).view(-1,1)
    #print(retOutput.size())
    my_dataset = utils.TensorDataset(adj, features, retOutput)
    return my_dataset


def init_weights(m):
    print(m)
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def train(device, dir='/work/qyuan/molecular_gcn/'):
    graph_gcn = gcn(58,32,512)
    graph_gcn = graph_gcn.to(device)
    graph_gcn.apply(init_weights)
    optimizer = optim.Adam(graph_gcn.parameters(), lr = 1e-3)
    criterion = nn.MSELoss()
    for epoch in range(200):
        running_loss = 0
        for idx in range(1, 26):
            dataset = loadInputs(idx, 1024,device, dir)
            data = utils.DataLoader(dataset, batch_size=128, shuffle=True, drop_last=False)
            for step, batch in enumerate(data):
                #print(step)

                A_inpt, X_inpt, y_true = batch[0], batch[1], batch[2]
                A_inpt = A_inpt.float()
                X_inpt = X_inpt.float()
                #print(A_inpt.size())
                y_true = y_true.float()
                y_true = y_true.to(device)
                optimizer.zero_grad()
                res = graph_gcn(X_inpt, A_inpt)
                #print(res.size())
                loss = criterion(res, y_true)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                if step == 7:
                    tqdm.write('Epoch {}: loss in batch {} step {} is {}'.format(epoch, idx, step, loss.item()))
        torch.save(graph_gcn.state_dict(), dir+'test_gcn_gate.ckpt')

def test(device, dir='/work/qyuan/molecular_gcn/'):
    result_df = pd.DataFrame(columns=['pve_true','pve_pred'])
    graph_gcn = gcn(58,32,512)
    graph_gcn = graph_gcn.to(device)
    if device == 'cpu':
        graph_gcn.load_state_dict(torch.load(dir+'test_gcn_gate.ckpt', map_location=lambda storage, loc:storage))
    else:
        graph_gcn.load_state_dict(torch.load(dir+'test_gcn_gate.ckpt'))
    for param in graph_gcn.parameters():
        param.requires_grad=False
    y_trues =[]; y_preds = []
    for idx in range(26,30):
        dataset = loadInputs(idx,1024,device,dir)
        data = utils.DataLoader(dataset, batch_size=128,shuffle=True, drop_last=False)
        for step, batch in enumerate(data):

            A_inpt, X_inpt, y_true = batch[0], batch[1], batch[2]
            A_inpt = A_inpt.float()
            X_inpt = X_inpt.float()
        #print(A_inpt.size())
            y_true = y_true.float()
            #if idx == 26 and step==1:
            #    tqdm.write(y_true)
            res = graph_gcn(X_inpt,A_inpt )
            y_trues.extend(y_true.flatten().data.numpy())
            res = res.cpu()
            y_preds.extend(res.flatten().data.numpy())
    result_df['pve_true'] = y_trues
    result_df['pve_pred'] = y_preds
    assert len(y_trues) == len(y_preds)
    #mae = abs(y_trues-y_preds)/len(y_preds)
    #tqdm.write('Mean absolute error for test set is {}'.format(mae))
    return result_df


if __name__ == '__main__':
    if torch.cuda.is_available():
        tqdm.write('cuda is available')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tqdm.write(str(torch.cuda.current_device()))
    #train(device=device)
    res_df = test(device)
    res_df.to_csv('/work/qyuan/molecular_gcn/test_pred_pve_gate.csv')
