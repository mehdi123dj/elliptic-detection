from my_data_set import MyEllipticBitcoinDataset
from torch_geometric.loader.dataloader import DataLoader
from model import AML_model 
import torch 
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from torch.nn import Softmax
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import os 
import argparse


data_path = '../data'
model_path = "../models"
save_path = "../positionning"

def get_embedding(data_path,model_name,split,time_step):
    data = MyEllipticBitcoinDataset(data_path,split = split)
    model = torch.load(model_name)
    Data = data[time_step]
    Embeddings = model.GNN_model(Data.x,Data.edge_index)
    softmax = Softmax(dim=1)
    preds = softmax(model.lin(Embeddings)).round()
    y_pred = np.array([preds[i].argmax().item() for i in range(len(preds))])
    Embeddings = Embeddings.detach().numpy()
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                    init='random').fit_transform(Embeddings)
    to_CSV(X_embedded,Data)
    return X_embedded


def to_CSV(X_embedded,Data):
    edge=Data.edge_index.detach().numpy()
    y=Data.y.detach().numpy()
    df_nodes=pd.DataFrame(columns=['noeuds','positionX','positionY','type'])
    df_edges=pd.DataFrame(columns=['source','destination'])
    
    T=[]
    nodes=[]
    positionX=[]
    positionY=[]
    
    for i in range(len(X_embedded)):
        positionX.append(X_embedded[i][0])
        positionY.append(X_embedded[i][1])
        T.append(y[i])
        nodes.append(i)
    df_nodes["noeuds"]=nodes
    df_nodes['positionX']=positionX
    df_nodes['positionY']=positionY
    df_nodes['type']=T

    df_nodes.to_csv(os.path.join(save_path,"bitcoinNodes.csv"),index=False)
    
    print(y[0])
    df_edges["source"]=edge[0]
    df_edges['destination']=edge[1]
    df_edges.to_csv(os.path.join(save_path,"bitcoinEdges.csv"),index=False)
        

def main(): 

    """
    Collect arguments and run.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-mn",
        "--model-name",
        default=os.path.join(model_path,"model_AML_SAGE.pt"),
        type=str,
    )

    parser.add_argument(
        "-s",
        "--split",
        default="val",
        type=str,
    )

    parser.add_argument(
        "-ts",
        "--time-step",
        default=2,
        type=int,
    )

    parser.add_argument(
        "-sf",
        "--save-file",
        default='embedding.npy',
        type=str,
    )

    args = parser.parse_args()


    plan_embedding = get_embedding(data_path,args.model_name,args.split,args.time_step)
    np.save(os.path.join(save_path,args.save_file),plan_embedding)


if __name__ == "__main__":
    main()