from torch import nn
import torch

if __name__ == '__main__':
    x = torch.tensor([0,5,3,0,1,1,0]).float() # 가짜 출력값
    x1 = torch.tensor([0,5,3,0,1,1,0]).float() # 가짜 출력값
    y = torch.tensor([1,4,2,1,0,1,1]).float()# 가짜 라벨값
    y1 = torch.tensor([0,0,2,5,3,0,0]).float()# 가짜 라벨값

    lossFunction = nn.BCEWithLogitsLoss()

    print(lossFunction(x,y))
    print(lossFunction(x,y1))
    #%%

