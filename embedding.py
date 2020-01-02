import torch
from torch import nn
from tensorboardX import SummaryWriter
import torchvision
# import torch.utils.data as Data
# from torch.utils.data import DataLoader
writer = SummaryWriter('runs/forward_example')
num_epochs=500
class feedforward_neural_network(nn.Module):
    def __init__(self, input_size, hidden, num_classes):
        super(feedforward_neural_network, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden)
        self.linear2 = nn.Linear(input_size, hidden)
        self.r = nn.Sigmoid()
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, x1,x2):
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)
        x1 = self.r(x1)
        x2 = self.r(x2)
        out = self.out(x1*x2)
        return out

model = feedforward_neural_network(input_size=8, hidden=64, num_classes=8)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.005)


embedding = nn.Embedding(8, 8)
embedding_y=[]
embedding_x1=[]
embedding_x2=[]

with open ("X.txt",'r') as f:
    for line in f:
        x = line.strip().split()
        embedding_x1.append([int(x[0])])
        embedding_x2.append([int(x[1])])
        embedding_y.append([int(x[2])])

input1 = torch.LongTensor(embedding_x1)
input2 = torch.LongTensor(embedding_x2)
output= torch.LongTensor(embedding_y).squeeze()
#print(input1.shape)
# torch_data=Data.TensorDataset(input1,input2,output)
# dataloader = DataLoader(dataset=torch_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

for epoch in range(num_epochs):
    correct=0
    for i in range(64):
        outputs = model(embedding(input1[i]),embedding(input2[i]))
        # print("输出：",outputs.shape)
        #print("原始输出：", output[i].unsqueeze(0))
        #writer.add_graph(model, input_to_model=(embedding(input1[i]), embedding(input2[i])), verbose=False)
        optimizer.zero_grad()
        loss = criterion(outputs, output[i].unsqueeze(0))
        loss.backward()
        optimizer.step()
        _, predictes = torch.max(outputs, 1)
        # print(predictes,output[i].unsqueeze(0),predictes == output[i].unsqueeze(0))
        correct += (predictes == output[i].unsqueeze(0)).sum()
        # print(correct)
        # correctTotal+=correct
    if (epoch+ 1) % 10 == 0:
        print('Epoch:[%d],Loss:%.4f' % (epoch + 1, loss.item()))
        print('%d total Accuracy of the model on the train : %f %%' % (epoch + 1, correct / 64.0 * 100.0))
torch.save(model.state_dict(), 'model_1.pkl')
writer.add_graph(model, input_to_model=(embedding(input1[1]), embedding(input2[1])), verbose=False)
writer.close()
#加载
# params=model.state_dict()
# for k,v in params.items():
#     print(k) #打印网络中的变量名
#     print(params['linear1.weight']) #打印conv1的weight
#     print(params['linear1.bias']) #打印conv1的bias
#
model.load_state_dict(torch.load('model_1.pkl'))
#测试

correct = 0
for i in range(64):
    outputs = model(embedding(input1[i]),embedding(input2[i]))
    _, predictes = torch.max(outputs, 1)
        #print(predictes,output[i].unsqueeze(0),predictes == output[i].unsqueeze(0))
    correct += (predictes == output[i].unsqueeze(0)).sum()
        #print(correct)
    # correctTotal+=correct
print('Accuracy of the model on the test : %f %%'%(correct/64.0*100.0))
# print("Auccacy is correc:",correctTotal)

num1=input("please input a number:")
num2=input("please input a number:")
x1=[]
x1.append(int(num1))
x2=[]
x2.append(int(num2))
x1=torch.LongTensor(x1)
x2=torch.LongTensor(x2)
y=model(embedding(x1),embedding(x2))
_, predictes = torch.max(y, 1)
print("The result is:",predictes.item())