import torch
import torch.nn as nn
import torch.nn.functional as F

class FFNNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNNModel, self).__init__()

        # an affine operation: y = Wx + b, linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 5*5 from image dimension
        
        # non-linearity
        self.relu = nn.ReLU()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        #x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

lead_time = 10
input_dim = 300*2
hidden_dim = 64
output_dim = lead_time * 2


class mjo_dataset(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
    def __len__(self):
        return self.length
mjo_dataset(x,y)

class ffnn_mjo:
    def __init__(self, dics, dics_ids, width, dims, num_epochs,lr=0.001) -> None:
        self.dics = dics
        self.dics_ids = dics_ids
        self.width = width
        self.input_dim = dims['input']
        self.hidden_dim = dims['hidden']
        self.output_dim = dims['output']
        self.num_epochs = num_epochs
        self.lr = lr

        #dataloader
        dataloader = torch.utils.data.DataLoader(dataset=dataset,shuffle=True,batch_size=batch_size)

    def train_mjo(self):
        input_dim = self.input_dim
        hidden_dim = self.hidden_dim
        output_dim = self.output_dim
        num_epochs = self.num_epochs
        lr = self.lr
        
        model = FFNNModel(input_dim,hidden_dim,output_dim) # instantiate the model class
        criterion = nn.MSELoss() # instantiate the loss class
        optimizer = torch.optim.SGD(model.parameters(), lr=lr) # instantiate the optimizer class

        # training the model
        iter = 0
        for epoch in range(num_epochs):
            # Load images with gradient accumulation capabilities
            images = images.view(-1, 28*28).requires_grad_()

            # clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # forward pass to get output/logits
            outputs = model(images)

            # calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # getting gradients w.r.t. parameters
            loss.backward()

            # updating parameters
            optimizer.step()