import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ..utils.dat_ops import rolling


class NNMJO:
    def __init__(self, npzfile, width=300, lead_time=60, 
                n=10000, start_train=0, n_offset = 0,
                start_val=None, v=2500) -> None:
        self.npzfile = npzfile
        self.width = width
        self.lead_time = lead_time
        self.n = n
        self.start_train = start_train
        self.d = width + lead_time
        self.n_train = self.d + n

        self.n_offset = n_offset
        if start_val is None:
            self.start_val = start_train + self.n_train + self.n_offset
        self.v = v
        self.n_val = self.d + v
        
        ##################################
        # Setup training data
        ##################################
        train_data = torch.tensor([])
        train_ids = torch.arange(start_train, start_train + self.n_train)
        for rmm in ['RMM1','RMM2']:  
            train_ij = torch.as_tensor(npzfile[rmm][train_ids], dtype=torch.float32)
            train_datarmm = rolling(train_ij[:-1], self.d) # (n, d) tensor
            
            train_data = train_data.reshape(train_datarmm.shape[0], self.d, -1)

            train_data = torch.cat(( train_data, train_datarmm[...,None] ), dim=-1) # (n, d, 2) numpy array

        train_x = train_data[:,:width,:].reshape(train_data.size(0), -1)
        train_y = train_data[:,width:,:].reshape(train_data.size(0), -1)

        self.train_data = train_data # (n, d, 2) tensor
        self.train_ids = train_ids
        self.train_dataset = TensorDataset(train_x, train_y)

        ##################################
        # Setup validation data
        ##################################
        val_data = torch.tensor([])
        val_ids = torch.arange(self.start_val, self.start_val + self.n_val)
        for rmm in ['RMM1','RMM2']:  
            val_ij = torch.as_tensor(npzfile[rmm][val_ids], dtype=torch.float32)
            val_datarmm = rolling(val_ij[:-1], self.d) # (v, d) tensor
            
            val_data = val_data.reshape(val_datarmm.shape[0], self.d, -1)

            val_data = torch.cat(( val_data, val_datarmm[...,None] ), dim=-1) # (v, d, 2) numpy array

        val_x = val_data[:,:width,:].reshape(val_data.size(0), -1) # (v, width*2) tensor
        val_y = val_data[:,width:,:].reshape(val_data.size(0), -1) # (v, lead_time*2) tensor

        self.val_data = val_data # (v, d, 2) tensor
        self.val_ids = val_ids
        self.val_dataset = TensorDataset(val_x, val_y)

    def train(self, model_cls, **kwargs):

        train_dataset = self.train_dataset
        val_dataset = self.val_dataset
        
        verbose = kwargs.get('verbose', True)
        hidden_dim = kwargs.get('hidden_dim', 64)
        num_epochs = kwargs.get('num_epochs', 10)
        lr = kwargs.get('lr', 0.1)
        seed = kwargs.get('seed', 99)

        train_x, train_y = train_dataset.tensors
        input_dim = train_x.size(1)
        output_dim = train_y.size(1)

        # Use batch size of 16
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed))
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed))
        best_val_loss = float('inf')  # Initialize with a very large value

        # Initialize the model
        model = model_cls(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, seed=seed)

        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Define loss function 
        criterion = nn.MSELoss()
        
        train_losses = [] # To store the losses for plotting
        val_losses = [] # To store the losses for plotting
        
        
        epochs_iter = tqdm(range(num_epochs), desc=f"Training {model_cls.__name__}")
        for i in epochs_iter:

            ###########################################
            # Train the model on the training set
            ###########################################
            # model.train()
            train_loss = 0.0


            # Within each iteration, we will go over each minibatch of data
            train_minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)
            for train_x, train_y in train_minibatch_iter:
                optimizer.zero_grad() # Zero gradients from previous iteration
                output = model(train_x) # Output from model
                loss = criterion(output, train_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            epochs_iter.set_postfix(loss=loss.item())

            # Calculate average training loss
            train_loss /= len(train_dataset)
            train_losses.append(train_loss)

            ###########################################
            # Evaluate the model on the validation set
            ###########################################
            val_loss = 0.0
            with torch.no_grad():
                model.eval()
                val_minibatch_iter = tqdm(val_loader, desc="Minibatch", leave=False)
                for val_x, val_y in val_minibatch_iter:
                    output = model(val_x) # Output from model
                    loss = criterion(output, val_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_dataset)
            val_losses.append(val_loss)
            if verbose:
                print(f"Epoch [{i+ 1}/{num_epochs}] Train Loss: {train_loss:.4f}  Validation Loss: {val_loss:.4f}")

            # Save the model if it performs better on validation set
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                checkpoints_path = os.path.join(dir_path, "data", "preds", "nn", "checkpoints", f"{model_cls.__name__}", f"best_model_epoch_{i + 1}.pth")
                torch.save(model.state_dict(), checkpoints_path)
        
        model.train_losses = train_losses
        model.val_losses = val_losses

        if verbose:
            print("Training Finished!")

        self.model = model

    def pred(self, start_test=None, test_ids=None, 
            n_pred=1, **kwargs):
        
        self.n_test = self.d + n_pred
        if start_test is not None and test_ids is None:
            test_ids = torch.arange(start_test, start_test + self.n_test)
        
        model = self.model
        verbose = kwargs.get('verbose', True)

        ##################################
        # Setup test data
        ##################################
        test_data = torch.tensor([])
        for rmm in ['RMM1','RMM2']:  
            test_ij = torch.as_tensor(self.npzfile[rmm][test_ids], dtype=torch.float32)
            test_datarmm = rolling(test_ij[:-1], self.d) # (n_pred, d) tensor
            
            test_data = test_data.reshape(test_datarmm.shape[0], self.d, -1)

            test_data = torch.cat(( test_data, test_datarmm[...,None] ), dim=-1) # (n_pred, d, 2) numpy array
        

        test_x = test_data[:,:self.width,:].reshape(test_data.size(0), -1) # (n_pred, width*2) tensor
        test_y = test_data[:,self.width:,:].reshape(test_data.size(0), -1) # (n_pred, lead_time*2) tensor
        with torch.no_grad():
            model.eval()
            preds_y = model(test_x)
        

        if verbose:
            error = torch.mean(torch.abs(preds_y - test_y))
            print(f"Test {model.__class__.__name__} MAE: {error.item()}")
        

        self.test_data = test_data # (n_pred, d, 2) tensor
        self.test_ids = test_ids
        self.test_dataset = TensorDataset(test_x, test_y)
        self.preds_y = preds_y.reshape(n_pred, -1, 2)
        self.obs_y = test_y.reshape(n_pred, -1, 2)
