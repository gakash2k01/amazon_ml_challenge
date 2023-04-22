import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR

from transformers import DistilBertPreTrainedModel, DistilBertModel, AutoConfig, AutoTokenizer
from transformers import DistilBertModel, DistilBertTokenizer

from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb
import pathlib, os, math
import warnings
from torch.utils.data import Dataset
from lion_pytorch import Lion
from utils.score import score_function

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def seed_everything(seed):
    '''
    Seeds everything so as to allow for reproducibility
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff0 = nn.Linear(769,128)
        self.ff1 = nn.Linear(128,16)
        self.ff2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        

    def forward(self, X):
        output = self.ff0(X)
        output = self.relu(output)
        output = self.ff1(output)
        output = self.relu(output)
        output = self.ff2(output)
        return output

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind):
        x_i = self.X.iloc[ind]
        y_i = self.y.iloc[ind]

        return torch.tensor(x_i), torch.tensor(y_i)

    
def train(model, iterator, optimizer):
    '''
    function: Training the model
    Input: model, iterator, optimizer
    Returns: epoch_loss
    '''
    epoch_loss = 0
    criterion = nn.MSELoss()
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    
    total = 0
    for batch_data in tqdm(iterator):
        print(batch_data)
        x_i, y_i = batch_data
        x_i = x_i.to(device)
        y_i = y_i.to(device)
        optimizer.zero_grad()
        # Obtaining the logits to obtain the loss
        predictions = model(x_i)
        loss = criterion(predictions, y_i)
        loss.backward()
        optimizer.step()
        total+=len(y_i)
        epoch_loss += loss.item()
    scheduler.step()
    return epoch_loss / total

def evaluate(model, iterator):
    '''
    function: Evaluating the model
    Input: model, iterator, pad_id
    Returns: epoch_loss, epoch_acc
    '''
    model.eval()
    final_pred = []
    # Predicted value
    comp_pred = []
    with torch.no_grad():
        for i, data in tqdm(iterator):
            inp_ids = data[0][0]
            inp_mask = data[0][1]
            input2 = data[1].to(device)
            target = data[2]
            model.to(device)
            inp_ids = inp_ids.to(device)
            inp_mask = inp_mask.to(device)
            predictions = model(input_ids=inp_ids, attention_mask=inp_mask, input2 = input2)
            predictions = predictions.squeeze().cpu().numpy()
            final_pred += list(predictions)
            comp_pred += list(target)
    return score_function(np.array(final_pred), np.array(comp_pred)), ((abs(np.array(final_pred)-np.array(comp_pred))).sum())/len(final_pred)



def run(model, root_dir):
    '''
    Function: Similar to the main function
    '''
    torch.cuda.empty_cache()
    seed_everything(SEED)
    print("Setting up data.")
    data = pd.read_csv(f'{root_dir}/embeddings.csv')
    data1 = pd.read_csv(f'{root_dir}/mini_train.csv')
    y = data['PRODUCT_LENGTH']
    X = data.drop(columns = ['PRODUCT_ID','PRODUCT_LENGTH', ''])
    print("Data shape:",X.shape)
    print("Full scale data without clipping.")
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Number of training samples:", len(x_train), len(y_train))
    # Dataloader
    train_dataset = CustomDataset(X=x_train, y=y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = num_workers, drop_last = False)
    val_dataset = CustomDataset(X=x_val, y=y_val)
    valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = num_workers, drop_last = False)
    optimizer = Lion(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    model = model.to(device)

    N_EPOCHS = num_epoch
    best_acc = 0.0
    for epoch in range(N_EPOCHS): 
        #training part
        train_loss = train(model, train_loader, optimizer)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | LR: {learning_rate:.6f}')
        
        # Validation
        valid_loss, val_mae = evaluate(model, valid_loader)
        print(f'Val. mae: {val_mae:.2f}\t Val. score: {valid_loss:.2f}%')
        if(valid_loss>best_acc):
            best_acc = valid_loss
            torch.save(model.state_dict(), f'{base_path}/weights/best_model2.pth')
            print("Model saved.")
        if wandb_login:
            wandb.log({"Training loss": train_loss, "Validation MAE": val_mae, "Validation Score": valid_loss, "lr": learning_rate}, step=epoch)


if __name__ == "__main__":
    # Helps make all paths relative
    base_path = pathlib.Path().absolute()

    # Input of the required hyperparameters
    BATCH_SIZE = 120
    learning_rate = 1e-3
    device = 'cuda:3'
    wandb_login = False

    SEED = 42
    num_epoch = 100
    num_workers = 8
    # Path to the dataset
    root_dir = f"{base_path}/dataset"
    if not os.path.exists(root_dir):
        print("Dataset missing.")
    
    #Loading the pretrained model and tokenizer
    
    model = Decoder()
    if wandb_login:
        # For logging
        wandb.login()
        wandb.init(project="amazon_ml_challenge", entity="gakash2001")
        wandb.config = {
            "learning_rate": learning_rate,
            "epochs": num_epoch,
            "batch_size": BATCH_SIZE
        }
    run(model, root_dir)
