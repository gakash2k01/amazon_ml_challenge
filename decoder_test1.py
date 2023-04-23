import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb
import pathlib, os
import warnings
from sklearn.preprocessing import OneHotEncoder
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
        self.ff1 = nn.Linear(13421,128)
        self.ff2 = nn.Linear(768+128, 120)
        self.ff3 = nn.Linear(120, 24)
        self.ff4 = nn.Linear(24, 1)
        self.relu = nn.ReLU()
        

    def forward(self, X, X1):
        X1 = self.ff1(X1)
        X1 = self.relu(X1)
        X2 = torch.cat((X, X1), dim=1)
        X2 = X2.to(self.ff2.weight.dtype)
        output = self.ff2(X2)
        output = self.relu(output)
        output = self.ff3(output)
        output = self.relu(output)
        output = self.ff4(output)
        return output

class CustomDataset(Dataset):
    def __init__(self, data, data1):
        self.data = data
        self.data1 = data1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        x_i = self.data.iloc[ind]
        x1_i = self.data1.iloc[ind]

        return torch.tensor(x_i), torch.tensor(x1_i)

    
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
        x_i, x1_i, y_i = batch_data
        x_i = x_i.to(device).float()
        x1_i = x1_i.to(device).float()
        y_i = y_i.to(device).float()
        optimizer.zero_grad()
        # Obtaining the logits to obtain the loss
        predictions = model(x_i, x1_i)
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
    with torch.no_grad():
        for batch_data in tqdm(iterator):
            x_i, x1_i = batch_data
            x_i = x_i.to(device).float()
            x1_i = x1_i.to(device).float()
            predictions = model(x_i, x1_i)
            predictions = predictions.squeeze().cpu().numpy()
            final_pred += list(predictions)
    # final_pred = torch.tensor(final_pred)
    return final_pred

def run(model, root_dir):
    '''
    Function: Similar to the main function
    '''
    torch.cuda.empty_cache()
    seed_everything(SEED)
    print("Setting up data.")
    data = pd.read_csv(f'{root_dir}/test_embeddings.csv')
    data1 = pd.read_csv(f'{root_dir}/mini_test.csv')
    
    # create a sample DataFrame with a categorical column containing values from 0 to 13420
    df = pd.DataFrame({
        'PRODUCT_TYPE_ID': range(13421),
        'value': [i**2 for i in range(13421)]
    })
    encoder = OneHotEncoder(sparse=False)
    _ = encoder.fit_transform(df[['PRODUCT_TYPE_ID']])
    onehot_array1 = encoder.transform(pd.DataFrame(data1['PRODUCT_TYPE_ID']))
    test_onehot_df = pd.DataFrame(onehot_array1, columns=encoder.get_feature_names_out(['PRODUCT_TYPE_ID']))
    # Dataloader
    test_dataset = CustomDataset(data = data, data1 = test_onehot_df)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers = num_workers, drop_last = False)
    model = model.to(device)

    pred = evaluate(model, test_loader)
    sub = pd.DataFrame({'PRODUCT_ID': data1['PRODUCT_ID'], 'PRODUCT_LENGTH': pred})
    sub.to_csv('submission_files/sub1.csv')


if __name__ == "__main__":
    # Helps make all paths relative
    base_path = pathlib.Path().absolute()

    # Input of the required hyperparameters
    BATCH_SIZE = 120 *2
    device = 'cuda:1'

    SEED = 42
    num_workers = 4
    # Path to the dataset
    root_dir = f"{base_path}/dataset"
    if not os.path.exists(root_dir):
        print("Dataset missing.")
    
    model = Decoder()
    model_path = "weights/best_model3.pth"
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    run(model, root_dir)
