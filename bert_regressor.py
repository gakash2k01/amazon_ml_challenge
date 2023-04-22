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


def tokenize(text):
    '''
    Function: converts words to tokens.
    Input: Word
    Output: tokens, attention-mask
    '''
    res = tokenizer.encode_plus(text, padding="max_length", truncation=True, max_length=512)
    return torch.tensor(res.input_ids), torch.tensor(res.attention_mask)


def read_dataset(sent):
    '''
    Reading dataset and preprocessing it to get it in the desired forma
    '''
    res = []
    for _, row in tqdm(sent.iterrows()):
        # Converting the labels to the given format to make it easier to train.
        inp = row['TITLE']
        res.append([tokenize(inp), row['PRODUCT_LENGTH']])
    return res


class BertRegresser(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = DistilBertModel(config)
        #The output layer that takes the [CLS] representation and gives an output
        self.cls_layer1 = nn.Linear(config.hidden_size,128)
        self.relu1 = nn.ReLU()
        self.ff1 = nn.Linear(128,128)
        self.tanh1 = nn.Tanh()
        self.ff2 = nn.Linear(128,1)

    def forward(self, input_ids, attention_mask):
        #Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #Obtain the representations of [CLS] heads
        logits = outputs.last_hidden_state[:,0,:]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        output = self.ff1(output)
        output = self.tanh1(output)
        output = self.ff2(output)
        return output
    
def train(model, iterator, optimizer):
    '''
    function: Training the model
    Input: model, iterator, optimizer
    Returns: epoch_loss
    '''
    epoch_loss = 0
    criterion = nn.L1Loss()
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    model.train()
    total = 0
    for data in tqdm(iterator):
        inp_ids = data[0][0]
        inp_mask = data[0][1]
        target = data[1]
        model.to(device)
        optimizer.zero_grad()
        inp_ids = inp_ids.to(device)
        inp_mask = inp_mask.to(device)
        target = target.to(device)
        # Obtaining the logits to obtain the loss
        predictions = model(input_ids=inp_ids, attention_mask=inp_mask)
        loss = criterion(predictions, target.type_as(predictions))
        loss.backward()
        optimizer.step()
        total+=len(data)
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
        for data in tqdm(iterator):
            inp_ids = data[0][0]
            inp_mask = data[0][1]
            target = data[1]
            model.to(device)
            inp_ids = inp_ids.to(device)
            inp_mask = inp_mask.to(device)
            predictions = model(input_ids=inp_ids, attention_mask=inp_mask)
            predictions = predictions.squeeze().cpu().numpy()
            final_pred += list(predictions)
            comp_pred += list(target)
    return score_function(np.array(final_pred), np.array(comp_pred)), ((abs(np.array(final_pred)-np.array(comp_pred))).sum())/len(final_pred)



def run(model, tokenizer, root_dir):
    '''
    Function: Similar to the main function
    '''
    torch.cuda.empty_cache()
    seed_everything(SEED)
    print("Setting up.")
    # Maximum number of characters in a sentence. Set to 512.
    max_input_length = tokenizer.max_model_input_sizes[model_name]
    pad_token = tokenizer.pad_token
    
    # Padding helps prevent size mistmatch
    pad_id = tokenizer.convert_tokens_to_ids(pad_token)
    
    # Reading dataset and preprocessing it to get it in the desired format
    print("Setting up data.")
    train_data = pd.read_csv(f'{root_dir}/train.csv')
    # train_data = train_data.sample(5000)
    train_data = train_data.dropna(axis=0)
    print("Data shape:",train_data.shape)
    train_data['PRODUCT_LENGTH'].clip(upper=5000)
    train_ds = read_dataset(train_data)
    train_ds, valid_ds = train_test_split(train_ds, test_size=0.2, random_state=42)
    print("Number of training samples:", len(train_ds))
    # Dataloader
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers)

    valid_loader = DataLoader(
        dataset=valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers)
    
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.8)
    model = model.to(device)

    N_EPOCHS = num_epoch
    best_acc = 0.0
    for epoch in range(N_EPOCHS):
        
        #training part
        train_loss = train(model, train_loader, optimizer)
        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | LR: {scheduler.get_lr()[0]:.6f}')
        
        # Validation
        valid_loss, val_mae = evaluate(model, valid_loader)
        print(f'Val. mae: {val_mae:.2f}\t Val. score: {valid_loss:.2f}%')
        if(valid_loss>best_acc):
            best_acc = valid_loss
            torch.save(model.state_dict(), f'{base_path}/weights/best_model.pth')
            print("Model saved.")
        if wandb_login:
            wandb.log({"Training loss": train_loss, "Validation MAE": val_mae, "Validation Score": valid_loss, "lr": scheduler.get_lr()[0]}, step=epoch)
        scheduler.step()


if __name__ == "__main__":
    # Helps make all paths relative
    base_path = pathlib.Path().absolute()

    # Input of the required hyperparameters
    BATCH_SIZE = 240
    learning_rate = 0.1
    model_name = 'distilbert-base-uncased'
    device = 'cuda:2'
    wandb_login = True

    SEED = 42
    # Since the dataset is simple, 1 epoch is sufficient to finetune.
    num_epoch = 100
    num_workers = 16
    # Path to the dataset
    root_dir = f"{base_path}/dataset"
    if not os.path.exists(root_dir):
        print("Dataset missing.")
    
    #Loading the pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_name)
    model = BertRegresser(config=config)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    
    if wandb_login:
        # For logging
        wandb.login()
        wandb.init(project="amazon_ml_challenge", entity="gakash2001")
        wandb.config = {
            "learning_rate": learning_rate,
            "epochs": num_epoch,
            "batch_size": BATCH_SIZE
        }
    run(model, tokenizer, root_dir)
