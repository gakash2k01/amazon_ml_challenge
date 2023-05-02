import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from lsg_converter import LSGConverter
from transformers import AutoTokenizer, BertModel
from tqdm import tqdm
import numpy as np
import pandas as pd
import pathlib, os, math
import warnings

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
    
def make_embed(df):
    df=df.fillna("Not Avaliable")
    ls_embed=[]
    ls_id=[]

    for i in tqdm(range(df.shape[0])):
        rw=df.iloc[i]
        x=[rw["TITLE"]+" "+rw["BULLET_POINTS"]+" "+rw["DESCRIPTION"]]
        id=rw["PRODUCT_ID"]
        inputs = tokenizer(x, return_tensors="pt").to(device)
        outputs = model(**inputs)
        last_hidden_states = outputs.logits[:,0,:]
        last_hidden_states=last_hidden_states.cpu().detach().numpy().squeeze(0)
        ls_embed.append(last_hidden_states)
        ls_id.append(id)
    nw_df=pd.DataFrame(ls_embed)

    return nw_df

def run(root_dir):
    '''
    Function: Similar to the main function
    '''
    torch.cuda.empty_cache()
    seed_everything(SEED)
    print("Setting up data.")
    test_df=pd.read_csv("dataset/test.csv")
    # train_df=pd.read_csv("dataset/train.csv")
    # train_df=train_df.fillna("Not Avaliable")
    test_df=test_df.fillna("Not Avaliable")
    print("Creating embeddings.")
    test_embed=make_embed(test_df)
    # train_embed=make_embed(train_df)
    print(test_embed.shape)
    # print(train_embed.shape)
    test_embed.to_csv('dataset/test_embed.csv', index = False)
    # train_embed.to_csv('dataset/train_embed.csv', index = False)

if __name__ == "__main__":
    # Helps make all paths relative
    base_path = pathlib.Path().absolute()

    # Input of the required hyperparameters
    model_name = 'distilbert-base-uncased'

    SEED = 42
    num_workers = 16
    # Path to the dataset
    root_dir = f"{base_path}/dataset"
    if not os.path.exists(root_dir):
        print("Dataset missing.")
    
    #Loading the pretrained model and tokenizer
    converter = LSGConverter(max_sequence_length=4096)
    model, tokenizer = converter.convert_from_pretrained("bert-base-uncased", num_global_tokens=7)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    run(root_dir)
