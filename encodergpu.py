import torch
import pandas as pd
from torch.utils.data import DataLoader
from lsg_converter import LSGConverter
from transformers import DistilBertModel, DistilBertTokenizer
from tqdm import tqdm
import numpy as np
import pandas as pd
import pathlib, os, math
import warnings
from torch.utils.data import Dataset

def CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        rw = self.df.iloc[idx]
        x = rw["TITLE"] + " " + rw["BULLET_POINTS"] + " " + rw["DESCRIPTION"]
        return torch.Tensor(x)

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

def read_dataset(sent):
    '''
    Reading dataset and preprocessing it to get it in the desired forma
    '''
    res = []
    for _, row in tqdm(sent.iterrows()):
        # Converting the labels to the given format to make it easier to train.
        inp = f"TITLE {row['TITLE']} BULLET_POINTS {row['BULLET_POINTS']} DESCRIPTION {row['DESCRIPTION']}"
        res.append([inp, row['PRODUCT_TYPE_ID']])
    return res
    
def make_embed(X_df):
    X_df=X_df.fillna("Not Avaliable")
    # ls_embed=np.empty((0,768))
    ls_id=np.empty((0))
    data = read_dataset(X_df)
    dataloader = DataLoader(dataset=data, batch_size=BATCH_SIZE  ,shuffle=False)
    for rw in tqdm(dataloader):
        x, id = rw
        # inputs = tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        # outputs = model(**inputs)
        # last_hidden_states = outputs.last_hidden_state[:, 0, :]
        # last_hidden_states=last_hidden_states.cpu().detach().numpy()
        # ls_embed = np.concatenate((ls_embed, last_hidden_states), axis=0)
        ls_id =  np.concatenate((ls_id, id), axis=0)
    # nw_df = pd.DataFrame(ls_embed)
    nw_df1 = pd.DataFrame(ls_id)

    return nw_df1

def run(root_dir):
    '''
    Function: Similar to the main function
    '''
    torch.cuda.empty_cache()
    seed_everything(SEED)
    print("Setting up data.")
    test_df=pd.read_csv("dataset/test.csv")
    # train_df=pd.read_csv("dataset/train.csv").sample(1000)
    # train_df=train_df.fillna("Not Avaliable")
    # test_df=test_df.fillna("Not Avaliable")
    print("Creating embeddings.")
    test_file=make_embed(test_df)
    # train_embed, test_file=make_embed(train_df)
    # print(test_embed.shape)
    # print(train_embed.shape)
    # test_embed.to_csv('dataset/test_embed.csv', index = False)
    test_file.to_csv('dataset/test_embed1.csv', index = False)
    # train_embed.to_csv('dataset/train_embed.csv', index = False)

if __name__ == "__main__":
    # Helps make all paths relative
    base_path = pathlib.Path().absolute()

    # Input of the required hyperparameters
    BATCH_SIZE = 1024
    # model_name = 'distilbert-base-uncased'

    SEED = 42
    num_workers = 8
    # Path to the dataset
    root_dir = f"{base_path}/dataset"
    if not os.path.exists(root_dir):
        print("Dataset missing.")
    
    #Loading the pretrained model and tokenizer
    # converter = LSGConverter(max_sequence_length=4096)
    # model, tokenizer = converter.convert_from_pretrained("bert-base-uncased", num_global_tokens=7, batch_size = BATCH_SIZE)
    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    # device = "cuda:3" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    run(root_dir)
