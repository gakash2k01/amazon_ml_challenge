import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import DistilBertPreTrainedModel, DistilBertModel, AutoConfig, AutoTokenizer
from transformers import DistilBertModel, DistilBertTokenizer

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
        inp = f"TITLE {row['TITLE']} BULLET_POINTS {row['BULLET_POINTS']} DESCRIPTION {row['DESCRIPTION']}"
        res.append([tokenize(inp), row['PRODUCT_TYPE_ID']])
    return res


class BertRegresser(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = DistilBertModel(config)
        print(config.hidden_size)

    def forward(self, input_ids, attention_mask):
        #Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.last_hidden_state[:,0,:]
        return logits

def evaluate(model, iterator, train_ds):
    '''
    function: Evaluating the model
    Input: model, iterator, pad_id
    Returns: epoch_loss, epoch_acc
    '''
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in tqdm(iterator):
            inp_ids = data[0][0].to(device)
            inp_mask = data[0][1].to(device)
            model.to(device)
            predictions = model(input_ids=inp_ids, attention_mask=inp_mask)
            predictions = predictions.cpu().numpy()
            embeddings += list(predictions)
    embedding = pd.DataFrame(embeddings)
    embedding.to_csv('dataset/embeddings1.csv', index = False)
    train_ds.to_csv('dataset/mini_train1.csv', index = False)
    

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
    train_data = train_data.sample(100000)
    train_data = train_data.fillna("Not given")
    print("Data shape:",train_data.shape)
    print("Full scale data with clipping.")
    train_ds = read_dataset(train_data)
    print("Number of training samples:", len(train_ds))
    # Dataloader

    test_loader = DataLoader(
        dataset=train_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers)
    model = model.to(device)
 
    # Testing
    evaluate(model, test_loader, train_data)


if __name__ == "__main__":
    # Helps make all paths relative
    base_path = pathlib.Path().absolute()

    # Input of the required hyperparameters
    BATCH_SIZE = 2048
    model_name = 'distilbert-base-uncased'
    device = 'cuda:1'

    SEED = 42
    num_workers = 16
    # Path to the dataset
    root_dir = f"{base_path}/dataset"
    if not os.path.exists(root_dir):
        print("Dataset missing.")
    
    #Loading the pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_name)
    model = BertRegresser(config=config)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    run(model, tokenizer, root_dir)
