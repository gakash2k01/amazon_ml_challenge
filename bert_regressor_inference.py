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
        #The output layer that takes the [CLS] representation and gives an output
        self.cls_layer1 = nn.Linear(config.hidden_size,128)
        self.relu1 = nn.ReLU()
        self.ff1 = nn.Linear(128,128)
        self.tanh1 = nn.Tanh()
        self.ff2 = nn.Linear(128,16)
        self.sig1 = nn.Sigmoid()
        self.ff3 = nn.Linear(17, 1)

    def forward(self, input_ids, attention_mask, input2):
        #Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #Obtain the representations of [CLS] heads
        logits = outputs.last_hidden_state[:,0,:]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        output = self.ff1(output)
        output = self.tanh1(output)
        output = self.ff2(output)
        output = self.relu1(output)
        # input2 = self.sig1(input2)
        output2 = torch.cat((input2.unsqueeze(1), output), dim=1)
        output2 = self.ff3(output2)
        return output2

def evaluate(model, iterator, submission):
    '''
    function: Evaluating the model
    Input: model, iterator, pad_id
    Returns: epoch_loss, epoch_acc
    '''
    model.eval()
    final_pred = []
    with torch.no_grad():
        for data in tqdm(iterator):
            inp_ids = data[0][0].to(device)
            inp_mask = data[0][1].to(device)
            input2 = data[1].to(device).to(device)
            model.to(device)
            predictions = model(input_ids=inp_ids, attention_mask=inp_mask, input2 = input2)
            predictions = predictions.squeeze().cpu().numpy()
            final_pred += list(predictions)
    submission['PRODUCT_LENGTH'] = final_pred
    submission.to_csv('submission_files/bert_regressor.csv', index = False)
    

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
    test_data = pd.read_csv(f'{root_dir}/test.csv')
    submission = pd.read_csv(f'{root_dir}/sample_submission.csv')
    test_data = test_data.fillna("Not given")
    print("Data shape:",test_data.shape)
    print("Full scale data with clipping.")
    test_ds = read_dataset(test_data)
    print("Number of training samples:", len(test_ds))
    # Dataloader

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers)
    model = model.to(device)
 
    # Testing
    evaluate(model, test_loader, submission)


if __name__ == "__main__":
    # Helps make all paths relative
    base_path = pathlib.Path().absolute()

    # Input of the required hyperparameters
    BATCH_SIZE = 240
    model_name = 'distilbert-base-uncased'
    device = 'cuda:0'

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
    model_path = "weights/best_model.pth"
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    run(model, tokenizer, root_dir)
