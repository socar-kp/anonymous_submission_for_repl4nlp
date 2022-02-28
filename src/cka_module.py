# from CKAesther import linear_CKA, kernel_CKA
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import BartTokenizer, BartModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import ElectraForPreTraining, ElectraTokenizerFast

from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
import numpy as np
import pandas as pd
from torch_cka import CKA
import argparse

from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset


def bert():
    # config = BertConfig.from_pretrained('bert-base-uncased',
    #                                 output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased') #, config=config)
    
    count = 0
    model1_layers=[]
    for name, layer in model.named_modules():
        if name.endswith('attention.output.LayerNorm'):
            print(name)
            count += 1
            model1_layers.append(name)
        # print(name)

    print('- LayerNorm count :', len(model1_layers))
    
    return model, tokenizer, model1_layers

def bart():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartModel.from_pretrained('facebook/bart-base')
    
    count = 0
    model1_layers=[]
    for name, layer in model.named_modules():
        if name.endswith('self_attn_layer_norm'):
            print(name)
            count += 1
            model1_layers.append(name)

    print('- LayerNorm count :', len(model1_layers))
    
    return model, tokenizer, model1_layers

def roberta():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    
    count = 0
    model1_layers=[]
    for name, layer in model.named_modules():
        if name.endswith('attention.output.LayerNorm'):
            print(name)
            count += 1
            model1_layers.append(name)

    print('- LayerNorm count :', len(model1_layers))
    
    return model, tokenizer, model1_layers

def electra():
    model = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
    tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")
    
    count = 0
    model1_layers=[]
    for name, layer in model.named_modules():
        if name.endswith('attention.output.LayerNorm'):
            print(name)
            count += 1
            model1_layers.append(name)

    print('- LayerNorm count :', len(model1_layers))
    
    return model, tokenizer, model1_layers


class IMDbDataset(Dataset):
  
    def __init__(self, dataset, tokenizer):
        self.dataset = pd.DataFrame(dataset)
        self.tokenizer = tokenizer

        print(self.dataset.describe())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 0:2].values
        text = row[0]
        y = row[1]

        inputs = self.tokenizer(
            text, 
            return_tensors='pt',
            truncation=True,
            max_length=64,
            pad_to_max_length=True,
            add_special_tokens=True
            )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return input_ids, attention_mask, y

def save_matrix(args, matrix):
    
    file_name = f"./result/{args.model1.upper()}-{args.model2.upper()}.csv"
    np.savetxt(file_name, cka.hsic_matrix, delimiter=",")
    
    return 0
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str)
    parser.add_argument('--model2', type=str)
    args = parser.parse_args()
    
    if args.model1 == 'bert':
        model1, tokenizer1, model1_layers = bert()
    elif args.model1 == 'bart':
        model1, tokenizer1, model1_layers = bart()
    elif args.model1 == 'roberta':
        model1, tokenizer1, model1_layers = roberta()
    elif args.model1 == 'electra':
        model1, tokenizer1, model1_layers = electra()
    
    if args.model2 == 'bert':
        model2, tokenizer2, model2_layers = bert()
    elif args.model2 == 'bart':
        model2, tokenizer2, model2_layers = bart()
    elif args.model2 == 'roberta':
        model2, tokenizer2, model2_layers = roberta()
    elif args.model2 == 'electra':
        model2, tokenizer2, model2_layers = electra()
        
        
    batch_size = 8
    
    dataset = load_dataset('imdb', split='train')

    # model 1 
    test_dataset = IMDbDataset(dataset, tokenizer1)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # model2
    test_dataset2 = IMDbDataset(dataset, tokenizer2)
    test_dataloader2 = torch.utils.data.DataLoader(test_dataset2, batch_size=batch_size)
    
    # cka
    cka = CKA(model1, model2,
        model1_name=args.model1, model2_name=args.model2,
        model1_layers=model1_layers, model2_layers=model2_layers,
        device='cpu') #device='cuda')

    cka.compare(test_dataloader, test_dataloader2)
    
    save_path=f"./result/{args.model1.upper()}-{args.model2.upper()}_compare.png"
    cka.plot_results(save_path=save_path)
    
    save_matrix(args, cka.hsic_matrix)