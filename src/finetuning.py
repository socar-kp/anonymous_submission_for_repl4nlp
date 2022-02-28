import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader, Dataset

from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from transformers import BartTokenizer, BartForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import ElectraTokenizer, ElectraForSequenceClassification

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_cka import CKA
import argparse
from datasets import load_dataset

from earlystopping import EarlyStopping

import neptune.new as neptune

from sklearn.metrics import f1_score

def bert():
    # config = BertConfig.from_pretrained('bert-base-uncased',
    #                                 output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased') #, config=config)
    
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
    model = BartForSequenceClassification.from_pretrained('facebook/bart-base')
    
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
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')
    
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
    tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
    model = ElectraForSequenceClassification.from_pretrained("google/electra-base-discriminator")
    
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
        self.dataset = dataset
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

def train(model, train_dataloader, optimizer, criterion, device):
    
    model.train()
    train_acc, train_loss = 0, 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, label_id = batch
        
        # output = batch_size * num_class
        output = model(input_ids, attention_mask) # output == logits
        output = output[0]
        
        # print(output[0])
        # CrossEntropyLoss
        loss = criterion(output, label_id)

        train_loss += loss.item()
        acc = (torch.argmax(output, -1) == label_id.squeeze()).sum().item()
        train_acc += acc / label_id.size(0)
        
        loss.backward()
        optimizer.step()
        
    return train_loss / len(train_dataloader), train_acc / len(train_dataloader)
    
def test(model, test_dataloader, criterion, device):
    test_loss, test_acc = 0, 0
    outputs, targets = [], []

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, label_id = batch
            output = model(input_ids, attention_mask)
            output = output[0]

            # CrossEntropyLoss
            loss = criterion(output, label_id)

            test_loss += loss.item()
            acc = (torch.argmax(output, -1) == label_id.squeeze()).sum().item()
            test_acc += acc / label_id.size(0)
            
            outputs.extend(torch.argmax(output, -1).cpu())
            targets.extend(label_id.squeeze().cpu())

    assert len(outputs) == len(targets)
    
    f1_micro = f1_score(outputs, targets, average='micro')

    return test_loss / len(test_dataloader), test_acc / len(test_dataloader), f1_micro
    
def save_matrix(args, matrix):
    
    file_name = f"./result/{args.model1.upper()}-{args.model2.upper()}.csv"
    np.savetxt(file_name, cka.hsic_matrix, delimiter=",")
    
    return 0
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--mode', type=str, default='test')
    args = parser.parse_args()
    
    if args.model1 == 'bert':
        model1, tokenizer1, model1_layers = bert()
    elif args.model1 == 'bart':
        model1, tokenizer1, model1_layers = bart()        
    elif args.model1 == 'roberta':
        model1, tokenizer1, model1_layers = roberta()
    elif args.model1 == 'electra':
        model1, tokenizer1, model1_layers = electra()
            
    batch_size = 64
    
    train_data = pd.DataFrame(load_dataset('imdb', split='train'))
    
    temp_valid = train_data.iloc[:round(len(train_data)*0.1),:]
    temp_valid2 = train_data.iloc[round(len(train_data)*0.9):,:]
    valid_data = pd.concat([temp_valid, temp_valid2])
    
    test_data = pd.DataFrame(load_dataset('imdb', split='test'))

    # model 1 
    train_dataset = IMDbDataset(train_data, tokenizer1)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    
    valid_dataset = IMDbDataset(valid_data, tokenizer1)
    valid_sampler = torch.utils.data.SequentialSampler(valid_dataset)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    test_dataset = IMDbDataset(test_data, tokenizer1)
    test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1.to(device)
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model1.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model1.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    t_total = len(train_dataloader) * args.epochs
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    warmup_ratio = 0.1
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
    
    run = neptune.init(
    project="cheese/cka-nlp-model",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NWRhNjU0Yi1kNzY5LTRhZTgtOTA4Yi1kMjFhMjNlYmUxMTIifQ==",
)  # your credentials
    
    run['parameters'] = args
    
    if args.mode == 'train':
        # earlystopping
        early_stopping = EarlyStopping(patience = 10, verbose = True, path="./result/finetuning/"+str(args.model1)+"_checkpoint.pt")
    
        # model train
        for _epoch in tqdm(range(args.epochs), desc='Epoch'):
            train_loss, train_acc = train(model1, train_dataloader, optimizer, criterion, device)
            valid_loss, valid_acc = test(model1, valid_dataloader, criterion, device)
            print("[Epoch: %d] train loss : %5.2f | train accuracy : %5.2f" % (_epoch, train_loss, train_acc))
            print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (_epoch, valid_loss, valid_acc))

            run['train/loss'].log(train_loss)
            run['train/acc'].log(train_acc)

            run['valid/loss'].log(valid_loss)
            run['valid/acc'].log(valid_acc)

            scheduler.step()

            early_stopping(valid_loss, model1)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        
    else:
        # best model load해서 test acc를 찍어봄
        model1.load_state_dict(torch.load("./result/finetuning/"+str(args.model1)+"_checkpoint.pt"))

        test_loss, test_acc, f1_micro = test(model1, test_dataloader, criterion, device)
        print("test loss : %5.2f | test accuracy : %5.2f" % (test_loss, test_acc))
        print("test f1 micro score : %5.5f" % (f1_micro))
        
        run['test/loss'].log(test_loss)
        run['test/acc'].log(test_acc)

    run.stop()