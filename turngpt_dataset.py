import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json


class TurnGPTDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, num_turns=0):
        self.tokenizer = tokenizer

        self.data = TurnGPTDataset.split_dialog(dataframe, num_turns)

        self.max_len = max_len

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        dialog = [value["text"] for name, value in self.data.iloc[index].iteritems() 
                                if value is not None]

        #print(self.tokenizer.pad_token)
        tokenized = self.tokenizer(dialog, include_pre_space=True, include_end_ts=True, 
                                        padding="max_length", max_length=self.max_len, 
                                        truncation = True, return_tensors="pt")
        tokenized["labels"]  = tokenized["input_ids"]   

        return tokenized

    @staticmethod
    def split_dialog(dataframe, num_turns=0):
        n, m = dataframe.shape
        print(f'Dataset contains {n} dialogs')      
        if num_turns == 0:
            return dataframe
        else:
            print(f'Splitting dialogs into {num_turns} turns each')
            instances = []
            for i in range(n):
                for j in range(m-num_turns+1):
                    elements = []
                    for k in range(num_turns):
                        if dataframe.iloc[i, j+k] is not None: 
                            elements.append(dataframe.iloc[i, j+k])
                    if len(elements) == num_turns:
                        instances.append(elements)
        print(f'Dataset was split and it contains {len(instances)} dialogs now')
        return pd.DataFrame(instances)            
        

class IncrTurnGPTDataset(Dataset):

    def __init__(self, path, tokenizer, max_len, num_turns, device):
        self.tokenizer = tokenizer

        self.data = IncrTurnGPTDataset.split_dialog(path, num_turns)

        self.max_len = max_len
        self.num_turns = num_turns
        self.device = device

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        dialog = [value for name, value in self.data.iloc[index].iteritems() 
                                if value is not None and value != "" and name != self.num_turns]

        tokenized = self.tokenizer(dialog, include_pre_space=True, include_end_ts=True, 
                                        padding="max_length", max_length=self.max_len, 
                                        truncation = True, return_tensors="pt")

        tokenized["input_ids"] = tokenized["input_ids"].to(self.device)
        tokenized["token_type_ids"] = tokenized["token_type_ids"].to(self.device)
        tokenized["attention_mask"] = tokenized["attention_mask"].to(self.device)
        tokenized["labels"]  = self.data.iloc[index][self.num_turns] 

        return tokenized

    @staticmethod
    def split_dialog(path, num_turns=0):
        f = open(path)
        dialogs = json.load(f)

        instances = []
        for dialog in dialogs:
            utterances = []
            for utterance in dialog:
                n_partial = len(utterance["partial"])
                if n_partial == 0:
                   utt = utterance["text"].strip()
                elif n_partial == 1:
                    utt = utterance["partial"][0].strip()
                else:
                    for idx in range(n_partial-1):
                        utt = utterance["partial"][idx].strip()
                        turns = ["" for _ in range(num_turns - len(utterances + [utt]))]
                        instances.append(utterances + [utt] + turns + [0])

                    utt = utterance["partial"][n_partial-1].strip()

                utterances.append(utt)
                if len(utterances) == num_turns:
                    instances.append(utterances + [1])
                    utterances.pop(0)
                else:
                    turns = ["" for _ in range(num_turns - len(utterances))]
                    instances.append(utterances + turns + [1])

        print(f'Dataset was split and it contains {len(instances)} dialogs now')
        return pd.DataFrame(instances)    

    @staticmethod
    def split_dialog_context(path, num_turns=0, context_size=0):
        f = open(path)
        dialogs = json.load(f)

        instances = []
        for dialog in dialogs:
            utterances = []
            for utterance in dialog:
                n_partial = len(utterance["partial"])
                if n_partial == 0:
                   utt = utterance["text"].strip()
                elif n_partial == 1:
                    utt = utterance["partial"][0].strip()
                else:
                    for idx in range(n_partial-1):
                        utt = utterance["partial"][idx].strip()

                        aux_utterances = []
                        split_utt = utt.split()
                        remaining_context = context_size - len(split_utt)
                        if remaining_context > 0:
                            aux_utt = utt
                            for utter in reversed(utterances):
                                split_utter = utter.split()
                                if remaining_context - len(split_utter) > 0:
                                    aux_utterances = [utter] + aux_utterances
                                    remaining_context = remaining_context - len(split_utter)
                                else:
                                    aux_utter = split_utter[len(split_utter)-remaining_context:]
                                    aux_utter = " ".join(aux_utter)
                                    aux_utterances = [aux_utter] + aux_utterances
                                    break                                
                        else:
                            aux_utt = split_utt[len(split_utt)-context_size:]
                            aux_utt = " ".join(aux_utt)

                        turns = ["" for _ in range(num_turns - len(aux_utterances + [aux_utt]))]

                        instances.append(aux_utterances + [aux_utt] + turns + [0])

                    utt = utterance["partial"][n_partial-1].strip()

                utterances.append(utt)

                remaining_context = context_size
                aux_utterances = []
                for utter in reversed(utterances):
                    split_utter = utter.split()
                    if remaining_context - len(split_utter) > 0:
                        aux_utterances = [utter] + aux_utterances
                        remaining_context = remaining_context - len(split_utter)
                    else:
                        aux_utter = split_utter[len(split_utter)-remaining_context:]
                        aux_utter = " ".join(aux_utter)
                        aux_utterances = [aux_utter] + aux_utterances
                        break
                turns = ["" for _ in range(num_turns - len(aux_utterances))]
                instances.append(aux_utterances + turns + [1])

                if len(utterances) == num_turns:
                    utterances.pop(0)

        print(f'Dataset was split and it contains {len(instances)} dialogs now')
        return pd.DataFrame(instances)   


class TurnGPTDatasetV2(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer

        self.data = dataframe

        self.max_len = max_len

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        dialog = [value for name, value in self.data.iloc[index].iteritems() 
                                if value is not None and int(name) % 2 == 0]

        tokenized = self.tokenizer(dialog, include_pre_space=True, include_end_ts=True,
                                        padding="max_length", max_length=self.max_len, 
                                        truncation = True, return_tensors="pt")
        tokenized["labels"]  = tokenized["input_ids"]   

        return tokenized


class ClassifierDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer

        self.data = dataframe

        self.max_len = max_len

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        dialog = [value for name, value in self.data.iloc[index].iteritems()
                                if value is not None and int(name) % 2 == 0]

        #print(self.tokenizer.pad_token, "\t", self.tokenizer.pad_token_id)

        tokenized = self.tokenizer(dialog, include_pre_space=True, include_end_ts=False,
                                        padding="max_length", max_length=self.max_len,
                                        truncation = True, return_tensors="pt")
        
        #tokenized["labels"]  =  torch.tensor([self.data.iloc[index][3]])

        for key in tokenized:
            tokenized[key] = tokenized[key].reshape(-1)
            #print(key, "\t", tokenized[key], "\t", tokenized[key].shape)

        #labels = [0 for _ in range(2)]
        #labels[self.data.iloc[index][3]] = 1
        #tokenized["labels"] = torch.tensor(labels)
        tokenized["labels"]  =  torch.tensor([self.data.iloc[index][3]])

        #print(tokenized["labels"])

        return tokenized
        

def get_incremental_dataset(path, tokenizer, max_len, num_turns=0, device="cpu"):
    print(f'Loading dataset from {path}')
    return IncrTurnGPTDataset(path, tokenizer, max_len, num_turns, device)
        
def get_dataset(path, tokenizer, max_len, num_turns=0):
    print(f'Loading dataset from {path}')
    df = pd.read_json(path)
    return TurnGPTDataset(df, tokenizer, max_len, num_turns)

def get_datasetV2(path, tokenizer, max_len, num_turns=0):
    print(f'Loading dataset from {path}')
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    print(f'Loading {df.shape[0]} instances...')
    return TurnGPTDatasetV2(df, tokenizer, max_len)

def get_classifier_dataset(path, tokenizer, max_len, balance=None):
    print(f'Loading dataset from {path}')
    df = pd.read_csv(path, sep="\t", encoding="utf-8")

    if balance is not None:
        print(f'Resizing {df.shape[0]} instances ...')
        classes_df = {0: df.loc[df['3'] == 0], 1: df.loc[df['3'] == 1]}
        len_index = 0 if classes_df[0].shape[0] > classes_df[1].shape[0] else 1
        diff_len = abs(classes_df[0].shape[0] - classes_df[1].shape[0])

        n_undersampling = classes_df[len_index].shape[0] - int(diff_len * balance)
        n_oversampling = int(diff_len * (1-balance))

        undersample_df = classes_df[len_index].sample(n=n_undersampling, random_state=42, ignore_index=True)
        oversample_df = pd.concat([classes_df[1-len_index], 
                classes_df[1-len_index].sample(n=n_oversampling, random_state=42, replace=True, ignore_index=True)], 
                ignore_index=True)
        balanced_df = pd.concat([undersample_df, oversample_df], ignore_index=True)
        print(f'Loading {balanced_df.shape[0]} balanced instances ...')
        return ClassifierDataset(balanced_df, tokenizer, max_len)

    print(f'Loading {df.shape[0]} instances...')
    return ClassifierDataset(df, tokenizer, max_len)
