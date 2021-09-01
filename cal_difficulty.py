import argparse
import json
import os
import pickle

import numpy as np
import scipy
import torch
import torch.nn as nn
import transformers
from sklearn.metrics import accuracy_score, f1_score
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from preprocess_dataset import get_dd_corpus
from selection_model import TransformerRanker
from utils import (RankingDataset, get_uttr_token, load_model, set_random_seed)

def cal_cc_difficulty(dataloader, model, device, tensor_save_fname):

    cc_d = None

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader)):
            c_ids_list, r_ids_list, data_idx = (batch[0], batch[1], batch[2])
            bs = c_ids_list.shape[0]
            
            c_ids_list = c_ids_list.reshape(bs, 300).to(device)
            r_ids_list = r_ids_list.reshape(bs, 300).to(device)
            
            output = model(c_ids_list, r_ids_list)
            temp_cc_d = torch.diagonal(output, 0)

            if cc_d == None:
                cc_d = temp_cc_d.to(device, dtype=torch.float16)
            else:
                cc_d = torch.cat((cc_d, temp_cc_d), 0)

        cc_d = cc_d.reshape(1, -1)
        cc_d = 1 - cc_d/torch.max(cc_d)
        cc_d_score, cc_d_ranking = torch.sort(cc_d, dim=1, descending=False)

        data = [cc_d_score] + [cc_d_ranking] + [data_idx]

        assert len(data) == 3
        with open(tensor_save_fname, "wb") as f:
            pickle.dump(data, f)


def main(args):
    set_random_seed(args.random_seed)

    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    UTTR_TOKEN = get_uttr_token()

    special_tokens_dict = {"additional_special_tokens": [UTTR_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = TransformerRanker(len(tokenizer), 256, 8, 3)
    model = load_model(model, args.model_path.format(args.random_seed), args.target_epoch, len(tokenizer))
    model.to(device)

    raw_dd_train, raw_dd_dev = get_dd_corpus("train"), get_dd_corpus("valid")

    print("Load begin!")
    
    train_dataset = RankingDataset(
        raw_dd_train,
        tokenizer,
        "train",
        300,
        UTTR_TOKEN,
        "./data/ranking/text_ranking_{}.pck",
        "./data/ranking/tensor_ranking_{}.pck",
    )

    dev_dataset = RankingDataset(
        raw_dd_dev,
        tokenizer,
        "dev",
        300,
        UTTR_TOKEN,
        "./data/ranking/text_ranking_{}.pck",
        "./data/ranking/tensor_ranking_{}.pck",
    )

    train_ranking_fname, val_ranking_fname = args.tensor_save_path+'cc_ranking_train.pck', args.tensor_save_path+'cc_ranking_valid.pck'

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
    validloader = DataLoader(dev_dataset, batch_size=args.batch_size, drop_last=True)

    cal_cc_difficulty(trainloader, model, device, train_ranking_fname)
    cal_cc_difficulty(validloader, model, device, val_ranking_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--corpus", default="dd", choices=["persona", "dd"])
    parser.add_argument("--setname", default="test", choices=["valid", "test"])
    parser.add_argument("--log_path", type=str, default="result")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./logs/ranking_batch128_seed{}/model",
    )
    parser.add_argument("--target_epoch", type=int, default=29)
    parser.add_argument(
        "--tensor_save_path",
        type=str,
        default="./data/selection/dd_cand5/",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--model",
        default="ranking",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="random seed during training",
    )

    args = parser.parse_args()
    assert len(args.model_path.split("/")) == 4
    main(args)
