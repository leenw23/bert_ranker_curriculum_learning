import argparse
import json
import os

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


def main(args):
    set_random_seed(42)

    device = torch.device("cuda")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    UTTR_TOKEN = get_uttr_token()

    special_tokens_dict = {"additional_special_tokens": [UTTR_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = TransformerRanker(len(tokenizer), 256, 8, 3)
    model = load_model(model, args.model_path.format(42), args.target_epoch, len(tokenizer))
    model.to(device)

    raw_dd_train = get_dd_corpus("train")

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

    trainloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
    )

    cc_d = None

    with torch.no_grad():
        for step, batch in enumerate(tqdm(trainloader)):
            c_ids_list, r_ids_list = (batch[0], batch[1])
            bs = c_ids_list.shape[0]
            
            c_ids_list = c_ids_list.reshape(bs, 300).to(device)
            r_ids_list = r_ids_list.reshape(bs, 300).to(device)
            
            output = model(c_ids_list, r_ids_list)
            temp_cc_d = torch.diagonal(output, 0)

            if cc_d == None:
                cc_d = temp_cc_d.to(device)
            else:
                cc_d = torch.cat((cc_d, temp_cc_d), 0)

        print(cc_d.size())





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
    parser.add_argument("--target_epoch", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--model",
        default="ranking",
    )
    parser.add_argument(
        "--direct_threshold",
        type=float,
        default=-1,
        help="baseline threshold",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="random seed during training",
    )

    args = parser.parse_args()

    assert len(args.model_path.split("/")) == 4

    args.exp_name = f"{args.model}-{args.setname}"

    args.log_path = os.path.join(args.log_path, args.corpus)

    # os.makedirs(args.log_path, exist_ok=True)
    # args.output_fname = os.path.join(args.log_path, args.exp_name) + ".json"
    # print("\n", args.output_fname, "\n")

    # assert not os.path.exists(args.output_fname)
    # os.makedirs(os.path.dirname(args.output_fname), exist_ok=True)

    main(args)
