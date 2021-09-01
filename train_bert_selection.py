import argparse
import os
import pickle
from random import shuffle

import numpy as np
import scipy
import torch
import torch.nn as nn
import transformers
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

from preprocess_dataset import get_dd_corpus
from selection_model import BertSelect
from utils import (SelectionDataset, SelectionDataset_CC, dump_config, get_uttr_token, save_model, set_random_seed, write2tensorboard)


def main(args):
    set_random_seed(args.random_seed)

    dump_config(args)
    device = torch.device("cuda")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    UTTR_TOKEN = get_uttr_token()
    
    special_tokens_dict = {"additional_special_tokens": [UTTR_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    if not args.random_initialization:
        bert = BertModel.from_pretrained("bert-base-uncased")
    else:
        bert = BertModel(BertConfig())
    bert.resize_token_embeddings(len(tokenizer))
    model = BertSelect(bert)
    model = torch.nn.DataParallel(model)
    model.to(device)

    raw_dd_train, raw_dd_dev = get_dd_corpus("train"), get_dd_corpus("valid")

    print("Load begin!")

    if args.curriculum == "cc":
        train_dataset = SelectionDataset_CC(
            raw_dd_train,
            tokenizer,
            "train",
            300,
            args.retrieval_candidate_num,
            UTTR_TOKEN,
            "./data/selection/text_cand{}".format(args.retrieval_candidate_num) + "_{}.pck",
            "./data/selection/tensor_cand{}".format(args.retrieval_candidate_num) + "_{}.pck", 
            "./data/selection/dd_cand5/cc_ranking_train.pck", 
        )

        dev_dataset = SelectionDataset_CC(
            raw_dd_dev,
            tokenizer,
            "dev",
            300,
            args.retrieval_candidate_num,
            UTTR_TOKEN,
            "./data/selection/text_cand{}".format(args.retrieval_candidate_num) + "_{}.pck",
            "./data/selection/tensor_cand{}".format(args.retrieval_candidate_num) + "_{}.pck",
            "./data/selection/dd_cand5/cc_ranking_valid.pck", 
        )
        shuffle_status = False
    else:
        train_dataset = SelectionDataset(
            raw_dd_train,
            tokenizer,
            "train",
            300,
            args.retrieval_candidate_num,
            UTTR_TOKEN,
            "./data/selection/text_cand{}".format(args.retrieval_candidate_num) + "_{}.pck",
            "./data/selection/tensor_cand{}".format(args.retrieval_candidate_num) + "_{}.pck",
        )

        dev_dataset = SelectionDataset(
            raw_dd_dev,
            tokenizer,
            "dev",
            300,
            args.retrieval_candidate_num,
            UTTR_TOKEN,
            "./data/selection/text_cand{}".format(args.retrieval_candidate_num) + "_{}.pck",
            "./data/selection/tensor_cand{}".format(args.retrieval_candidate_num) + "_{}.pck",
        )
        shuffle_status = True

    trainloader = DataLoader(
        train_dataset,
        shuffle=shuffle_status,
        batch_size=args.batch_size,
        drop_last=True,
    )
    validloader = DataLoader(dev_dataset, batch_size=args.batch_size, drop_last=True)

    print("Load end!")

    """
    Training
    """
    crossentropy = CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
    )
    writer = SummaryWriter(args.board_path)

    save_model(model, "begin", args.model_path)
    global_step = 0

    if args.aum:
        data_len = len(train_dataset)
        aum_score = [[] for _ in range(data_len)]

    for epoch in range(args.epoch):
        print("Epoch {}".format(epoch))
        model.train()
        for step, batch in enumerate(tqdm(trainloader)):
            optimizer.zero_grad()
            ids_list, mask_list, label, data_idx = (
                batch[: args.retrieval_candidate_num],
                batch[args.retrieval_candidate_num : 2 * args.retrieval_candidate_num],
                batch[2 * args.retrieval_candidate_num],
                batch[2 * args.retrieval_candidate_num+1]
            )
            label = label.to(device)
            bs = label.shape[0]
            
            ids_list = (
                torch.cat(ids_list, 1)
                .reshape(bs * args.retrieval_candidate_num, 300)
                .to(device)
            )
            mask_list = (
                torch.cat(mask_list, 1)
                .reshape(bs * args.retrieval_candidate_num, 300)
                .to(device)
            )

            output = model(ids_list, mask_list)
            output = output.reshape(bs, -1)
            loss = crossentropy(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if args.aum:
                record_output = output.cpu().detach().numpy()
                pos_output = torch.Tensor(record_output[:, 0])
                neg_outputs = torch.Tensor(record_output[:, 1:])
                neg_max, _ = torch.max(neg_outputs, 1)
                batch_aum = pos_output - neg_max
                for b_idx, idx in enumerate(data_idx):
                    aum_score[idx].append(batch_aum[b_idx])

            write2tensorboard(writer, {"loss": loss}, "train", global_step)
            global_step += 1

        model.eval()
        loss_list = []
        try:
            with torch.no_grad():
                for step, batch in enumerate(tqdm(validloader)):
                    ids_list, mask_list, label = (
                        batch[: args.retrieval_candidate_num],
                        batch[
                            args.retrieval_candidate_num : 2
                            * args.retrieval_candidate_num
                        ],
                        batch[2 * args.retrieval_candidate_num],
                    )
                    label = label.to(device)
                    bs = label.shape[0]
                    ids_list = (
                        torch.cat(ids_list, 1)
                        .reshape(bs * args.retrieval_candidate_num, 300)
                        .to(device)
                    )
                    mask_list = (
                        torch.cat(mask_list, 1)
                        .reshape(bs * args.retrieval_candidate_num, 300)
                        .to(device)
                    )
                    output = model(ids_list, mask_list)
                    output = output.reshape(bs, -1)
                    loss = crossentropy(output, label)
                    loss_list.append(loss.cpu().detach().numpy())
                    write2tensorboard(writer, {"loss": loss}, "train", global_step)
                final_loss = sum(loss_list) / len(loss_list)
                write2tensorboard(writer, {"loss": final_loss}, "valid", global_step)
        except Exception as err:
            print(err)
        
        save_model(model, epoch, args.model_path)
        
        if args.aum:
            with open("./data/selection/aum_cand5/aum_ranking.pck", "wb") as f:
                pickle.dump(aum_score, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="aum_select_batch12_candi5",
    )
    parser.add_argument("--log_path", type=str, default="logs")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--curriculum", type=str, default="basic", choices=["basic", "cc"])
    parser.add_argument("--aum", type=bool, default=True)
    parser.add_argument(
        "--retrieval_candidate_num",
        type=int,
        default=5,
        help="Number of candidates including golden",
    )
    parser.add_argument(
        "--random_initialization", type=str, default="False", choices=["True", "False"]
    )
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()
    args.random_initialization = args.random_initialization == "True"
    args.exp_name += "_seed{}".format(args.random_seed)
    if args.random_initialization:
        args.exp_name += "_randinit"

    args.exp_path = os.path.join(args.log_path, args.exp_name)
    args.model_path = os.path.join(args.exp_path, "model")
    args.board_path = os.path.join(args.exp_path, "board")
    os.makedirs(args.model_path, exist_ok=False)
    os.makedirs(args.board_path, exist_ok=False)
    main(args)
