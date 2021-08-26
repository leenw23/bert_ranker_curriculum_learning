import argparse
import os
import pickle

import numpy as np
import scipy
import torch
import torch.nn as nn
import transformers
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from preprocess_dataset import get_dd_corpus
from selection_model import TransformerRanker
from utils import (RankingDataset, dump_config, get_uttr_token, save_model, set_random_seed, write2tensorboard)

def main(args):
    set_random_seed(args.random_seed)

    dump_config(args)
    device = torch.device("cuda")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    UTTR_TOKEN = get_uttr_token()
    special_tokens_dict = {"additional_special_tokens": [UTTR_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = TransformerRanker(len(tokenizer), 256, 8, 3)
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
    print("Load end!")

    trainloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
    )
    validloader = DataLoader(dev_dataset, batch_size=args.batch_size, drop_last=True)

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
    for epoch in range(args.epoch):
        print("Epoch {}".format(epoch))
        model.train()
        for step, batch in enumerate(tqdm(trainloader)):
            optimizer.zero_grad()
            c_ids_list, r_ids_list = (batch[0], batch[1])
            bs = c_ids_list.shape[0]
            
            c_ids_list = c_ids_list.reshape(bs, 300).to(device)
            r_ids_list = r_ids_list.reshape(bs, 300).to(device)

            output = model(c_ids_list, r_ids_list)

            label = [i for i in range(bs)]
            label = torch.tensor(label).to(device)

            loss = crossentropy(output, label)
            
            # max_score, max_idxs = torch.max(output, 1)
            # correct_predictions = (max_idxs == label).sum()

            # if step % 100 == 0:
            #     print("loss: ", loss)
            #     print("correct_predictions: ", correct_predictions)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            write2tensorboard(writer, {"loss": loss}, "train", global_step)
            global_step += 1

        model.eval()
        loss_list = []
        try:
            with torch.no_grad():
                for step, batch in enumerate(tqdm(validloader)):
                    c_ids_list, r_ids_list = (batch[0], batch[1])
                    bs = c_ids_list.shape[0]

                    c_ids_list = c_ids_list.reshape(bs, 300).to(device)
                    r_ids_list = r_ids_list.reshape(bs, 300).to(device)
                    
                    output = model(c_ids_list, r_ids_list)

                    label = [i for i in range(bs)]
                    label = torch.tensor(label).to(device)

                    loss = crossentropy(output, label)
                    
                    # max_score, max_idxs = torch.max(output, 1)
                    # correct_predictions = (max_idxs == label).sum()

                    # if step % 10 == 0:
                    #     print("loss: ", loss)
                    #     print("correct_predictions: ", correct_predictions)

                    loss_list.append(loss.cpu().detach().numpy())
                    write2tensorboard(writer, {"loss": loss}, "train", global_step)
                final_loss = sum(loss_list) / len(loss_list)
                write2tensorboard(writer, {"loss": final_loss}, "valid", global_step)
        except Exception as err:
            print(err)
        save_model(model, epoch, args.model_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="ranking_batch128",
    )
    parser.add_argument("--log_path", type=str, default="logs")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epoch", type=int, default=10)
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
