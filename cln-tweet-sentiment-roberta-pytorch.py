# To add a new cell, type ''
# To add a new markdown cell, type ''


# # Libraries

from IPython.display import display
import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, Union
import warnings
import random
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import tokenizers
from transformers import RobertaModel, RobertaTokenizer

warnings.filterwarnings("ignore")


# # Seed


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


# # Data Loader


def _get_input_data(
    row: pd.Series, tokenizer: tokenizers.ByteLevelBPETokenizer, max_len: int
) -> Tuple[torch.tensor, torch.tensor, str, torch.tensor]:
    tweet = " " + " ".join(row.text.lower().split())
    encoding = tokenizer.encode(tweet)
    sentiment_id = tokenizer.encode(row.sentiment).ids
    # <s> sentiment </s></s> encoding </s>
    ids = (
        [tokenizer.bos_token_id]
        + sentiment_id
        + [tokenizer.eos_token_id, tokenizer.eos_token_id]
        + encoding.ids
        + [tokenizer.eos_token_id]
    )
    offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]

    pad_len = max_len - len(ids)
    if pad_len > 0:
        ids += [1] * pad_len
        offsets += [(0, 0)] * pad_len

    ids = torch.tensor(ids)
    masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
    offsets = torch.tensor(offsets)

    assert ids.shape == torch.Size([96])
    assert masks.shape == torch.Size([96])
    assert offsets.shape == torch.Size([96, 2])

    return ids, masks, tweet, offsets



def _get_target_idx(row, tweet, offsets):
    selected_text = " " + " ".join(row.selected_text.lower().split())

    len_st = len(selected_text) - 1
    idx0 = None
    idx1 = None

    for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
        if " " + tweet[ind : ind + len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st - 1
            break

    char_targets = [0] * len(tweet)
    if idx0 is not None and idx1 is not None:
        for ct in range(idx0, idx1 + 1):
            char_targets[ct] = 1

    target_idx = []
    for j, (offset1, offset2) in enumerate(offsets):
        if sum(char_targets[offset1:offset2]) > 0:
            target_idx.append(j)

    start_idx = target_idx[0]
    end_idx = target_idx[-1]

    return start_idx, end_idx



def _process_row(
    row: pd.Series, tokenizer: tokenizers.ByteLevelBPETokenizer, max_len: int
) -> Dict[str, Union[torch.tensor, str, int]]:
    """
    Process row so it can be fed into Roberta.

    Parameters
    ----------
    row :
        Observation (either training or testing)

    Returns
    -------
    Dictionary that can be processed by training loop.

    Examples
    --------
    >>> row = pd.Series({'textID': 'cb774db0d1', 'text': ' I`d have responded, if I were going', 'selected_text': 'I`d have responded, if I were going', 'sentiment': 'neutral'})
    >>> data = _process_row(row)
    >>> data['tweet']
    ' i`d have responded, if i were going'' i`d have responded, if i were going'

    >>> data['ids'][data['masks']==1]
    tensor([    0,  7974,     2,     2,   939, 12905,   417,    33,  2334,     6,
            114,   939,    58,   164,     2])

    >>> left, right = data['offsets'][7]
    >>> data['tweet'][left:right]
    ' have'

    >>> [data['tweet'][data['offsets'][n][0]:data['offsets'][n][1]] for n in range(data['start_idx'], data['end_idx']+1)]
    [' i', '`', 'd', ' have', ' responded', ',', ' if', ' i', ' were', ' going']
    """
    data = {}

    ids, masks, tweet, offsets = _get_input_data(row, tokenizer, max_len)
    data["ids"] = ids
    data["masks"] = masks
    data["tweet"] = tweet
    data["offsets"] = offsets

    if "selected_text" in data:
        start_idx, end_idx = _get_target_idx(row, tweet, offsets)
        data["start_idx"] = start_idx
        data["end_idx"] = end_idx

    return data



class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=96):
        self.df = df
        self.max_len = max_len

        tokenizer = RobertaTokenizer.from_pretrained("../input/roberta-base/")
        tokenizer.save_vocabulary(".")
        for i in ["bos", "cls", "eos"]:
            attribute = f"{i}_token_id"
            setattr(self, attribute, getattr(tokenizer, attribute))

        # vocab.json and merges.txt come from save_vocabulary above
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file="./vocab.json",
            merges_file="./merges.txt",
            lowercase=True,
            add_prefix_space=True,
        )

    def __getitem__(self, index):
        row = self.df.iloc[index]

        data = _process_row(row, self.tokenizer, self.max_len)

        return data

    def __len__(self):
        return len(self.df)


def get_train_val_loaders(df, train_idx, val_idx, batch_size=8):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    return dataloaders_dict


def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df), batch_size=batch_size, shuffle=False, num_workers=2
    )
    return loader


# # Model


class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()

        self.roberta = RobertaModel.from_pretrained(
            "../input/roberta-base", output_hidden_states=True
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.roberta.config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        _, _, hs = self.roberta(input_ids, attention_mask)

        x = torch.stack([hs[-1], hs[-2], hs[-3], hs[-4]])
        x = torch.mean(x, 0)
        x = self.dropout(x)
        x = self.fc(x)
        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


# # Loss Function


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss


# # Evaluation Function


def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0] : offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def compute_jaccard_score(
    text, start_idx, end_idx, start_logits, end_logits, offsets
):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)

    true = get_selected_text(text, start_idx, end_idx, offsets)

    return jaccard(true, pred)


# # Training Function


def loop_through_data_loader(
    model, dataloaders_dict, criterion, optimizer, num_epochs, epoch, phase
):
    epoch_loss = 0.0
    epoch_jaccard = 0.0

    for data in dataloaders_dict[phase]:
        ids = data["ids"].cuda()
        masks = data["masks"].cuda()
        tweet = data["tweet"]
        offsets = data["offsets"].numpy()
        start_idx = data["start_idx"].cuda()
        end_idx = data["end_idx"].cuda()

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):

            start_logits, end_logits = model(ids, masks)

            loss = criterion(start_logits, end_logits, start_idx, end_idx)

            if phase == "train":
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * len(ids)

            start_idx = start_idx.cpu().detach().numpy()
            end_idx = end_idx.cpu().detach().numpy()
            start_logits = (
                torch.softmax(start_logits, dim=1).cpu().detach().numpy()
            )
            end_logits = (
                torch.softmax(end_logits, dim=1).cpu().detach().numpy()
            )

            for i in range(len(ids)):
                jaccard_score = compute_jaccard_score(
                    tweet[i],
                    start_idx[i],
                    end_idx[i],
                    start_logits[i],
                    end_logits[i],
                    offsets[i],
                )
                epoch_jaccard += jaccard_score

    epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
    epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)

    print(
        "Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}".format(
            epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard
        )
    )

    return model



def train_one_epoch(
    model, dataloaders_dict, criterion, optimizer, num_epochs, epoch
):
    # train
    model.train()
    model = loop_through_data_loader(
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        num_epochs,
        epoch,
        "train",
    )

    # evaluate
    model.eval()
    model = loop_through_data_loader(
        model, dataloaders_dict, criterion, optimizer, num_epochs, epoch, "val"
    )

    return model



def train_model(
    model, dataloaders_dict, criterion, optimizer, num_epochs, filename
):
    model.cuda()

    for epoch in range(num_epochs):
        model = train_one_epoch(
            model, dataloaders_dict, criterion, optimizer, num_epochs, epoch
        )

    torch.save(model.state_dict(), filename)


# # Training

if __name__ == '__main__':

    seed = 42
    seed_everything(seed)

    num_epochs = 3
    batch_size = 32
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    train_df = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
    train_df["text"] = train_df["text"].astype(str)
    train_df["selected_text"] = train_df["selected_text"].astype(str)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_df, train_df.sentiment), start=1
    ):
        print(f"Fold: {fold}")

        model = TweetModel()
        optimizer = optim.AdamW(model.parameters(), lr=3e-5, betas=(0.9, 0.999))
        criterion = loss_fn
        dataloaders_dict = get_train_val_loaders(
            train_df, train_idx, val_idx, batch_size
        )

        train_model(
            model,
            dataloaders_dict,
            criterion,
            optimizer,
            num_epochs,
            f"roberta_fold{fold}.pth",
        )

    # # Inference

    test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
    test_df["text"] = test_df["text"].astype(str)
    test_loader = get_test_loader(test_df)
    predictions = []
    models = []
    for fold in range(skf.n_splits):
        model = TweetModel()
        model.cuda()
        model.load_state_dict(torch.load(f"roberta_fold{fold+1}.pth"))
        model.eval()
        models.append(model)

    for data in test_loader:
        ids = data["ids"].cuda()
        masks = data["masks"].cuda()
        tweet = data["tweet"]
        offsets = data["offsets"].numpy()

        start_logits = []
        end_logits = []
        for model in models:
            with torch.no_grad():
                output = model(ids, masks)
                start_logits.append(
                    torch.softmax(output[0], dim=1).cpu().detach().numpy()
                )
                end_logits.append(
                    torch.softmax(output[1], dim=1).cpu().detach().numpy()
                )

        start_logits = np.mean(start_logits, axis=0)
        end_logits = np.mean(end_logits, axis=0)
        for i in range(len(ids)):
            start_pred = np.argmax(start_logits[i])
            end_pred = np.argmax(end_logits[i])
            if start_pred > end_pred:
                pred = tweet[i]
            else:
                pred = get_selected_text(
                    tweet[i], start_pred, end_pred, offsets[i]
                )
            predictions.append(pred)

    # # Submission

    sub_df = pd.read_csv(
        "../input/tweet-sentiment-extraction/sample_submission.csv"
    )
    sub_df["selected_text"] = predictions
    sub_df["selected_text"] = sub_df["selected_text"].apply(
        lambda x: x.replace("!!!!", "!") if len(x.split()) == 1 else x
    )
    sub_df["selected_text"] = sub_df["selected_text"].apply(
        lambda x: x.replace("..", ".") if len(x.split()) == 1 else x
    )
    sub_df["selected_text"] = sub_df["selected_text"].apply(
        lambda x: x.replace("...", ".") if len(x.split()) == 1 else x
    )
    sub_df.to_csv("submission.csv", index=False)
    display(sub_df.head())
