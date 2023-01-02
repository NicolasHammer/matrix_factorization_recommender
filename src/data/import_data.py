"""Import data to train/evaluate the recommendation systems on."""
from enum import Enum
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from pandas import DataFrame
from torch.utils.data import DataLoader, TensorDataset


class FeedbackMode(Enum):
    """The different tpyes of feedback for the dataset to provide."""

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"


class SplitMode(Enum):
    """The different types of splits on the MovieLens 100k dataset."""

    SEQ_AWARE = "seq-aware"
    RANDOM = "random"


def split_data_ml100k(
    data: DataFrame,
    num_users: int,
    split_mode: SplitMode = SplitMode.RANDOM,
    test_ratio: float = 0.1,
) -> Tuple[DataFrame, DataFrame]:
    """Split the dataset in random mode or sequence aware mode.

    Args:
        data: The MovieLens 100k dataset.
        num_users: The number of users in the dataset.
        split_mode: The type of split used for the MovieLens 100k dataset.
        test_ratio: The ratio of train to test data. This value should
            fall between 0.0 and 1.0.

    Return:
        train_data: The data to train the recommendation systems on.
        test-data: The data to test the recommendation systems on.
    """
    if split_mode == SplitMode.SEQ_AWARE:
        train_items, test_items, train_list = {}, {}, []

        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)

        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=(lambda k: k[3])))

        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = DataFrame(train_data)
        test_data = DataFrame(test_data)
    elif split_mode == SplitMode.RANDOM:
        mask: List[bool] = [
            x == 1 for x in np.random.uniform(0, 1, len(data)) < 1 - test_ratio
        ]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    else:
        return ValueError(
            f"split_mode must be {SplitMode.RANDOM} or {SplitMode.SEQ_AWARE}"
        )

    return train_data, test_data


def load_data_ml100k(
    data: DataFrame,
    num_users: int,
    num_items: int,
    feedback: FeedbackMode = FeedbackMode.EXPLICIT,
) -> Tuple[List[int], List[int], List[int], Union[np.ndarray, Dict[int, List[int]]]]:
    """Load the MovieLens 100k dataset.

    Args:
        data: The data to load.
        num_users: The number of users in the dataset.
        num_items: The number of items in the dataset.
        feedback: The type of feedback in the dataset.

    Returns:
        users: A list of user indices.
        items: A list of item indices that correspond with the user indices.
        scores: A list of scores given by a user for a particular item.
        inter: If feedback is explicit, then return an array of scores. If
            feedback is implicit, then return a dictionary of user indicies
            mapped to a list of item indices.
    """
    users, items, scores = [], [], []
    inter = (
        np.zeros((num_items, num_users)) if feedback == FeedbackMode.EXPLICIT else {}
    )
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == FeedbackMode.EXPLICIT else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == FeedbackMode.IMPLICIT:
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score

    return users, items, scores, inter


# Split and load the dataset
def split_and_load_ml100k(
    data: DataFrame,
    num_users: int,
    num_items: int,
    device: torch.device,
    split_mode: SplitMode = SplitMode.SEQ_AWARE,
    feedback: FeedbackMode = FeedbackMode.EXPLICIT,
    test_ratio: float = 0.1,
    batch_size: int = 256,
):
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio
    )
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback
    )
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback
    )
    train_set = TensorDataset(
        torch.from_numpy(np.array(train_u)).to(device=device),
        torch.from_numpy(np.array(train_i)).to(device=device),
        torch.from_numpy(np.array(train_r)).to(device=device),
    )
    test_set = TensorDataset(
        torch.from_numpy(np.array(test_u)).to(device=device),
        torch.from_numpy(np.array(test_i)).to(device=device),
        torch.from_numpy(np.array(test_r)).to(device=device),
    )
    train_iter = DataLoader(train_set, shuffle=True, batch_size=batch_size)
    test_iter = DataLoader(test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter
