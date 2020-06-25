import torch
import torch.nn as nn
import numpy as np

def compute_edge_length(vertices, faces):
    p1 = torch.index_select(vertices, 0, faces[:, 0])
    p2 = torch.index_select(vertices, 0, faces[:, 1])
    p3 = torch.index_select(vertices, 0, faces[:, 2])
    # get edge lentgh
    e1 = p2 - p1
    e2 = p3 - p1
    e3 = p2 - p3

    el1 = ((torch.sum(e1**2, 1))).mean()
    el2 = ((torch.sum(e2**2, 1))).mean()
    el3 = ((torch.sum(e3**2, 1))).mean()

    edge_length = (el1 + el2 + el3) / 6.
    return edge_length
