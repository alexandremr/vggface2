import pandas as pd
import csv
import os
import sys
import torch
import shutil
import pickle

#
# def load_state_dict(model, fname):
#     with open(fname, 'rb') as f:
#         weights = pickle.load(f, encoding='latin1')
#
#     own_state = model.state_dict()
#
#     for name, param in weights.items():
#         if name in own_state:
#             try:
#                 own_state[name].copy_(torch.from_numpy(param))
#             except Exception:
#                 raise RuntimeError(
#                     'While copying the parameter named {}, whose dimensions in the model are {} and whose ' \
#                     'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
#         else:
#             raise KeyError('unexpected key "{}" in state_dict'.format(name))
#

def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    #
    # for name, param in model.state_dict().items():
    #     print(name, param.size())

    # model_dict = model.state_dict()
    # model_keys = model_dict.keys()
    # counter = 0
    # for name, param in weights.items():
    #     for model_key in model_keys:
    #         if name in model_key:
    #             try:
    #                 model_dict[model_key].copy_(torch.from_numpy(param))
    #                 print('[✔] ', "%03d " % (counter,), name, ' |  ', list(torch.from_numpy(param).size()), ' | ', list(model_dict[model_key].size()),'  |')
    #                 break
    #             except Exception:
    #                 raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
    #                                    'dimensions in the checkpoint are {}.'.format(name, model_dict[name].size(), param.size()))
    #
    #     else:
    #         print('[✖] ', "%03d " % (counter,), name, ' |  ', ['x'], ' | ', ['x'],'  |')
    #     counter += 1
    weights.items()
    model_dict = model.state_dict()
    model_keys = model_dict.keys()
    counter = 0
    for model_key in model_keys:
        for name, param in weights.items():
            if name == model_key.replace('module.', ''):
                try:
                    model_dict[model_key].copy_(torch.from_numpy(param))
                    print('[✔] ', "%03d " % (counter,), name, model_key, ' |  ', list(torch.from_numpy(param).size()), ' | ', list(model_dict[model_key].size()),'  |')
                    break
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                       'dimensions in the checkpoint are {}.'.format(name, model_dict[name].size(), param.size()))

        else:
            if len(model_dict[model_key].size()) == 0:
                print('[✔] ', "%03d " % (counter,), '------------', model_key, ' |  ', ['x'], ' | ', list(model_dict[model_key].size()), '  |')
            else:
                print('[✖] ', "%03d " % (counter,), '------------', model_key, ' |  ', ['x'], ' | ', list(model_dict[model_key].size()),'  |')
        counter += 1