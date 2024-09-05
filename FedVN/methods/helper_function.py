
import torch
import numpy as np
import copy
from utils.util_dataset import NewData
from torch.utils import data


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.state_dict()))
    idx = 0
    for name, param in dict(mdl.state_dict()).items():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl


def get_mdl_params(model_list, n_par=None):
    if n_par is None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in dict(exp_mdl.state_dict()).items():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in dict(mdl.state_dict()).items():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

def get_mdl_nonbn_idx(model_list, name_par=None):
    if name_par is None:
        exp_mdl = model_list[0]
        name_par = []
        for name, param in exp_mdl.named_parameters():
            name_par.append(name)

    idx_list = [[]] * len(model_list)
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in dict(mdl.state_dict()).items():
            temp = param.data.cpu().numpy().reshape(-1)
            if name in name_par:
                idx_list[i].extend(list(range(idx, idx + len(temp))))
            idx += len(temp)
    return np.array(idx_list)

def get_mdl_bn_idx(model_list, name_par=None):
    if name_par is None:
        exp_mdl = model_list[0]
        name_par = []
        for name, param in exp_mdl.named_parameters():
            name_par.append(name)

    idx_list = [[]] * len(model_list)
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in dict(mdl.state_dict()).items():
            temp = param.data.cpu().numpy().reshape(-1)
            if name not in name_par:
                idx_list[i].extend(list(range(idx, idx + len(temp))))
            idx += len(temp)
    return np.array(idx_list)



# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name, w_decay=None):
    acc_overall = 0
    loss_overall = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    # batch_size = min(6000, data_x.shape[0])
    batch_size = min(2000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(NewData(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)

            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_overall += loss.item()

            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct

    loss_overall /= n_tst
    if w_decay is not None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay / 2 * np.sum(params * params)

    model.train()
    return loss_overall, acc_overall / n_tst


# --- Helper functions

def avg_models(mdl, clnt_models, weight_list):
    n_node = len(clnt_models)
    dict_list = list(range(n_node))
    for i in range(n_node):
        dict_list[i] = copy.deepcopy(dict(clnt_models[i].state_dict()))

    param_0 = clnt_models[0].state_dict()

    for name, param in param_0:
        param_ = weight_list[0] * param.data
        for i in list(range(1, n_node)):
            param_ = param_ + weight_list[i] * dict_list[i][name].data
        dict_list[0][name].data.copy_(param_)

    mdl.load_state_dict(dict_list[0])

    # Remove dict_list from memory
    del dict_list

    return mdl