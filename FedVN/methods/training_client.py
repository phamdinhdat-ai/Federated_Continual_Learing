import numpy as np 
import copy

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils import data
import torch.utils
from utils.util_dataset import NewData


from .helper_function import set_client_from_params, get_acc_loss, get_mdl_bn_idx
from .helper_function import get_mdl_nonbn_idx, get_mdl_params
from .helper_function import avg_models


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')




def train_model(model, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, epoch, print_per, weight_decay,
                dataset_name, sch_step=1, sch_gamma=1):
    history = dict(
        loss = [], 
        acc = [], 
        val_loss = [], 
        val_acc = []
    )
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(NewData(trn_x, trn_y, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # model.train()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    testset = NewData(tst_x, tst_y)
    test_loader = torch.utils.data.DataLoader(testset, batch_size  = batch_size, shuffle = True)
    
    # Put tst_x=False if no tst data given
    
    # print_test = not isinstance(tst_x, bool)
    # print("=" *100 + "print the first test" + "="*100)
    # loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    # if print_test:
    #     loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
    #     print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
    #           % (0, acc_trn, loss_trn, acc_tst, loss_tst, optimizer.param_groups[0]['lr']))
    # else:
    #     print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f"
    #           % (0, acc_trn, loss_trn, optimizer.param_groups[0]['lr']))
    
    
    model.train()

    for e in range(epoch):
        # Training
        loss_overall = 0
        acc_overall = 0 
        
        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_x = batch_x.clone().detach().requires_grad_(True)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            loss = loss / list(batch_y.size())[0]
            
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm = 5)  # Clip gradients to prevent exploding
            optimizer.step()
          
            y_pred = y_pred.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct
            loss_overall += loss.item()
            
        loss_e = loss_overall / int(np.ceil(n_trn / batch_size))
        acc_e = acc_overall / n_trn
        
        model.eval()
        loss_t_overal = 0 
        acc_t_overal = 0
        # model = model.to(device)
        n_test = tst_x.shape[0]
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                y_pred = model(batch_x)
                # print("Model dtype: ",model.type)
                # print("Output dtype: ", y_pred.type)
                loss  = loss_fn(y_pred , batch_y.reshape(-1).long())
                y_pred = y_pred.detach().cpu().numpy()
                y_pred = np.argmax(y_pred, axis = 1).reshape(-1).astype(np.int32)
                batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
                batch_correct = np.sum(y_pred == batch_y)
                acc_t_overal += batch_correct / len(batch_y)
                loss_t_overal +=loss.item() 
                
            loss_t_e = loss_t_overal / (batch_size * len(test_loader))
            acc_t_e  = acc_t_overal / len(test_loader)
        history['loss'].append(loss_e)
        history['acc'].append(acc_e)
        history['val_loss'].append(loss_t_e)
        history['val_acc'].append(acc_t_e)
        
        print(f"Epoch: {e} | Loss: {loss_e} | Acc: {acc_e}  | Test loss: {loss_t_e} | Test Acc: {acc_t_e}")
            
            
            
                
        # if (e + 1) % print_per == 0:
        #     loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
        #     if print_test:
        #         loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        #         print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
        #               % (e + 1, acc_trn, loss_trn, acc_tst, loss_tst, optimizer.param_groups[0]['lr']))
        #     else:
        #         print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" % (
        #             e + 1, acc_trn, loss_trn, optimizer.param_groups[0]['lr']))

            # model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model, history