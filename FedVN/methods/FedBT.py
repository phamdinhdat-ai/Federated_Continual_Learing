import copy 
import os 
from pathlib import Path
import numpy as np

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter

from .training_client import train_model
from dataset.BreathDataset import BreathDataset
from utils.util_dataset import NewData
  
from .helper_function import get_acc_loss, get_mdl_bn_idx, get_mdl_params, get_mdl_nonbn_idx
from .helper_function import set_client_from_params

device  = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')



def train_Fed_common(data_obj, act_prob, learning_rate, batch_size, epoch,
                     com_amount, print_per, weight_decay,
                     model_func, init_model, sch_step, sch_gamma,
                     save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1):
    
    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f' % (
    save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed
    
    n_clnts = data_obj.n_clients
    print("Clients dataset shape: ", data_obj.clnt_x.shape)
    clnt_x =  data_obj.clnt_x.mean(axis  = 3)
    clnt_y =  data_obj.clnt_y

    
    
    
    cent_x = np.concatenate(clnt_x, axis = 0)
    cent_y = np.concatenate(clnt_y, axis = 0)
    
    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnts)])
    weight_list = weight_list.reshape((n_clnts, 1))

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))
    fed_mdls_all = list(range(n_save_instances))

    trn_perf_sel = np.zeros((com_amount, 2))
    trn_perf_all = np.zeros((com_amount, 2))
    tst_perf_sel = np.zeros((com_amount, 2))
    tst_perf_all = np.zeros((com_amount, 2))
    
    
    tst_x = torch.tensor(data_obj.tst_x.mean(axis = 2))
    tst_y = data_obj.tst_y
    

    
    n_par = len(get_mdl_params([model_func])[0])

    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnts).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    
    print(clnt_params_list.shape)
    
    saved_itr = -1 
    
    writer = SummaryWriter('%sRuns/%s/%s' % (data_path, data_obj.name, suffix))
    print("Set up Oke !!!! ")
    if not trial:
        
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i
                
                fed_model = model_func
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                
                fed_model.to(device)
                
                #Freeze model 
                for params in fed_model.parameters():
                    params.requires_grad = False
                    
                
                fed_mdls_sel[saved_itr // save_period] = fed_model

                ###
                fed_model = model_func
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_all.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr // save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    trn_perf_sel[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_perf_all[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_trn_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    tst_perf_sel[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_perf_all[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))

    if trial or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, com_amount))):
        # clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func.to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.state_dict())))

            all_model = model_func.to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.state_dict())))

        else:
            # Load recent one
            avg_model = model_func.to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

            all_model = model_func.to(device)
            all_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_all.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            # Fix randomness
            inc_seed = 0
            while True:
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnts)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))
            clnt_models = list(range(n_clnts))
            for clnt in selected_clnts:
                print("---------Training Clients %d -----------" % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                print(f"Data shape train for client {clnt}: ", trn_x.shape)
                print(f"Label shape train for client {clnt}: ", trn_y.shape)
                
                print(f"Label shape test for client {clnt}: ", tst_x.shape)
                print(f"Label shape test for client {clnt}: ", tst_y.shape)
            
                
                
                clnt_models[clnt]  = model_func.to(device)
                clnt_models[clnt].load_state_dict(copy.deepcopy(avg_model.state_dict()))
                
                for param in clnt_models[clnt].parameters():
                    param.requires_grad = True
                    
                clnt_models[clnt] = train_model(model=clnt_models[clnt],
                                                trn_x=trn_x,
                                                trn_y=trn_y, 
                                                tst_x=tst_x, 
                                                tst_y=tst_y,
                                                learning_rate=learning_rate,
                                                batch_size=batch_size, 
                                                epoch=epoch, 
                                                print_per=print_per, 
                                                weight_decay=weight_decay, 
                                                dataset_name="Breathing", 
                                                sch_gamma=1, 
                                                sch_step=1)
                
                clnt_params_list[clnt] = get_mdl_params([clnt_models[clnt]], n_par=n_par)[0]
                
                
            # set the client with new params
            # # Scale with weights

            avg_model = set_client_from_params(model_func, np.sum(
                clnt_params_list[selected_clnts] * weight_list[selected_clnts] / np.sum(weight_list[selected_clnts]),
                axis=0))
            all_model = set_client_from_params(model_func,
                                               np.sum(clnt_params_list * weight_list / np.sum(weight_list), axis=0))

            ###
            loss_tst, acc_tst = get_acc_loss(tst_x, tst_y,
                                             avg_model, data_obj.dataset_name, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            print("**** Communication sel %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             avg_model, data_obj.dataset_name, 0)
            trn_perf_sel[i] = [loss_tst, acc_tst]
            print("**** Communication sel %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            ###
            loss_tst, acc_tst = get_acc_loss(tst_x, tst_y,
                                             all_model, data_obj.dataset_name, 0)
            tst_perf_all[i] = [loss_tst, acc_tst]

            print("**** Communication all %3d, Test Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            ###
            loss_tst, acc_tst = get_acc_loss(cent_x, cent_y,
                                             all_model, data_obj.dataset_name, 0)
            trn_perf_all[i] = [loss_tst, acc_tst]
            print("**** Communication all %3d, Cent Accuracy: %.4f, Loss: %.4f"
                  % (i + 1, acc_tst, loss_tst))

            writer.add_scalars('Loss/train_wd',
                               {
                                   'Sel clients':
                                       get_acc_loss(cent_x, cent_y, avg_model, data_obj.dataset_name, weight_decay)[0],
                                   'All clients':
                                       get_acc_loss(cent_x, cent_y, all_model, data_obj.dataset_name, weight_decay)[0]
                               }, i
                               )

            writer.add_scalars('Loss/train',
                               {
                                   'Sel clients': trn_perf_sel[i][0],
                                   'All clients': trn_perf_all[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/train',
                               {
                                   'Sel clients': trn_perf_sel[i][1],
                                   'All clients': trn_perf_all[i][1]
                               }, i
                               )

            writer.add_scalars('Loss/test',
                               {
                                   'Sel clients': tst_perf_sel[i][0],
                                   'All clients': tst_perf_all[i][0]
                               }, i
                               )

            writer.add_scalars('Accuracy/test',
                               {
                                   'Sel clients': tst_perf_sel[i][1],
                                   'All clients': tst_perf_all[i][1]
                               }, i
                               )

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))
                torch.save(all_model.state_dict(), '%sModel/%s/%s/%dcom_all.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_perf_sel[:i + 1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_sel[:i + 1])

                np.save('%sModel/%s/%s/%dcom_trn_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_perf_all[:i + 1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_all[:i + 1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)

                if (i + 1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%s/%s/%dcom_trn_perf_all.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_all.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))


            if ((i + 1) % save_period == 0):
                fed_mdls_sel[i // save_period] = avg_model
                fed_mdls_all[i // save_period] = all_model
                
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all
