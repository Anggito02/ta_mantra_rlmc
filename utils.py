import torch

import numpy as np
import pandas as pd
from tqdm import trange
from collections import Counter

from sktime.performance_metrics.forecasting import \
    mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

DATA_DIR = 'dataset'
# SCALE_MEAN, SCALE_STD = np.load(f'{DATA_DIR}/scaler.npy')
# def inv_trans(x): return x * SCALE_STD + SCALE_MEAN

def get_model_info(error_array):
    model_rank = np.zeros_like(error_array)
    sort_res   = error_array.argsort(1)
    model_rank[1:] = sort_res[:-1]
    model_rank[0]  = sort_res[-1]
    return model_rank


def compute_mape_error(y, bm_preds):
    mape_loss_df = pd.DataFrame()
    for i in trange(bm_preds.shape[1], desc='[Compute Error]'):     # Compute learner dari tiap learner pakai iterasi dimensi ke 1
        model_mape_loss = [mean_absolute_percentage_error(
            y[j], bm_preds[j, i, :],
            symmetric=True) for j in range(len(y))]
        mape_loss_df[i] = model_mape_loss                           # Simpan mape error dari tiap learner di mape_loss_df
    return mape_loss_df

def compute_mape_error_new(y, bm_preds):
    mape_loss_np = []
    for i in trange(bm_preds.shape[1], desc='[Compute Error]'):     # Compute learner dari tiap learner pakai iterasi dimensi ke 1
        mape_loss_df_variate = [[] for _ in range(bm_preds.shape[-1])]
        for v in range(bm_preds.shape[-1]):
            model_mape_loss = [mean_absolute_percentage_error(
                y[j, :, v], bm_preds[j, i, :, v],
                symmetric=True) for j in range(len(y))]
            mape_loss_df_variate[v] = model_mape_loss                           # Simpan mape error dari tiap learner di mape_loss_df
        mape_loss_np.append(mape_loss_df_variate)
    mape_loss_np = np.array(mape_loss_np)
    mape_loss_np = mape_loss_np.reshape(mape_loss_np.shape[2], mape_loss_np.shape[0], mape_loss_np.shape[1])
    return mape_loss_np


def compute_mae_error(y, bm_preds):
    loss_df = pd.DataFrame()
    for i in trange(bm_preds.shape[1], desc='[Compute Error]'):
        model_mae_loss = [mean_absolute_error(
            y[j], bm_preds[j, i, :],
            symmetric=True) for j in range(len(y))]
        loss_df[i] = model_mae_loss
    return loss_df

def compute_mse_error(y, bm_preds):
    loss_df = pd.DataFrame()
    for i in trange(bm_preds.shape[1], desc='[Compute Error]'):
        model_mse_loss = [mean_squared_error(
            y[j], bm_preds[j, i, :],
            symmetric=True) for j in range(len(y))]
        loss_df[i] = model_mse_loss
    return loss_df

def unify_input_data():
    ### RAW DATASET PREPROCESS ###
    train_val_X   = np.load('dataset/train_val_X.npy')    # (62795, 120, 7)
    test_X        = np.load('dataset/test_X.npy')         # (6867, 120, 7)
    train_val_y   = np.load('dataset/train_val_y.npy')    # (62795, 24)
    test_y        = np.load('dataset/test_y.npy')         # (6867, 24)

    L = len(test_y)
    train_X = train_val_X[:-L]
    valid_X = train_val_X[-L:]
    train_y = train_val_y[:-L]
    valid_y = train_val_y[-L:]
    ### END RAW DATASET PREPROCESS ###

    # predictions
    MODEL_NAMES = ['lstm1', 'lstm2', 'gru1', 'gru2', 'cnn1', 'cnn2',
                   'transformer1', 'transformer2', 'repeat']            # Ini nanti berisi 1 learner Mantra
    merge_data = []

    ### GET ALL MODEL PREDICTIONS ###
    bm_train_preds = np.load('dataset/bm_train_preds.npz')      # Berupa value hasil loss function train dari semua learner Mantra, tetapi terpisah
    
    for model_name in MODEL_NAMES:
        model_pred = bm_train_preds[model_name]
        model_pred = np.expand_dims(model_pred, axis=1)     # (xxx, 24) -> (xxx, 1, 24)
        merge_data.append(model_pred)
    ### MERGE MODEL PREDICTIONS DATA ###
    merge_data = np.concatenate(merge_data, axis=1)  # (62795, 9, 24) = (xxx, 1, 24) + (xxx, 1 , 24) + ... + (xxx, 1, 24) 9x

    ### SEPARATE TRAINING AND VALIDATION DATA ###
    train_preds = merge_data[:-L]           # (:-L dari xxx, 9, 24)
    valid_preds = merge_data[-L:]           # (-L: dari xxx, 9, 24)
    np.save('dataset/bm_train_preds.npy', train_preds)      
    np.save('dataset/bm_valid_preds.npy', valid_preds)          # Berupa value hasil loss function validation dari 1 learner Mantra

    ### GET TEST DATA OF ALL MODELS ###
    merge_test_data = []
    bm_train_preds = np.load('dataset/bm_test_preds.npz')

    for model_name in MODEL_NAMES:
        model_pred = bm_train_preds[model_name]
        model_pred = np.expand_dims(model_pred, axis=1)
        merge_test_data.append(model_pred)
    test_preds = np.concatenate(merge_test_data, axis=1)  # (62795, 9, 24)
    np.save('dataset/bm_test_preds.npy', test_preds)            # Berupa value hasil loss function test dari 1 learner Mantra

    train_error_np = compute_mape_error(train_y, train_preds)
    valid_error_np = compute_mape_error(valid_y, valid_preds)
    test_error_np  = compute_mape_error(test_y , test_preds)

    np.savez('dataset/input.npz',
             train_X=train_X,
             valid_X=valid_X,
             test_X=test_X,
             train_y=train_y,
             valid_y=valid_y,
             test_y=test_y,
             train_error=train_error_np,
             valid_error=valid_error_np,
             test_error=test_error_np
            )
    
def unify_input_data_new(data_path):
    train_x = np.load(f'{data_path}/dataset/input_train_x.npy')    
    train_y = np.load(f'{data_path}/dataset/input_train_y.npy')
    vali_x  = np.load(f'{data_path}/dataset/input_vali_x.npy')
    vali_y  = np.load(f'{data_path}/dataset/input_vali_y.npy')
    test_x  = np.load(f'{data_path}/dataset/input_test_x.npy')
    test_y  = np.load(f'{data_path}/dataset/input_test_y.npy')

    # predictions
    merge_data = []
    train_preds_npz = np.load(f'{data_path}/rl_bm/bm_train_preds.npz')
    for model_name in train_preds_npz.keys():
        train_preds = train_preds_npz[model_name]
        train_preds = np.expand_dims(train_preds, axis=1)
        merge_data.append(train_preds)
    train_preds_merge_data = np.concatenate(merge_data, axis=1)

    merge_data = []
    valid_preds_npz = np.load(f'{data_path}/rl_bm/bm_vali_preds.npz')
    for model_name in valid_preds_npz.keys():
        valid_preds = valid_preds_npz[model_name]
        valid_preds = np.expand_dims(valid_preds, axis=1)
        merge_data.append(valid_preds)
    valid_preds_merge_data = np.concatenate(merge_data, axis=1)

    merge_data = []
    test_preds_npz = np.load(f'{data_path}/rl_bm/bm_test_preds.npz')
    for model_name in test_preds_npz.keys():
        test_preds = test_preds_npz[model_name]
        test_preds = np.expand_dims(test_preds, axis=1)
        merge_data.append(test_preds)
    test_preds_merge_data = np.concatenate(merge_data, axis=1)
    
    # save preds
    np.save(f'{data_path}/rl_bm/bm_train_preds.npy', train_preds_merge_data)
    np.save(f'{data_path}/rl_bm/bm_valid_preds.npy', valid_preds_merge_data)
    np.save(f'{data_path}/rl_bm/bm_test_preds.npy', test_preds_merge_data)

    train_error_df = compute_mse_error(train_y, train_preds_merge_data)
    valid_error_df = compute_mse_error(vali_y, valid_preds_merge_data)
    test_error_df  = compute_mse_error(test_y , test_preds_merge_data)

    np.savez(f'{data_path}/dataset/input_rl.npz',
             train_X=train_x,
             valid_X=vali_x,
             test_X=test_x,
             train_y=train_y,
             valid_y=vali_y,
             test_y=test_y,
             train_error=train_error_df,
             valid_error=valid_error_df,
             test_error=test_error_df
            )


def load_data(data_path):
    input_data = np.load(f'{data_path}/dataset/input_rl.npz')
    train_X = input_data['train_X']
    valid_X = input_data['valid_X']
    test_X  = input_data['test_X' ]
    train_y = input_data['train_y']
    valid_y = input_data['valid_y']
    test_y  = input_data['test_y' ]
    train_error = input_data['train_error']  # (55928, 9)
    valid_error = input_data['valid_error']  # (6867,  9)
    test_error  = input_data['test_error' ]  # (6867,  9)
    return (train_X, valid_X, test_X, train_y, valid_y, test_y,
            train_error, valid_error, test_error)


def plot_best_data(train_error, valid_error, test_error):
    import matplotlib.pyplot as plt
    train_min_model = Counter(train_error.argmin(1))
    valid_min_model = Counter(valid_error.argmin(1))
    test_min_model  = Counter(test_error.argmin(1))

    train_best_num = [train_min_model[i] for i in range(9)]
    valid_best_num = [valid_min_model[i] for i in range(9)]
    test_best_num  = [test_min_model[i] for i in range(9)]

    labels  = [f'M{i}' for i in range(1, 10)]

    fig, ax = plt.subplots()
    width = 0.3
    x = np.arange(len(labels)) * 1.5
    _ = ax.bar(x - width, train_best_num, width, label='train')
    _ = ax.bar(x, valid_best_num, width, label='valid')
    _ = ax.bar(x + width, test_best_num, width, label='test')

    ax.set_title('Jena Climate Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.savefig('jena_best_model.png', dpi=300)


############
# evaluate #
############
def evaluate_agent(agent, test_states, test_bm_preds, test_y):
    with torch.no_grad():
        weights = agent.select_action(test_states)  # (2816, 9)
    act_counter = Counter(weights.argmax(1))
    act_sorted  = sorted([(k, v) for k, v in act_counter.items()])
    weights = np.expand_dims(weights, -1)  # (2816, 9, 1)
    weighted_y = weights * test_bm_preds  # (2816, 9, 24)
    weighted_y = weighted_y.sum(1)  # (2816, 24)
    mse_loss = mean_squared_error(test_y, weighted_y)
    mae_loss = mean_absolute_error(test_y, weighted_y)
    mape_loss = mean_absolute_percentage_error(test_y, weighted_y)
    return mse_loss, mae_loss, mape_loss, act_sorted

def evaluate_agent_test(agent, test_states, test_bm_preds, test_y):
    with torch.no_grad():
        weights = agent.select_action(test_states)  # (2816, 9)
    weights = np.expand_dims(weights, -1)  # (2816, 9, 1)
    weighted_y = weights * test_bm_preds  # (2816, 9, 24)
    weighted_y = weighted_y.sum(1)  # (2816, 24)

    return weighted_y

def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    # print(pred)
    # print(true)
    # print('test shape:', pred.shape, true.shape)
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe