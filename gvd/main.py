import argparse
import json
import math
import os
import random
import typing as typ

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import csaf.config as cconf
import csaf.trace as ctr

RunsType = typ.List[typ.Union[typ.Tuple[bool, ctr.TimeTrace], Exception]]
sns.set()

color_codes = {'1': 'brown', '2': 'darkred', '3': 'red', '4': 'salmon', '5': 'orangered', '6': 'sienna',
               '7': 'saddlebrown', '8': 'sandybrown', '9': 'peru', '10': 'darkorange', '11': 'orange',
               '12': 'goldenrod', '13': 'gold', '14': 'khaki', '15': 'darkkhaki', '16': 'olive', '17': 'yellow',
               '18': 'yellowgreen', '19': 'chartreuse', '20': 'lightgreen', '21': 'darkgreen', '22': 'lime',
               '23': 'springgreen', '24': 'turquoise', '25': 'darkslategrey', '26': 'cyan', '27': 'lightblue',
               '28': 'deepskyblue', '29': 'steelblue', '30': 'dodgerblue', '31': 'slategrey', '32': 'royalblue',
               '33': 'navy', '34': 'blue', '35': 'indigo', '36': 'darkviolet', '37': 'plum', '38': 'magenta',
               '39': 'hotpink', '40': 'pink'}


class RecurrentIQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, gru_size, quantile_embedding_dim, num_quantile_sample, device,
                 fc1_units=32, fc2_units=64, fc3_units=32):
        super(RecurrentIQN, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.gru_size = gru_size
        self.quantile_embedding_dim = quantile_embedding_dim
        self.num_quantile_sample = num_quantile_sample
        self.device = device

        self.gru = nn.GRUCell(num_inputs, gru_size)
        self.post_gru = nn.Linear(gru_size, fc1_units)
        self.fc = nn.Linear(fc1_units, num_outputs)

        self.phi = nn.Linear(self.quantile_embedding_dim, 32)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, state, hx, tau, num_quantiles):
        input_size = state.size()[0]  # batch_size(train) or 1(get_action)
        tau = tau.expand(input_size * num_quantiles, self.quantile_embedding_dim)
        pi_mtx = torch.Tensor(np.pi * np.arange(0, self.quantile_embedding_dim)).expand(input_size * num_quantiles,
                                                                                        self.quantile_embedding_dim)
        cos_tau = torch.cos(tau * pi_mtx).to(self.device)

        phi = self.phi(cos_tau)
        phi = F.relu(phi)

        state_tile = state.expand(input_size, num_quantiles, self.num_inputs)
        state_tile = state_tile.flatten().view(-1, self.num_inputs).to(self.device)

        ghx = self.gru(state_tile, hx)
        x = self.post_gru(ghx)
        x = self.fc(x * phi)

        z = x.view(-1, num_quantiles, self.num_outputs)

        z = z.transpose(1, 2)  # [input_size, num_output, num_quantile]
        return z, ghx

    @classmethod
    def train_model(cls, model, optimizer, hx, states, actions, target, batch_size, num_tau_sample, device):
        tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
        states = states.reshape(states.shape[0], 1, -1)
        # states_actions = torch.cat((states, actions.unsqueeze(1)), 2)
        # z_a, hx = model(states_actions, hx, tau, num_tau_sample)
        z_a, hx = model(states, hx, tau, num_tau_sample)
        z_a = torch.mean(z_a, dim=1)
        T_z = target.to(device).unsqueeze(1).expand(-1, num_tau_sample)

        error_loss = T_z - z_a
        huber_loss = F.smooth_l1_loss(z_a, T_z.detach(), reduction='none')
        tau = torch.arange(0, 1, 1 / num_tau_sample).view(1, num_tau_sample)

        loss = (tau.to(device) - (error_loss < 0).float()).abs() * huber_loss
        loss = loss.sum(dim=1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, hx

    @classmethod
    def eval_model(cls, model, hx, states, actions, target, batch_size, num_tau_sample, device):
        tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
        states = states.reshape(states.shape[0], 1, -1)
        # states_actions = torch.cat((states, actions.unsqueeze(1)), 2)
        # z_a, hx = model(states_actions, hx, tau, num_tau_sample)
        z_a, hx = model(states, hx, tau, num_tau_sample)
        z_a = torch.mean(z_a, dim=1)
        T_z = target.to(device).unsqueeze(1).expand(-1, num_tau_sample)

        error_loss = T_z - z_a
        huber_loss = F.smooth_l1_loss(z_a, T_z.detach(), reduction='none')
        tau = torch.arange(0, 1, 1 / num_tau_sample).view(1, num_tau_sample)

        loss = (tau.to(device) - (error_loss < 0).float()).abs() * huber_loss
        loss = loss.sum(dim=1).mean()
        return loss, hx

    @classmethod
    def test_model(cls, model, hx, states, actions, target, batch_size, num_tau_sample, device):
        tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
        states = states.reshape(states.shape[0], 1, -1)
        # states_actions = torch.cat((states, actions.unsqueeze(1)), 2)
        # z_a, hx = model(states_actions, hx, tau, num_tau_sample)
        z_a, hx = model(states, hx, tau, num_tau_sample)
        z_a = torch.mean(z_a, dim=1)
        T_z = target.to(device).unsqueeze(1).expand(-1, num_tau_sample)

        error_loss = T_z - z_a
        huber_loss = F.smooth_l1_loss(z_a, T_z.detach(), reduction='none')
        tau = torch.arange(0, 1, 1 / num_tau_sample).view(1, num_tau_sample)

        loss = (tau.to(device) - (error_loss < 0).float()).abs() * huber_loss
        loss = loss.sum(dim=1).mean()
        return z_a.squeeze(0), loss, hx

    @classmethod
    def feed_forward(cls, model, hx, states, batch_size, num_tau_sample):
        tau = torch.Tensor(np.random.rand(batch_size * num_tau_sample, 1))
        states = states.reshape(states.shape[0], 1, -1)
        # states_actions = torch.cat((states, actions.unsqueeze(1)), 2)
        # z_a, hx = model(states_actions, hx, tau, num_tau_sample)
        z_a, hx = model(states, hx, tau, num_tau_sample)
        z_a = torch.mean(z_a, dim=1)
        return z_a.squeeze(0), hx


def get_action(state, target_net, epsilon, env, num_quantile_sample):
    if np.random.rand() <= epsilon:
        return env.action_space.sample(), None
    else:
        action, z = target_net.get_action(state, num_quantile_sample)
        return action, z


def construct_gvd_data_undiscounted(input_len, dataset, batch_size, horizon, device, gvd_name, sum_type):
    states, actions = [], []
    episodes_states = []
    episodes_actions = []
    episodes_returns = []
    episodes_len = []
    all_states = []
    assert len(dataset) != 0, "Memory is empty!"
    for i, data in enumerate(dataset):
        for j in range(len(data['time'])):
            all_states.append(np.array(data['state'][j]))

    for i, data in enumerate(dataset):
        for j in range(len(data['time'])):
            states.append(np.array(data['state'][j]))
            actions.append(np.array(data['actions'][j]))

        normalized_states = (np.array(states) - np.array(all_states).min(axis=0)) / (np.array(all_states).max(axis=0) - np.array(all_states).min(axis=0))
        returns = np.zeros(normalized_states.shape[0])
        for j in range(normalized_states.shape[0]):
            feature_index = int(gvd_name)
            if sum_type == "delta":
                returns[j] = sum(np.diff(normalized_states[j: j + horizon + 1, feature_index]))
            elif sum_type == "abs_delta":
                returns[j] = sum(abs(np.diff(normalized_states[j: j + horizon + 1, feature_index])))
            elif sum_type == "time_avg":
                returns[j] = sum(np.diff(normalized_states[j: j + horizon + 1, feature_index])) / len(normalized_states[j: j + horizon, feature_index])
            else:
                assert False, "Undefined/unknown method given to calculate the target return for GVDs. Notice the" \
                              " given arguments!"

        episodes_states.append(states)
        episodes_actions.append(actions)
        episodes_returns.append(returns)
        episodes_len.append(len(states))
        states = []
        actions = []

    max_len = len(max(episodes_states, key=len))
    for i, _ in enumerate(episodes_states):
        episodes_states[i] = np.concatenate((episodes_states[i], np.zeros((max_len - len(episodes_states[i]), input_len))), axis=0)
        episodes_actions[i] = np.concatenate((episodes_actions[i], np.zeros((max_len - len(episodes_actions[i]), num_actions))), axis=0)
        episodes_returns[i] = np.concatenate((episodes_returns[i], np.zeros((max_len - len(episodes_returns[i])))), axis=0)

        episodes_states[i] = torch.Tensor(episodes_states[i]).to(device)
        episodes_actions[i] = torch.Tensor(episodes_actions[i]).to(device)
        episodes_returns[i] = torch.Tensor(episodes_returns[i]).to(device)

    episodes_states = torch.stack(episodes_states)
    episodes_actions = torch.stack(episodes_actions)
    episodes_returns = torch.stack(episodes_returns)
    episodes_len = torch.Tensor(episodes_len).to(device)[:, None, None]

    tensor_dataset = torch.utils.data.TensorDataset(episodes_states, episodes_actions, episodes_returns, episodes_len)
    all_indices = np.arange(len(episodes_states))
    np.random.shuffle(all_indices)
    train_indices = all_indices[:int(len(all_indices) * 90 / 100)]
    test_indices = all_indices[int(len(all_indices) * 90 / 100):]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_dl = DataLoader(tensor_dataset, batch_size, sampler=train_sampler)
    test_dl = DataLoader(tensor_dataset, batch_size, sampler=test_sampler)
    return train_dl, test_dl, max_len


def construct_test_gvd_data_undiscounted(input_len, dataset, train_dataset, batch_size, horizon, device, sum_type):
    states, actions = [], []
    episodes_states = []
    episodes_actions = []
    episodes_returns = []
    episodes_len = []
    all_training_states = []
    assert len(dataset) != 0, "Memory is empty!"

    for i, data in enumerate(train_dataset):
        for j in range(len(data['time'])):
            all_training_states.append(np.array(data['state'][j]))

    for j in range(len(dataset['time'])):
        states.append(np.array(dataset['state'][j]))
        actions.append(np.array(dataset['actions'][j]))

    normalized_states = (np.array(states) - np.array(all_training_states).min(axis=0)) / (np.array(all_training_states).max(axis=0) - np.array(all_training_states).min(axis=0))
    returns = []
    for j in range(normalized_states.shape[0]):
        if sum_type == "delta":
            returns.append(sum(np.diff(normalized_states[j: j + horizon + 1], axis=0)))
        elif sum_type == "abs_delta":
            returns.append(sum(abs(np.diff(normalized_states[j: j + horizon + 1], axis=0))))
        elif sum_type == "time_avg":
            if j == normalized_states.shape[0] - 1:
                returns.append(np.zeros(input_len))
            else:
                returns.append(sum(np.diff(normalized_states[j: j + horizon + 1], axis=0)) / len(normalized_states[j: j + horizon]))
        else:
            assert False, "Undefined/unknown method given to calculate the target return for GVDs. Notice the" \
                          " given arguments!"

    episodes_states.append(states)
    episodes_actions.append(actions)
    episodes_returns.append(np.array(returns))
    episodes_len.append(len(states))

    max_len = len(max(episodes_states, key=len))
    episodes_states[0] = torch.Tensor(episodes_states[0]).to(device)
    episodes_actions[0] = torch.Tensor(episodes_actions[0]).to(device)
    episodes_returns[0] = torch.Tensor(episodes_returns[0]).to(device)

    episodes_states = torch.stack(episodes_states)
    episodes_actions = torch.stack(episodes_actions)
    episodes_returns = torch.stack(episodes_returns)
    episodes_len = torch.Tensor(episodes_len).to(device)[:, None, None]

    tensor_dataset = torch.utils.data.TensorDataset(episodes_states, episodes_actions, episodes_returns, episodes_len)
    all_indices = np.arange(len(episodes_states))
    test_sampler = SubsetRandomSampler(all_indices)
    test_dl = DataLoader(tensor_dataset, batch_size, sampler=test_sampler)
    return test_dl, max_len


def plot_gvd_data_undiscounted(input_len, train_dataset, dataset, result_folder, horizon, sum_type):
    states = []
    episodes_returns = {}
    all_training_states = []
    for i in range(input_len):
        episodes_returns[i] = []
    assert len(dataset) != 0, "Memory is empty!"

    for i, data in enumerate(train_dataset):
        for j in range(len(data['time'])):
            all_training_states.append(np.array(data['state'][j]))

    for j in range(len(dataset['time'])):
        states.append(np.array(dataset['state'][j]))
    normalized_states = (np.array(states) - np.array(all_training_states).min(axis=0)) / (np.array(all_training_states).max(axis=0) - np.array(all_training_states).min(axis=0))
    for j in range(normalized_states.shape[0]):
        for k in range(input_len):
            feature_index = k
            if sum_type == "delta":
                episodes_returns[feature_index].append(sum(np.diff(normalized_states[j: j + horizon + 1, feature_index])))
            elif sum_type == "abs_delta":
                episodes_returns[feature_index].append(sum(abs(np.diff(normalized_states[j: j + horizon + 1, feature_index]))))
            elif sum_type == "time_avg":
                episodes_returns[feature_index].append(sum(np.diff(normalized_states[j: j + horizon + 1, feature_index])) / len(normalized_states[j: j + horizon, feature_index]))
            else:
                assert False, "Undefined/unknown method given to calculate the target return for GVDs. Notice the" \
                          " given arguments!"

    fig, axs = plt.subplots(math.ceil(input_len / 3), 3, figsize=(20, 20))
    r, c = 0, 0
    for i in range(input_len):
        axs[r, c].plot(episodes_returns[i], color='teal')
        axs[r, c].set(xlabel='samples', ylabel='feature values')
        axs[r, c].set_title("Feature: " + str(i))
        if r < math.ceil(input_len / 3) - 1:
            r += 1
        else:
            c += 1
            r = 0
    # fig.show()
    fig.suptitle("Feature values\n")
    fig.tight_layout()
    fig.savefig(os.path.join(result_folder, "feature_values.png"))


def learn_undiscounted(model, optimizer, memory, max_len, gru_size, num_tau_sample, device):
    total_loss = 0
    count = 0
    model.train()

    for s_batch, a_batch, mc_returns, _ in memory:
        h_gvfs = None
        for i in range(max_len):
            s, a, mc_return = s_batch[:, i, :], a_batch[:, i, :], mc_returns[:, i]
            if h_gvfs is None:
                h_gvfs = torch.zeros(len(s_batch) * num_tau_sample, gru_size)
            loss, h_gvfs = RecurrentIQN.train_model(model, optimizer, h_gvfs.detach().to(device), s, a, mc_return, len(s_batch), num_tau_sample, device)
            total_loss += loss
            count += 1

    return total_loss / count


def evaluation_undiscounted(args, model, memory, max_len, gru_size, num_tau_sample, device, best_gvf_total_loss, is_test=False):
    total_loss = 0
    count = 0
    model.eval()
    for s_batch, a_batch, mc_returns, _ in memory:
        h_gvfs = None
        for i in range(max_len):
            s, a, mc_return = s_batch[:, i, :], a_batch[:, i, :], mc_returns[:, i]
            if h_gvfs is None:
                h_gvfs = torch.zeros(len(s_batch) * num_tau_sample, gru_size)

            loss, h_gvfs = RecurrentIQN.eval_model(model, h_gvfs.detach().to(device), s, a, mc_return, len(s_batch), num_tau_sample, device)
            total_loss += loss
            count += 1
    if not is_test:
        print("GVD avg loss is:", round(total_loss.item() / count, 3))
        if total_loss.item() / count <= best_gvf_total_loss:
            print("Saving the best model!")
            best_gvf_total_loss = total_loss.item() / count
            save_model(model, "./models/" + args.env_name + "/" + args.gvd_name + "_gvd_" + args.target_return_type + "_h_" + str(args.horizon[0]) + ".pt")
    return round(total_loss.item() / count, 3), best_gvf_total_loss


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)


def plot_losses(train_loss, test_loss, result_folder, horizon, gvd_name, info, bootstrapped):
    plt.plot(train_loss, label="training loss")
    plt.plot(test_loss, label="test loss")
    plt.legend()
    if not bootstrapped:
        plt.savefig(os.path.join(result_folder, "losses_" + gvd_name + "_" + info + "_h" + str(horizon) + ".png"))
    else:
        plt.savefig(os.path.join(result_folder, "losses_" + gvd_name + "_" + info + "_h" + str(horizon) + "_bootstrap.png"))
    plt.clf()


def update_recurrent_gvds(args, train_memory, test_memory, r_gvd_model, optimizer, device, horizon, max_len):
    all_train_losses, all_test_losses = [], []
    best_gvf_total_loss = float("inf")
    for i in range(args.num_iterations):
        total_loss = learn_undiscounted(r_gvd_model, optimizer, train_memory, max_len, args.gru_units, args.num_tau_sample, device)
        if i % args.test_interval == 0:
            print("train loss : {}".format(total_loss))
            all_train_losses.append(total_loss)
            avg_eval_loss, best_gvf_total_loss = evaluation_undiscounted(args, r_gvd_model, test_memory, max_len, args.gru_units, args.num_tau_sample, device, best_gvf_total_loss)
            all_test_losses.append(avg_eval_loss)
            plot_losses(all_train_losses, all_test_losses, os.path.join(iqn_base_path, args.env_name), horizon,
                        args.gvd_name, args.target_return_type, bootstrapped=False)


def test_undiscounted(model, memory, max_len, gru_size, num_tau_sample, device, gvd_name):
    total_loss = 0
    count = 0
    model.eval()
    dists = []
    mcs = []
    for s_batch, a_batch, mc_returns, _ in memory:
        h_gvfs = None
        for i in range(max_len):
            s, a, mc_return = s_batch[:, i, :], a_batch[:, i, :], mc_returns[:, i, gvd_name]
            if h_gvfs is None:
                h_gvfs = torch.zeros(len(s) * num_tau_sample, gru_size)

            distributional_return, loss, h_gvfs = RecurrentIQN.test_model(model, h_gvfs.detach().to(device), s, a, mc_return, len(s), num_tau_sample, device)
            dists.append(distributional_return.squeeze(0).detach().cpu().numpy())
            mcs.append(mc_return.item())
            total_loss += loss
            count += 1
    return mcs, dists


def local_outlier_factor(distribution, actual_return):
    lof = LocalOutlierFactor(n_neighbors=8)
    lof.fit_predict(np.append(distribution, actual_return).reshape(-1, 1))
    score = abs(lof.negative_outlier_factor_[-1])
    return score


def k_nearest_neighbors(distribution, actual_return):
    neigh = NearestNeighbors(n_neighbors=8)
    neigh.fit(distribution.reshape(-1, 1))
    distances, indices = neigh.kneighbors(np.array(actual_return).reshape(-1, 1))
    return distances.sum()


def isolation_forest(distribution, actual_return):
    clf = IsolationForest(n_estimators=10, contamination=0.03)
    clf.fit(distribution.reshape(-1, 1))
    score = abs(clf.score_samples(np.array(actual_return).reshape(-1, 1)))[0]
    return score
    # return 0


def oneclass_svm(distribution, actual_return):
    clf = OneClassSVM(gamma='scale', nu=0.03)
    clf.fit(distribution.reshape(-1, 1))
    score = clf.score_samples(np.array(actual_return).reshape(-1, 1))[0]
    return score


def measure_as(as_method, value_dist, ac_return):
    if as_method == "lof":
        score = local_outlier_factor(value_dist, round(ac_return, 5))
    elif as_method == "knn":
        score = k_nearest_neighbors(value_dist, round(ac_return, 5))
    elif as_method == "iforest":
        score = isolation_forest(value_dist, round(ac_return, 5))
    elif as_method == "svm":
        score = oneclass_svm(value_dist, round(ac_return, 5))
    else:
        assert False, "Anomaly score measuring method is not given properly! Check '--score_calc_method'!"
    return score


def anomaly_detection(all_models, memory, max_len, gru_size, num_tau_sample, device, as_method, h, merge_type):
    ep_scores = {}
    ep_dists = {}
    mcs = {}
    hs_dict = {}
    for _, rgvd_name in all_models:
        ep_scores[rgvd_name] = []
        ep_dists[rgvd_name] = []
        mcs[rgvd_name] = []
        hs_dict[rgvd_name] = torch.zeros(num_tau_sample, gru_size)

    for s_batch, a_batch, mc_returns, _ in memory:
        for i in range(max_len):
            s, a, mc_return = s_batch[:, i, :], a_batch[:, i, :], mc_returns[:, i]

            for rgvd_model, rgvd_name in all_models:
                distributional_return, h_gvfs = RecurrentIQN.feed_forward(rgvd_model, hs_dict[rgvd_name].detach().to(device),
                                                                          s, len(s), num_tau_sample)
                hs_dict[rgvd_name] = h_gvfs
                anomaly_score = measure_as(as_method, distributional_return.squeeze(0).detach().cpu().numpy(),
                                           mc_return.squeeze(0)[int(rgvd_name)].item())
                ep_scores[rgvd_name].append(anomaly_score)
                ep_dists[rgvd_name].append(distributional_return.squeeze(0).detach().cpu().numpy())
                mcs[rgvd_name].append(mc_return.squeeze(0)[int(rgvd_name)].item())

    if merge_type == "avg":
        scores_merged = np.zeros(max_len)
        for key, values in ep_scores.items():
            scores_merged += np.array(values).copy()
    elif merge_type == "max":
        scores_merged = []
        for key, values in ep_scores.items():
            scores_merged.append(values)
        scores_merged = np.array(scores_merged).max(axis=0)
    return ep_scores, scores_merged, mcs, ep_dists


def plot_rgvd_accuracy(results, result_folder, horizon, info, plot_dist=True):
    fig, axs = plt.subplots(math.ceil(len(results) / 3), 3, figsize=(20, 20))
    r, c = 0, 0
    for key in results.keys():
        if plot_dist:
            axs[r, c].plot(results[key][1], color='limegreen')
            # for i in range(len(results[key][1])):
            #     scattered_dist = np.zeros((len(results[key][1][i]))) + i
            #     axs[r, c].scatter(scattered_dist, results[key][1][i], color='limegreen', s=10)
        else:
            axs[r, c].plot(np.array(results[key][1]).mean(axis=1), color='limegreen')
            axs[r, c].plot(np.array(results[key][1]).max(axis=1), color='palegreen')
            axs[r, c].plot(np.array(results[key][1]).min(axis=1), color='palegreen')
        axs[r, c].plot(results[key][0], color='teal')
        axs[r, c].set(xlabel='step', ylabel='return')
        axs[r, c].set_title("GVD: " + key.split("_")[0])
        if r < math.ceil(len(results) / 3) - 1:
            r += 1
        else:
            c += 1
            r = 0
    if plot_dist:
        labels = ["actual MC returns", "rGVD returns"]
        fig.legend(labels=labels, labelcolor=['teal', 'limegreen'], handlelength=0)
    else:
        labels = ["actual MC returns", "rGVD returns mean", "rGVD returns min & max"]
        fig.legend(labels=labels, labelcolor=['teal', 'limegreen', 'palegreen'], handlelength=0)
    # fig.show()
    fig.suptitle("Recurrent GVD accuracy\nhorizon: " + str(horizon) + "\n")
    fig.tight_layout()
    fig.savefig(os.path.join(result_folder, "rGVD_accuracy_" + info + "_h" + str(horizon) + ".png"))


def format_openai(runs: RunsType, model_conf: cconf.SystemConfig, component_name="autopilot") -> typ.List[dict]:
    """processes workgroup runs into a data structure serializable into json

    follows an openai gym like format:

    state, next_state, action, time, did_terminate, is_exception
    """
    out = []
    for r in runs:
        if isinstance(r, Exception):
            out.append({"is_exception": True})
        else:
            outd = {"is_exception": False}
            trajs = r[1]
            cio = ctr.get_component_io(component_name, trajs, model_conf)

            outd["did_terminate"] = not r[0]
            outd["time"] = cio["times"].tolist()
            outd["state"] = cio["inputs"].tolist()
            outd["actions"] = cio["outputs"].tolist()
            outd["next_state"] = np.vstack((cio["inputs"][1:], cio["inputs"][-1])).tolist()
            out.append(outd)
    return out


def ground_collision_condition(cname, outs):
    """ground collision premature termnation condition"""
    return cname == "plant" and outs["states"][11] <= 0.0


def merged_confusion_matrix(nominal_scores, anom_scores):
    scores = np.append(nominal_scores, anom_scores)

    norm_labels = np.zeros(len(nominal_scores))
    anorm_labels = np.ones(len(anom_scores))

    labels = np.append(norm_labels, anorm_labels)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc = sklearn.metrics.auc(fpr, tpr)
    results = (fpr, tpr, thresholds, auc)
    return results


def merged_confusion_matrix_single(scores, anom_occurrence):
    fpr, tpr, thresholds = roc_curve(anom_occurrence, scores)
    auc = sklearn.metrics.auc(fpr, tpr)
    results = (fpr, tpr, thresholds, auc)
    return results


def separated_confusion_matrix(nominal_scores, anom_scores):
    results = {}
    for key in nominal_scores.keys():
        scores = np.append(nominal_scores[key], anom_scores[key])

        norm_labels = np.zeros(len(nominal_scores[key]))
        anorm_labels = np.ones(len(anom_scores[key]))

        labels = np.append(norm_labels, anorm_labels)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc = sklearn.metrics.auc(fpr, tpr)
        results[key] = (fpr, tpr, thresholds, auc)
    return results


def separated_confusion_matrix_single(scores, anom_occurrence):
    results = {}
    for key in scores.keys():
        fpr, tpr, thresholds = roc_curve(anom_occurrence, scores[key])
        auc = sklearn.metrics.auc(fpr, tpr)
        results[key] = (fpr, tpr, thresholds, auc)
    return results


def plot_merged_roc(results, dir, env_name, method, target_type, horizons):
    fig, axs = plt.subplots(figsize=(4, 5))
    for key in results.keys():
        axs.plot(results[key][0], results[key][1],
                 label="H:" + str(key) + " - AUC:" + str(round(results[key][3], 2)),
                 color=color_codes[str(key)])
    for i in range(len(results)):
        if i == 0:
            axs.plot(np.arange(2), np.arange(2), label="Random", color='purple')
        axs.set(xlabel='FPR', ylabel='TPR')
        axs.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)

    fig.suptitle("Combined ROC Curve\nhorizon: " + str(horizons) + "\npr type: " + target_type)
    fig.tight_layout()
    fig.savefig(os.path.join(dir, "Combined_ROC_AUC_" + method + "_" + env_name + "_" + target_type + "_h_" +
                             str(horizons) + ".png"))
    # fig.show()
    plt.clf()
    plt.cla()
    plt.close()
    print("Combined ROC AUC plot saved in", dir)


def plot_separated_roc(all_results, dir, env_name, method, gvd_names, target_type, horizon):
    num_distinct_gvds = len(gvd_names)
    fig, axs = plt.subplots(math.ceil(num_distinct_gvds / 2), 2, figsize=(8, 28))
    for key in all_results.keys():
        r, c = 0, 0
        for gvd_n in gvd_names:
            axs[r, c].plot(all_results[key][gvd_n][0], all_results[key][gvd_n][1],
                           label="H:" + str(key) + " - AUC:" + str(round(all_results[key][gvd_n][3], 2)),
                           color=color_codes[str(key)])
            if r < math.ceil(num_distinct_gvds / 2) - 1:
                r += 1
            else:
                c += 1
                r = 0
    r, c = 0, 0
    for gvd_n in gvd_names:
        axs[r, c].plot(np.arange(2), np.arange(2), label="Random", color='purple')
        axs[r, c].set(xlabel='FPR', ylabel='TPR')
        axs[r, c].set_title("GVD: " + gvd_n)
        axs[r, c].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        if r < math.ceil(num_distinct_gvds / 2) - 1:
            r += 1
        else:
            c += 1
            r = 0
    fig.suptitle("ROC Curve\nhorizon: " + str(horizon) + "\npr type: " + target_type)
    fig.tight_layout()
    fig.savefig(os.path.join(dir, "ROC_AUC_" + method + "_" + env_name + "_" + target_type + "_h_" + str(horizon) + ".png"))
    # fig.show()
    plt.clf()
    plt.cla()
    plt.close()
    print("ROC AUC plot saved in", dir)


def plot_online_anomaly_detection(dots, dir, method, env_name, target_type, horizon):
    for dot in dots:
        plt.plot(dot[0], dot[1], 'o')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.suptitle("Anomaly detection for a chosen threshold\nhorizon: " + str(horizon) + "\npr type: " + target_type)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "Online_detection_" + method + "_" + env_name + "_" + target_type + "_h_" + str(horizon) + ".png"))
    plt.clf()
    plt.cla()
    plt.close()
    print("ROC AUC plot saved in", dir)


def plot_nominal_vs_anomalous_data(n_vs_a_data, result_folder, info, horizon, anomaly_occurrence=None):
    fig, axs = plt.subplots(math.ceil(len(n_vs_a_data) / 3), 3, figsize=(20, 20))
    r, c = 0, 0
    if anomaly_occurrence is None:
        for key, value in n_vs_a_data.items():
            axs[r, c].plot(value[0], color='teal')
            axs[r, c].plot(value[1], color='limegreen')
            axs[r, c].set(xlabel='step', ylabel='return')
            axs[r, c].set_title("GVD: " + key)
            if r < math.ceil(len(n_vs_a_data) / 3) - 1:
                r += 1
            else:
                c += 1
                r = 0
        labels = ["nominal data", "anomalous data"]
        fig.legend(labels=labels, labelcolor=['teal', 'limegreen'], handlelength=0)

    else:
        for key, value in n_vs_a_data.items():
            axs[r, c].plot(value, color='teal')
            axs[r, c].axvline(x=anomaly_occurrence.tolist().index(1) - horizon, color='black')
            axs[r, c].set(xlabel='step', ylabel='return')
            axs[r, c].set_title("GVD: " + key)
            if r < math.ceil(len(n_vs_a_data) / 3) - 1:
                r += 1
            else:
                c += 1
                r = 0
        labels = ["data", "when noise starts"]
        fig.legend(labels=labels, labelcolor=['teal', 'black'], handlelength=0)

    fig.suptitle("Nominal vs. Anomalous Data\nhorizon: " + str(h) + "\n")
    fig.tight_layout()
    # fig.show()
    fig.savefig(os.path.join(result_folder, "nom_vs_anom_data_" + info + "_h" + str(horizon) + ".png"))
    plt.clf()
    plt.cla()
    plt.close()


def plot_studying_anomaly_scores(nominal_dist, anomalous_dist, nominal_mc, anomalous_mc, anomaly_scores_n, anomaly_scores_a, h, result_folder, gvd_number):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    for i in range(len(nominal_dist[gvd_number])):
        scattered_dist = np.zeros((len(nominal_dist[gvd_number][i]))) + i
        axs[0].scatter(scattered_dist, nominal_dist[gvd_number][i], color='goldenrod', s=0.75)
    axs[0].plot(nominal_mc[gvd_number], label='nominal')
    for i in range(len(anomalous_dist[gvd_number])):
        scattered_dist = np.zeros((len(anomalous_dist[gvd_number][i]))) + i
        axs[1].scatter(scattered_dist, anomalous_dist[gvd_number][i], color='goldenrod', s=0.75)
    axs[1].plot(anomalous_mc[gvd_number], label='anomalous')
    axs[2].plot(anomaly_scores_n[gvd_number], label='nominal AS')
    axs[2].plot(anomaly_scores_a[gvd_number], label='anomalous AS')
    axs[0].legend(handlelength=0, loc='upper right')
    axs[1].legend(handlelength=0, loc='upper right')
    axs[2].legend(loc='upper right')
    fig.suptitle("Anomaly scores analysis\nhorizon: " + str(h) + "\nrGVD: " + gvd_number + "\n")
    fig.tight_layout()
    # fig.show()
    fig.savefig(os.path.join(result_folder, "AS_analysis_rGVD_" + gvd_number + "_h" + str(h) + ".png"))


def input_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--update_gvds', action='store_true', default=False,
                        help="To update a GVD")
    parser.add_argument('--test_gvds', action='store_true', default=False,
                        help="To test the trained GVD model")
    parser.add_argument('--plot_features', action='store_true', default=False,
                        help="To plot the training features")
    parser.add_argument('--online_anomaly_detection', action='store_true', default=False,
                        help="Anomaly detection done ONLINE by first calculating proper threshold OFFLINE")
    parser.add_argument('--noisy_data_path', type=str, default="",
                        help="Path to the pre-generated noisy data. If none, generates a fresh one")
    parser.add_argument('--recurrent_gvd', action='store_true', default=False,
                        help="To update a GVD using a fixed horizon and undiscounted MC returns")
    parser.add_argument('--pair_anomaly_detection', action='store_true', default=False,
                        help="Run a pair of episodes (one nominal, one anomalous) and detect anomaly")
    parser.add_argument('--single_anomaly_detection', action='store_true', default=False,
                        help="Run a single episode where the anomaly is inserted into the system in the middle of the run,"
                             " and detect anomaly")
    parser.add_argument('--anomaly_inserted', type=int, default=20,
                        help="Time when the anomaly is inserted into the system")
    parser.add_argument('--seed', type=int, default=500,
                        help="Seed for environment setup")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument('--horizon', nargs='+', type=int,
                        help='Horizon used for Monte-Carlo return calculation')
    parser.add_argument('--gru_units', type=int, default=32,
                        help="Number of cells in the GRU")
    parser.add_argument('--num_quantile_sample', type=int, default=32,
                        help="Number of quantile samples for IQN")
    parser.add_argument('--num_tau_sample', type=int, default=16,
                        help="Number of tau samples for IQN")
    parser.add_argument('--quantile_embedding_dim', type=int, default=64,
                        help="Qunatiles embedding dimension in IQN")
    parser.add_argument('--test_interval', type=int, default=10,
                        help="Intervals between train and test")
    parser.add_argument('--num_iterations', type=int, default=100000,
                        help="Number of iterations to update GVDs")
    parser.add_argument('--env_name', type=str,
                        help="Name of the main environment: to train, test, update gvds, find threshold, and calculate "
                             "performance on normal envs")
    parser.add_argument('--data_path', type=str,
                        help="path to the dataset json file")
    parser.add_argument('--test_data_path', type=str,
                        help="path to the test dataset json file")
    parser.add_argument('--gvd_name', type=str,
                        help="Name of GVD")
    parser.add_argument('--target_return_type', type=str, choices=["actual", "delta", "abs_delta", "time_avg"],
                        help="Method to calculate the target return for GVDs")
    parser.add_argument('--score_calc_method', type=str, choices=["lof", "knn", "iforest", "svm"],
                        help="Method to measure the anomaly score")
    parser.add_argument('--merge_type', type=str, choices=["avg", "max"],
                        help="Method to merge anomaly scores")
    parser.add_argument("--ic_file", type=str,
                        help="Path to Initial Conditions JSON")
    parser.add_argument("--config_file", type=str,
                        help="Path to CSAF System TOML")
    parser.add_argument("--t_max", type=int, default=35,
                        help="Maximum time to simulate each run")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = input_arg_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iqn_base_path = "./models"
    if not os.path.exists(os.path.join(iqn_base_path, args.env_name)):
        os.mkdir(os.path.join(iqn_base_path, args.env_name))

    test_performances = []
    gvd_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    num_inputs = 17
    num_actions = 4
    num_outputs = 1

    if args.update_gvds:
        print("Loading GVD training data!")
        with open(args.data_path) as f:
            memory = json.load(f)
        print("GVD data loaded!")

        recurrent_gvd_model = RecurrentIQN(num_inputs, num_outputs, args.gru_units, args.quantile_embedding_dim,
                                           args.num_quantile_sample, device)
        gvd_path = os.path.join(iqn_base_path, args.env_name, args.gvd_name + "_gvd_" + args.target_return_type + "_h_"
                                + str(args.horizon[0]) + ".pt")
        if os.path.exists(gvd_path):
            print("Loading pre-trained model!")
            recurrent_gvd_model.load_state_dict(torch.load(gvd_path, map_location=device))
            print("Pre-trained model loaded!")
        optimizer = optim.Adam(recurrent_gvd_model.parameters(), lr=args.lr)
        recurrent_gvd_model.to(device)
        recurrent_gvd_model.train()
        train_rb, test_rb, max_len = construct_gvd_data_undiscounted(num_inputs, memory, args.batch_size, args.horizon[0],
                                                                     device, args.gvd_name, args.target_return_type)
        update_recurrent_gvds(args, train_rb, test_rb, recurrent_gvd_model, optimizer, device, args.horizon[0], max_len)

    elif args.test_gvds:
        print("Loading GVD training data!")
        with open(args.data_path) as f:
            train_memory = json.load(f)
        print("GVD training data loaded!")
        test_results = {}
        print("Loading GVD test data!")
        with open(args.test_data_path) as f:
            memory = json.load(f)
        memory = memory[random.randint(0, len(memory) - 1)]
        # memory = memory[45]
        # memory = memory[33]
        print("GVD test data loaded!")
        test_rb, max_len = construct_test_gvd_data_undiscounted(num_inputs, memory, train_memory, args.batch_size,
                                                                args.horizon[0], device, args.target_return_type)
        for rgvd_name in gvd_names:
            recurrent_gvd_model = RecurrentIQN(num_inputs, num_outputs, args.gru_units, args.quantile_embedding_dim,
                                               args.num_quantile_sample, device)
            gvd_path = os.path.join(iqn_base_path, args.env_name,
                                    rgvd_name + "_gvd_" + args.target_return_type + "_h_" + str(args.horizon[0]) + ".pt")
            recurrent_gvd_model.load_state_dict(torch.load(gvd_path, map_location=device))
            recurrent_gvd_model.to(device)
            recurrent_gvd_model.eval()

            actual_returns, dist_returns = test_undiscounted(recurrent_gvd_model, test_rb, max_len, args.gru_units,
                                                             args.num_tau_sample, device, int(rgvd_name))

            test_results[rgvd_name] = (actual_returns, dist_returns)
        plot_rgvd_accuracy(test_results, os.path.join(iqn_base_path, args.env_name), args.horizon[0],
                           args.target_return_type, plot_dist=True)

    elif args.plot_features:
        print("Loading GVD training data!")
        with open(args.data_path) as f:
            train_memory = json.load(f)
        print("GVD data loaded!")
        print("Loading plotting data!")
        with open(args.test_data_path) as f:
            memory = json.load(f)
        print("Plotting data loaded!")
        memory = memory[random.randint(0, len(memory) - 1)]
        plot_gvd_data_undiscounted(num_inputs, train_memory, memory, os.path.join(iqn_base_path, args.env_name), args.horizon[0],
                                   args.target_return_type)

    elif args.pair_anomaly_detection:
        print("Loading noisy data!")
        with open(args.noisy_data_path) as f:
            anomalous_memory = json.load(f)
        print("Noisy data loaded!")
        print("Loading GVD training data!")
        with open(args.data_path) as f:
            train_memory = json.load(f)
        print("GVD training data loaded!")
        print("Loading nominal data!")
        with open(args.test_data_path) as f:
            nominal_memory = json.load(f)
        print("Nominal data loaded!")
        all_data_merged_results = []
        all_data_separated_results = []
        for data_index in range(len(anomalous_memory)):
            single_anomalous_memory = anomalous_memory[data_index]
            single_nominal_memory = nominal_memory[data_index]
            merged_results = {}
            separated_results = {}
            for h in args.horizon:
                all_rgvd_models = []
                for rgvd_name in gvd_names:
                    recurrent_gvd_model = RecurrentIQN(num_inputs, num_outputs, args.gru_units, args.quantile_embedding_dim,
                                                       args.num_quantile_sample, device)
                    gvd_path = os.path.join(iqn_base_path, args.env_name,
                                            rgvd_name + "_gvd_" + args.target_return_type + "_h_" + str(h) + ".pt")
                    recurrent_gvd_model.load_state_dict(torch.load(gvd_path, map_location=device))
                    recurrent_gvd_model.to(device)
                    recurrent_gvd_model.eval()
                    all_rgvd_models.append((recurrent_gvd_model, rgvd_name))

                anomalous_rb, anom_max_len = construct_test_gvd_data_undiscounted(num_inputs, single_anomalous_memory, train_memory,
                                                                                  args.batch_size, h, device, args.target_return_type)

                rgvds_anomaly_scores_a, merged_anomaly_scores_a, mcs_anom, dists_anom = anomaly_detection(all_rgvd_models, anomalous_rb, anom_max_len,
                                                                                    args.gru_units, args.num_tau_sample, device,
                                                                                    args.score_calc_method, h, args.merge_type)

                nominal_rb, nom_max_len = construct_test_gvd_data_undiscounted(num_inputs, single_nominal_memory, train_memory,
                                                                               args.batch_size, h, device, args.target_return_type)

                rgvds_anomaly_scores_n, merged_anomaly_scores_n, mcs_nom, dists_nom = anomaly_detection(all_rgvd_models, nominal_rb, nom_max_len,
                                                                                    args.gru_units, args.num_tau_sample, device,
                                                                                    args.score_calc_method, h, args.merge_type)
                # plot_studying_anomaly_scores(dists_nom, dists_anom, mcs_nom, mcs_anom, rgvds_anomaly_scores_n,
                #                              rgvds_anomaly_scores_a, h, os.path.join(iqn_base_path, args.env_name), gvd_number='11')
                merged_results[h] = merged_confusion_matrix(merged_anomaly_scores_n, merged_anomaly_scores_a)
                separated_results[h] = separated_confusion_matrix(rgvds_anomaly_scores_n, rgvds_anomaly_scores_a)
                data_comparison = {}
                for rgvd_name in gvd_names:
                    anom_mcs = []
                    nom_mcs = []
                    for _, _, mc_returns, _ in anomalous_rb:
                        for i in range(min(nom_max_len, anom_max_len)):
                            mc_return = mc_returns[:, i, int(rgvd_name)]
                            anom_mcs.append(mc_return.item())
                    for _, _, mc_returns, _ in nominal_rb:
                        for i in range(min(nom_max_len, anom_max_len)):
                            mc_return = mc_returns[:, i, int(rgvd_name)]
                            nom_mcs.append(mc_return.item())
                    data_comparison[rgvd_name] = (nom_mcs, anom_mcs)
                plot_nominal_vs_anomalous_data(data_comparison, os.path.join(iqn_base_path, args.env_name), args.env_name, h)
                print("Processing for h =", str(h), "is done! Move on to the next step!")
            # plot_merged_roc(merged_results, os.path.join(iqn_base_path, args.env_name), args.env_name, args.score_calc_method,
            #                   args.target_return_type, args.horizon)
            # plot_separated_roc(separated_results, os.path.join(iqn_base_path, args.env_name), args.env_name, args.score_calc_method,
            #                   gvd_names, args.target_return_type, args.horizon)
            all_data_merged_results.append(merged_results)
            all_data_separated_results.append(separated_results)

        each_horizon_merged_auc = {}
        each_horizon_separated_auc = {}
        for h in args.horizon:
            each_horizon_merged_auc[h] = []
            for item in all_data_merged_results:
                each_horizon_merged_auc[h].append(item[h][3])
            each_horizon_separated_auc[h] = []
            for gvd_n in gvd_names:
                tmp_storage = []
                for item in all_data_separated_results:
                    tmp_storage.append(item[h][gvd_n][3])
                each_horizon_separated_auc[h].append(tmp_storage)

        print("Number of runs:", len(anomalous_memory))
        for h in args.horizon:
            print("-------- horizon", h, "--------")
            print("Max combined AUC:", round(np.array(each_horizon_merged_auc[h]).max(), 3))
            print("Min combined AUC:", round(np.array(each_horizon_merged_auc[h]).min(), 3))
            print("Average combined AUC:", round(np.array(each_horizon_merged_auc[h]).mean(), 3))
            print("Individual feature with max AUC (feature #, AUC):", (np.array(each_horizon_separated_auc[h]).mean(axis=1).argmax(),
                                                                        round(np.array(each_horizon_separated_auc[h]).mean(axis=1).max(), 3)))
            print("Individual feature with min AUC (feature #, AUC):", (np.array(each_horizon_separated_auc[h]).mean(axis=1).argmin(),
                                                                        round(np.array(each_horizon_separated_auc[h]).mean(axis=1).min(), 3)))
            print("Individual features' AUCs (feature #, AUC):", end=' ')
            for k in range(len(each_horizon_separated_auc[h])):
                print((k, round(np.array(each_horizon_separated_auc[h][k]).mean(), 3)), end=' ')
            print("\n")
    elif args.single_anomaly_detection:
        print("Loading combined data!")
        with open(args.noisy_data_path) as f:
            combined_memory = json.load(f)
        print("Combined data loaded!")
        print("Loading GVD training data!")
        with open(args.data_path) as f:
            train_memory = json.load(f)
        print("GVD training data loaded!")
        all_data_merged_results = []
        all_data_separated_results = []
        for data_index in range(len(combined_memory)):
            if len(combined_memory[data_index]['time']) > args.anomaly_inserted:
                when_anomaly_occurs = np.zeros(len(combined_memory[data_index]['time']))

                single_run = combined_memory[data_index]
                merged_results = {}
                separated_results = {}
                for h in args.horizon:
                    when_anomaly_occurs[args.anomaly_inserted - h:] = 1
                    all_rgvd_models = []
                    for rgvd_name in gvd_names:
                        recurrent_gvd_model = RecurrentIQN(num_inputs, num_outputs, args.gru_units,
                                                           args.quantile_embedding_dim,
                                                           args.num_quantile_sample, device)
                        gvd_path = os.path.join(iqn_base_path, args.env_name,
                                                rgvd_name + "_gvd_" + args.target_return_type + "_h_" + str(h) + ".pt")
                        recurrent_gvd_model.load_state_dict(torch.load(gvd_path, map_location=device))
                        recurrent_gvd_model.to(device)
                        recurrent_gvd_model.eval()
                        all_rgvd_models.append((recurrent_gvd_model, rgvd_name))

                    data_rb, data_max_len = construct_test_gvd_data_undiscounted(num_inputs, single_run, train_memory,
                                                                                      args.batch_size, h, device,
                                                                                      args.target_return_type)

                    rgvds_anomaly_scores, merged_anomaly_scores, monte_carlos, distributions = anomaly_detection(all_rgvd_models, data_rb, data_max_len,
                                                                                                args.gru_units, args.num_tau_sample, device,
                                                                                                args.score_calc_method, h, args.merge_type)

                    merged_results[h] = merged_confusion_matrix_single(merged_anomaly_scores, when_anomaly_occurs)
                    separated_results[h] = separated_confusion_matrix_single(rgvds_anomaly_scores, when_anomaly_occurs)
                    data_comparison = {}
                    for rgvd_name in gvd_names:
                        mcs = []
                        for _, _, mc_returns, _ in data_rb:
                            for i in range(data_max_len):
                                mc_return = mc_returns[:, i, int(rgvd_name)]
                                mcs.append(mc_return.item())
                        data_comparison[rgvd_name] = (mcs)
                    plot_nominal_vs_anomalous_data(data_comparison, os.path.join(iqn_base_path, args.env_name),
                                                   args.env_name, h, anomaly_occurrence=when_anomaly_occurs)
                    print("Processing for h =", str(h), "is done! Move on to the next step!")
                all_data_merged_results.append(merged_results)
                all_data_separated_results.append(separated_results)

        each_horizon_merged_auc = {}
        each_horizon_separated_auc = {}
        for h in args.horizon:
            each_horizon_merged_auc[h] = []
            for item in all_data_merged_results:
                each_horizon_merged_auc[h].append(item[h][3])
            each_horizon_separated_auc[h] = []
            for gvd_n in gvd_names:
                tmp_storage = []
                for item in all_data_separated_results:
                    tmp_storage.append(item[h][gvd_n][3])
                each_horizon_separated_auc[h].append(tmp_storage)

        print("Number of runs:", len(combined_memory))
        for h in args.horizon:
            print("-------- horizon", h, "--------")
            print("Max combined AUC:", round(np.array(each_horizon_merged_auc[h]).max(), 3))
            print("Min combined AUC:", round(np.array(each_horizon_merged_auc[h]).min(), 3))
            print("Average combined AUC:", round(np.array(each_horizon_merged_auc[h]).mean(), 3))
            print("Individual feature with max AUC (feature #, AUC):",
                  (np.array(each_horizon_separated_auc[h]).mean(axis=1).argmax(),
                   round(np.array(each_horizon_separated_auc[h]).mean(axis=1).max(), 3)))
            print("Individual feature with min AUC (feature #, AUC):",
                  (np.array(each_horizon_separated_auc[h]).mean(axis=1).argmin(),
                   round(np.array(each_horizon_separated_auc[h]).mean(axis=1).min(), 3)))
            print("Individual features' AUCs (feature #, AUC):", end=' ')
            for k in range(len(each_horizon_separated_auc[h])):
                print((k, round(np.array(each_horizon_separated_auc[h][k]).mean(), 3)), end=' ')
            print("\n")

    elif args.online_anomaly_detection:
        print("Loading noisy data!")
        with open(args.noisy_data_path) as f:
            anomalous_memory = json.load(f)
        print("Noisy data loaded!")
        print("Loading GVD training data!")
        with open(args.data_path) as f:
            train_memory = json.load(f)
        print("GVD training data loaded!")
        print("Loading nominal data!")
        with open(args.test_data_path) as f:
            nominal_memory = json.load(f)
        print("Nominal data loaded!")
        half_anomalous_memory = anomalous_memory[:int(len(anomalous_memory) / 2)]
        half_nominal_memory = nominal_memory[:int(len(nominal_memory) / 2)]
        merged_results = {}
        print("Calculating the average threshold!")
        for data_index in range(len(half_anomalous_memory)):
            print("Data counter:", data_index)
            single_anomalous_memory = half_anomalous_memory[data_index]
            single_nominal_memory = half_nominal_memory[data_index]
            h = args.horizon[0]
            all_rgvd_models = []
            for rgvd_name in gvd_names:
                recurrent_gvd_model = RecurrentIQN(num_inputs, num_outputs, args.gru_units,
                                                   args.quantile_embedding_dim,
                                                   args.num_quantile_sample, device)
                gvd_path = os.path.join(iqn_base_path, args.env_name,
                                        rgvd_name + "_gvd_" + args.target_return_type + "_h_" + str(h) + ".pt")
                recurrent_gvd_model.load_state_dict(torch.load(gvd_path, map_location=device))
                recurrent_gvd_model.to(device)
                recurrent_gvd_model.eval()
                all_rgvd_models.append((recurrent_gvd_model, rgvd_name))

            anomalous_rb, anom_max_len = construct_test_gvd_data_undiscounted(num_inputs, single_anomalous_memory,
                                                                              train_memory,
                                                                              args.batch_size, h, device,
                                                                              args.target_return_type)

            _, merged_anomaly_scores_a, _, _ = anomaly_detection(all_rgvd_models, anomalous_rb, anom_max_len, args.gru_units,
                                                                 args.num_tau_sample, device, args.score_calc_method, h,
                                                                 args.merge_type)

            nominal_rb, nom_max_len = construct_test_gvd_data_undiscounted(num_inputs, single_nominal_memory,
                                                                           train_memory,
                                                                           args.batch_size, h, device,
                                                                           args.target_return_type)

            _, merged_anomaly_scores_n, _, _ = anomaly_detection(all_rgvd_models, nominal_rb, nom_max_len, args.gru_units,
                                                                 args.num_tau_sample, device, args.score_calc_method, h,
                                                                 args.merge_type)
            # (fpr, tpr, thresholds, auc) = merged_confusion_matrix(...)
            merged_results[data_index] = merged_confusion_matrix(merged_anomaly_scores_n, merged_anomaly_scores_a)
        chosen_ths, roc_dots = [], []
        for key, value in merged_results.items():
            distances = []
            fpr = value[0]
            tpr = value[1]
            thresholds = value[2]
            for i in range(len(fpr)):
                # if FP and FN costs are equal and +ve vs. -ve cases have the same proportion in data
                distances.append(math.sqrt((1 - fpr[i]) ** 2 + tpr[i] ** 2))
            chosen_ths.append(thresholds[np.argmax(distances)])
        avg_chosen_th = np.array(chosen_ths).mean()
        print("Average threshold is calculated:", avg_chosen_th)
        print("Using the average threshold, start detecting anomalies now!")
        all_data_merged_results = []
        all_data_separated_results = []
        half_anomalous_memory = anomalous_memory[int(len(anomalous_memory) / 2):]
        half_nominal_memory = nominal_memory[int(len(nominal_memory) / 2):]
        for data_index in range(len(half_anomalous_memory)):
            print("Data counter:", data_index)
            single_anomalous_memory = half_anomalous_memory[data_index]
            single_nominal_memory = half_nominal_memory[data_index]
            h = args.horizon[0]
            all_rgvd_models = []
            for rgvd_name in gvd_names:
                recurrent_gvd_model = RecurrentIQN(num_inputs, num_outputs, args.gru_units,
                                                   args.quantile_embedding_dim,
                                                   args.num_quantile_sample, device)
                gvd_path = os.path.join(iqn_base_path, args.env_name,
                                        rgvd_name + "_gvd_" + args.target_return_type + "_h_" + str(h) + ".pt")
                recurrent_gvd_model.load_state_dict(torch.load(gvd_path, map_location=device))
                recurrent_gvd_model.to(device)
                recurrent_gvd_model.eval()
                all_rgvd_models.append((recurrent_gvd_model, rgvd_name))

            anomalous_rb, anom_max_len = construct_test_gvd_data_undiscounted(num_inputs, single_anomalous_memory,
                                                                              train_memory,
                                                                              args.batch_size, h, device,
                                                                              args.target_return_type)

            _, merged_anomaly_scores_a, _, _ = anomaly_detection(all_rgvd_models, anomalous_rb, anom_max_len, args.gru_units,
                                                                 args.num_tau_sample, device, args.score_calc_method, h,
                                                                 args.merge_type)

            nominal_rb, nom_max_len = construct_test_gvd_data_undiscounted(num_inputs, single_nominal_memory,
                                                                           train_memory,
                                                                           args.batch_size, h, device,
                                                                           args.target_return_type)

            _, merged_anomaly_scores_n, _, _ = anomaly_detection(all_rgvd_models, nominal_rb, nom_max_len, args.gru_units,
                                                                 args.num_tau_sample, device, args.score_calc_method, h,
                                                                 args.merge_type)
            tn = (merged_anomaly_scores_n <= avg_chosen_th).sum()
            fp = (merged_anomaly_scores_n > avg_chosen_th).sum()
            fn = (merged_anomaly_scores_a <= avg_chosen_th).sum()
            tp = (merged_anomaly_scores_a > avg_chosen_th).sum()
            fpr = fp / (tn + fp)
            tpr = tp / (tp + fn)
            roc_dots.append((fpr, tpr))
        print("DONE!")
        print(roc_dots)
        plot_online_anomaly_detection(roc_dots, os.path.join(iqn_base_path, args.env_name), args.score_calc_method, args.env_name,
                                      args.target_return_type, args.horizon)