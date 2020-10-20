import torch

import utils
import wandb
import os
from tqdm import tqdm
import numpy as np
from dataloaders import loader
from torch.nn import functional as F

from models.BayesBeat import Bayesian_Deepbeat

SAVE_PATH = 'saved_model/bayesbeat_cpu.pt'
ENSEMBLES = 1
DEEPBEAT_DATA_PATH = "data/"


def get_uncertainty(model, input_signal, T=15, normalized=False):
    input_signal = input_signal.view(1, -1)
    input_signal = input_signal.unsqueeze(0)
    input_signals = input_signal.repeat(T, 1, 1)

    net_out, _ = model(input_signals)
    if normalized:
        prediction = F.softplus(net_out)
        p_hat = prediction / torch.sum(prediction, dim=1).unsqueeze(1)
    else:
        p_hat = F.softmax(net_out, dim=1)
    p_hat = p_hat.detach().cpu().numpy()
    p_bar = np.mean(p_hat, axis=0)

    temp = p_hat - np.expand_dims(p_bar, 0)
    epistemic = np.dot(temp.T, temp) / T
    epistemic = np.diag(epistemic)

    aleatoric = np.diag(p_bar) - (np.dot(p_hat.T, p_hat) / T)
    aleatoric = np.diag(aleatoric)

    return torch.Tensor(p_bar.reshape(1, 2)), epistemic, aleatoric


def evaluate_with_uncertainity(model, generator, prefix='val_', wandb_log=False, uncertainity_bound=0.03):
    # EVALUATE SIGNAL BY SIGNAL WITH UNCERTAINTY CALCULATION

    sigs = np.empty((0, 800))
    REPEAT = 15
    stat_map_overall = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
    output_map_all = {}
    output_map_all['true'] = torch.empty((0, 2), device='cpu')
    output_map_all['pred'] = torch.empty((0, 2), device='cpu')

    for i, batch in enumerate(tqdm(iter(generator))):
        signal, qa, rhythm = batch
        signal, rhythm = signal.to(device, non_blocking=True), rhythm.to(device, non_blocking=True)
        if signal.shape[0] > 1:
            raise Exception(
                "Uncertainity requires batch size = 1 in current build, full paralleled batch support in future.")

        log_outputs, ep, al = get_uncertainty(model, signal, REPEAT)
        sigs = np.vstack((sigs, signal.cpu().numpy().reshape(1, 800)))
        if al[1] > uncertainity_bound:
            continue

        stat_map_batch = utils.batch_stat_deepbeat(rhythm_true=rhythm, rhythm_pred=log_outputs)
        utils.accumulate_stat(stat_map=stat_map_overall, **stat_map_batch)
        utils.accumulate_responses(output_map_all, rhythm, log_outputs)

    metrics_map = utils.metrics_from_stat(**stat_map_overall, prefix=prefix, output_map_all=output_map_all)

    utils.print_stat_map(metrics_map)

    if wandb_log:
        wandb.log(metrics_map)
    return metrics_map[prefix + 'F1']


if __name__ == '__main__':
    model = Bayesian_Deepbeat()

    device = 'cpu'
    print("Device: " + device)

    # BATCH_SIZE SHOULD BE 1 IN CURRENT BUILD. FULL MINIBATCH INFERENCE WILL BE ADDED IN FUTURE.

    test_generator = loader.get_weighted_generator(DEEPBEAT_DATA_PATH + 'test',
                                                   batch_size=1,
                                                   replacement=False, is_train=False, shuffle=False, remove_poor=False)

    best_val_f1_pos = -1

    model.load_state_dict(torch.load(os.path.join(SAVE_PATH))['state_dict'])
    model = model.to(device)
    model.eval()
    bounds = [0.05]
    with torch.no_grad():
        for bound in bounds:
            print('**************BOUND=: ' + str(bound) + '*************')
            print("-------------------------------------------------------")
            test_f1_pos = evaluate_with_uncertainity(model=model, generator=test_generator, prefix='test_',
                                                         uncertainity_bound=bound)
