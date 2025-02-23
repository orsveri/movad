import datetime
import torch
import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from losses import build_loss
from dota import gt_cls_target


def test(cfg, model, testdata_loader, epoch,
         filename):
    # Tensorboard
    writer = SummaryWriter(cfg.output + '/tensorboard/eval_{}'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    targets_all = []
    outputs_all = []
    toas_all = []
    teas_all = []
    idxs_all = []
    info_all = []
    frames_counter = []

    rnn_type = cfg.get('rnn_type', 'lstm')
    rnn_cell_num = cfg.get('rnn_cell_num', 1)

    index_l = 0
    criterion = build_loss(cfg)
    fb = cfg.NF
    model.eval()
    for j, (video_data, data_info) in tqdm(
            enumerate(testdata_loader), total=len(testdata_loader),
            desc='Epoch: %d / %d' % (epoch, cfg.epochs)):
        video_data = video_data.to(cfg.device, non_blocking=True)
        data_info = data_info.to(cfg.device, non_blocking=True)

        # [B, F, C, W, H] -> [B, C, F, W, H]
        video_data = torch.swapaxes(video_data, 1, 2)

        t_shape = (video_data.shape[0], video_data.shape[2] - fb)
        targets = torch.full(t_shape, -100).to(video_data.device)
        outputs = torch.full(t_shape, -100, dtype=float).to(video_data.device)

        idx_batch = data_info[:, 1]
        toa_batch = data_info[:, 2]
        tea_batch = data_info[:, 3]
        info_batch = data_info[:, 7:11]
        rnn_state = None

        if rnn_type == 'gru':
            rnn_state = torch.randn(
                rnn_cell_num, 1, cfg.rnn_state_size).to(cfg.device)

        # FIXME start from zero!
        for i in range(fb, video_data.shape[2]):
            target = gt_cls_target(i, toa_batch, tea_batch).long()
            x = video_data[:, :, i-fb:i]

            output, rnn_state = model(x, rnn_state)

            if cfg.get('apply_softmax', True):
                output = output.softmax(dim=1)

            loss = criterion(output, target)
            writer.add_scalar('losses/test', loss.item(), index_l)
            index_l += 1

            if not cfg.get('apply_softmax', True):
                output = output.softmax(dim=1)

            targets[:, i-fb] = target.clone()
            outputs[:, i-fb] = output[:, 1].clone()

        # collect results for each video
        targets_all.append(targets.view(-1).tolist())
        outputs_all.append(outputs.view(-1).tolist())
        toas_all.append(toa_batch.tolist())
        teas_all.append(tea_batch.tolist())
        idxs_all.append(idx_batch.tolist())
        info_all.append(info_batch.tolist())
        frames_counter.append(video_data.shape[2])

    # collect results for all dataset
    toas_all = np.array(toas_all).reshape(-1)
    teas_all = np.array(teas_all).reshape(-1)
    idxs_all = np.array(idxs_all).reshape(-1)
    info_all = np.array(info_all).reshape(-1, 4)
    frames_counter = np.array(frames_counter).reshape(-1)

    print('save file {}'.format(filename))
    with open(filename, 'wb') as f:
        pickle.dump({
            'targets': targets_all,
            'outputs': outputs_all,
            'toas': toas_all,
            'teas': teas_all,
            'idxs': idxs_all,
            'info': info_all,
            'frames_counter': frames_counter,
        }, f)


def test_filenames_csv(cfg, model, testdata_loader, epoch,
         filename):

    targets_all = []
    logits = []
    clip_names_all = []
    timesteps_all = []
    ttc_all = []

    rnn_type = cfg.get('rnn_type', 'lstm')
    rnn_cell_num = cfg.get('rnn_cell_num', 1)

    fb = cfg.NF
    model.eval()
    for j, (video_data, data_info, timesteps_labels, ttcs) in tqdm(
            enumerate(testdata_loader), total=len(testdata_loader),
            desc='Epoch: %d / %d' % (epoch, cfg.epochs)):
        # DoTA
        #clip_name = testdata_loader.dataset.keys[int(data_info[0, 1].item())]
        # DADA
        clip_id, frame_seq = testdata_loader.dataset.dataset_samples[int(data_info[0, 1].item())]
        clip_name = testdata_loader.dataset.clip_names[clip_id]

        video_data = video_data.to(cfg.device, non_blocking=True)

        # [B, F, C, W, H] -> [B, C, F, W, H]
        video_data = torch.swapaxes(video_data, 1, 2)
        rnn_state = None
        if rnn_type == 'gru':
            rnn_state = torch.randn(
                rnn_cell_num, 1, cfg.rnn_state_size).to(cfg.device)

        # FIXME start from zero! # orsveri: I think it's ok
        for i in range(fb, video_data.shape[2]+1):
            x = video_data[:, :, i-fb:i]

            output, rnn_state = model(x, rnn_state)

            targets_all.append(timesteps_labels[0, i-1, 1].item())
            logits.append(output[0].detach().cpu().numpy())
            timesteps_all.append(timesteps_labels[0, i-1, 0].item())
            clip_names_all.append(clip_name)
            ttc_all.append(ttcs[0, i-1].item())

    print('save file {}'.format(filename))

    logits = np.array(logits)

    df = pd.DataFrame({
        "clip": clip_names_all,
        "filename": timesteps_all,
        "logits_safe": logits[:, 0],
        "logits_risk": logits[:, 1],
        "label": targets_all,
        "ttc": ttc_all  # TODO
    })
    df.to_csv(filename, index=True, header=True)
