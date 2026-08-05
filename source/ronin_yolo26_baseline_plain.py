import os
import sys
import time
from os import path as osp
import csv

import joblib
import numpy as np
import torch
import json

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from data_glob_speed import *
from transformations import *
from metric import compute_ate_rte
from model_resnet1d import *
from model_yolo26_1d import YOLO26_1D_Regressor
from model_mobilenet1d import MobileNetV2_1D
from model_shufflenet1d import ShuffleNetV2_1D
from model_efficientnet_lite1d import EfficientNetLite0_1D
from model_tinycnn1d import TinyCNN1D
from model_lighttcn1d import LightTCN1D
from rf_utils import apply_rf_correction, build_sequence_features

_input_channel, _output_channel = 6, 2
_fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.2, 'trans_planes': 128}


def get_model(backbone, model_dropout=0.2, use_attention=False):
    if backbone == 'yolo26':
        network = YOLO26_1D_Regressor(
            in_channels=6,
            num_outputs=2,
            base_ch=32,
            widths=(64, 128, 256),
            n_blocks=(1, 2, 2),
            dropout=model_dropout,
            use_attention=use_attention,
        )
        print("[YOLO26-1D] Plain baseline mode: training from scratch.")
        print(f"[YOLO26-1D] Attention: {use_attention}, dropout: {model_dropout}")
    elif backbone == 'mobilenetv2':
        network = MobileNetV2_1D(in_channels=6, num_outputs=2, width_mult=0.9, dropout=model_dropout)
        print("[MobileNetV2-1D] Plain baseline mode: training from scratch.")
    elif backbone == 'shufflenetv2':
        network = ShuffleNetV2_1D(in_channels=6, num_outputs=2, dropout=model_dropout)
        print("[ShuffleNetV2-1D] Plain baseline mode: training from scratch.")
    elif backbone == 'efficientnet_lite0':
        network = EfficientNetLite0_1D(in_channels=6, num_outputs=2, width_mult=1.0, dropout=model_dropout)
        print("[EfficientNet-Lite0-1D] Plain baseline mode: training from scratch.")
    elif backbone == 'tinycnn':
        network = TinyCNN1D(in_channels=6, num_outputs=2, dropout=model_dropout)
        print("[TinyCNN-1D] Plain baseline mode: training from scratch.")
    elif backbone == 'lighttcn':
        network = LightTCN1D(in_channels=6, num_outputs=2, dropout=model_dropout)
        print("[LightTCN-1D] Plain baseline mode: training from scratch.")
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    if hasattr(network, 'get_num_params'):
        print(f"[{backbone}] Parameters: {network.get_num_params():,}")
    return network


def run_test(network, data_loader, device, eval_mode=True, norm_mean=None, norm_std=None):
    targets_all = []
    preds_all = []
    if eval_mode:
        network.eval()
    for bid, (feat, targ, _, _) in enumerate(data_loader):
        feat = feat.to(device)
        if norm_mean is not None and norm_std is not None:
            feat = (feat - norm_mean) / (norm_std + 1e-8)
        pred = network(feat).cpu().detach().numpy()
        targets_all.append(targ.detach().numpy())
        preds_all.append(pred)
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    return targets_all, preds_all


def get_stats_path(dataset_name):
    if dataset_name == 'ridi':
        return '../output/stats/ridi_channel_stats.npy'
    return '../output/stats/ronin_channel_stats.npy'


def add_summary(writer, loss, step, mode):
    names = '{0}_loss/loss_x,{0}_loss/loss_y,{0}_loss/loss_z,{0}_loss/loss_sin,{0}_loss/loss_cos'.format(
        mode).split(',')

    for i in range(loss.shape[0]):
        writer.add_scalar(names[i], loss[i], step)
    writer.add_scalar('{}_loss/avg'.format(mode), np.mean(loss), step)


def get_dataset(root_dir, data_list, args, **kwargs):
    mode = kwargs.get('mode', 'train')

    random_shift, shuffle, transforms, grv_only = 0, False, None, False
    if mode == 'train':
        random_shift = args.step_size // 2
        shuffle = True
        transforms = RandomHoriRotate(math.pi * 2)
    elif mode == 'val':
        shuffle = True
    elif mode == 'test':
        shuffle = False
        grv_only = True

    if args.dataset == 'ronin':
        seq_type = GlobSpeedSequence
    elif args.dataset == 'ridi':
        from data_ridi import RIDIGlobSpeedSequence
        seq_type = RIDIGlobSpeedSequence
    dataset = StridedSequenceDataset(
        seq_type, root_dir, data_list, args.cache_path, args.step_size, args.window_size,
        random_shift=random_shift, transform=transforms,
        shuffle=shuffle, grv_only=grv_only, max_ori_error=args.max_ori_error)

    global _input_channel, _output_channel
    _input_channel, _output_channel = dataset.feature_dim, dataset.target_dim
    return dataset


def get_dataset_from_list(root_dir, list_path, args, **kwargs):
    with open(list_path) as f:
        data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    return get_dataset(root_dir, data_list, args, **kwargs)


def train(args, **kwargs):
    # Loading data
    start_t = time.time()
    train_dataset = get_dataset_from_list(args.root_dir, args.train_list, args, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    end_t = time.time()
    print('Training set loaded. Feature size: {}, target size: {}. Time usage: {:.3f}s'.format(
        train_dataset.feature_dim, train_dataset.target_dim, end_t - start_t))
    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset = get_dataset_from_list(args.root_dir, args.val_list, args, mode='val')
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')

    summary_writer = None
    if args.out_dir is not None:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        write_config(args)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))

    global _fc_config
    _fc_config['in_dim'] = args.window_size // 32 + 1

    network = get_model(
        args.backbone,
        model_dropout=args.model_dropout,
        use_attention=args.use_attention,
    ).to(device)
    print('Number of train samples: {}'.format(len(train_dataset)))
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
    total_params = network.get_num_params()
    print('Total number of parameters: ', total_params)

    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, verbose=True
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10
        )
    print(f"[Optimizer] Adam lr={args.lr:.2e}, weight_decay={args.weight_decay:.2e}")

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        checkpoints = torch.load(args.continue_from)
        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))

    if args.out_dir is not None and osp.exists(osp.join(args.out_dir, 'logs')):
        summary_writer = SummaryWriter(osp.join(args.out_dir, 'logs'))
        summary_writer.add_text('info', 'total_param: {}'.format(total_params))

    step = 0
    best_val_loss = np.inf
    epoch_metrics_path = _init_epoch_metrics_csv(args.out_dir, _output_channel)

    print('Start from epoch {}'.format(start_epoch))
    total_epoch = start_epoch
    train_losses_all, val_losses_all = [], []

    # Get the initial loss.
    init_train_targ, init_train_pred = run_test(
        network, train_loader, device, eval_mode=False
    )

    init_train_loss = np.mean((init_train_targ - init_train_pred) ** 2, axis=0)
    train_losses_all.append(np.mean(init_train_loss))
    print('-------------------------')
    print('Init: average loss: {}/{:.6f}'.format(init_train_loss, train_losses_all[-1]))
    if summary_writer is not None:
        add_summary(summary_writer, init_train_loss, 0, 'train')

    if val_loader is not None:
        init_val_targ, init_val_pred = run_test(
            network, val_loader, device
        )
        init_val_loss = np.mean((init_val_targ - init_val_pred) ** 2, axis=0)
        val_losses_all.append(np.mean(init_val_loss))
        print('Validation loss: {}/{:.6f}'.format(init_val_loss, val_losses_all[-1]))
        if summary_writer is not None:
            add_summary(summary_writer, init_val_loss, 0, 'val')

    try:
        for epoch in range(start_epoch, args.epochs):
            start_t = time.time()
            network.train()
            train_outs, train_targets = [], []
            use_tqdm = (not getattr(args, 'no_tqdm', False)) and (tqdm is not None)
            batch_iter = train_loader
            if use_tqdm:
                batch_iter = tqdm(
                    train_loader,
                    total=len(train_loader),
                    desc=f"Epoch {epoch + 1}/{args.epochs}",
                    unit="batch",
                    leave=False,
                    file=sys.stderr,
                )
            running_loss = 0.0
            for batch_id, (feat, targ, _, _) in enumerate(batch_iter):
                feat, targ = feat.to(device), targ.to(device)
                optimizer.zero_grad()
                pred = network(feat)
                train_outs.append(pred.cpu().detach().numpy())
                train_targets.append(targ.cpu().detach().numpy())
                loss = criterion(pred, targ)
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().item())
                if use_tqdm and batch_id % 10 == 0:
                    denom = max(1, batch_id + 1)
                    batch_iter.set_postfix(loss=f"{(running_loss / denom):.6f}")
                step += 1
            train_outs = np.concatenate(train_outs, axis=0)
            train_targets = np.concatenate(train_targets, axis=0)
            train_losses = np.average((train_outs - train_targets) ** 2, axis=0)

            end_t = time.time()
            print('-------------------------')
            print('Epoch {}, time usage: {:.3f}s, average loss: {}/{:.6f}'.format(
                epoch, end_t - start_t, train_losses, np.average(train_losses)))
            train_losses_all.append(np.average(train_losses))

            if summary_writer is not None:
                add_summary(summary_writer, train_losses, epoch + 1, 'train')
                summary_writer.add_scalar('optimizer/lr', optimizer.param_groups[0]['lr'], epoch)

            if val_loader is not None:
                network.eval()
                val_outs, val_targets = run_test(
                    network, val_loader, device
                )
                val_losses = np.average((val_outs - val_targets) ** 2, axis=0)
                avg_loss = np.average(val_losses)
                print('Validation loss: {}/{:.6f}'.format(val_losses, avg_loss))
                scheduler.step(avg_loss)
                print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
                if summary_writer is not None:
                    add_summary(summary_writer, val_losses, epoch + 1, 'val')
                val_losses_all.append(avg_loss)
                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    if args.out_dir and osp.isdir(args.out_dir):
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Model saved to ', model_path)
                _append_epoch_metrics(
                    epoch_metrics_path,
                    epoch=epoch,
                    seconds=(time.time() - start_t),
                    lr=optimizer.param_groups[0]['lr'],
                    train_losses=train_losses,
                    val_losses=val_losses,
                    best_val_loss=best_val_loss,
                )
            else:
                if args.out_dir is not None and osp.isdir(args.out_dir):
                    model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                    torch.save({'model_state_dict': network.state_dict(),
                                'epoch': epoch,
                                'optimizer_state_dict': optimizer.state_dict()}, model_path)
                    print('Model saved to ', model_path)
                _append_epoch_metrics(
                    epoch_metrics_path,
                    epoch=epoch,
                    seconds=(time.time() - start_t),
                    lr=optimizer.param_groups[0]['lr'],
                    train_losses=train_losses,
                    val_losses=None,
                    best_val_loss=best_val_loss,
                )

            total_epoch = epoch

    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training complete')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': total_epoch}, model_path)
        print('Checkpoint saved to ', model_path)

    return train_losses_all, val_losses_all


def recon_traj_with_preds(dataset, preds, seq_id=0, **kwargs):
    """
    Reconstruct trajectory with predicted global velocities.
    """
    ts = dataset.ts[seq_id]
    ind = np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=int)
    dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
    pos = np.zeros([preds.shape[0] + 2, 2])
    pos[0] = dataset.gt_pos[seq_id][0, :2]
    pos[1:-1] = np.cumsum(preds[:, :2] * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    ts_ext = np.concatenate([[ts[0] - 1e-06], ts[ind], [ts[-1] + 1e-06]], axis=0)
    pos = interp1d(ts_ext, pos, axis=0)(ts)
    return pos


def test_sequence(args):
    if args.test_path is not None:
        if args.test_path[-1] == '/':
            args.test_path = args.test_path[:-1]
        root_dir = osp.split(args.test_path)[0]
        test_data_list = [osp.split(args.test_path)[1]]
    elif args.test_list is not None:
        root_dir = args.root_dir
        with open(args.test_list) as f:
            test_data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    else:
        raise ValueError('Either test_path or test_list must be specified.')

    if args.out_dir is not None and not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if not torch.cuda.is_available() or args.cpu:
        device = torch.device('cpu')
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        device = torch.device('cuda:0')
        checkpoint = torch.load(args.model_path)

    # Load the first sequence to update the input and output size
    _ = get_dataset(root_dir, [test_data_list[0]], args)

    global _fc_config
    _fc_config['in_dim'] = args.window_size // 32 + 1

    network = get_model(
        args.backbone,
        model_dropout=args.model_dropout,
        use_attention=args.use_attention,
    )

    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.model_path, device))

    rf_model = None
    if args.use_rf_postprocess:
        if args.rf_model_path is None:
            raise ValueError('When --use_rf_postprocess is set, --rf_model_path is required')
        rf_model = joblib.load(args.rf_model_path)
        print('RF model {} loaded.'.format(args.rf_model_path))

    preds_seq, targets_seq, losses_seq, ate_all, rte_all = [], [], [], [], []
    traj_lens = []

    pred_per_min = 200 * 60

    # Timing instrumentation
    inference_times_neural = []  # ms per sample (neural only)
    inference_times_rf = []  # ms per sample (RF only, if enabled)
    inference_times_total = []  # ms per sample (neural + RF)
    sequence_times = {}  # ms per sequence
    sequence_times_neural = {}  # ms per sequence (neural only)
    sequence_times_rf = {}  # ms per sequence (RF only, if enabled)

    for data in test_data_list:
        seq_dataset = get_dataset(root_dir, [data], args, mode='test')
        seq_loader = DataLoader(seq_dataset, batch_size=1024, shuffle=False)
        ind = np.array([i[1] for i in seq_dataset.index_map if i[0] == 0], dtype=int)

        seq_start_neural = time.time()
        targets, preds = run_test(network, seq_loader, device, True)
        seq_elapsed_neural = (time.time() - seq_start_neural) * 1000.0
        sequence_times_neural[data] = seq_elapsed_neural

        seq_elapsed_rf = 0.0
        feat_sparse = seq_dataset.features[0][ind]
        ts_sparse = seq_dataset.ts[0][ind]

        if rf_model is not None:
            seq_start_rf = time.time()
            rf_X, _ = build_sequence_features(preds, feat_sparse, ts_sparse, args.rf_hist_window)
            residual_hat = rf_model.predict(rf_X).astype(np.float32)
            preds = apply_rf_correction(preds, residual_hat, alpha=args.rf_alpha, residual_clip=args.rf_clip)
            seq_elapsed_rf = (time.time() - seq_start_rf) * 1000.0
            sequence_times_rf[data] = seq_elapsed_rf

        seq_elapsed_total = seq_elapsed_neural + seq_elapsed_rf
        sequence_times[data] = seq_elapsed_total

        losses = np.mean((targets - preds) ** 2, axis=0)
        preds_seq.append(preds)
        targets_seq.append(targets)
        losses_seq.append(losses)

        # Per-sample timing
        num_samples = preds.shape[0]
        time_per_sample_neural = seq_elapsed_neural / num_samples if num_samples > 0 else 0.0
        time_per_sample_rf = seq_elapsed_rf / num_samples if num_samples > 0 else 0.0
        time_per_sample_total = seq_elapsed_total / num_samples if num_samples > 0 else 0.0
        inference_times_neural.extend([time_per_sample_neural] * num_samples)
        inference_times_rf.extend([time_per_sample_rf] * num_samples)
        inference_times_total.extend([time_per_sample_total] * num_samples)

        pos_pred = recon_traj_with_preds(seq_dataset, preds)[:, :2]
        pos_gt = seq_dataset.gt_pos[0][:, :2]

        traj_lens.append(np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1)))
        ate, rte = compute_ate_rte(pos_pred, pos_gt, pred_per_min)
        ate_all.append(ate)
        rte_all.append(rte)
        pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)

        if args.use_rf_postprocess:
            print('Sequence {}, loss {} / {}, ate {:.6f}, rte {:.6f}, neural={:.2f}ms, rf={:.2f}ms, total={:.2f}ms, time_per_sample={:.4f}ms'.format(
                data, losses, np.mean(losses), ate, rte, seq_elapsed_neural, seq_elapsed_rf, seq_elapsed_total, time_per_sample_total))
        else:
            print('Sequence {}, loss {} / {}, ate {:.6f}, rte {:.6f}, seq_time={:.2f}ms, time_per_sample={:.4f}ms'.format(
                data, losses, np.mean(losses), ate, rte, seq_elapsed_neural, time_per_sample_neural))

        # Plot figures
        kp = preds.shape[1]
        if kp == 2:
            targ_names = ['vx', 'vy']
        elif kp == 3:
            targ_names = ['vx', 'vy', 'vz']

        plt.figure('{}'.format(data), figsize=(16, 9))
        plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
        plt.plot(pos_pred[:, 0], pos_pred[:, 1])
        plt.plot(pos_gt[:, 0], pos_gt[:, 1])
        plt.title(data)
        plt.axis('equal')
        plt.legend(['Predicted', 'Ground truth'])
        plt.subplot2grid((kp, 2), (kp - 1, 0))
        plt.plot(pos_cum_error)
        plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])
        for i in range(kp):
            plt.subplot2grid((kp, 2), (i, 1))
            plt.plot(ind, preds[:, i])
            plt.plot(ind, targets[:, i])
            plt.legend(['Predicted', 'Ground truth'])
            plt.title('{}, error: {:.6f}'.format(targ_names[i], losses[i]))
        plt.tight_layout()

        if args.show_plot:
            plt.show()

        if args.out_dir is not None and osp.isdir(args.out_dir):
            np.save(osp.join(args.out_dir, data + '_gsn.npy'),
                    np.concatenate([pos_pred[:, :2], pos_gt[:, :2]], axis=1))
            plt.savefig(osp.join(args.out_dir, data + '_gsn.png'))

        plt.close('all')

    losses_seq = np.stack(losses_seq, axis=0)
    losses_avg = np.mean(losses_seq, axis=1)
    
    # Print timing summary
    print('\n' + '='*80)
    if args.use_rf_postprocess:
        print('[TIMING SUMMARY] YOLO26 + RF Postprocessing')
        print(f"Mean time per sample (neural): {np.mean(inference_times_neural):.4f} ms")
        print(f"Mean time per sample (RF): {np.mean(inference_times_rf):.4f} ms")
        print(f"Mean time per sample (total): {np.mean(inference_times_total):.4f} ms")
    else:
        print('[TIMING SUMMARY] YOLO26 Baseline Plain')
        print(f"Mean time per sample: {np.mean(inference_times_neural):.4f} ms")
    print('='*80 + '\n')
    
    # Export a csv file
    if args.out_dir is not None and osp.isdir(args.out_dir):
        with open(osp.join(args.out_dir, 'losses.csv'), 'w') as f:
            if losses_seq.shape[1] == 2:
                f.write('seq,vx,vy,avg,ate,rte\n')
            else:
                f.write('seq,vx,vy,vz,avg,ate,rte\n')
            for i in range(losses_seq.shape[0]):
                f.write('{},'.format(test_data_list[i]))
                for j in range(losses_seq.shape[1]):
                    f.write('{:.6f},'.format(losses_seq[i][j]))
                f.write('{:.6f},{:6f},{:.6f}\n'.format(losses_avg[i], ate_all[i], rte_all[i]))

    print('----------\nOverall loss: {}/{}, avg ATE:{}, avg RTE:{}'.format(
        np.average(losses_seq, axis=0), np.average(losses_avg), np.mean(ate_all), np.mean(rte_all)))
    return losses_avg


def write_config(args):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f)


def _init_epoch_metrics_csv(out_dir, target_dim):
    if out_dir is None:
        return None
    path = osp.join(out_dir, 'epoch_metrics.csv')
    if osp.exists(path):
        return path
    os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ['epoch', 'seconds', 'lr']
        if target_dim == 2:
            header += ['train_vx_mse', 'train_vy_mse', 'train_avg_mse', 'val_vx_mse', 'val_vy_mse', 'val_avg_mse']
        elif target_dim == 3:
            header += [
                'train_vx_mse', 'train_vy_mse', 'train_vz_mse', 'train_avg_mse',
                'val_vx_mse', 'val_vy_mse', 'val_vz_mse', 'val_avg_mse',
            ]
        else:
            header += ['train_avg_mse', 'val_avg_mse']
        header += ['best_val_avg_mse']
        w.writerow(header)
    return path


def _append_epoch_metrics(csv_path, epoch, seconds, lr, train_losses, val_losses, best_val_loss):
    if csv_path is None:
        return
    train_losses = np.asarray(train_losses).reshape(-1)
    val_losses_arr = None if val_losses is None else np.asarray(val_losses).reshape(-1)

    train_avg = float(np.mean(train_losses)) if train_losses.size else ''
    val_avg = float(np.mean(val_losses_arr)) if val_losses_arr is not None and val_losses_arr.size else ''

    row = [int(epoch), float(seconds), float(lr)]
    row += [float(x) for x in train_losses.tolist()]
    row += [train_avg]
    if val_losses_arr is None:
        row += [''] * (train_losses.size + 1)
    else:
        row += [float(x) for x in val_losses_arr.tolist()]
        row += [val_avg]
    row += [float(best_val_loss) if best_val_loss is not None and np.isfinite(best_val_loss) else '']

    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(row)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str)
    parser.add_argument('--val_list', type=str, default=None)
    parser.add_argument('--test_list', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default=None, help='Path to data directory')
    parser.add_argument('--cache_path', type=str, default=None, help='Path to cache folder to store processed data')
    parser.add_argument('--dataset', type=str, default='ronin', choices=['ronin', 'ridi'])
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--optim', type=str, default='musgd', choices=['musgd', 'adam', 'adamw'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--ns_steps', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--backbone', type=str, default='yolo26',
                        choices=['yolo26', 'mobilenetv2', 'shufflenetv2', 'efficientnet_lite0', 'tinycnn', 'lighttcn'])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--run_ekf', action='store_true')
    parser.add_argument('--fast_test', action='store_true')
    parser.add_argument('--show_plot', action='store_true')
    parser.add_argument('--no_tqdm', action='store_true',
                        help='Disable tqdm progress bar during training.')

    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--feature_sigma', type=float, default=0.00001)
    parser.add_argument('--target_sigma', type=float, default=0.00001)
    parser.add_argument('--model_dropout', type=float, default=0.2,
                        help='Dropout probability in YOLO26 head for plain baseline.')
    parser.add_argument('--use_attention', action='store_true',
                        help='Enable PSA attention block in YOLO26 plain baseline.')
    parser.add_argument('--use_rf_postprocess', action='store_true')
    parser.add_argument('--rf_model_path', type=str, default=None)
    parser.add_argument('--rf_alpha', type=float, default=1.0)
    parser.add_argument('--rf_clip', type=float, default=0.75)
    parser.add_argument('--rf_hist_window', type=int, default=50)

    args = parser.parse_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test_sequence(args)
    else:
        raise ValueError('Undefined mode')
