import argparse
import csv
import json
import os
import time
from os import path as osp

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor


def parse_csv_ints(text):
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def parse_csv_depths(text):
    out = []
    for token in [x.strip() for x in text.split(',') if x.strip()]:
        if token.lower() == 'none':
            out.append(None)
        else:
            out.append(int(token))
    return out


def parse_csv_floats_or_sqrt(text):
    out = []
    for token in [x.strip() for x in text.split(',') if x.strip()]:
        if token.lower() == 'sqrt':
            out.append('sqrt')
        else:
            out.append(float(token))
    return out


def build_param_grid(args):
    return {
        'n_estimators': parse_csv_ints(args.n_estimators),
        'max_depth': parse_csv_depths(args.max_depth),
        'min_samples_leaf': parse_csv_ints(args.min_samples_leaf),
        'max_features': parse_csv_floats_or_sqrt(args.max_features),
    }


def load_npz_dataset(path):
    data = np.load(path, allow_pickle=True)
    return {
        'X': data['X'].astype(np.float32),
        'y': data['y'].astype(np.float32),
        'feature_names': data['feature_names'].tolist() if 'feature_names' in data else None,
    }


def create_model(params, output_mode, random_state, n_jobs):
    base = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        random_state=random_state,
        n_jobs=n_jobs,
    )

    if output_mode == 'multi':
        return base
    if output_mode == 'independent':
        return MultiOutputRegressor(base, n_jobs=n_jobs)
    raise ValueError(f'Unknown output_mode: {output_mode}')


def evaluate_residual_mse(y_true, y_pred):
    mse_x = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    mse_y = mean_squared_error(y_true[:, 1], y_pred[:, 1])
    return mse_x, mse_y, (mse_x + mse_y) / 2.0


def main():
    parser = argparse.ArgumentParser(description='Train RF residual postprocessor for YOLO26 predictions.')
    parser.add_argument('--train_npz', type=str, required=True)
    parser.add_argument('--val_npz', type=str, required=True)
    parser.add_argument('--out_model', type=str, required=True)

    parser.add_argument('--output_mode', type=str, default='multi', choices=['multi', 'independent'])
    parser.add_argument('--n_estimators', type=str, default='200,400,800')
    parser.add_argument('--max_depth', type=str, default='12,16,24,None')
    parser.add_argument('--min_samples_leaf', type=str, default='1,5,20')
    parser.add_argument('--max_features', type=str, default='sqrt,0.5,1.0')

    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--trial_log_path', type=str, default=None)

    args = parser.parse_args()

    train = load_npz_dataset(args.train_npz)
    val = load_npz_dataset(args.val_npz)

    X_train, y_train = train['X'], train['y']
    X_val, y_val = val['X'], val['y']

    print('Loaded datasets:')
    print(f'  train: X={X_train.shape}, y={y_train.shape}')
    print(f'  val:   X={X_val.shape}, y={y_val.shape}')

    baseline_pred = np.zeros_like(y_val)
    b_mse_x, b_mse_y, b_mse_avg = evaluate_residual_mse(y_val, baseline_pred)
    print('Validation baseline (no RF correction) residual MSE:')
    print(f'  mse_x={b_mse_x:.6f}, mse_y={b_mse_y:.6f}, mse_avg={b_mse_avg:.6f}')

    grid = build_param_grid(args)
    print('Tuning grid:')
    print(json.dumps(grid, indent=2, default=str))

    total_trials = (
        len(grid['n_estimators']) *
        len(grid['max_depth']) *
        len(grid['min_samples_leaf']) *
        len(grid['max_features'])
    )
    print(f'Total trials: {total_trials}')

    best = None
    trials = 0
    trial_rows = []
    start_time = time.time()

    for n_estimators in grid['n_estimators']:
        for max_depth in grid['max_depth']:
            for min_samples_leaf in grid['min_samples_leaf']:
                for max_features in grid['max_features']:
                    params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_leaf': min_samples_leaf,
                        'max_features': max_features,
                    }
                    trials += 1
                    trial_start = time.time()

                    print('-' * 90)
                    print(f'Trial {trials}/{total_trials} START')
                    print(f'  params: {params}')
                    print('  phase: fitting model on train split')

                    model = create_model(
                        params=params,
                        output_mode=args.output_mode,
                        random_state=args.random_state,
                        n_jobs=args.n_jobs,
                    )
                    model.fit(X_train, y_train)
                    print('  phase: evaluating model on val split')
                    y_hat = model.predict(X_val).astype(np.float32)

                    mse_x, mse_y, mse_avg = evaluate_residual_mse(y_val, y_hat)
                    elapsed_trial = time.time() - trial_start

                    is_best = False
                    if best is None or mse_avg < best['mse_avg']:
                        is_best = True
                        best = {
                            'model': model,
                            'params': params,
                            'mse_x': mse_x,
                            'mse_y': mse_y,
                            'mse_avg': mse_avg,
                            'trial': trials,
                        }

                    elapsed_total = time.time() - start_time
                    avg_trial_time = elapsed_total / trials
                    remaining = total_trials - trials
                    eta_seconds = remaining * avg_trial_time

                    print('  metrics: ' f'mse_x={mse_x:.6f}, mse_y={mse_y:.6f}, mse_avg={mse_avg:.6f}')
                    print(f'  trial_time_sec: {elapsed_trial:.2f}')
                    print(
                        f'  progress: {trials}/{total_trials} '
                        f'({(100.0 * trials / total_trials):.1f}%), '
                        f'eta_sec={eta_seconds:.1f}'
                    )
                    if is_best:
                        print('  status: NEW_BEST')
                    else:
                        print(
                            '  status: not best '
                            f"(best_trial={best['trial']}, best_mse_avg={best['mse_avg']:.6f})"
                        )

                    trial_rows.append({
                        'trial': trials,
                        'total_trials': total_trials,
                        'n_estimators': n_estimators,
                        'max_depth': 'None' if max_depth is None else max_depth,
                        'min_samples_leaf': min_samples_leaf,
                        'max_features': max_features,
                        'mse_x': float(mse_x),
                        'mse_y': float(mse_y),
                        'mse_avg': float(mse_avg),
                        'trial_time_sec': float(elapsed_trial),
                        'elapsed_total_sec': float(elapsed_total),
                        'eta_sec': float(eta_seconds),
                        'is_best': int(is_best),
                    })

    if best is None:
        raise RuntimeError('No RF model was trained')

    out_dir = osp.dirname(args.out_model)
    if out_dir and not osp.isdir(out_dir):
        os.makedirs(out_dir)

    joblib.dump(best['model'], args.out_model)

    summary_path = osp.splitext(args.out_model)[0] + '_summary.json'
    summary = {
        'train_npz': args.train_npz,
        'val_npz': args.val_npz,
        'output_mode': args.output_mode,
        'best_params': best['params'],
        'val_mse_x': float(best['mse_x']),
        'val_mse_y': float(best['mse_y']),
        'val_mse_avg': float(best['mse_avg']),
        'baseline_zero_residual_mse_avg': float(b_mse_avg),
        'feature_names': train['feature_names'],
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    trial_log_path = args.trial_log_path
    if trial_log_path is None:
        trial_log_path = osp.splitext(args.out_model)[0] + '_trials.csv'

    trial_log_dir = osp.dirname(trial_log_path)
    if trial_log_dir and not osp.isdir(trial_log_dir):
        os.makedirs(trial_log_dir)

    with open(trial_log_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'trial',
                'total_trials',
                'n_estimators',
                'max_depth',
                'min_samples_leaf',
                'max_features',
                'mse_x',
                'mse_y',
                'mse_avg',
                'trial_time_sec',
                'elapsed_total_sec',
                'eta_sec',
                'is_best',
            ],
        )
        writer.writeheader()
        for row in trial_rows:
            writer.writerow(row)

    print('Saved best RF model:')
    print(f'  model: {args.out_model}')
    print(f'  summary: {summary_path}')
    print(f'  trials_log: {trial_log_path}')
    print(f"  best_params: {best['params']}")
    print('  best_val_mse: ' f"x={best['mse_x']:.6f}, y={best['mse_y']:.6f}, avg={best['mse_avg']:.6f}")


if __name__ == '__main__':
    main()
