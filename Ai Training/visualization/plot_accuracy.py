"""Parse and visualize Ai Training/accuracy_log.txt

Features:
- Parse the log format produced by model_training.py
- Plot validation accuracy per epoch for a selected run or for the best run
- Produce a summary CSV listing the best validation accuracy per run
- Compute average validation accuracy for filtered subsets (e.g. all runs with dropout=0.5)

Usage examples:
  python plot_accuracy.py --best        # prints best run and plots its epoch curve
  python plot_accuracy.py --plot 5     # plots run number 5 (1-indexed)
  python plot_accuracy.py --summary    # writes best_per_run_summary.csv
  python plot_accuracy.py --filter "dropout=0.5" --avg  # compute average best val_acc for filtered runs

Requires: pandas, matplotlib
"""
import argparse
import re
import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'accuracy_log.txt')

RUN_HEADER_RE = re.compile(r"RUN\s+(\d+)/(\d+)\s+PARAMETERS:\s+dropout=([0-9.e-]+),\s*batch_size=([0-9]+),\s*learning_rate=([0-9.e-]+),\s*color_jitter=([0-9.e-]+)")
CSV_LINE_RE = re.compile(r"^([0-9.eE+-]+),([0-9]+),([0-9.eE+-]+),([0-9.eE+-]+),(\d+),([0-9.]+),([0-9.]+)\s*$")


def parse_log(path=LOG_PATH):
    runs = []  # list of dicts: each run has params and a list of epoch dicts
    if not os.path.exists(path):
        raise FileNotFoundError(f"Log file not found: {path}")

    current_run = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            if not line.strip():
                continue
            m = RUN_HEADER_RE.search(line)
            if m:
                # start new run
                run_num = int(m.group(1))
                total = int(m.group(2))
                dropout = float(m.group(3))
                batch_size = int(m.group(4))
                lr = float(m.group(5))
                cj = float(m.group(6))
                current_run = {
                    'run_idx': run_num,
                    'total_runs': total,
                    'dropout': dropout,
                    'batch_size': batch_size,
                    'learning_rate': lr,
                    'color_jitter': cj,
                    'epochs': []
                }
                runs.append(current_run)
                continue
            # try CSV line
            m2 = CSV_LINE_RE.match(line)
            if m2 and current_run is not None:
                d = float(m2.group(1))
                b = int(m2.group(2))
                lr = float(m2.group(3))
                cj = float(m2.group(4))
                epoch = int(m2.group(5))
                train_acc = float(m2.group(6))
                val_acc = float(m2.group(7))
                current_run['epochs'].append({
                    'epoch': epoch,
                    'train_acc': train_acc,
                    'val_acc': val_acc
                })
            else:
                # ignore unexpected lines
                continue

    return runs


def runs_to_dataframe(runs):
    rows = []
    for r in runs:
        for e in r['epochs']:
            rows.append({
                'run_idx': r['run_idx'],
                'dropout': r['dropout'],
                'batch_size': r['batch_size'],
                'learning_rate': r['learning_rate'],
                'color_jitter': r['color_jitter'],
                'epoch': e['epoch'],
                'train_acc': e['train_acc'],
                'val_acc': e['val_acc']
            })
    return pd.DataFrame(rows)


def summarize_best_per_run(df):
    # returns DataFrame with one row per run, best val_acc and epoch
    best = df.loc[df.groupby('run_idx')['val_acc'].idxmax()].copy()
    best = best.sort_values(['val_acc'], ascending=False).reset_index(drop=True)
    return best


def plot_run(df, run_idx):
    sub = df[df['run_idx'] == run_idx]
    if sub.empty:
        print(f"No data for run {run_idx}")
        return
    plt.figure(figsize=(8,5))
    plt.plot(sub['epoch'], sub['train_acc'], label='train_acc')
    plt.plot(sub['epoch'], sub['val_acc'], label='val_acc')
    plt.title(f"Run {run_idx}: dropout={sub['dropout'].iloc[0]}, batch={sub['batch_size'].iloc[0]}, lr={sub['learning_rate'].iloc[0]}, cj={sub['color_jitter'].iloc[0]}")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def find_best_combination(best_df):
    if best_df.empty:
        return None
    best = best_df.iloc[0]
    return best.to_dict()


def filter_df(df, filter_expr):
    # filter_expr example: "dropout=0.5,batch_size=32"
    if not filter_expr:
        return df
    exprs = [x.strip() for x in filter_expr.split(',') if x.strip()]
    sub = df.copy()
    for e in exprs:
        if '=' not in e:
            continue
        k,v = e.split('=',1)
        k=k.strip(); v=v.strip()
        # try numeric
        try:
            fv = float(v)
            if fv.is_integer():
                fv = int(fv)
            sub = sub[sub[k]==fv]
        except Exception:
            sub = sub[sub[k]==v]
    return sub


def avg_for_subset(best_df, filter_expr):
    sub = filter_df(best_df, filter_expr)
    if sub.empty:
        return None
    return {
        'count': len(sub),
        'avg_best_val_acc': sub['val_acc'].mean(),
        'std_best_val_acc': sub['val_acc'].std()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', type=int, help='Plot run number (1-indexed)')
    parser.add_argument('--best', action='store_true', help='Show best run and plot its epochs')
    parser.add_argument('--summary', action='store_true', help='Write best_per_run_summary.csv')
    parser.add_argument('--filter', type=str, help='Filter expression like "dropout=0.5,batch_size=32"')
    parser.add_argument('--avg', action='store_true', help='Compute average best val_acc for filter expression')
    args = parser.parse_args()

    runs = parse_log()
    df = runs_to_dataframe(runs)
    if df.empty:
        print('No run data parsed from log.')
        return

    best_df = summarize_best_per_run(df)

    if args.summary:
        out = os.path.join(os.path.dirname(LOG_PATH), 'best_per_run_summary.csv')
        best_df.to_csv(out, index=False)
        print('Wrote', out)

    if args.filter and args.avg:
        res = avg_for_subset(best_df, args.filter)
        if res is None:
            print('No runs match filter')
        else:
            print(f"Filter={args.filter} count={res['count']} avg_best_val_acc={res['avg_best_val_acc']:.4f} std={res['std_best_val_acc']:.4f}")

    if args.best:
        best = find_best_combination(best_df)
        if best is None:
            print('No best run found')
        else:
            print('Best run:')
            print(best)
            plot_run(df, int(best['run_idx']))

    if args.plot:
        plot_run(df, args.plot)

if __name__ == '__main__':
    main()
