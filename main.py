#!/usr/bin/env python3
"""
Copyright (c) Ammar Kheder
Licensed under the MIT License.

v2 Finland: prediction PM2.5 residuelle multi-horizon (6/12/24/48 h).
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_preprocessing import load_and_preprocess_data, split_and_normalize_data
from src.dataset import AirQualityDataset
from src.model import LSTMAttentionModel, iTransformerModel
from src.training import train_model, evaluate_model
from src.knn_interpolation import knn_prediction

# v2 Finland: chemins data LUMI
DATA_PATH = '/scratch/project_462000640/ammar/finland_data/finland_2019_2024.csv'
TRAFFIC_PATH = '/scratch/project_462000640/ammar/finland_data/finland_traffic_2019_2024.csv'

SEQ_LENGTHS = [336, 72, 48, 24]                 # boucle multi-window
PREDICTION_HORIZONS = [6, 12, 24, 48]           # v2 Finland: multi-horizon
HORIZON_WEIGHTS = [1.0, 0.7, 0.5, 0.3]          # v2 Finland: loss ponderee
TARGET = 'PM2.5 (μg/m3)'

# v2 Finland: features modele (raw coords/calendrier exclus, distances encodent le spatial)
FEATURES = [
    'pm10', 'no2', 'no', 'aqi',
    'era5_temp', 'era5_rh', 'era5_wind', 'era5_pressure',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'hdd', 'stagnation', 'hdd_rolling_24h', 'hdd_rolling_72h',
    'is_heating_season', 'is_road_dust_season', 'is_newyear', 'is_juhannus',
    'pm25_lag_1', 'pm25_lag_24', 'pm25_roll_24',
    'min_distance', 'mean_distance', 'max_distance', 'std_distance',
    'traffic_volume', 'mean_speed', 'has_traffic',
]


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def main():
    global PREDICTION_HORIZONS  # v2 Finland: --horizons rebind le global
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=3)   # v2 Finland: overfit des E2
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--stride', type=int, default=6)
    ap.add_argument('--model', choices=['lstm', 'itransformer'],
                    default='lstm')  # v2 Finland: backbone selectionnable
    ap.add_argument('--no_log1p', action='store_true',
                    help='cible = z-score brut (clip) sans log1p')
    ap.add_argument('--scaler', choices=['standard', 'minmax'],
                    default='standard')  # v2 Finland: ablation normalisation
    ap.add_argument('--hotspot_alpha', type=float, default=0.0,
                    help='cran_pm: upweight pics (0=off, ~2.0 actif)')
    ap.add_argument('--under_penalty', type=float, default=1.0,
                    help='cran_pm: penalite sous-estimation (1=off, ~2.0 actif)')
    ap.add_argument('--seq_lengths', type=int, nargs='+', default=SEQ_LENGTHS)
    ap.add_argument('--horizons', type=int, nargs='+',
                    default=PREDICTION_HORIZONS)  # v2 Finland: horizons configurables
    ap.add_argument('--limit_rows', type=int, default=0,
                    help='dry run: garder seulement les N premieres lignes')
    args = ap.parse_args()
    set_seed(42)
    PREDICTION_HORIZONS = list(args.horizons)  # rebind local pour tout main()

    tp = TRAFFIC_PATH if os.path.exists(TRAFFIC_PATH) else None
    data = load_and_preprocess_data(DATA_PATH, traffic_path=tp)
    feats = [f for f in FEATURES if f in data.columns]
    missing = [f for f in FEATURES if f not in data.columns]
    if missing:
        print(f"[v2 Finland] features absentes ignorees: {missing}")
    if args.limit_rows:
        data = data.groupby('station_name').head(
            args.limit_rows // max(1, data['station_name'].nunique())).copy()

    # v2 Finland: embargo = max(seq) + max(horizon) pour bloquer toute
    # fuite par chevauchement de fenetres (stride=1) entre train/val/test.
    gap_hours = max(args.seq_lengths) + max(PREDICTION_HORIZONS)
    use_log1p = not args.no_log1p
    train_df, val_df, test_df, scaler_t = split_and_normalize_data(
        data, feats, TARGET, gap_hours=gap_hours, use_log1p=use_log1p,
        scaler=args.scaler)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | features={len(feats)} | horizons={PREDICTION_HORIZONS}")

    results = []
    for seq_len in args.seq_lengths:
        print(f"\n===== SEQ_LENGTH = {seq_len} =====")
        ds_kw = dict(features=feats, target=TARGET, seq_length=seq_len,
                     prediction_horizons=PREDICTION_HORIZONS, stride=args.stride)
        tr = AirQualityDataset(train_df, **ds_kw)
        va = AirQualityDataset(val_df, **ds_kw)
        te = AirQualityDataset(test_df, **ds_kw)
        print(f"samples: train={len(tr)} val={len(va)} test={len(te)}")
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            print("  -> pas assez de fenetres contigues, skip"); continue
        trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=4)
        tel = DataLoader(te, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # v2 Finland: capacite reduite (overfit massif des epoch 2) ->
        # num_layers 3->2, attention_dim_head 128->32, dropout 0.3
        if args.model == 'itransformer':
            model = iTransformerModel(input_dim=len(feats), seq_length=seq_len,
                                      horizons=PREDICTION_HORIZONS, d_model=64,
                                      depth=2, heads=4, dropout=0.3,
                                      residual=True).to(device)
        else:
            model = LSTMAttentionModel(input_dim=len(feats),
                                       horizons=PREDICTION_HORIZONS,
                                       hidden_dim=128, num_layers=2,
                                       attention_heads=4, attention_dim_head=32,
                                       dropout_rate=0.3, residual=True).to(device)
        model.device = device
        criterion = torch.nn.MSELoss()
        # v2 Finland: lr plus bas + weight_decay fort -> anti-overfit (val
        # decroche des epoch 2 sinon, peu de signal generalisable).
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
        # v2 Finland: Cosine au lieu de OneCycleLR (OneCycle se desynchronise
        # de l'early-stopping -> LR jamais complete son cycle).
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
        model, _, _ = train_model(model, trl, val, criterion, optimizer,
                                  scheduler, num_epochs=args.epochs,
                                  patience=args.patience,
                                  horizon_weights=HORIZON_WEIGHTS,
                                  hotspot_alpha=args.hotspot_alpha,
                                  under_penalty=args.under_penalty)
        test_loss, yt, yp = evaluate_model(model, tel, criterion, scaler_t,
                                           horizon_weights=HORIZON_WEIGHTS,
                                           use_log1p=use_log1p)
        for hi, h in enumerate(PREDICTION_HORIZONS):
            m = ~np.isnan(yt[:, hi])
            if m.sum() == 0:
                continue
            e = yp[m, hi] - yt[m, hi]
            results.append(dict(seq_length=seq_len, horizon=h,
                                rmse=float(np.sqrt(np.mean(e ** 2))),
                                mae=float(np.mean(np.abs(e))),
                                n=int(m.sum())))
        print(f"test_loss={test_loss:.4f}")

    if results:
        out = pd.DataFrame(results)
        out.to_csv("results_v2_finland.csv", index=False)
        print("\n=== RESULTATS (results_v2_finland.csv) ===")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
