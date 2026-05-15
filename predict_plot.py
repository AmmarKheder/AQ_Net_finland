#!/usr/bin/env python3
"""
Copyright (c) Ammar Kheder
Licensed under the MIT License.

v2 Finland: diagnostic predictions vs verite-terrain PAR STATION.
Recharge le checkpoint, infere sur le TEST (meme split seed=42),
sort: tableau metriques/station/horizon + PNG (serie temporelle + scatter).
"""

import os, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(42); np.random.seed(42); torch.manual_seed(42)

from src.data_preprocessing import load_and_preprocess_data, split_and_normalize_data
from src.dataset import AirQualityDataset
from src.model import LSTMAttentionModel
from main import DATA_PATH, FEATURES, TARGET, PREDICTION_HORIZONS

SEQ = 24
STRIDE = 6
CKPT = "best_lstm_attention_model.pth"
OUTDIR = "plots_v2"
os.makedirs(OUTDIR, exist_ok=True)


def r2(yt, yp):
    ss = np.sum((yt - yt.mean()) ** 2)
    return 1.0 - np.sum((yp - yt) ** 2) / ss if ss > 0 else float("nan")


def main():
    d = load_and_preprocess_data(DATA_PATH, traffic_path=None)
    feats = [f for f in FEATURES if f in d.columns]
    tr, va, te, sc = split_and_normalize_data(d, feats, TARGET)
    # coord -> nom station (le dataset n'expose que coord/sid)
    coord2name = (te.drop_duplicates(['longitude', 'latitude'])
                    .set_index(['longitude', 'latitude'])['station_name'].to_dict())

    ds = AirQualityDataset(te, feats, TARGET, seq_length=SEQ,
                           prediction_horizons=PREDICTION_HORIZONS, stride=STRIDE)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    print(f"test samples: {len(ds)}  stations: {te['station_name'].nunique()}")

    model = LSTMAttentionModel(input_dim=len(feats),
                               horizons=PREDICTION_HORIZONS, hidden_dim=128,
                               attention_heads=2, dropout_rate=0.1, residual=True)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval(); model.device = torch.device("cpu")

    rows = []
    with torch.no_grad():
        for x, y, cur, mask, dates, sids, coords in dl:
            pred, _ = model(x, cur)
            pred = pred.numpy(); y = y.numpy(); mask = mask.numpy()
            lon = coords[0].numpy(); lat = coords[1].numpy()
            for b in range(len(pred)):
                nm = coord2name.get((float(lon[b]), float(lat[b])), f"st{int(sids[b])}")
                for hi, h in enumerate(PREDICTION_HORIZONS):
                    if mask[b, hi] > 0:
                        rows.append((nm, dates[b], h,
                                     float(y[b, hi]), float(pred[b, hi])))
    df = pd.DataFrame(rows, columns=["station", "ts", "horizon", "yt_s", "yp_s"])
    # de-normalisation (espace StandardScaler -> ug/m3)
    df["y_true"] = sc.inverse_transform(df[["yt_s"]].values).ravel()
    df["y_pred"] = np.clip(
        sc.inverse_transform(df[["yp_s"]].values).ravel(), 0, None)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["station", "horizon", "ts"])
    df.to_csv("predictions_test_v2.csv", index=False)

    # --- tableau metriques par station x horizon ---
    print("\n=== METRIQUES PAR STATION x HORIZON ===")
    print(f"{'station':32s} {'h':>3s} {'n':>6s} {'RMSE':>6s} {'MAE':>5s} {'R2':>6s}")
    summ = []
    for (st, h), g in df.groupby(["station", "horizon"]):
        e = g["y_pred"].values - g["y_true"].values
        rmse = np.sqrt(np.mean(e ** 2)); mae = np.mean(np.abs(e))
        r = r2(g["y_true"].values, g["y_pred"].values)
        summ.append(dict(station=st, horizon=h, n=len(g),
                         rmse=rmse, mae=mae, r2=r))
        print(f"{st:32.32s} {h:3d} {len(g):6d} {rmse:6.2f} {mae:5.2f} {r:6.3f}")
    pd.DataFrame(summ).to_csv("metrics_per_station_v2.csv", index=False)

    print("\n=== GLOBAL PAR HORIZON ===")
    for h in PREDICTION_HORIZONS:
        g = df[df.horizon == h]
        e = g["y_pred"].values - g["y_true"].values
        print(f"  {h:3d}h  RMSE={np.sqrt(np.mean(e**2)):5.2f} "
              f"MAE={np.mean(np.abs(e)):5.2f} R2={r2(g['y_true'].values, g['y_pred'].values):6.3f}")

    # --- PNG par station : serie temporelle (6h & 24h) + scatter ---
    for st, g in df.groupby("station"):
        fig, ax = plt.subplots(2, 2, figsize=(16, 8))
        fig.suptitle(st, fontsize=13)
        for col, h in enumerate([6, 24]):
            gh = g[g.horizon == h].sort_values("ts")
            sl = gh.iloc[:600]                          # ~ derniers points lisibles
            ax[0, col].plot(sl["ts"], sl["y_true"], label="vrai", lw=0.9)
            ax[0, col].plot(sl["ts"], sl["y_pred"], label="pred", lw=0.9, alpha=0.8)
            ax[0, col].set_title(f"+{h}h  serie (600 pts)"); ax[0, col].legend()
            ax[1, col].scatter(gh["y_true"], gh["y_pred"], s=4, alpha=0.3)
            mx = max(gh["y_true"].max(), gh["y_pred"].max(), 1)
            ax[1, col].plot([0, mx], [0, mx], 'r--', lw=1)
            rr = r2(gh["y_true"].values, gh["y_pred"].values)
            rm = np.sqrt(np.mean((gh["y_pred"].values - gh["y_true"].values) ** 2))
            ax[1, col].set_title(f"+{h}h  scatter  RMSE={rm:.2f} R2={rr:.3f}")
            ax[1, col].set_xlabel("vrai (ug/m3)"); ax[1, col].set_ylabel("pred")
        fig.tight_layout()
        safe = st.replace("/", "_").replace(" ", "_")
        fig.savefig(f"{OUTDIR}/{safe}.png", dpi=90)
        plt.close(fig)
    print(f"\nPNG -> {OUTDIR}/  ({df['station'].nunique()} stations)")
    print("CSV -> predictions_test_v2.csv , metrics_per_station_v2.csv")


if __name__ == "__main__":
    main()
