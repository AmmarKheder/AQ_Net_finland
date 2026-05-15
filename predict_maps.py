#!/usr/bin/env python3
"""
Copyright (c) Ammar Kheder
Licensed under the MIT License.

v2 Finland: cartes Finlande + graphiques sur les JOURNEES INTERESSANTES
(episodes de forte pollution PM2.5) — verite vs prediction. Colormap RdYlBu_r.
"""
import os, random, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

random.seed(42); np.random.seed(42); torch.manual_seed(42)
from src.data_preprocessing import load_and_preprocess_data, split_and_normalize_data
from src.dataset import AirQualityDataset
from src.model import LSTMAttentionModel
from main import DATA_PATH, FEATURES, TARGET, PREDICTION_HORIZONS

SEQ, STRIDE = 48, 6
CKPT = "best_lstm_attention_model.pth"
OUT = "maps_v2"; os.makedirs(OUT, exist_ok=True)
CMAP = "RdYlBu_r"                                  # bleu=propre, rouge=pollue


def main():
    d = load_and_preprocess_data(DATA_PATH, traffic_path=None)
    feats = [f for f in FEATURES if f in d.columns]
    tr, va, te, sc = split_and_normalize_data(d, feats, TARGET,
                                              gap_hours=SEQ + max(PREDICTION_HORIZONS))
    coord2name = (te.drop_duplicates(['longitude', 'latitude'])
                    .set_index(['longitude', 'latitude'])['station_name'].to_dict())

    ds = AirQualityDataset(te, feats, TARGET, seq_length=SEQ,
                           prediction_horizons=PREDICTION_HORIZONS, stride=STRIDE)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    model = LSTMAttentionModel(input_dim=len(feats), horizons=PREDICTION_HORIZONS,
                               hidden_dim=128, num_layers=2, attention_heads=4,
                               attention_dim_head=32, dropout_rate=0.3,
                               residual=True)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval(); model.device = torch.device("cpu")

    rows = []
    with torch.no_grad():
        for x, y, cur, mask, dates, sids, coords in dl:
            pred, _ = model(x, cur)
            pred = pred.numpy(); y = y.numpy(); mask = mask.numpy()
            lon = coords[0].numpy(); lat = coords[1].numpy()
            for b in range(len(pred)):
                nm = coord2name.get((float(lon[b]), float(lat[b])), "?")
                for hi, h in enumerate(PREDICTION_HORIZONS):
                    if mask[b, hi] > 0:
                        rows.append((nm, float(lon[b]), float(lat[b]),
                                     pd.Timestamp(dates[b]), h,
                                     float(y[b, hi]), float(pred[b, hi])))
    df = pd.DataFrame(rows, columns=["station", "lon", "lat", "ts",
                                     "horizon", "yt_s", "yp_s"])
    df["y_true"] = sc.inverse_transform(df[["yt_s"]].values).ravel()
    df["y_true"] = np.expm1(df["y_true"]).clip(0)
    df["y_pred"] = np.expm1(
        sc.inverse_transform(df[["yp_s"]].values).ravel()).clip(0)

    # --- journees interessantes = jours avec la + forte moyenne reseau PM2.5 ---
    h0 = PREDICTION_HORIZONS[0]
    g = df[df.horizon == h0].copy()
    g["day"] = g["ts"].dt.floor("D")
    day_mean = g.groupby("day")["y_true"].mean().sort_values(ascending=False)
    episodes = list(day_mean.head(4).index)
    print("Journees les + polluees (test):")
    for dd in episodes:
        print(f"  {dd.date()}  moyenne reseau = {day_mean[dd]:.1f} ug/m3")

    # ===== CARTES : verite vs prediction par episode (horizon +24h) =====
    HMAP = 24 if 24 in PREDICTION_HORIZONS else PREDICTION_HORIZONS[-1]
    for dd in episodes:
        sub = df[(df.horizon == HMAP) & (df.ts.dt.floor("D") == dd)]
        if sub.empty:
            continue
        st = sub.groupby(["station", "lon", "lat"]).agg(
            y_true=("y_true", "mean"), y_pred=("y_pred", "mean")).reset_index()
        vmax = max(st.y_true.max(), st.y_pred.max())
        vmin = min(st.y_true.min(), st.y_pred.min())
        fig, ax = plt.subplots(1, 2, figsize=(15, 8),
                               sharex=True, sharey=True)
        for k, (col, lab) in enumerate([("y_true", "Observé"),
                                        ("y_pred", f"Prédit +{HMAP}h")]):
            scd = ax[k].scatter(st.lon, st.lat, c=st[col], cmap=CMAP,
                                vmin=vmin, vmax=vmax, s=260,
                                edgecolor="k", linewidth=0.6)
            for _, r in st.iterrows():
                ax[k].annotate(f"{r[col]:.0f}", (r.lon, r.lat),
                               fontsize=7, ha="center", va="center")
            ax[k].set_title(f"{lab}", fontsize=13)
            ax[k].set_xlabel("Longitude"); ax[k].set_ylabel("Latitude")
            ax[k].grid(alpha=0.25)
        cb = fig.colorbar(scd, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label("PM2.5 (µg/m³)")
        fig.suptitle(f"Finlande — épisode {dd.date()}  "
                     f"(moyenne réseau {day_mean[dd]:.1f} µg/m³)",
                     fontsize=15)
        fig.savefig(f"{OUT}/carte_{dd.date()}.png", dpi=110,
                    bbox_inches="tight")
        plt.close(fig)

    # ===== GRAPHIQUE : serie temporelle autour du PIRE episode =====
    worst = episodes[0]
    win0 = worst - pd.Timedelta(days=3)
    win1 = worst + pd.Timedelta(days=4)
    gg = df[(df.horizon == HMAP) & (df.ts >= win0) & (df.ts <= win1)]
    top_st = (gg.groupby("station")["y_true"].max()
                .sort_values(ascending=False).head(4).index)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, s in zip(axes.ravel(), top_st):
        z = gg[gg.station == s].sort_values("ts")
        ax.plot(z.ts, z.y_true, label="observé", lw=1.4)
        ax.plot(z.ts, z.y_pred, label=f"prédit +{HMAP}h", lw=1.4, alpha=.8)
        ax.axvspan(worst, worst + pd.Timedelta(days=1), color="red",
                   alpha=0.08)
        ax.set_title(s, fontsize=11); ax.grid(alpha=.25); ax.legend(fontsize=8)
        ax.set_ylabel("PM2.5 µg/m³")
    fig.suptitle(f"Épisode {worst.date()} — séries observé vs prédit "
                 f"(+{HMAP}h), top 4 stations", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT}/episode_{worst.date()}_series.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)

    df.to_csv("predictions_final_v2.csv", index=False)
    print(f"\nPNG -> {OUT}/  ({len(episodes)} cartes + 1 serie)")
    print("CSV -> predictions_final_v2.csv")


if __name__ == "__main__":
    main()
