#!/usr/bin/env python3
"""
Copyright (c) Ammar Kheder
Licensed under the MIT License.

v2 Finland: dataset multi-horizon, residuel, gap-aware.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AirQualityDataset(Dataset):
    """
    v2 Finland :
      - prediction_horizons : liste (ex. [6, 12, 24, 48])
      - decoupage GAP-AWARE : une fenetre n'est gardee que si l'input est
        horaire strictement contigu (aucune heure manquante) ET pm25_current
        present. Les cibles manquantes par horizon -> masque (0) au lieu de
        jeter l'echantillon.
      - __getitem__ -> (x, y, pm25_current, mask, date, station_id, coord)
      - stockage paresseux (slicing a la volee) : indispensable au volume
        (20 stations x 52k h x seq jusqu'a 336).
    """

    def __init__(self, data, features, target, seq_length=336,
                 prediction_horizons=(6, 12, 24, 48), stride=1):
        self.features = list(features)
        self.target = target
        self.seq_length = int(seq_length)
        self.horizons = list(prediction_horizons)
        self.max_h = max(self.horizons)
        self.stride = int(stride)

        self._stations = []   # liste de dicts par station (arrays reindexes)
        self._index = []      # (station_k, i_start) des fenetres valides

        for sid, ((lon, lat), sdata) in enumerate(
                data.groupby(['longitude', 'latitude'])):
            sdata = sdata.sort_values('date')
            t = pd.to_datetime(sdata['date'].values)
            # grille horaire complete -> revele les trous (lignes inserees NaN)
            full = pd.date_range(t.min(), t.max(), freq='h')
            sdata = sdata.set_index(t).reindex(full)
            present = sdata[target].notna().values.copy()  # heure reellement mesuree
            F = sdata[self.features].to_numpy(dtype=np.float32)  # NaN sur trous
            y = sdata[target].to_numpy(dtype=np.float32)
            dates = full.astype('datetime64[ns]')
            k = len(self._stations)
            self._stations.append(dict(F=F, y=y, present=present,
                                       dates=dates, sid=sid, coord=(lon, lat)))
            T = len(full)
            L, mh = self.seq_length, self.max_h
            for i in range(0, T - L - mh, self.stride):
                cur = i + L - 1
                # input contigu (aucun trou) + valeur courante presente
                if not present[i:i + L].all():
                    continue
                if not present[cur]:
                    continue
                # au moins 1 horizon avec verite-terrain
                ok = False
                for h in self.horizons:
                    j = cur + h
                    if j < T and present[j]:
                        ok = True
                        break
                if ok:
                    self._index.append((k, i))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        k, i = self._index[idx]
        st = self._stations[k]
        L, cur = self.seq_length, i + self.seq_length - 1
        x = torch.from_numpy(st['F'][i:i + L])               # (L, n_feat)
        pm25_current = torch.tensor(st['y'][cur], dtype=torch.float32)
        y = torch.zeros(len(self.horizons), dtype=torch.float32)
        mask = torch.zeros(len(self.horizons), dtype=torch.float32)
        T = len(st['y'])
        for hi, h in enumerate(self.horizons):
            j = cur + h
            if j < T and st['present'][j]:
                y[hi] = float(st['y'][j])
                mask[hi] = 1.0
        return (x, y, pm25_current, mask,
                str(st['dates'][cur]), st['sid'], st['coord'])
