#!/usr/bin/env python3
"""
Copyright (c) Ammar Kheder
Licensed under the MIT License.

v2 Finland: adapte au nouveau format FMI (finland_2019_2024.csv).
"""

import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

# v2 Finland: criteres d'inclusion station (decisions utilisateur)
MIN_PM25_COV = 0.80                     # PM2.5 >= 80% des heures pleines
MIN_MET_COV = 0.70                      # temp/rh/wind/pressure tous >= 70%
# v2 Finland: polluants conserves comme features (co/o3/so2/bc_pm25 supprimes)
DROP_POLLUTANTS = ['co', 'o3', 'so2', 'bc_pm25', 'odorous_sulphur_compounds']
MEDIAN_IMPUTE = ['pm10', 'no2', 'no', 'aqi']   # imputation mediane PAR STATION


def load_raw_data(data_path):
    # v2 Finland: nouveau format FMI = timestamp UTC + colonnes minuscules
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values(['timestamp', 'longitude', 'latitude']).reset_index(drop=True)
    return df


def _station_coverage_filter(df):
    # v2 Finland: garde PM2.5>=80% ET temp/rh/wind/pressure tous >=70%
    span = df['timestamp']
    full_h = int((span.max() - span.min()).total_seconds() // 3600) + 1
    kept = []
    for st, s in df.groupby('station_name'):
        c = {k: s[k].notna().sum() / full_h
             for k in ('pm25', 'temp', 'rh', 'wind', 'pressure')}
        if (c['pm25'] >= MIN_PM25_COV
                and min(c['temp'], c['rh'], c['wind'], c['pressure']) >= MIN_MET_COV):
            kept.append(st)
    return df[df['station_name'].isin(kept)].copy(), sorted(kept)


def load_and_preprocess_data(data_path, traffic_path=None):
    data = load_raw_data(data_path)

    # v2 Finland: filtre stations (criteres de couverture)
    data, kept = _station_coverage_filter(data)
    print(f"[v2 Finland] {len(kept)} stations train retenues : {kept}")

    # v2 Finland: calendrier derive du timestamp UTC (remplace year/month/day/hour bruts)
    data['date'] = data['timestamp']
    data['year'] = data['timestamp'].dt.year
    data['month'] = data['timestamp'].dt.month
    data['day'] = data['timestamp'].dt.day
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['z'] = 1.0
    data = data.drop(columns=['timestamp'])  # 'date' devient canonique

    # v2 Finland: renommages (prefixe era5_ conserve pour compat pipeline)
    data = data.rename(columns={
        'pm25': 'PM2.5 (μg/m3)',
        'temp': 'era5_temp', 'rh': 'era5_rh',
        'wind': 'era5_wind', 'pressure': 'era5_pressure',
    })
    target = 'PM2.5 (μg/m3)'

    # v2 Finland: suppression polluants trop creux
    data = data.drop(columns=[c for c in DROP_POLLUTANTS if c in data.columns])

    data = data.sort_values(['date', 'longitude', 'latitude']).reset_index(drop=True)

    # v2 Finland: merge traffic sur (date, nearest_pm25_station <-> station_name)
    if traffic_path is not None:
        if os.path.exists(traffic_path):
            tr = pd.read_csv(traffic_path, usecols=[
                'timestamp', 'nearest_pm25_station', 'traffic_volume', 'mean_speed'])
            tr['date'] = pd.to_datetime(tr['timestamp'], utc=True)
            tr = (tr.groupby(['date', 'nearest_pm25_station'], as_index=False)
                    .agg(traffic_volume=('traffic_volume', 'sum'),
                         mean_speed=('mean_speed', 'mean')))
            data = data.merge(
                tr, how='left',
                left_on=['date', 'station_name'],
                right_on=['date', 'nearest_pm25_station'])
            data = data.drop(columns=[c for c in ['nearest_pm25_station']
                                      if c in data.columns])
            # v2 Finland: has_traffic distingue "0 = pas de capteur" de "0 reel"
            data['has_traffic'] = data['traffic_volume'].notna().astype(int)
            data['traffic_volume'] = data['traffic_volume'].fillna(0.0)
            data['mean_speed'] = data['mean_speed'].fillna(0.0)
        else:
            print(f"[v2 Finland] traffic absent ({traffic_path}) -> traffic=0")
            data['traffic_volume'] = 0.0
            data['mean_speed'] = 0.0
            data['has_traffic'] = 0

    # v2 Finland: imputation mediane PAR STATION (jamais la cible)
    for c in MEDIAN_IMPUTE:
        if c in data.columns:
            data[c] = data.groupby('station_name')[c].transform(
                lambda x: x.fillna(x.median()))
            data[c] = data[c].fillna(data[c].median())  # fallback global

    # v2 Finland: meteo imputee AVANT les features derivees (sinon NaN propages)
    for c in ('era5_temp', 'era5_rh', 'era5_wind', 'era5_pressure'):
        data[c] = data.groupby('station_name')[c].transform(
            lambda x: x.fillna(x.median()))
        data[c] = data[c].fillna(data[c].median())

    # features cycliques (logique existante conservee)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)

    # v2 Finland: features derivees meteo / saison / calendrier
    data['hdd'] = (17 - data['era5_temp']).clip(lower=0)
    data['stagnation'] = data['hdd'] / (data['era5_wind'] + 0.1)
    data['hdd_rolling_24h'] = data.groupby('station_name')['hdd'] \
        .transform(lambda x: x.rolling(24, min_periods=1).sum())
    data['hdd_rolling_72h'] = data.groupby('station_name')['hdd'] \
        .transform(lambda x: x.rolling(72, min_periods=1).sum())
    data['is_heating_season'] = data['month'].isin(
        [10, 11, 12, 1, 2, 3, 4]).astype(int)
    data['is_road_dust_season'] = data['month'].isin([3, 4]).astype(int)
    data['is_newyear'] = ((data['month'] == 1) & (data['day'] == 1)
                          & (data['hour'] < 6)).astype(int)
    data['is_juhannus'] = ((data['month'] == 6)
                           & (data['day'].between(20, 26))).astype(int)

    # v2 Finland: lags / rolling PM2.5 autoregressifs (par station, decales -> pas de fuite)
    g = data.groupby('station_name')[target]
    data['pm25_lag_1'] = g.shift(1)
    data['pm25_lag_24'] = g.shift(24)
    data['pm25_roll_24'] = g.transform(
        lambda x: x.shift(1).rolling(24, min_periods=1).mean())
    for c in ('pm25_lag_1', 'pm25_lag_24', 'pm25_roll_24'):
        data[c] = data.groupby('station_name')[c].transform(
            lambda x: x.fillna(x.median()))
        data[c] = data[c].fillna(0.0)

    # distances inter-stations (logique existante, sur le reseau retenu)
    unique_coords = data[['longitude', 'latitude']].drop_duplicates().values
    dm = cdist(unique_coords, unique_coords, metric='euclidean')
    np.fill_diagonal(dm, np.inf)
    us = pd.DataFrame(unique_coords, columns=['longitude', 'latitude'])
    us['min_distance'] = np.nanmin(np.where(dm == np.inf, np.nan, dm), axis=1)
    us['mean_distance'] = np.nanmean(np.where(dm == np.inf, 0, dm), axis=1)
    us['max_distance'] = np.nanmax(np.where(dm == np.inf, np.nan, dm), axis=1)
    us['std_distance'] = np.nanstd(np.where(dm == np.inf, 0, dm), axis=1)
    data = pd.merge(data, us, on=['longitude', 'latitude'], how='left')

    # v2 Finland: filet de securite -> aucune NaN dans les features (jamais la cible)
    feat_num = [c for c in data.columns if c not in
                ('date', 'station_name', target)]
    for c in feat_num:
        if data[c].dtype.kind in 'fiu' and data[c].isna().any():
            data[c] = data.groupby('station_name')[c].transform(
                lambda x: x.fillna(x.median()))
            data[c] = data[c].fillna(data[c].median())

    data = data.sort_values(['date', 'longitude', 'latitude']).reset_index(drop=True)
    return data


def split_and_normalize_data(data, features, target):
    station_ids = data.groupby(['longitude', 'latitude']).ngroups
    hidden_ratio = 0.1
    import numpy as np
    np.random.seed(42)
    hidden_ids = np.random.choice(range(station_ids), size=int(hidden_ratio * station_ids), replace=False)
    data['visibility'] = data.groupby(['longitude', 'latitude']).ngroup().apply(lambda x: 'hidden' if x in hidden_ids else 'visible')
    visible_data = data[data['visibility'] == 'visible'].copy()
    train_size = int(0.7 * len(visible_data))
    val_size = int(0.1 * len(visible_data))
    train_visible = visible_data.iloc[:train_size].copy()
    val_visible = visible_data.iloc[train_size:train_size + val_size].copy()
    test_visible = visible_data.iloc[train_size + val_size:].copy()
    train_visible[features + [target]] = train_visible[features + [target]].astype(float)
    val_visible[features + [target]] = val_visible[features + [target]].astype(float)
    test_visible[features + [target]] = test_visible[features + [target]].astype(float)
    # v2 Finland: clip cible (negatifs bruit capteur -> 0, cap outlier 200 ug/m3)
    # PUIS log1p : PM2.5 tres asymetrique -> stabilise la variance, gros levier AQ.
    # (de-transform = expm1 cote evaluate_model). Aucune info future, zero fuite.
    for d in (train_visible, val_visible, test_visible):
        d.loc[:, target] = np.log1p(d[target].clip(lower=0.0, upper=200.0))
    # v2 Finland: StandardScaler/z-score (sur l'espace log) au lieu de MinMax.
    # Scalers fit sur le TRAIN uniquement (pas de fuite val/test).
    from sklearn.preprocessing import StandardScaler
    scaler_features = StandardScaler()
    scaler_target = StandardScaler()
    train_features_norm = scaler_features.fit_transform(train_visible[features].values)
    val_features_norm = scaler_features.transform(val_visible[features].values)
    test_features_norm = scaler_features.transform(test_visible[features].values)
    train_visible.loc[:, features] = train_features_norm
    val_visible.loc[:, features] = val_features_norm
    test_visible.loc[:, features] = test_features_norm
    scaler_target.fit(train_visible[[target]].values)
    train_visible.loc[:, target] = scaler_target.transform(train_visible[[target]].values)
    val_visible.loc[:, target] = scaler_target.transform(val_visible[[target]].values)
    test_visible.loc[:, target] = scaler_target.transform(test_visible[[target]].values)
    return train_visible, val_visible, test_visible, scaler_target
