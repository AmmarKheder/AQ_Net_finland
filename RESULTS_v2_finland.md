# PM2.5 multi-horizon forecasting — Finland (v2)

Prévision PM2.5 par station, modèle résiduel multi-horizon. Branche `v2-finland`.

## Données

- **Source** : FMI Open Data (qualité de l'air + météo) + Fintraffic/digitraffic (trafic)
- **Période** : 2019-01-01 → 2024-12-31 (horaire, UTC)
- **Stations** : 20 stations finlandaises retenues (PM2.5 ≥ 80 % & météo temp/rh/wind/pressure ≥ 70 % de couverture sur 2019-2024)
- **Features** (32) : pm10, no2, no, aqi, era5_temp/rh/wind/pressure, calendrier cyclique
  (hour/month/day_of_week sin-cos), hdd/stagnation + rolling, saisons (heating/road-dust/
  newyear/juhannus), lags PM2.5 (1h/24h/roll24), distances inter-stations, trafic
  (volume/vitesse/has_traffic)
- **Cible** : PM2.5 (µg/m³), clip [0,200], `log1p` puis z-score (fit train uniquement)

## Méthodologie (rigueur — point fort du mémoire)

- **Prédiction résiduelle** : `pred(t+h) = PM2.5(t) + Δ_h` (têtes init ~0 → démarre en
  persistance, apprend l'écart). Pas de Softplus sur Δ, clamp ≥ 0 en sortie.
- **Multi-horizon** : 6/12/24/48 h, loss MSE masquée par horizon, pondérée [1, 0.7, 0.5, 0.3]
- **Découpage gap-aware** : fenêtres uniquement sur segments horaires contigus
- **Split temporel strict + EMBARGO** : par station, train < val < test avec une bande
  purgée de `seq+horizon` h à chaque coupe → **zéro fuite vérifiée empiriquement**
  (0 timestamp commun entre fenêtres train/val, gap ≥ 217 h)
- **Normalisation fit sur le train uniquement** (pas de fuite val/test)
- **Setup honnête** : past-only, aucune feature future (pas de météo/CAMS future)

## Résultats — modèle final (LSTM régularisé + résiduel, seq=48)

Test set (période la plus récente, jamais vue), PM2.5 µg/m³ :

| Horizon | RMSE | MAE | **r** | R² | persist r | persist R² |
|---|---|---|---|---|---|---|
| 6 h  | 3.38 | 2.00 | **0.645** | 0.400 | 0.610 | 0.219 |
| 12 h | 3.77 | 2.31 | **0.529** | 0.251 | 0.464 | -0.073 |
| 24 h | 4.01 | 2.52 | **0.436** | 0.150 | 0.384 | -0.238 |
| 48 h | 4.35 | 2.78 | **0.295** | 0.010 | 0.260 | -0.479 |

### Horizons courts (`[1, 3, 6, 12]`)

| Horizon | RMSE | MAE | **r** | R² | persist r | persist R² |
|---|---|---|---|---|---|---|
| 1 h  | 1.72 | 0.89 | **0.917** | 0.839 | 0.916 | 0.829 |
| 3 h  | 2.57 | 1.53 | **0.794** | 0.629 | 0.788 | 0.560 |
| 6 h  | 3.36 | 1.99 | **0.642** | 0.404 | 0.610 | 0.219 |
| 12 h | 3.76 | 2.29 | **0.526** | 0.255 | 0.464 | -0.073 |

Double lecture honnête : chiffres absolus élevés à court terme (1 h r≈0.92,
3 h r≈0.79) **mais** la persistance y est déjà très forte (1 h R²=0.83) → le
modèle ≈ persistance à 1 h, l'**apport réel (skill)** se situe à 6-48 h où il
dépasse nettement la persistance (négative à 12 h+).

**Lecture** : le modèle bat la persistance à **tous** les horizons ; la marge (skill score)
**grandit avec l'horizon** (la persistance devient R² négatif dès 12 h, le modèle reste
positif). Le narratif défendable = **skill score vs persistance/climatologie**, pas le
R² absolu (plafonné par le régime air-propre).

## Ablations (toutes confirment le plafond data)

| Variante | r 6h | Effet |
|---|---|---|
| LSTM régularisé (final) | 0.642 | référence |
| + trafic (32 feat) | 0.645 | ~nul (bruit) |
| iTransformer (simplifié) | 0.380 | pire |
| iTransformer fidèle (RevIN) | 0.352 | pire (RevIN retire le niveau, néfaste en résiduel) |
| sans log1p (z-score brut) | 0.629 | neutre / léger - |
| MinMax au lieu de z-score | 0.603 | collapse vers persistance (signal écrasé) |
| loss anti-pics cran_pm (hotspot+asym) | 0.603 | nettement pire (biais sans signal) |

→ 8 confirmations indépendantes : **architecture/loss/normalisation/features ne déplacent
pas le R²**. Le goulot est l'information dans la donnée (air propre finlandais : moy
~5 µg/m³, 23 % ≤ 2 µg/m³, autocorrélation qui s'effondre, persistance R² < 0 dès 12 h).

## Limites (assumées, à discuter dans le mémoire)

- Faible signal intrinsèque en air propre → R² absolu modeste, surtout 24/48 h
- Le modèle **sous-estime les épisodes** (lissage vers la moyenne — propre à la MSE sur
  cible asymétrique faible signal) ; cartes épisodes le montrent
- 20 stations, un seul pays

## Perspectives (future work)

- Données EEA Europe horaires (3067 stations, déjà identifiées) + ERA5/CAMS au point
- Horizons très courts (1-3 h) pour un usage opérationnel
- Reformulation en prévision de **dépassement de seuil** (classification d'épisodes)

## Figures

- `maps_v2/carte_<date>.png` — cartes Finlande cartopy (RdYlBu_r), observé vs prédit
  sur les journées les plus polluées
- `maps_v2/episode_*_series.png` — séries temporelles observé vs prédit (épisode)
- `plots_v2/<station>.png` — diagnostics par station (séries + scatter + R²)

## Reproduire

```bash
sbatch dryrun_v2.sbatch          # modèle final (6/12/24/48 h)
sbatch dryrun_short.sbatch       # horizons courts [1,3,6,12]
python3 predict_maps.py          # cartes épisodes
python3 predict_plot.py          # diagnostics par station
```
