# PM2.5 multi-horizon forecasting — Finland

Prévision résiduelle multi-horizon du PM2.5 par station en Finlande
(données FMI + trafic Fintraffic), 2019–2024. Mémoire de M.Sc.

➡️ **Résultats, méthodologie et ablations complets : [`RESULTS_v2_finland.md`](RESULTS_v2_finland.md)**

## Résumé

- **Données** : FMI Open Data (qualité de l'air + météo) + Fintraffic/digitraffic
  (trafic), 20 stations finlandaises, horaire, 2019-01-01 → 2024-12-31, UTC.
- **Modèle** : prédiction **résiduelle** `pred(t+h) = PM2.5(t) + Δ_h`,
  multi-horizon (6/12/24/48 h, et variante courte 1/3/6/12 h), backbone
  sélectionnable (LSTM+attention, iTransformer, PatchTST).
- **Rigueur** : split temporel **par station + embargo** (zéro fuite vérifiée),
  découpage gap-aware, normalisation fit train-only, setup *past-only*
  (aucune feature future).
- **Résultat clé** : bat la persistance à tous les horizons (le skill grandit
  avec l'horizon) ; chiffres absolus présentables à court terme
  (1 h r≈0.92, 3 h r≈0.79). 8 ablations confirment le plafond de
  prévisibilité du régime air-propre.

## Structure

```
main.py                  entrainement multi-horizon (--model, --horizons, ...)
src/
  data_preprocessing.py  chargement FMI + merge trafic, features, split+embargo
  dataset.py             fenetres gap-aware, residuel, masque par horizon
  model.py               LSTMAttention / iTransformer / PatchTST (residuels)
  training.py            loss MSE masquee ponderee, metriques r/R2/persistance
  knn_interpolation.py   interpolation spatiale stations cachees
predict_maps.py          cartes Finlande (cartopy, RdYlBu_r) sur episodes
predict_plot.py          diagnostics par station (series + scatter)
slurm/*.sbatch           jobs LUMI (modele final + ablations)
maps_v2/  plots_v2/       figures du memoire
RESULTS_v2_finland.md    resultats + ablations + limites + perspectives
```

## Données

Non versionnées (volumineuses). Reproduites par les scripts de
téléchargement (sur le cluster) : `finland_2019_2024.csv` (FMI AQ+météo)
et `finland_traffic_2019_2024.csv` (Fintraffic). Chemins dans `main.py`.

## Reproduire (LUMI)

```bash
sbatch slurm/dryrun_v2.sbatch        # modele final LSTM (6/12/24/48 h)
sbatch slurm/dryrun_short.sbatch     # horizons courts [1,3,6,12]
sbatch slurm/dryrun_itr.sbatch       # ablation iTransformer
sbatch slurm/dryrun_patchtst.sbatch  # ablation PatchTST
python3 predict_maps.py              # cartes episodes
python3 predict_plot.py              # diagnostics par station
```

`requirements.txt` : pandas, numpy, scipy, scikit-learn, matplotlib, torch,
einops (+ pyarrow, cartopy pour les figures).
