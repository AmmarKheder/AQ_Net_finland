#!/usr/bin/env python3
"""
Copyright (c) Ammar Kheder
Licensed under the MIT License.

v2 Finland: loss multi-horizon masquee + ponderee, baseline persistance.
"""

import numpy as np
import torch
import torch.nn as nn


def _masked_mse_per_h(pred, y, mask, hotspot_alpha=0.0, hotspot_thr=0.5,
                      hotspot_maxw=4.0, under_penalty=1.0):
    # v2 Finland: MSE/horizon masquee. Reprend la strategie cran_pm pour
    # les pics : (1) hotspot weighting (upweight les fortes valeurs),
    # (2) penalite asymetrique de sous-estimation (pred<y puni + fort).
    # y est en espace z-score(log1p) -> seuil/echelle en z (mean~0, std~1).
    w = mask.clone()
    if hotspot_alpha > 0:
        excess = (y - hotspot_thr).clamp(min=0.0, max=hotspot_maxw)
        w = w * (1.0 + hotspot_alpha * excess)
    if under_penalty > 1.0:
        under = (pred < y).float()                   # 1 si sous-estimation
        w = w * (1.0 + (under_penalty - 1.0) * under)
    se = (pred - y) ** 2 * w
    denom = w.sum(dim=0).clamp(min=1.0)              # garde le graphe connecte
    return se.sum(dim=0) / denom                     # (n_horizons,)


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, num_epochs=50, patience=10,
                horizon_weights=None, entropy_weight=0.0,
                hotspot_alpha=0.0, under_penalty=1.0):
    n_h = len(model.horizons)
    if horizon_weights is None:
        horizon_weights = [1.0] * n_h
    w = torch.tensor(horizon_weights, dtype=torch.float32, device=model.device)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    use_amp = torch.cuda.is_available()           # v2 Finland: AMP seulement sur GPU
    scaler_grad = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for x, y, pm25_cur, mask, _, _, _ in train_loader:
            x = x.to(model.device); y = y.to(model.device)
            pm25_cur = pm25_cur.to(model.device); mask = mask.to(model.device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred, attn = model(x, pm25_cur)
                # train: loss ponderee pics (cran_pm) ; val: loss plate
                # (selection modele sur objectif non biaise)
                loss_per_h = _masked_mse_per_h(
                    pred, y, mask, hotspot_alpha=hotspot_alpha,
                    under_penalty=under_penalty)
                loss = (loss_per_h * w).sum() / w.sum()
                if entropy_weight:
                    ent = -torch.sum(attn * torch.log(attn + 1e-10), dim=-1)
                    loss = loss + entropy_weight * torch.mean(ent)
            scaler_grad.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_grad.step(optimizer)
            scaler_grad.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, pm25_cur, mask, _, _, _ in val_loader:
                x = x.to(model.device); y = y.to(model.device)
                pm25_cur = pm25_cur.to(model.device)
                mask = mask.to(model.device)
                pred, _ = model(x, pm25_cur)
                lph = _masked_mse_per_h(pred, y, mask)
                val_loss += ((lph * w).sum() / w.sum()).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_lstm_attention_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs.")
                break
        print(f"Epoch {epoch+1}/{num_epochs} - Train {train_loss:.4f} "
              f"- Val {val_loss:.4f}")
    return model, train_losses, val_losses


def evaluate_model(model, test_loader, criterion, scaler_target,
                   horizon_weights=None, use_log1p=True):
    """
    v2 Finland: renvoie (test_loss, y_true, y_pred) en unites reelles,
    shape (N, n_horizons), NaN ou la verite-terrain manque.
    Logue aussi le baseline PERSISTANCE (pred = valeur courante) par horizon.
    """
    model.load_state_dict(torch.load("best_lstm_attention_model.pth"))
    model.eval()
    horizons = model.horizons
    n_h = len(horizons)
    w = (torch.tensor(horizon_weights, dtype=torch.float32, device=model.device)
         if horizon_weights is not None
         else torch.ones(n_h, device=model.device))

    test_loss = 0.0
    yt, yp, ym, ycur = [], [], [], []
    with torch.no_grad():
        for x, y, pm25_cur, mask, _, _, _ in test_loader:
            x = x.to(model.device); y = y.to(model.device)
            pm25_cur = pm25_cur.to(model.device); mask = mask.to(model.device)
            pred, _ = model(x, pm25_cur)
            lph = _masked_mse_per_h(pred, y, mask)
            test_loss += ((lph * w).sum() / w.sum()).item()
            yt.append(y.cpu().numpy()); yp.append(pred.cpu().numpy())
            ym.append(mask.cpu().numpy())
            ycur.append(pm25_cur.cpu().numpy())
    test_loss /= len(test_loader)

    yt = np.concatenate(yt); yp = np.concatenate(yp)
    ym = np.concatenate(ym).astype(bool)
    ycur = np.concatenate(ycur)

    # v2 Finland: de-transform = inverse z-score, puis expm1 si log1p utilise
    def inv(a):
        z = scaler_target.inverse_transform(a.reshape(-1, 1)).reshape(a.shape)
        return np.expm1(z) if use_log1p else z
    yt_r, yp_r = inv(yt), inv(yp)
    ycur_r = inv(ycur)
    yp_r = np.clip(yp_r, 0.0, None)            # v2 Finland: PM2.5 >= 0

    # v2 Finland: r (Pearson) en metrique principale + R2 conserve (reviewers)
    def _r(a, b):
        if len(a) < 2 or a.std() == 0 or b.std() == 0:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    print("  horizon |  RMSE   MAE     r      R2   | persist r   persist R2")
    for hi, h in enumerate(horizons):
        m = ym[:, hi]
        if m.sum() == 0:
            continue
        yt_h = yt_r[m, hi]
        ss_tot = np.sum((yt_h - yt_h.mean()) ** 2)
        e = yp_r[m, hi] - yt_h
        rmse = np.sqrt(np.mean(e ** 2)); mae = np.mean(np.abs(e))
        r = _r(yp_r[m, hi], yt_h)
        r2 = 1.0 - np.sum(e ** 2) / ss_tot if ss_tot > 0 else float('nan')
        pr = _r(ycur_r[m], yt_h)                     # persistance = valeur courante
        pe = ycur_r[m] - yt_h
        pr2 = 1.0 - np.sum(pe ** 2) / ss_tot if ss_tot > 0 else float('nan')
        print(f"   {h:3d}h   | {rmse:6.2f} {mae:5.2f} {r:6.3f} {r2:6.3f} |"
              f"   {pr:6.3f}    {pr2:6.3f}")

    yt_r[~ym] = np.nan
    yp_r[~ym] = np.nan
    return test_loss, yt_r, yp_r
