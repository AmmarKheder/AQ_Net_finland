#!/usr/bin/env python3
"""
Copyright (c) Ammar Kheder
Licensed under the MIT License.

v2 Finland: loss multi-horizon masquee + ponderee, baseline persistance.
"""

import numpy as np
import torch
import torch.nn as nn


def _masked_mse_per_h(pred, y, mask):
    # v2 Finland: MSE par horizon en ignorant les cibles manquantes (mask=0)
    se = (pred - y) ** 2 * mask
    denom = mask.sum(dim=0).clamp(min=1.0)          # garde le graphe connecte
    return se.sum(dim=0) / denom                    # (n_horizons,)


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, num_epochs=50, patience=10,
                horizon_weights=None, entropy_weight=0.01):
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
                loss_per_h = _masked_mse_per_h(pred, y, mask)
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
                   horizon_weights=None):
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

    inv = lambda a: scaler_target.inverse_transform(
        a.reshape(-1, 1)).reshape(a.shape)
    yt_r, yp_r = inv(yt), inv(yp)
    ycur_r = scaler_target.inverse_transform(ycur.reshape(-1, 1)).ravel()
    yp_r = np.clip(yp_r, 0.0, None)            # v2 Finland: PM2.5 >= 0

    print("  horizon |   RMSE   MAE  |  persistance RMSE")
    for hi, h in enumerate(horizons):
        m = ym[:, hi]
        if m.sum() == 0:
            continue
        e = yp_r[m, hi] - yt_r[m, hi]
        rmse = np.sqrt(np.mean(e ** 2)); mae = np.mean(np.abs(e))
        pe = ycur_r[m] - yt_r[m, hi]               # persistance = valeur courante
        prmse = np.sqrt(np.mean(pe ** 2))
        print(f"   {h:3d}h   | {rmse:6.2f} {mae:5.2f} |  {prmse:6.2f}")

    yt_r[~ym] = np.nan
    yp_r[~ym] = np.nan
    return test_loss, yt_r, yp_r
