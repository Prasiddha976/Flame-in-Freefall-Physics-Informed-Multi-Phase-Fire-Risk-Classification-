"""
Publication-Grade Figure Generator (300 DPI PNG + Vector PDF) for 'Flame in Freefall'
Generates 4 IEEE-compliant figures directly from the v4 NASA PSI dataset and ExtraTrees pipelines:
  1. fig1_dataset_harmonization.png/.pdf   - NASA PSI Dataset Breakdown & Empirical Severity Distribution
  2. fig2_benchmark_ablation.png/.pdf      - Multi-Algorithm Benchmark & Raw vs. Compact Physics Ablation
  3. fig3_safety_calibration.png/.pdf      - Asymmetric ECLSS Safety Threshold Trade-Off & OOF Confusion Matrix
  4. fig4_extinction_parity_shap.png/.pdf  - Pure Droplet Radiative Extinction Parity Plot & Physics Feature Importance
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from train_dual_ml_engine import (
    CATEGORICAL_FEATURES,
    COMPACT_PHYSICS_NUMERIC,
    RAW_PRE_IGNITION_NUMERIC,
    build_preprocessor,
    engineer_physics_features,
    get_candidate_classifiers,
    get_candidate_regressors,
)

# IEEE-style publication formatting
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

PALETTE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "amber": "#D97706",
    "crimson": "#B91C1C",
    "slate": "#64748B",
    "light_blue": "#3B82F6",
    "emerald": "#059669",
}


def apply_safety_threshold(probs: np.ndarray, safe_conf_threshold: float = 0.50) -> np.ndarray:
    preds = np.zeros(len(probs), dtype=int)
    for i, (p0, p1, p2) in enumerate(probs):
        if p0 >= safe_conf_threshold:
            preds[i] = 0
        else:
            preds[i] = 2 if p2 >= p1 * 0.85 else 1
    return preds


def save_fig(fig: plt.Figure, out_dir: str, basename: str):
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"{basename}.png")
    pdf_path = os.path.join(out_dir, f"{basename}.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Exported {png_path} & {pdf_path}")


def generate_all_figures(
    data_path: str = "organized_data/master_microgravity_fires.csv",
    out_dir: str = "figures",
):
    os.makedirs(out_dir, exist_ok=True)
    raw_df = pd.read_csv(data_path)
    feat_df = engineer_physics_features(raw_df)

    phys_cols = COMPACT_PHYSICS_NUMERIC + CATEGORICAL_FEATURES
    raw_cols = RAW_PRE_IGNITION_NUMERIC + CATEGORICAL_FEATURES
    reg_cols = [c for c in phys_cols if c != "solid_thermally_thin_spread_proxy"]
    raw_reg_cols = [c for c in raw_cols if c != "sample_thickness_um"]

    flame_df = feat_df[feat_df["fuel_category"] != "Wire Insulation"].reset_index(drop=True)
    y_cls = flame_df["fire_risk_level"].astype(int).to_numpy()
    groups_cls = flame_df["campaign_group_id"]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    ext_df = feat_df[
        (feat_df["fuel_category"] == "Liquid Droplet")
        & (feat_df["true_radiative_extinction_s"].notna())
        & (feat_df["true_radiative_extinction_s"] <= 120.0)
    ].reset_index(drop=True)
    y_reg = ext_df["true_radiative_extinction_s"].astype(float).to_numpy()
    gkf = GroupKFold(n_splits=5)

    # =========================================================================
    # FIGURE 1: Harmonized NASA PSI Dataset & Empirical Severity Distribution
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    # Left: Fuel Family Counts across NASA Investigations
    fam_counts = raw_df["fuel_family"].value_counts()
    fam_labels = [f.replace("_", "\n") for f in fam_counts.index]
    bars1 = ax1.barh(fam_labels[::-1], fam_counts.values[::-1], color=PALETTE["navy"], edgecolor="black", linewidth=0.6)
    ax1.set_xlabel("Number of Verified NASA PSI Runs (N = 407)")
    ax1.set_title("(a) Fuel Family Distribution Across Archives")
    ax1.set_xlim(0, max(fam_counts.values) * 1.18)
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 2.5, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", fontsize=8.5)

    # Right: Stacked Empirical Severity Distribution by Mission Dataset
    missions = ["FLEX (PSI-69)", "BASS-II (PSI-25)", "SAME (PSI-102)"]
    sev_names = ["Safe (0)", "Warning (1)", "Critical (2)"]
    sev_colors = [PALETTE["emerald"], PALETTE["amber"], PALETTE["crimson"]]
    bottoms = np.zeros(len(missions))

    for level, sname, col in zip([0, 1, 2], sev_names, sev_colors):
        vals = [
            len(raw_df[(raw_df["mission_dataset"] == m) & (raw_df["fire_risk_level"] == level)])
            for m in missions
        ]
        b = ax2.bar(["FLEX\n(N=274)", "BASS-II\n(N=129)", "SAME\n(N=4)"], vals, bottom=bottoms, label=sname, color=col, edgecolor="black", linewidth=0.6, width=0.55)
        for rect, v in zip(b, vals):
            if v >= 15:
                ax2.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + v / 2, f"{v}", ha="center", va="center", color="white", fontweight="bold", fontsize=8.5)
        bottoms += np.array(vals)

    ax2.set_ylabel("Experiment Run Count")
    ax2.set_title("(b) Post-Test Empirical Severity Ground Truth")
    ax2.legend(loc="upper right", frameon=True)
    save_fig(fig, out_dir, "fig1_dataset_harmonization")

    # =========================================================================
    # FIGURE 2: Multi-Algorithm Benchmark & Compact Physics Ablation
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.9))

    # Left: Multi-Algorithm Comparison (Engine 1 OOF Acc & Engine 2 OOF R2)
    cls_models = ["RandomForest", "XGBoost", "ExtraTrees"]
    cls_accs = []
    for name in cls_models:
        oof = np.zeros(len(flame_df), dtype=int)
        for tr_i, va_i in sgkf.split(flame_df[phys_cols], y_cls, groups_cls):
            pipe = Pipeline([("prep", build_preprocessor(phys_cols)), ("clf", get_candidate_classifiers()[name])])
            pipe.fit(flame_df[phys_cols].iloc[tr_i], y_cls[tr_i])
            oof[va_i] = pipe.predict(flame_df[phys_cols].iloc[va_i])
        cls_accs.append(accuracy_score(y_cls, oof) * 100.0)

    reg_models = ["RandomForest", "RidgeLinear", "XGBoost", "ExtraTrees"]
    reg_r2s = []
    for name in reg_models:
        oof_r = np.zeros(len(ext_df), dtype=float)
        is_lin = name == "RidgeLinear"
        for tr_i, va_i in gkf.split(ext_df[reg_cols], y_reg, ext_df["campaign_group_id"]):
            pipe = Pipeline([("prep", build_preprocessor(reg_cols, impute_for_linear=is_lin)), ("reg", get_candidate_regressors()[name])])
            pipe.fit(ext_df[reg_cols].iloc[tr_i], y_reg[tr_i])
            oof_r[va_i] = np.clip(pipe.predict(ext_df[reg_cols].iloc[va_i]), 0.1, None)
        reg_r2s.append(r2_score(y_reg, oof_r) * 100.0)

    x_pos = np.arange(len(reg_models))
    cls_aligned = [cls_accs[0], np.nan, cls_accs[1], cls_accs[2]]  # align with reg_models
    w = 0.36
    b1 = ax1.bar(x_pos - w / 2, cls_aligned, width=w, label="Engine 1: Severity OOF Acc (%)", color=PALETTE["navy"], edgecolor="black", linewidth=0.6)
    b2 = ax1.bar(x_pos + w / 2, reg_r2s, width=w, label="Engine 2: Extinction OOF R² (%)", color=PALETTE["teal"], edgecolor="black", linewidth=0.6)

    for rect in b1:
        h = rect.get_height()
        if not np.isnan(h):
            ax1.text(rect.get_x() + rect.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", fontsize=8)
    for rect in b2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", fontsize=8)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(reg_models)
    ax1.set_ylim(0, 82)
    ax1.set_ylabel("Out-of-Fold Performance (%)")
    ax1.set_title("(a) Multi-Algorithm Grouped OOF Benchmark")
    ax1.legend(loc="upper left", frameon=True)

    # Right: Ablation Study (Raw Telemetry vs. Compact Physics Invariants using ExtraTrees)
    ablation_labels = ["Raw Pre-Ignition\nTelemetry (8 Feats)", "Compact Physics\nInvariants (7 Feats)"]
    ab_cls_accs, ab_reg_r2s = [], []

    for c_cols, r_cols in [(raw_cols, raw_reg_cols), (phys_cols, reg_cols)]:
        oof_c = np.zeros(len(flame_df), dtype=int)
        for tr_i, va_i in sgkf.split(flame_df[c_cols], y_cls, groups_cls):
            pipe = Pipeline([("prep", build_preprocessor(c_cols)), ("clf", get_candidate_classifiers()["ExtraTrees"])])
            pipe.fit(flame_df[c_cols].iloc[tr_i], y_cls[tr_i])
            oof_c[va_i] = pipe.predict(flame_df[c_cols].iloc[va_i])
        ab_cls_accs.append(accuracy_score(y_cls, oof_c) * 100.0)

        oof_r = np.zeros(len(ext_df), dtype=float)
        for tr_i, va_i in gkf.split(ext_df[r_cols], y_reg, ext_df["campaign_group_id"]):
            pipe = Pipeline([("prep", build_preprocessor(r_cols)), ("reg", get_candidate_regressors()["ExtraTrees"])])
            pipe.fit(ext_df[r_cols].iloc[tr_i], y_reg[tr_i])
            oof_r[va_i] = np.clip(pipe.predict(ext_df[r_cols].iloc[va_i]), 0.1, None)
        ab_reg_r2s.append(r2_score(y_reg, oof_r) * 100.0)

    x_ab = np.arange(2)
    b3 = ax2.bar(x_ab - w / 2, ab_cls_accs, width=w, label="Engine 1: OOF Accuracy (%)", color=PALETTE["light_blue"], edgecolor="black", linewidth=0.6)
    b4 = ax2.bar(x_ab + w / 2, ab_reg_r2s, width=w, label="Engine 2: Extinction OOF R² (%)", color=PALETTE["amber"], edgecolor="black", linewidth=0.6)

    for rect in list(b3) + list(b4):
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2, h + 1.0, f"{h:.2f}%", ha="center", fontsize=8.5, fontweight="bold")

    ax2.set_xticks(x_ab)
    ax2.set_xticklabels(ablation_labels)
    ax2.set_ylim(0, 82)
    ax2.set_ylabel("Out-of-Fold Metric (%)")
    ax2.set_title("(b) Physics Ablation Study (ExtraTrees)")
    ax2.legend(loc="upper left", frameon=True)
    save_fig(fig, out_dir, "fig2_benchmark_ablation")

    # =========================================================================
    # FIGURE 3: Asymmetric ECLSS Safety Threshold Calibration & Confusion Matrix
    # =========================================================================
    oof_probs = np.zeros((len(flame_df), 3), dtype=float)
    for tr_i, va_i in sgkf.split(flame_df[phys_cols], y_cls, groups_cls):
        pipe = Pipeline([("prep", build_preprocessor(phys_cols)), ("clf", get_candidate_classifiers()["ExtraTrees"])])
        pipe.fit(flame_df[phys_cols].iloc[tr_i], y_cls[tr_i])
        oof_probs[va_i] = pipe.predict_proba(flame_df[phys_cols].iloc[va_i])

    taus = np.linspace(0.30, 0.70, 17)
    miss_rates, false_esc_rates, overall_accs = [], [], []
    n_crit = int((y_cls == 2).sum())
    n_safe = int((y_cls == 0).sum())

    for t in taus:
        p_t = apply_safety_threshold(oof_probs, safe_conf_threshold=t)
        cm_t = confusion_matrix(y_cls, p_t)
        miss_rates.append((cm_t[2, 0] / n_crit) * 100.0)
        false_esc_rates.append(((cm_t[0, 1] + cm_t[0, 2]) / n_safe) * 100.0)
        overall_accs.append(accuracy_score(y_cls, p_t) * 100.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.9))

    ax1.plot(taus, miss_rates, marker="o", markersize=4, color=PALETTE["crimson"], linewidth=2.0, label="Catastrophic Miss Rate (Critical → Safe %)")
    ax1.plot(taus, false_esc_rates, marker="s", markersize=4, color=PALETTE["amber"], linewidth=1.8, linestyle="--", label="Safe False-Escalation Rate (Safe → Warn/Crit %)")
    ax1.plot(taus, overall_accs, marker="^", markersize=4, color=PALETTE["navy"], linewidth=1.8, label="Overall OOF Classification Accuracy (%)")
    ax1.axvline(0.45, color=PALETTE["emerald"], linestyle=":", linewidth=1.6, label="Selected ECLSS Operating Point (τ_safe = 0.45)")
    ax1.set_xlabel("Safe Confidence Threshold (τ_safe)")
    ax1.set_ylabel("Percentage of Runs (%)")
    ax1.set_title("(a) Asymmetric ECLSS Safety Trade-Off Curve")
    ax1.set_ylim(0, 95)
    ax1.legend(loc="upper left", fontsize=7.8, frameon=True)

    # Right: Normalized OOF Confusion Matrix at τ_safe = 0.45
    preds_cal = apply_safety_threshold(oof_probs, safe_conf_threshold=0.45)
    cm_cal = confusion_matrix(y_cls, preds_cal)
    cm_norm = cm_cal.astype(float) / cm_cal.sum(axis=1, keepdims=True)

    im = ax2.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=0.80)
    classes = ["Safe (0)", "Warning (1)", "Critical (2)"]
    ax2.set_xticks(np.arange(3))
    ax2.set_yticks(np.arange(3))
    ax2.set_xticklabels(classes)
    ax2.set_yticklabels(classes)
    ax2.set_xlabel("Predicted ECLSS Risk Level")
    ax2.set_ylabel("Actual Post-Test Fire Severity")
    ax2.set_title("(b) Calibrated OOF Confusion Matrix (τ_safe = 0.45)")
    ax2.grid(False)

    for i in range(3):
        for j in range(3):
            txt_col = "white" if cm_norm[i, j] > 0.42 else "black"
            ax2.text(j, i, f"{cm_cal[i, j]}\n({cm_norm[i, j]*100:.1f}%)", ha="center", va="center", color=txt_col, fontsize=8.5, fontweight="bold")

    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    save_fig(fig, out_dir, "fig3_safety_calibration")

    # =========================================================================
    # FIGURE 4: Radiative Extinction Parity Plot & Physics Feature Importance
    # =========================================================================
    oof_reg_preds = np.zeros(len(ext_df), dtype=float)
    for tr_i, va_i in gkf.split(ext_df[reg_cols], y_reg, ext_df["campaign_group_id"]):
        pipe = Pipeline([("prep", build_preprocessor(reg_cols)), ("reg", get_candidate_regressors()["ExtraTrees"])])
        pipe.fit(ext_df[reg_cols].iloc[tr_i], y_reg[tr_i])
        oof_reg_preds[va_i] = np.clip(pipe.predict(ext_df[reg_cols].iloc[va_i]), 0.1, None)

    final_r2 = r2_score(y_reg, oof_reg_preds)
    final_mae = mean_absolute_error(y_reg, oof_reg_preds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.9))

    # Left: Parity Plot (Actual vs. Predicted Radiative Extinction Time)
    fuels = ext_df["fuel_family"].astype(str)
    for fam, col, mk in [("Methanol", PALETTE["navy"], "o"), ("Heptane", PALETTE["crimson"], "^")]:
        mask = fuels == fam
        ax1.scatter(y_reg[mask], oof_reg_preds[mask], c=col, marker=mk, alpha=0.75, edgecolors="black", linewidth=0.4, s=36, label=f"{fam} (n={mask.sum()})")

    lim = max(y_reg.max(), oof_reg_preds.max()) * 1.05
    ax1.plot([0, lim], [0, lim], "k--", linewidth=1.3, label="Ideal 1:1 Parity")
    ax1.fill_between([0, lim], [0 - final_mae, lim - final_mae], [0 + final_mae, lim + final_mae], color=PALETTE["slate"], alpha=0.15, label=f"±1 MAE Band ({final_mae:.2f} s)")
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)
    ax1.set_xlabel("Observed FLEX Radiative Extinction Time (s)")
    ax1.set_ylabel("Predicted Extinction Time (s)")
    ax1.set_title(f"(a) Pure Droplet Extinction Parity (OOF R² = {final_r2:.3f})")
    ax1.legend(loc="upper left", frameon=True)

    # Right: Gini Feature Importance of Orthogonal Physics Invariants (Classifier)
    clf_full = Pipeline([("prep", build_preprocessor(phys_cols)), ("clf", get_candidate_classifiers()["ExtraTrees"])])
    clf_full.fit(flame_df[phys_cols], y_cls)

    ohe_names = list(clf_full.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    all_out_names = [c for c in phys_cols if c not in CATEGORICAL_FEATURES] + ohe_names
    importances = clf_full.named_steps["clf"].feature_importances_

    # Aggregate OneHot categorical importances back into Fuel Family & Fuel Category
    agg_imp = {}
    clean_map = {
        "o2_partial_pressure_kpa": "O₂ Partial Pressure (P_O₂)",
        "gas_thermal_conductivity": "Gas Thermal Conductivity (k_mix)",
        "o2_to_inert_ratio": "O₂-to-Inert Ratio",
        "initial_size_mm": "Initial Fuel Size (D₀ / W₀)",
        "droplet_d2_timescale_proxy": "Droplet D² Timescale (D₀²/P_O₂)",
        "solid_thermally_thin_spread_proxy": "Solid Convective Proxy (V·P_O₂/τ)",
        "radiative_quenching_number": "Radiative Quenching Number",
    }
    for name, val in zip(all_out_names, importances):
        if name.startswith("fuel_family_"):
            agg_imp["Fuel Chemical/Geometric Family"] = agg_imp.get("Fuel Chemical/Geometric Family", 0.0) + val
        elif name.startswith("fuel_category_"):
            agg_imp["Fuel Phase Category"] = agg_imp.get("Fuel Phase Category", 0.0) + val
        else:
            agg_imp[clean_map.get(name, name)] = val

    imp_series = pd.Series(agg_imp).sort_values(ascending=True)
    bars = ax2.barh(imp_series.index, imp_series.values * 100.0, color=PALETTE["teal"], edgecolor="black", linewidth=0.6)
    ax2.set_xlabel("Relative Split Importance (%)")
    ax2.set_title("(b) Physics Invariant Importance (Engine 1)")
    ax2.set_xlim(0, max(imp_series.values * 100.0) * 1.2)
    for bar in bars:
        w = bar.get_width()
        ax2.text(w + 0.4, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8)

    save_fig(fig, out_dir, "fig4_extinction_parity_shap")
    print("\n[ALL 4 PUBLICATION FIGURES GENERATED IN 'figures/' AT 300 DPI + VECTOR PDF]")


if __name__ == "__main__":
    generate_all_figures()