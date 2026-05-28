import os
import random
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import shap
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss,
                             confusion_matrix, roc_curve, auc)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, Dense, Dropout,
                                     GlobalAveragePooling1D,
                                     BatchNormalization, Concatenate,
                                     MultiHeadAttention, Lambda)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from config import TARGET, FEATURES

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Random seed
def set_random_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_random_seeds(42)
warnings.filterwarnings('ignore')


# Model
def create_improved_model(input_shape, learning_rate=0.001):
    inputs = Input(shape=input_shape)

    # Branch_H (horizontal branch): transpose then Conv1D along feature axis
    # captures inter-variable relationships at the same time point
    x_t = Lambda(lambda t: tf.transpose(t, perm=[0, 2, 1]))(inputs)
    x1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x_t)
    x1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x1)
    x1 = BatchNormalization(momentum=0.99, epsilon=1e-5)(x1)
    x1 = Dropout(0.3)(x1)
    x1 = GlobalAveragePooling1D()(x1)

    # Branch_L (longitudinal branch): pointwise conv along time axis
    # captures per-variable temporal feature representation
    x2 = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
    x2 = BatchNormalization(momentum=0.99, epsilon=1e-5)(x2)
    x2 = Dropout(0.3)(x2)
    x2 = GlobalAveragePooling1D()(x2)

    # Branch_C (crossover branch): multi-head attention for cross-temporal cross-variable dependencies
    x3 = MultiHeadAttention(num_heads=1, key_dim=16)(inputs, inputs)
    x3 = BatchNormalization(momentum=0.99, epsilon=1e-5)(x3)
    x3 = Dropout(0.3)(x3)
    x3 = GlobalAveragePooling1D()(x3) 

 
    x = Concatenate()([x1, x2, x3]) 
    x = Dense(units=128, activation='relu')(x)
    x = Dropout(0.4)(x)

    # Classification
    outputs = Dense(units=1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Data configuration
N_FEATURES = len(FEATURES)

METRIC_KEYS  = ['acc', 'roc_auc', 'sens', 'spec', 'ppv', 'npv', 'f1', 'pr_auc', 'brier']
METRIC_NAMES = ['Accuracy', 'ROC AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', 'PR-AUC', 'Brier Score']

# Time series construction
def build_time_series(patient_data):
    ts_data, tgt_data, t3_idx = [], [], []
    if len(patient_data) < 3:
        return np.array([]), np.array([]), []
    for i in range(len(patient_data) - 2):
        t1 = patient_data.iloc[i][FEATURES].values.astype(float)
        t2 = patient_data.iloc[i + 1][FEATURES].values.astype(float)
        t3 = patient_data.iloc[i + 2][FEATURES].values.astype(float)
        ts_data.append(np.concatenate([t1, t2, t3]))
        tgt_data.append(patient_data.iloc[i + 2][TARGET])
        t3_idx.append(patient_data.index[i + 2])
    return np.array(ts_data), np.array(tgt_data), t3_idx

def load_dataset(df):
    X_list, y_list, idx_list = [], [], []
    for _, pdata in df.groupby('ID'):
        ts, tgt, t3i = build_time_series(pdata)
        if len(ts):
            X_list.append(ts); y_list.append(tgt); idx_list.extend(t3i)
    return np.vstack(X_list), np.hstack(y_list), np.array(idx_list)

# Metric computation
def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'acc':     accuracy_score(y_true, y_pred),
        'sens':    tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'spec':    tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'ppv':     tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'npv':     tn / (tn + fn) if (tn + fn) > 0 else 0.0,
        'f1':      f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc':  average_precision_score(y_true, y_prob),
        'brier':   brier_score_loss(y_true, y_prob),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }

# Main experiment function
def run_experiment(df, label):
    X, y, _ = load_dataset(df)
    print(f'\n[{label}] Total samples: {len(y)}  Positive: {int(np.sum(y==1))}  Negative: {int(np.sum(y==0))}')

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics     = []
    fold_cms         = []
    fold_roc         = []
    all_shap_imp     = []
    all_shap_vals_2d = []
    all_feat_vals_2d = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f'\n  ===== Fold {fold_idx+1}/5 =====')
        print(f'  Train: {len(y_train)} (Pos:{int(np.sum(y_train==1))}, Neg:{int(np.sum(y_train==0))})')
        print(f'  Test:  {len(y_test)} (Pos:{int(np.sum(y_test==1))}, Neg:{int(np.sum(y_test==0))})')

        X_tr3 = X_train.reshape(-1, 3, N_FEATURES)
        X_te3 = X_test.reshape(-1, 3, N_FEATURES)

        model = create_improved_model(input_shape=(3, N_FEATURES))
        cb_list = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0, min_lr=1e-6),
        ]
        cw_vals = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(y_train), y=y_train)
        cw_dict = {int(c): w for c, w in zip(np.unique(y_train), cw_vals)}
        model.fit(X_tr3, y_train, epochs=100, batch_size=32,
                  validation_data=(X_te3, y_test), verbose=0,
                  class_weight=cw_dict,
                  callbacks=cb_list)

        y_prob = model.predict(X_te3, verbose=0).ravel()
        y_pred = (y_prob > 0.6).astype(int)

        m = compute_metrics(y_test, y_pred, y_prob)
        fold_metrics.append(m)
        fold_cms.append(confusion_matrix(y_test, y_pred, labels=[0, 1]))

        fpr_i, tpr_i, _ = roc_curve(y_test, y_prob)
        fold_roc.append((fpr_i, tpr_i, auc(fpr_i, tpr_i)))

        print(f'  Fold {fold_idx+1} -> '
              f'Acc:{m["acc"]:.4f} | Sens:{m["sens"]:.4f} | Spec:{m["spec"]:.4f} | '
              f'PPV:{m["ppv"]:.4f} | NPV:{m["npv"]:.4f} | F1:{m["f1"]:.4f} | '
              f'ROC:{m["roc_auc"]:.4f} | PR:{m["pr_auc"]:.4f} | Brier:{m["brier"]:.4f}')

        # SHAP
        bg_idx = np.random.default_rng(fold_idx).choice(
            len(X_tr3), min(100, len(X_tr3)), replace=False)
        exp = shap.GradientExplainer(model, X_tr3[bg_idx])
        sv  = exp.shap_values(X_te3)
        if isinstance(sv, list):
            sv = sv[0]
        all_shap_imp.append(np.mean(np.mean(np.abs(sv), axis=1), axis=0))
        all_shap_vals_2d.append(np.mean(sv, axis=1))
        all_feat_vals_2d.append(np.mean(X_te3, axis=1))
        print(f'  Fold {fold_idx+1} SHAP done')

    # Confusion matrix
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for i, (cm_i, ax) in enumerate(zip(fold_cms, axes)):
        im = ax.imshow(cm_i, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'Fold {i+1}', fontsize=12)
        ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive']); ax.set_yticklabels(['Negative', 'Positive'])
        thresh_c = cm_i.max() / 2.0
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm_i[r, c]), ha='center', va='center', fontsize=14,
                        color='white' if cm_i[r, c] > thresh_c else 'black')
        fig.colorbar(im, ax=ax)
    plt.suptitle(f'{label} - 5-Fold Cross-Validation Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.show()

    # ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_fpr    = np.linspace(0, 1, 100)
    tprs_interp = []
    for i, (fpr_i, tpr_i, roc_i) in enumerate(fold_roc):
        tp_i = np.interp(mean_fpr, fpr_i, tpr_i); tp_i[0] = 0.0
        tprs_interp.append(tp_i)
        ax.plot(fpr_i, tpr_i, alpha=0.4, lw=1.5, label=f'Fold {i+1} (AUC={roc_i:.3f})')
    mean_tpr = np.mean(tprs_interp, axis=0); mean_tpr[-1] = 1.0
    std_tpr  = np.std(tprs_interp, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, 'b-', lw=2.5, label=f'Mean ROC (AUC={mean_auc:.3f})')
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    alpha=0.15, color='blue', label='± 1 std')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{label} - ROC Curve (5-Fold)')
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.show()

    # SHAP feature importance
    avg_imp  = np.mean(all_shap_imp, axis=0)
    sorted_i = np.argsort(avg_imp)[::-1]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh([FEATURES[i] for i in sorted_i], avg_imp[sorted_i], color='steelblue')
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'{label} - SHAP Feature Importance')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    # SHAP beeswarm plot
    shap_concat = np.vstack(all_shap_vals_2d)
    feat_concat = np.vstack(all_feat_vals_2d)
    plt.figure()
    shap.summary_plot(shap_concat, feat_concat, feature_names=list(FEATURES),
                      max_display=N_FEATURES, show=True)

    # Metric summary
    print(f'\n── {label} Summary ──────────────────────────────────────────')
    for fi, m in enumerate(fold_metrics):
        print(f'  Fold {fi+1}: '
              f'Acc={m["acc"]:.4f} | Sens={m["sens"]:.4f} | Spec={m["spec"]:.4f} | '
              f'PPV={m["ppv"]:.4f} | NPV={m["npv"]:.4f} | F1={m["f1"]:.4f} | '
              f'ROC={m["roc_auc"]:.4f} | PR={m["pr_auc"]:.4f} | Brier={m["brier"]:.4f}')
    print(f'\n  Summary (mean ± std)')
    print(f'  {"-"*50}')
    for k, n in zip(METRIC_KEYS, METRIC_NAMES):
        vals = [m[k] for m in fold_metrics]
        print(f'  {n:<14}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')


print('\n' + '='*60)
print('  Experiment: data.xlsx')
print('='*60)

data_orig = pd.read_excel('data.xlsx')
run_experiment(data_orig, label='MFFNet')

print('\nExperiment complete!')