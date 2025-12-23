import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_recall_curve, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import warnings

warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostClassifier
    USE_CATBOOST = True
except:
    USE_CATBOOST = False

TRAIN_FEATURES = "train_features.csv"
TEST_FEATURES = "test_features.csv"
DEBUG_FILE = "debug_predictions.csv"
SUBMISSION_FILE = "submission.csv"
IS_PRODUCTION = True

SCALE_WEIGHT = 15.0
N_ESTIMATORS = 1000
LEARNING_RATE = 0.015

def smart_veto(X, probs, threshold=0.5):
    veto_mask = np.zeros(len(X), dtype=bool)
    
    if 'detection_coincidence' in X.columns:
        veto_mask |= (X['detection_coincidence'] < 2) & (probs < 0.3)
    
    if 'total_pre_bumps' in X.columns:
        veto_mask |= (X['total_pre_bumps'] > 30) & (probs < 0.4)
    
    sig_cols = [c for c in X.columns if 'peak_significance' in c]
    if sig_cols:
        max_sig = X[sig_cols].max(axis=1)
        veto_mask |= (max_sig < 3.0) & (probs < 0.2)
    
    if 'cooling_ratio' in X.columns:
        veto_mask |= (X['cooling_ratio'] < -50) & (probs < 0.3)
    
    return veto_mask

def train_and_evaluate(is_production=False):
    try:
        df_train = pd.read_csv(TRAIN_FEATURES)
        if is_production:
            df_test = pd.read_csv(TEST_FEATURES, encoding='latin-1')
    except:
        return

    if 'object_id' in df_train.columns:
        df_train = df_train.set_index('object_id')
    
    drop_cols = ['target', 'truth', 'group', 'Z_err', 'SpecType']
    features = [c for c in df_train.columns if c not in drop_cols]
    
    target_col = 'truth' if 'truth' in df_train.columns else 'target'
    X = df_train[features].fillna(0)
    y = df_train[target_col]

    selector = SelectFromModel(
        lgb.LGBMClassifier(
            n_estimators=100,
            class_weight={0: 1, 1: SCALE_WEIGHT},
            random_state=42,
            verbose=-1
        ),
        threshold='0.5*median'
    )
    selector.fit(X, y)
    
    selected_features = [f for f, s in zip(features, selector.get_support()) if s]
    X_selected = X[selected_features]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    final_models = []
    best_thresholds = []
    fold_scores = []
    feature_importances = np.zeros(len(selected_features))
    mistakes_list = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y)):
        X_train = X_selected.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_val = X_selected.iloc[val_idx]
        y_val = y.iloc[val_idx]

        mono_constraints = []
        for feat in selected_features:
            if feat == 'total_pre_bumps':
                mono_constraints.append(-1)
            elif feat == 'total_post_bumps':
                mono_constraints.append(-1)
            elif feat == 'detection_coincidence':
                mono_constraints.append(1)
            elif 'flux_ratio_' in feat:
                mono_constraints.append(1)
            elif feat == 'cooling_ratio':
                mono_constraints.append(1)
            elif 'neumann' in feat:
                mono_constraints.append(-1)
            elif feat == 'peak_overlap':
                mono_constraints.append(1)
            elif 'peak_significance' in feat:
                mono_constraints.append(1)
            else:
                mono_constraints.append(0)

        clf_xgb = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=SCALE_WEIGHT,
            eval_metric='logloss',
            random_state=42 + fold,
            n_jobs=-1,
            monotone_constraints=tuple(mono_constraints)
        )

        clf_lgb = lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            num_leaves=50,
            max_depth=7,
            class_weight={0: 1, 1: SCALE_WEIGHT},
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1
        )

        model = VotingClassifier(
            estimators=[('xgb', clf_xgb), ('lgb', clf_lgb)],
            voting='soft',
            weights=[1.2, 1.0]
        )

        model.fit(X_train, y_train)
        probs_xgb_lgb = model.predict_proba(X_val)[:, 1]

        probs_list = [probs_xgb_lgb]
        use_catboost_fold = False

        if USE_CATBOOST:
            try:
                clf_cat = CatBoostClassifier(
                    iterations=N_ESTIMATORS,
                    learning_rate=LEARNING_RATE,
                    depth=6,
                    scale_pos_weight=SCALE_WEIGHT,
                    random_state=42 + fold,
                    verbose=0,
                    allow_writing_files=False
                )
                clf_cat.fit(X_train, y_train)
                probs_cat = clf_cat.predict_proba(X_val)[:, 1]
                probs_list.append(probs_cat)
                use_catboost_fold = True
            except:
                pass

        if use_catboost_fold:
            probs = (probs_xgb_lgb * 2.2 + probs_cat) / 3.2
        else:
            probs = probs_xgb_lgb

        final_models.append({
            'main': model,
            'catboost': clf_cat if use_catboost_fold else None,
            'use_cat': use_catboost_fold
        })

        precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_thresholds.append(best_thresh)

        preds = (probs >= best_thresh).astype(int)
        veto_mask = smart_veto(X_val, probs, best_thresh)
        preds[veto_mask] = 0

        fold_scores.append(f1_score(y_val, preds))

        if fold == 0:
            clf_xgb.fit(X_train, y_train)
            feature_importances += clf_xgb.feature_importances_

        val_meta = df_train.iloc[val_idx].copy()
        val_meta['prob'] = probs
        val_meta['pred'] = preds
        val_meta['truth'] = y_val
        val_meta['fold'] = fold + 1
        val_meta['vetoed'] = veto_mask

        relevant_mask = (val_meta['truth'] == 1) | (val_meta['pred'] == 1)
        mistakes_list.append(val_meta[relevant_mask])

    if mistakes_list:
        pd.concat(mistakes_list).to_csv(DEBUG_FILE)

    if is_production:
        if 'object_id' in df_test.columns:
            df_test = df_test.set_index('object_id')

        X_test = df_test.reindex(columns=selected_features, fill_value=0).fillna(0)

        all_probs = []
        for model_dict in final_models:
            probs_main = model_dict['main'].predict_proba(X_test)[:, 1]
            if model_dict['use_cat'] and model_dict['catboost'] is not None:
                probs_cat = model_dict['catboost'].predict_proba(X_test)[:, 1]
                probs = (probs_main * 2.2 + probs_cat) / 3.2
            else:
                probs = probs_main
            all_probs.append(probs)

        avg_probs = np.mean(all_probs, axis=0)
        veto_mask_test = smart_veto(X_test, avg_probs, np.mean(best_thresholds))
        avg_probs[veto_mask_test] = 0.0

        final_classes = (avg_probs >= np.mean(best_thresholds)).astype(int)

        submission = pd.DataFrame({
            'object_id': X_test.index.values,
            'prediction': final_classes
        })
        submission.to_csv(SUBMISSION_FILE, index=False)

if __name__ == "__main__":
    train_and_evaluate(is_production=IS_PRODUCTION)
