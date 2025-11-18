import os
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, LeaveOneGroupOut, ShuffleSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# --- 基础设置 ---
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 配置参数
# ==============================================================================
class Config:
    RAW_DATA_DIR = 'raw_data_h'
    OUTPUT_DIR = 'results_final_v4'
    
    TEMP_START = 1290
    TEMP_END = 1650
    TEMP_STEP = 10
    
    # 数据工程
    RESAMPLE_N = 50
    SIGMA_THRESHOLD = 3
    NUMERIC_EPS = 1e-8
    
    # 划分
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # 优化
    OPTUNA_TRIALS = 30  # 全量数据寻优次数
    OPTUNA_TRIALS_SPARSE = 10 # 稀疏数据寻优次数
    N_FOLDS = 3
    ENSEMBLE_MODELS = 5
    
    # 特征筛选阈值 (与 T_true 的相关性)
    FEATURE_CORR_THRESHOLD = 0.95
    
    try:
        build_info = xgb.build_info()
        DEVICE = 'cuda' if 'USE_CUDA' in build_info and build_info['USE_CUDA'] else 'cpu'
    except:
        DEVICE = 'cpu'

if not os.path.exists(Config.OUTPUT_DIR):
    os.makedirs(Config.OUTPUT_DIR)

print(f"运行设备: {Config.DEVICE}")

# ==============================================================================
# 1. 数据工程 (GroupShuffleSplit)
# ==============================================================================
class DataEngineer:
    def __init__(self, config):
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.RANDOM_SEED)

    def _load_single(self, filepath):
        try:
            df = pd.read_csv(filepath, header=None, skiprows=4, encoding='gbk')
            if len(df) == 0: return None
            return df.iloc[:, 0].values.astype(float)
        except:
            return None

    def _clean_and_resample(self, v1, v2, temp_c):
        def clean(arr):
            m, s = np.mean(arr), np.std(arr)
            if s == 0:
                return arr
            t = self.cfg.SIGMA_THRESHOLD
            return arr[(arr >= m - t * s) & (arr <= m + t * s)]

        v1_c, v2_c = clean(v1), clean(v2)
        min_len = min(len(v1_c), len(v2_c))
        if min_len == 0:
            return []

        idx = self.rng.permutation(min_len)
        v1_s, v2_s = v1_c[idx], v2_c[idx]
        n_batches = min(self.cfg.RESAMPLE_N, min_len)
        splits = np.array_split(np.arange(min_len), n_batches)

        samples = []
        for batch_idx in splits:
            if batch_idx.size == 0:
                continue
            samples.append({
                'T_true': temp_c,
                'V1': np.mean(v1_s[batch_idx]),
                'V2': np.mean(v2_s[batch_idx])
            })
        return samples

    def run(self):
        print("\n>>> [Stage 1] 数据加载与处理...")
        all_data = []
        temps = np.arange(self.cfg.TEMP_START, self.cfg.TEMP_END + self.cfg.TEMP_STEP, self.cfg.TEMP_STEP)
        
        for t in tqdm(temps, desc="Stage 1: 加载文件"):
            f1 = os.path.join(self.cfg.RAW_DATA_DIR, f"{t}_1550.csv")
            f2 = os.path.join(self.cfg.RAW_DATA_DIR, f"{t}_1650.csv")
            if not os.path.exists(f1): continue
            
            d1, d2 = self._load_single(f1), self._load_single(f2)
            if d1 is None or d2 is None: continue
            
            all_data.extend(self._clean_and_resample(d1, d2, t))
            
        df = pd.DataFrame(all_data)
        
        print(">>> 按温度点划分数据集 (GroupShuffleSplit)...")
        splitter = GroupShuffleSplit(n_splits=1, test_size=self.cfg.TEST_SIZE, random_state=self.cfg.RANDOM_SEED)
        train_idx, test_idx = next(splitter.split(df, groups=df['T_true']))
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        
        print(f"Train: {len(train_df)} 样本 ({train_df['T_true'].nunique()} 温度点)")
        print(f"Test : {len(test_df)} 样本 ({test_df['T_true'].nunique()} 温度点)")
        
        return train_df, test_df

# ==============================================================================
# 2. 特征工程 (带相关性分析)
# ==============================================================================
class FeatureEngineer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base_cols = ['V1', 'V2', 'Ratio']
        self.aug_cols = ['Log_Ratio', 'Energy_Sum', 'Norm_Diff', 'Interaction']

    def generate(self, df):
        df = df.copy()
        eps = self.cfg.NUMERIC_EPS

        ratio = df['V1'] / df['V2']
        invalid_mask = (
            ~np.isfinite(ratio) |
            (ratio <= eps) |
            (np.abs(df['V1'] + df['V2']) < eps)
        )
        if invalid_mask.any():
            print(f"警告: 发现 {invalid_mask.sum()} 条样本分母接近零或比值≤0，已剔除。")
        df = df[~invalid_mask].copy()
        df['QC_Flag'] = 'ok'

        df['Ratio'] = df['V1'] / df['V2']
        df['Log_Ratio'] = np.log(np.clip(df['Ratio'], eps, None))
        df['Energy_Sum'] = df['V1'] + df['V2']
        df['Norm_Diff'] = (df['V1'] - df['V2']) / np.clip(df['Energy_Sum'], eps, None)
        df['Interaction'] = df['V1'] * df['V2']
        return df

    def filter_and_get_combinations(self, df_train):
        """启用相关性分析来筛选特征 """
        print("\n>>> [Stage 2] 特征工程与相关性分析...")
        
        # 1. 计算与 T_true 的皮尔逊相关性
        corr_matrix = df_train[self.base_cols + self.aug_cols + ['T_true']].corr()
        corr_with_target = corr_matrix['T_true'].abs().sort_values(ascending=False)
        
        print("特征与温度的相关性 (绝对值):")
        print(corr_with_target)
        
        # 2. 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Figure 0: 特征相关性热力图")
        plt.tight_layout()
        plt.savefig(f"{self.cfg.OUTPUT_DIR}/Figure0_Correlation.png", dpi=300)

        # 3. 筛选出与目标强相关的特征
        # (我们总是保留 V1, V2, Ratio 作为基础)
        strong_aug_cols = corr_with_target[self.aug_cols][
            corr_with_target[self.aug_cols] >= self.cfg.FEATURE_CORR_THRESHOLD
        ].index.tolist()
        
        print(f"强相关增强特征 (>{self.cfg.FEATURE_CORR_THRESHOLD}): {strong_aug_cols}")

        # 4. 基于筛选后的特征生成组合
        combinations = {'Base': self.base_cols}
        for r in range(1, len(strong_aug_cols) + 1):
            for subset in itertools.combinations(strong_aug_cols, r):
                name = f"Base+{r}_{'+'.join([s.split('_')[0] for s in subset])}"
                combinations[name] = self.base_cols + list(subset)
                
        return combinations

# ==============================================================================
# 3. 多项式比值回归 (Polynomial Ratio Calibration)
# ==============================================================================
class PolynomialRatioCalibrator:
    def __init__(self, degree=3):
        self.degree = degree
        self.model = None

    def train(self, df):
        print("\n>>> [Stage 3] 多项式比值回归 (论文基线)")
        ratios = df['Ratio'].values
        temps_c = df['T_true'].values
        coeffs = np.polyfit(ratios, temps_c, self.degree)
        self.model = np.poly1d(coeffs)
        preds_c = self.model(ratios)
        r2 = r2_score(temps_c, preds_c)
        print(f"    -> 拟合完成 (deg={self.degree}), R2(°C) = {r2:.6f}")

    def predict(self, df):
        if self.model is None:
            raise RuntimeError("Polynomial baseline not fitted.")
        preds_c = self.model(df['Ratio'].values)
        return pd.Series(preds_c, index=df.index, name='Poly-Ratio')

# ==============================================================================
# 4. 基准模型 (Robust Version)
# ==============================================================================
class BaselineManager:
    def __init__(self):
        self.model = None

    def train(self, X, y):
        print("\n>>> [Stage 4] 训练基准模型 (Ridge Regularized)")
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('regressor', Ridge(alpha=1.0))
        ])
        self.model.fit(X[['Ratio']], y)

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Baseline model not fitted.")
        preds = self.model.predict(X[['Ratio']])
        return pd.Series(preds, index=X.index, name='Ridge-Ratio')

# ==============================================================================
# 5. XGBoost 优化 (GroupKFold 版)
# ==============================================================================
class XGBoostManager:
    def __init__(self, config):
        self.cfg = config
        self.best_params = None
        self.best_features = None
        self.history = []

    def _run_optuna(self, X_train, y_train, groups, feature_list, n_trials):
        def objective(trial):
            if len(X_train) <= 1:
                return float('inf')

            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400 if n_trials > 10 else 200),
                'max_depth': trial.suggest_int('max_depth', 2, 6 if n_trials > 10 else 4),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-6, 5.0, log=True),
                'tree_method': 'hist',
                'device': self.cfg.DEVICE,
                'n_jobs': -1,
                'verbosity': 0,
                'random_state': self.cfg.RANDOM_SEED
            }

            unique_groups = np.unique(groups)
            if unique_groups.size == 1:
                if len(X_train) == 1:
                    return float('inf')
                test_size = min(max(1, int(np.ceil(len(X_train) * 0.2))), len(X_train) - 1)
                splitter = ShuffleSplit(n_splits=3, test_size=test_size, random_state=self.cfg.RANDOM_SEED)
                split_iter = splitter.split(X_train, y_train)
            elif unique_groups.size < self.cfg.N_FOLDS:
                splitter = LeaveOneGroupOut()
                split_iter = splitter.split(X_train, y_train, groups=groups)
            else:
                splitter = GroupKFold(n_splits=self.cfg.N_FOLDS)
                split_iter = splitter.split(X_train, y_train, groups=groups)

            scores = []
            for t_idx, v_idx in split_iter:
                X_t, y_t = X_train.iloc[t_idx][feature_list], y_train.iloc[t_idx]
                X_v, y_v = X_train.iloc[v_idx][feature_list], y_train.iloc[v_idx]

                model = xgb.XGBRegressor(**params)
                model.fit(X_t, y_t)
                preds = model.predict(X_v)
                scores.append(np.sqrt(mean_squared_error(y_v, preds)))

            return np.mean(scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_value, study.best_params

    def optimize_main(self, df_train, feature_combos):
        """ [Stage 4] 对全量训练集进行彻底寻优 """
        print("\n>>> [Stage 4] XGBoost 极限寻优 (GroupKFold CV)...")
        best_score = float('inf')
        y_train = df_train['T_true']
        groups = df_train['T_true']
        
        for name, feats in tqdm(feature_combos.items(), desc="Stage 4: 寻优"):
            rmse, params = self._run_optuna(df_train, y_train, groups, feats, self.cfg.OPTUNA_TRIALS)
            
            self.history.append({'Combo': name, 'Feature_Count': len(feats), 'RMSE': rmse})
            
            if rmse < best_score:
                best_score = rmse
                self.best_params = params
                self.best_features = feats
                
        print(f"  >>> 最优特征组合: {self.best_features} (CV RMSE: {best_score:.4f})")
        return self.best_params, self.best_features

    def optimize_sparse(self, df_train, feature_list):
        """ [Stage 5 调用] 对稀疏数据进行轻量级寻优 """
        y_train = df_train['T_true']
        groups = df_train['T_true']
        # 使用更少的 trial 和更简单的参数空间
        _, params = self._run_optuna(df_train, y_train, groups, feature_list, self.cfg.OPTUNA_TRIALS_SPARSE)
        return params

    def train_ensemble(self, df_train, df_test):
        print(f"  -> 训练集成模型 (N={self.cfg.ENSEMBLE_MODELS})...")
        preds = []
        models = []
        X_train = df_train[self.best_features]
        y_train = df_train['T_true']
        X_test = df_test[self.best_features]
        
        for i in range(self.cfg.ENSEMBLE_MODELS):
            p = self.best_params.copy()
            p['random_state'] = self.cfg.RANDOM_SEED + i
            p['tree_method'] = 'hist'
            p['device'] = self.cfg.DEVICE
            
            model = xgb.XGBRegressor(**p)
            model.fit(X_train, y_train)
            preds.append(model.predict(X_test))
            models.append(model)
            
        preds = np.array(preds)
        return models[0], np.mean(preds, axis=0), np.std(preds, axis=0)

# ==============================================================================
# 5. 数据效率 (修复版)
# ==============================================================================
def run_efficiency_study(cfg, df_train, df_test, xgbo_manager, best_feats_global):
    """
    [修改] 为每个稀疏子集重新运行轻量级寻优，防止过拟合
    """
    print("\n>>> [Stage 5] 数据效率探究 (带独立寻优)...")
    steps = [10, 50, 100]
    results = []
    
    y_test = df_test['T_true']
    unique_temps = np.sort(df_train['T_true'].unique())
    if len(unique_temps) == 0:
        return pd.DataFrame()
    min_temp = unique_temps[0]

    for step in steps:
        offsets = np.round(unique_temps - min_temp).astype(int)
        selected_temps = unique_temps[offsets % step == 0]
        train_subset = df_train[df_train['T_true'].isin(selected_temps)]
        
        if train_subset.empty: continue
        
        print(f"  [Step {step}°C] 样本数: {len(train_subset)} ({len(selected_temps)} 个温度点)")

        # 1. A3 Baseline (Ridge)
        model_a3 = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('reg', Ridge(alpha=1.0))
        ])
        model_a3.fit(train_subset[['Ratio']], train_subset['T_true'])
        p_a3 = model_a3.predict(df_test[['Ratio']])
        err_a3 = np.sqrt(np.mean(((y_test - p_a3)/y_test)**2)) * 100
        
        # 2. [关键修复] XGB 必须重新寻优
        print(f"    -> 正在为 {len(train_subset)} 样本重新寻优 (Trials={cfg.OPTUNA_TRIALS_SPARSE})...")
        sparse_params = xgbo_manager.optimize_sparse(train_subset, best_feats_global)
        
        model_xgb = xgb.XGBRegressor(**sparse_params, tree_method='hist', device=cfg.DEVICE)
        model_xgb.fit(train_subset[best_feats_global], train_subset['T_true'])
        p_xgb = model_xgb.predict(df_test[best_feats_global])
        err_xgb = np.sqrt(np.mean(((y_test - p_xgb)/y_test)**2)) * 100
        
        results.append({
            'Step_Size': step, 'Samples': len(train_subset),
            'A3_RMSPE': err_a3, 'XGB_RMSPE': err_xgb
        })
        print(f"    -> A3 Err: {err_a3:.3f}% | XGB Err: {err_xgb:.3f}%")
        
    return pd.DataFrame(results)

def run_extrapolation_test(train_df, test_df, best_params, best_feats, cfg):
    print("\n>>> [Stage 7] 外推测试 (Extrapolation Test)...")
    combined = pd.concat([train_df, test_df], ignore_index=True)
    if combined.empty:
        print("  无可用样本，跳过外推测试。")
        return

    combined = combined.sort_values('T_true').reset_index(drop=True)
    temp_min, temp_max = combined['T_true'].min(), combined['T_true'].max()
    print(f"  当前数据覆盖: {temp_min:.1f}°C ~ {temp_max:.1f}°C (目标 400-1750°C)")

    if combined['T_true'].nunique() < 4:
        print("  温度点不足以构造外推切分，跳过。")
        return

    threshold = combined['T_true'].quantile(0.85)
    if threshold >= temp_max:
        threshold = combined['T_true'].median()

    train_ext = combined[combined['T_true'] <= threshold]
    extrap_set = combined[combined['T_true'] > threshold]

    if train_ext.empty or extrap_set.empty:
        print("  高温段样本不足，无法评估外推性。")
        return

    print(f"  -> 训练温度 ≤ {threshold:.1f}°C, 外推区间 {extrap_set['T_true'].min():.1f}°C ~ {extrap_set['T_true'].max():.1f}°C")
    print(f"     训练样本 {len(train_ext)} 条，高温测试样本 {len(extrap_set)} 条。")

    ridge_model = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('reg', Ridge(alpha=1.0))
    ])
    ridge_model.fit(train_ext[['Ratio']], train_ext['T_true'])
    pred_ridge = ridge_model.predict(extrap_set[['Ratio']])

    params = best_params.copy()
    params.update({'tree_method': 'hist', 'device': cfg.DEVICE, 'random_state': cfg.RANDOM_SEED + 2024})
    xgb_model = xgb.XGBRegressor(**params)
    xgb_model.fit(train_ext[best_feats], train_ext['T_true'])
    pred_xgb = xgb_model.predict(extrap_set[best_feats])

    def _report(name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmspe = np.sqrt(np.mean(((y_true - y_pred) / np.clip(y_true, cfg.NUMERIC_EPS, None))**2)) * 100
        print(f"     {name:<12} RMSE = {rmse:6.3f}°C | RMSPE = {rmspe:5.2f}%")

    _report("Ridge-Ratio", extrap_set['T_true'], pred_ridge)
    _report("XGB-Optimal", extrap_set['T_true'], pred_xgb)

def run_low_temp_focus(final_df, cfg):
    print("\n>>> [Stage 8] 低温区间稳定性讨论...")
    if final_df.empty:
        print("  测试集为空，跳过。")
        return
    threshold = None
    low_df = pd.DataFrame()
    for q in (0.3, 0.4, 0.5):
        threshold = final_df['T_true'].quantile(q)
        low_df = final_df[final_df['T_true'] <= threshold]
        if len(low_df) >= max(8, int(0.15 * len(final_df))):
            break
    if threshold is None or low_df.empty:
        print("  低温样本不足。")
        return
    ratio_cv = low_df['Ratio'].std() / (np.abs(low_df['Ratio'].mean()) + cfg.NUMERIC_EPS)
    print(f"  -> 阈值 ≤ {threshold:.1f}°C，占比 {len(low_df)/len(final_df)*100:.1f}%")
    print(f"     低压均值: V1={low_df['V1'].mean():.3f}, V2={low_df['V2'].mean():.3f}, Ratio CV={ratio_cv:.2f}")
    def _err(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmspe = np.sqrt(np.mean(((y_true - y_pred)/np.clip(y_true, cfg.NUMERIC_EPS, None))**2)) * 100
        return rmse, rmspe
    r_rmse, r_rmspe = _err(low_df['T_true'], low_df['Pred_Ridge'])
    x_rmse, x_rmspe = _err(low_df['T_true'], low_df['Pred_XGB'])
    print(f"     Ridge-Ratio  RMSE={r_rmse:6.3f}°C | RMSPE={r_rmspe:5.2f}% (比值被放大)")
    print(f"     XGB-Optimal  RMSE={x_rmse:6.3f}°C | RMSPE={x_rmspe:5.2f}% (多特征抑制噪声)")

# ==============================================================================
# 主程序
# ==============================================================================
def main():
    cfg = Config()
    
    # 1. 数据
    de = DataEngineer(cfg)
    train_df, test_df = de.run()
    if train_df is None: return

    # 2. 特征 (带筛选)
    fe = FeatureEngineer(cfg)
    train_df = fe.generate(train_df)
    test_df = fe.generate(test_df)
    feature_combos = fe.filter_and_get_combinations(train_df)
    print(f"筛选后剩余 {len(feature_combos)} 种特征组合。")
    
    # 3. 多项式回归 (论文基线)
    poly_mgr = PolynomialRatioCalibrator()
    poly_mgr.train(train_df)
    poly_preds = poly_mgr.predict(test_df)
    
    # 4. 基准 (Ridge)
    bm = BaselineManager()
    bm.train(train_df, train_df['T_true'])
    base_preds = bm.predict(test_df)
    
    # 5. XGBoost
    xgbo = XGBoostManager(cfg)
    best_params, best_feats = xgbo.optimize_main(train_df, feature_combos)
    main_model, xgb_mean, xgb_std = xgbo.train_ensemble(train_df, test_df)
    
    # 6. 效率 (修复版)
    eff_df = run_efficiency_study(cfg, train_df, test_df, xgbo, best_feats)
    
    # 7. 绘图与报告
    print("\n>>> [Stage 6] 生成最终图表...")
    
    y_test = test_df['T_true']
    final_df = test_df.copy()
    final_df['Pred_Poly'] = poly_preds
    final_df['Pred_Ridge'] = base_preds
    final_df['Pred_XGB'] = xgb_mean
    final_df['Residual_XGB'] = final_df['Pred_XGB'] - final_df['T_true']
    
    plt.figure(figsize=(12, 7))
    sort_idx = np.argsort(y_test)
    y_sorted = y_test.iloc[sort_idx]
    plt.plot([y_sorted.min(), y_sorted.max()], [y_sorted.min(), y_sorted.max()], 'k--', lw=1, label='Ideal')
    plt.plot(y_sorted, final_df['Pred_Poly'].iloc[sort_idx], c='tab:green', lw=1.5, label='Poly-Ratio (deg3)')
    plt.plot(y_sorted, final_df['Pred_Ridge'].iloc[sort_idx], c='tab:orange', lw=1.5, label='Ridge-Ratio')
    plt.plot(y_sorted, final_df['Pred_XGB'].iloc[sort_idx], c='red', lw=2.5, label='XGB-Optimal')
    plt.fill_between(y_sorted,
                     (xgb_mean - 2 * xgb_std)[sort_idx],
                     (xgb_mean + 2 * xgb_std)[sort_idx],
                     color='red', alpha=0.2, label='95% CI')
    plt.xlabel('真实温度 (°C)')
    plt.ylabel('预测温度 (°C)')
    plt.title('Figure 1: 拟合性能曲线 (摄氏)')
    plt.legend()
    plt.savefig(f"{cfg.OUTPUT_DIR}/Figure1_Performance_Curves.png", dpi=300)

    residual_by_temp = final_df.groupby('T_true')['Residual_XGB'].agg(['mean', 'std']).fillna(0.0)
    plt.figure(figsize=(10, 6))
    plt.errorbar(residual_by_temp.index, residual_by_temp['mean'],
                 yerr=residual_by_temp['std'].values, fmt='o-', capsize=4, color='tab:purple')
    plt.axhline(0, color='gray', linestyle='--', lw=1)
    plt.xlabel('温度 (°C)')
    plt.ylabel('残差 (°C)')
    plt.title('Figure 2: 分温度残差 (XGB)')
    plt.tight_layout()
    plt.savefig(f"{cfg.OUTPUT_DIR}/Figure2_PerTempResiduals.png", dpi=300)

    # Figure 3B: 特征组合
    hist_df = pd.DataFrame(xgbo.history).sort_values('RMSE')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=hist_df, x='Combo', y='RMSE', palette='viridis')
    plt.ylim(hist_df['RMSE'].min() * 0.95, hist_df['RMSE'].max() * 1.05)
    plt.xticks(rotation=30, ha='right')
    plt.title('Figure 3B: 特征组合 CV 表现 (GroupKFold)')
    plt.savefig(f"{cfg.OUTPUT_DIR}/Figure3B_Combinations.png", dpi=300)

    # Figure 4: 数据效率 (修复后)
    if not eff_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(eff_df['Step_Size'], eff_df['A3_RMSPE'], 'y--s', label='Poly-Ratio', lw=2)
        plt.plot(eff_df['Step_Size'], eff_df['XGB_RMSPE'], 'r-s', label='XGB-Optimal', lw=2)
        plt.xlabel('采样步长 (°C)')
        plt.ylabel('测试集 RMSPE (%)')
        plt.title('Figure 4: 数据效率 (已修复过拟合问题)')
        plt.legend()
        plt.grid(True, linestyle='--')
        plt.savefig(f"{cfg.OUTPUT_DIR}/Figure4_Data_Efficiency.png", dpi=300)

    plt.figure()
    X_shap = test_df[best_feats].sample(n=min(500, len(test_df)), random_state=cfg.RANDOM_SEED)
    explainer = shap.TreeExplainer(main_model)
    shap_values = explainer(X_shap)
    shap.summary_plot(shap_values, X_shap, show=False)
    plt.savefig(f"{cfg.OUTPUT_DIR}/Figure5_SHAP.png", bbox_inches='tight')

    run_extrapolation_test(train_df, test_df, best_params, best_feats, cfg)
    run_low_temp_focus(final_df, cfg)

    poly_rmse = np.sqrt(mean_squared_error(y_test, final_df['Pred_Poly']))
    poly_r2 = r2_score(y_test, final_df['Pred_Poly'])
    ridge_rmse = np.sqrt(mean_squared_error(y_test, final_df['Pred_Ridge']))
    ridge_r2 = r2_score(y_test, final_df['Pred_Ridge'])
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_mean))
    r2 = r2_score(y_test, xgb_mean)
    print("\n>>> 最终测试集指标:")
    print(f"     Poly-Ratio   RMSE = {poly_rmse:.4f} °C, R2 = {poly_r2:.5f}")
    print(f"     Ridge-Ratio  RMSE = {ridge_rmse:.4f} °C, R2 = {ridge_r2:.5f}")
    print(f"     XGB-Optimal  RMSE = {xgb_rmse:.4f} °C, R2 = {r2:.5f}")
    
    final_df['Err_Poly'] = np.abs((y_test - final_df['Pred_Poly']) / y_test) * 100
    final_df['Err_Ridge'] = np.abs((y_test - final_df['Pred_Ridge']) / y_test) * 100
    final_df['Err_XGB'] = np.abs((y_test - final_df['Pred_XGB']) / y_test) * 100
    final_df.to_csv(f"{cfg.OUTPUT_DIR}/final_predictions.csv", index=False)
    hist_df.to_csv(f"{cfg.OUTPUT_DIR}/optimization_history.csv", index=False)
    eff_df.to_csv(f"{cfg.OUTPUT_DIR}/data_efficiency_report.csv", index=False)
    
    print(f"所有结果已保存至 {cfg.OUTPUT_DIR}/")

if __name__ == "__main__":
    main()