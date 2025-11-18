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
import json
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, LeaveOneGroupOut, ShuffleSplit, train_test_split
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
    RAW_DATA_DIR = 'raw_data_l'
    OUTPUT_DIR = 'results_final_l'
    
    TEMP_START = 400
    TEMP_END = 990
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
    TEMP_BIN = 10
    # 检查点设置：是否从保存的最优配置快速启动
    RESUME_FROM_CHECKPOINT = True
    BEST_CONFIG_FILE = 'best_config.json'
    
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
# 1. 数据工程
# ==============================================================================
class DataEngineer:
    def __init__(self, config):
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.RANDOM_SEED)
        # 固定随机数生成器，保证相同原始数据在多次运行中得到完全一致的重采样结果

    def _load_single(self, filepath):
        try:
            # 原始 CSV 前几行通常写入元数据，因此跳过后仅抓取探头电压列
            df = pd.read_csv(filepath, header=None, skiprows=4, encoding='gbk')
            if len(df) == 0: return None
            return df.iloc[:, 0].values.astype(float)
        except:
            return None

    def _clean_and_resample(self, v1, v2, temp_c):
        """对同一温度点的两个探头读数进行 3σ 清洗，并以 array_split 等份平均"""
        def clean(arr):
            # 通过对称阈值剔除异常点，保留稳定片段
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
        # 利用固定 RNG 打乱顺序，避免序列性偏差
        v1_s, v2_s = v1_c[idx], v2_c[idx]
        n_batches = min(self.cfg.RESAMPLE_N, min_len)
        # array_split 确保所有样本被均匀分入批次，而不是简单截断前 RESAMPLE_N 条
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
            # 两个探头文件缺一不可，缺失则跳过该温度点
            f1 = os.path.join(self.cfg.RAW_DATA_DIR, f"{t}_1550.csv")
            f2 = os.path.join(self.cfg.RAW_DATA_DIR, f"{t}_1650.csv")
            if not os.path.exists(f1): continue
            
            d1, d2 = self._load_single(f1), self._load_single(f2)
            if d1 is None or d2 is None: continue
            
            all_data.extend(self._clean_and_resample(d1, d2, t))
            
        df = pd.DataFrame(all_data)
        
        print(">>> 按样本随机划分数据集 (train_test_split 80/20)...")
        train_df, test_df = train_test_split(
            df,
            test_size=self.cfg.TEST_SIZE,
            random_state=self.cfg.RANDOM_SEED,
            shuffle=True
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
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
        # 先计算比值并构造质控掩码，避免除零或负值进入后续对数变换
        ratio = df['V1'] / df['V2']
        invalid_mask = (
            ~np.isfinite(ratio) |
            (ratio <= eps) |
            (np.abs(df['V1'] + df['V2']) < eps)
        )
        if invalid_mask.any():
            print(f"警告: 发现 {invalid_mask.sum()} 条样本分母接近零或比值≤0，已剔除。")
        df = df[~invalid_mask].copy()
        df['QC_Flag'] = 'ok'  # 可扩展为更多质控标签

        # 对数与归一化差分均使用 clip/epsilon 保护，防止数值不稳定
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
        self.effective_degree = None
        self.model = None

    def train(self, df):
        print("\n>>> [Stage 3] 多项式比值回归 (论文基线)")
        # 根据样本数动态下调多项式阶数，避免欠定导致的拟合震荡
        ratios = df['Ratio'].values
        temps_c = df['T_true'].values
        if len(ratios) < 2:
            raise ValueError("多项式回归至少需要两条样本。")
        deg_cap = min(self.degree, len(ratios) - 1)
        self.effective_degree = max(1, deg_cap)
        coeffs = np.polyfit(ratios, temps_c, self.effective_degree)
        self.model = np.poly1d(coeffs)
        preds_c = self.model(ratios)
        r2 = r2_score(temps_c, preds_c)
        print(f"    -> 拟合完成 (deg={self.effective_degree}), R2(°C) = {r2:.6f}")

    def predict(self, df):
        if self.model is None:
            raise RuntimeError("Polynomial baseline not fitted.")
        preds_c = self.model(df['Ratio'].values)
        return pd.Series(preds_c, index=df.index, name='Poly-Ratio')
    
    def format_equation(self):
        """返回 T=f(R) 的三次多项式字符串，便于图中展示"""
        if self.model is None:
            return ""
        coeffs = self.model.c
        degree = len(coeffs) - 1
        pieces = []
        for idx, coeff in enumerate(coeffs):
            power = degree - idx
            coeff_str = f"{coeff:+.4e}"
            if power == 0:
                pieces.append(coeff_str)
            elif power == 1:
                pieces.append(f"{coeff_str}·R")
            else:
                pieces.append(f"{coeff_str}·R^{power}")
        return "T = " + " ".join(pieces)

# ==============================================================================
# 4. XGBoost 优化 (GroupKFold 版)
# ==============================================================================
class XGBoostManager:
    def __init__(self, config):
        self.cfg = config
        self.best_params = None
        self.best_features = None
        self.history = []

    def _run_optuna(self, X_train, y_train, groups, feature_list, n_trials):
        def objective(trial):
            # 极端稀疏或单样本场景直接返回无效分数，防止 Optuna 染毒
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
                # 仅一个温度点时退化为 ShuffleSplit，以样本级别划分
                if len(X_train) == 1:
                    return float('inf')
                test_size = min(max(1, int(np.ceil(len(X_train) * 0.2))), len(X_train) - 1)
                splitter = ShuffleSplit(n_splits=3, test_size=test_size, random_state=self.cfg.RANDOM_SEED)
                split_iter = splitter.split(X_train, y_train)
            elif unique_groups.size < self.cfg.N_FOLDS:
                # 温度点不足折数时使用留一分组，最大化可用验证组合
                splitter = LeaveOneGroupOut()
                split_iter = splitter.split(X_train, y_train, groups=groups)
            else:
                # 正常情况下使用 GroupKFold，避免同一温度泄漏
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

    def save_best(self, feature_combos, path):
        """将最优特征组合名称和参数保存到 JSON，便于下次快速启动"""
        if self.best_params is None or self.best_features is None:
            return
        # 反查组合名称，方便人眼查看
        best_name = None
        for name, feats in feature_combos.items():
            if feats == self.best_features:
                best_name = name
                break
        payload = {
            "best_combo_name": best_name,
            "best_features": self.best_features,
            "best_params": self.best_params
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"  -> 最优配置已保存到 {path}")

    def load_best(self, feature_combos, path):
        """从 JSON 恢复最优参数和特征组合（若失败则返回 False）"""
        if not os.path.exists(path):
            print(f"  -> 未找到检查点文件 {path}，需要重新寻优。")
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            best_name = payload.get("best_combo_name")
            best_features = payload.get("best_features")
            best_params = payload.get("best_params")
            # 如果保存了组合名且当前组合字典中存在，则以当前字典里的特征为准
            if best_name is not None and best_name in feature_combos:
                self.best_features = feature_combos[best_name]
            else:
                # 退化到直接使用保存的特征列表
                self.best_features = best_features
            self.best_params = best_params
            print(f"  -> 已从检查点加载最优配置: 组合名={best_name}, 特征数={len(self.best_features)}")
            return True
        except Exception as e:
            print(f"  -> 加载检查点失败，将重新寻优。原因: {e}")
            return False

    def train_single(self, df_train, df_test):
        print("  -> 训练单个 XGBoost 模型 (无集成)...")
        X_train = df_train[self.best_features]
        y_train = df_train['T_true']
        X_test = df_test[self.best_features]

        params = self.best_params.copy()
        params.update({'tree_method': 'hist', 'device': self.cfg.DEVICE, 'random_state': self.cfg.RANDOM_SEED})
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        return model, preds

# ==============================================================================
# 5. 数据效率 (修复版)
# ==============================================================================
def run_efficiency_study(cfg, df_train, df_test, xgbo_manager, best_feats_global):
    """
    [修改] 为每个稀疏子集重新运行轻量级寻优，防止过拟合
    """
    print("\n>>> [Stage 5] 数据效率探究 (带独立寻优)...")
    steps = [10, 20, 30,40,50,60,70,80,90,100]
    results = []
    
    y_test = df_test['T_true']
    unique_temps = np.sort(df_train['T_true'].unique())
    if len(unique_temps) == 0:
        return pd.DataFrame()
    min_temp = unique_temps[0]

    for step in steps:
        offsets = np.round(unique_temps - min_temp).astype(int)
        # 通过步长抽取温度点，模拟节省实验次数的场景
        selected_temps = unique_temps[offsets % step == 0]
        train_subset = df_train[df_train['T_true'].isin(selected_temps)]
        
        if train_subset.empty: continue
        
        print(f"  [Step {step}°C] 样本数: {len(train_subset)} ({len(selected_temps)} 个温度点)")

        # 先训练 Ridge 以保有与论文基线一致的对照
        try:
            poly_model = PolynomialRatioCalibrator()
            poly_model.train(train_subset)
            p_poly = poly_model.predict(df_test)
            err_poly = np.sqrt(np.mean(((y_test - p_poly)/y_test)**2)) * 100
        except ValueError:
            err_poly = np.nan

        print(f"    -> 正在为 {len(train_subset)} 样本重新寻优 (Trials={cfg.OPTUNA_TRIALS_SPARSE})...")
        sparse_params = xgbo_manager.optimize_sparse(train_subset, best_feats_global)
        model_xgb = xgb.XGBRegressor(**sparse_params, tree_method='hist', device=cfg.DEVICE)
        model_xgb.fit(train_subset[best_feats_global], train_subset['T_true'])
        p_xgb = model_xgb.predict(df_test[best_feats_global])
        err_xgb = np.sqrt(np.mean(((y_test - p_xgb)/y_test)**2)) * 100

        print(f"    -> Poly Err: {err_poly:.3f}% | XGB Err: {err_xgb:.3f}%")
        results.append({
            'Step_Size': step,
            'Samples': len(train_subset),
            'Poly_RMSPE': err_poly,
            'XGB_RMSPE': err_xgb
        })
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
    # 85% 分位作为训练上限，剩余高温模拟“未校准区间”
    if threshold >= temp_max:
        threshold = combined['T_true'].median()

    train_ext = combined[combined['T_true'] <= threshold]
    extrap_set = combined[combined['T_true'] > threshold]

    if train_ext.empty or extrap_set.empty:
        print("  高温段样本不足，无法评估外推性。")
        return

    # 为导出做准备
    extrap_set = extrap_set.copy()

    print(f"  -> 训练温度 ≤ {threshold:.1f}°C, 外推区间 {extrap_set['T_true'].min():.1f}°C ~ {extrap_set['T_true'].max():.1f}°C")
    print(f"     训练样本 {len(train_ext)} 条，高温测试样本 {len(extrap_set)} 条。")

    poly_model = PolynomialRatioCalibrator()
    poly_model.train(train_ext)
    pred_poly = poly_model.predict(extrap_set)

    params = best_params.copy()
    params.update({'tree_method': 'hist', 'device': cfg.DEVICE, 'random_state': cfg.RANDOM_SEED + 2024})
    xgb_model = xgb.XGBRegressor(**params)
    xgb_model.fit(train_ext[best_feats], train_ext['T_true'])
    pred_xgb = xgb_model.predict(extrap_set[best_feats])

    # 追加到 df，方便导出
    extrap_set['Pred_Poly'] = pred_poly.values
    extrap_set['Pred_XGB'] = pred_xgb
    extrap_set['Residual_Poly'] = extrap_set['Pred_Poly'] - extrap_set['T_true']
    extrap_set['Residual_XGB'] = extrap_set['Pred_XGB'] - extrap_set['T_true']

    # 保存外推测试明细
    extrap_csv = os.path.join(cfg.OUTPUT_DIR, "extrapolation_detail.csv")
    extrap_set.to_csv(extrap_csv, index=False)

    def _report(name, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmspe = np.sqrt(np.mean(((y_true - y_pred) / np.clip(y_true, cfg.NUMERIC_EPS, None))**2)) * 100
        print(f"     {name:<12} RMSE = {rmse:6.3f}°C | RMSPE = {rmspe:5.2f}%")

    _report("Poly-Ratio", extrap_set['T_true'], extrap_set['Pred_Poly'])
    _report("XGB-Optimal", extrap_set['T_true'], extrap_set['Pred_XGB'])

    plt.figure(figsize=(8, 6))
    # 三种模型的散点可直观看出在高温区的偏移趋势
    ideal_min, ideal_max = extrap_set['T_true'].min(), extrap_set['T_true'].max()
    plt.plot([ideal_min, ideal_max], [ideal_min, ideal_max], 'k--', lw=1, label='Ideal')
    plt.scatter(extrap_set['T_true'], pred_poly, c='tab:green', label='Poly-Ratio', marker='s')
    plt.scatter(extrap_set['T_true'], pred_xgb, c='red', label='XGB-Optimal', marker='o')
    plt.xlabel('真实温度 (°C)')
    plt.ylabel('预测温度 (°C)')
    plt.title('Figure 7: 外推性能对比')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.OUTPUT_DIR}/Figure7_Extrapolation.png", dpi=300)

def to_temp_bin(series, bin_size, base):
    """将温度映射到固定 bin_size (°C) 的下界，确保分析步长一致"""
    return (np.round((series - base) / bin_size) * bin_size + base).astype(int)

def run_low_temp_focus(final_df, cfg):
    print("\n>>> [Stage 8] 低温区间稳定性讨论...")
    if final_df.empty:
        print("  测试集为空，跳过。")
        return
    threshold = None
    low_df = pd.DataFrame()
    for q in (0.3, 0.4, 0.5):
        # 动态调节分位数，确保低温段至少覆盖 15% 测试样本
        threshold = final_df['T_true'].quantile(q)
        low_df = final_df[final_df['T_true'] <= threshold]
        if len(low_df) >= max(8, int(0.15 * len(final_df))):
            break
    if threshold is None or low_df.empty:
        print("  低温样本不足。")
        return
    # 复制一份用于导出
    low_df = low_df.copy()

    ratio_cv = low_df['Ratio'].std() / (np.abs(low_df['Ratio'].mean()) + cfg.NUMERIC_EPS)
    # Ratio 的变异系数帮助判断探头在低区的噪声水平
    print(f"  -> 阈值 ≤ {threshold:.1f}°C，占比 {len(low_df)/len(final_df)*100:.1f}%")
    print(f"     低压均值: V1={low_df['V1'].mean():.3f}, V2={low_df['V2'].mean():.3f}, Ratio CV={ratio_cv:.2f}")
    def _err(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rmspe = np.sqrt(np.mean(((y_true - y_pred)/np.clip(y_true, cfg.NUMERIC_EPS, None))**2)) * 100
        return rmse, rmspe
    p_rmse, p_rmspe = _err(low_df['T_true'], low_df['Pred_Poly'])
    x_rmse, x_rmspe = _err(low_df['T_true'], low_df['Pred_XGB'])
    print(f"     Poly-Ratio   RMSE={p_rmse:6.3f}°C | RMSPE={p_rmspe:5.2f}%")
    print(f"     XGB-Optimal  RMSE={x_rmse:6.3f}°C | RMSPE={x_rmspe:5.2f}%")

    # 构造低温残差列并导出
    low_df['Residual_Poly'] = low_df['Pred_Poly'] - low_df['T_true']
    low_df['Residual_XGB'] = low_df['Pred_XGB'] - low_df['T_true']
    low_df['Temp_Bin'] = to_temp_bin(low_df['T_true'], cfg.TEMP_BIN, cfg.TEMP_START)
    low_detail_csv = os.path.join(cfg.OUTPUT_DIR, "low_temp_detail.csv")
    low_df.to_csv(low_detail_csv, index=False)

    plt.figure(figsize=(9, 5))
    # 分箱求均值用于画图
    residual_low = low_df.groupby('Temp_Bin')[['Pred_Poly', 'Pred_XGB', 'T_true']].mean().reset_index()
    residual_low['Poly_Residual'] = residual_low['Pred_Poly'] - residual_low['T_true']
    residual_low['XGB_Residual'] = residual_low['Pred_XGB'] - residual_low['T_true']
    plt.plot(residual_low['Temp_Bin'], residual_low['Poly_Residual'], 'g-o', label='Poly-Ratio')
    plt.plot(residual_low['Temp_Bin'], residual_low['XGB_Residual'], 'r-s', label='XGB-Optimal')
    plt.xticks(np.arange(residual_low['Temp_Bin'].min(), residual_low['Temp_Bin'].max() + cfg.TEMP_BIN, cfg.TEMP_BIN))
    plt.xlabel('温度 (°C)')
    plt.ylabel('残差 (°C)')
    plt.title('Figure 8: 低温区间残差对比')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.OUTPUT_DIR}/Figure8_LowTempResiduals.png", dpi=300)

# ==============================================================================
# 主程序
# ==============================================================================
def main():
    cfg = Config()
    
    # 1. 数据
    de = DataEngineer(cfg)
    train_df, test_df = de.run()
    if train_df is None:
        return

    # 2. 特征 (带筛选)
    fe = FeatureEngineer(cfg)
    train_df = fe.generate(train_df)
    test_df = fe.generate(test_df)
    feature_combos = fe.filter_and_get_combinations(train_df)
    print(f"筛选后剩余 {len(feature_combos)} 种特征组合。")

    # 保存带特征的原始 train/test，方便后续分析
    train_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "train_features.csv"), index=False)
    test_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "test_features.csv"), index=False)
    
    # 3. 多项式回归 (论文基线)
    poly_mgr = PolynomialRatioCalibrator()
    poly_mgr.train(train_df)
    poly_preds = poly_mgr.predict(test_df)
    
    # 4. XGBoost（支持检查点）
    xgbo = XGBoostManager(cfg)
    best_config_path = os.path.join(cfg.OUTPUT_DIR, cfg.BEST_CONFIG_FILE)

    used_checkpoint = False
    if cfg.RESUME_FROM_CHECKPOINT:
        # 尝试直接从检查点加载最优配置
        used_checkpoint = xgbo.load_best(feature_combos, best_config_path)

    if not used_checkpoint:
        # 正常寻优
        best_params, best_feats = xgbo.optimize_main(train_df, feature_combos)
        # 将当前最优配置保存下来（下次可快速启动）
        xgbo.save_best(feature_combos, best_config_path)
    else:
        # 从检查点恢复时，best_params / best_features 已在 load_best 里设置
        best_params, best_feats = xgbo.best_params, xgbo.best_features

    main_model, xgb_mean = xgbo.train_single(train_df, test_df)
    
    # 5. 效率 (修复版)
    eff_df = run_efficiency_study(cfg, train_df, test_df, xgbo, best_feats)
    
    # 6. 绘图与报告
    print("\n>>> [Stage 6] 生成最终图表...")
    
    y_test = test_df['T_true']
    final_df = test_df.copy()
    final_df['Pred_Poly'] = poly_preds
    final_df['Pred_XGB'] = xgb_mean
    final_df['Residual_Poly'] = final_df['Pred_Poly'] - final_df['T_true']
    final_df['Residual_XGB'] = final_df['Pred_XGB'] - final_df['T_true']
    final_df['Temp_Bin'] = to_temp_bin(final_df['T_true'], cfg.TEMP_BIN, cfg.TEMP_START)

    # 保存测试集预测与残差明细
    final_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "test_predictions_detail.csv"), index=False)

    plt.figure(figsize=(12, 7))
    # Figure 1: 三条拟合曲线（无置信区间）
    sort_idx = np.argsort(y_test.values)
    y_sorted = y_test.iloc[sort_idx].values
    plt.plot([y_sorted.min(), y_sorted.max()], [y_sorted.min(), y_sorted.max()], 'k--', lw=1, label='Ideal')
    plt.plot(y_sorted, final_df['Pred_Poly'].iloc[sort_idx], c='tab:green', lw=1.8, label='Poly-Ratio (deg3)')
    plt.plot(y_sorted, final_df['Pred_XGB'].iloc[sort_idx], c='red', lw=2.5, label='XGB-Optimal')
    poly_eq = poly_mgr.format_equation()
    plt.text(0.02, 0.02, f"Poly Cubic:\n{poly_eq}", transform=plt.gca().transAxes,
             fontsize=9, family='monospace', bbox=dict(facecolor='white', alpha=0.7, edgecolor='green'))
    plt.xlabel('真实温度 (°C)')
    plt.ylabel('预测温度 (°C)')
    plt.title('Figure 1: 拟合性能曲线 (摄氏)')
    plt.legend()
    plt.savefig(f"{cfg.OUTPUT_DIR}/Figure1_Performance_Curves.png", dpi=300)

    residual_by_temp = final_df.groupby('Temp_Bin')[['Residual_Poly', 'Residual_XGB']].mean().reset_index()
    # 保存分温度残差表
    residual_by_temp.to_csv(os.path.join(cfg.OUTPUT_DIR, "per_temp_residuals.csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(residual_by_temp['Temp_Bin'], residual_by_temp['Residual_Poly'], 'g-o', label='Poly-Ratio')
    plt.plot(residual_by_temp['Temp_Bin'], residual_by_temp['Residual_XGB'], 'r-s', label='XGB-Optimal')
    plt.xticks(np.arange(cfg.TEMP_START, cfg.TEMP_END + cfg.TEMP_BIN, cfg.TEMP_BIN))
    plt.xlabel('温度 (°C)')
    plt.ylabel('残差 (°C)')
    plt.title('Figure 2: 分温度残差 (步长 10°C)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.OUTPUT_DIR}/Figure2_PerTempResiduals.png", dpi=300)

    if not eff_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(eff_df['Step_Size'], eff_df['Poly_RMSPE'], 'g--o', label='Poly-Ratio', lw=2)
        plt.plot(eff_df['Step_Size'], eff_df['XGB_RMSPE'], 'r-s', label='XGB-Optimal', lw=2)
        plt.xlabel('采样步长 (°C)')
        plt.ylabel('测试集 RMSPE (%)')
        plt.title('Figure 4: 数据效率 (双模型对比)')
        plt.legend()
        plt.grid(True, linestyle='--')
        plt.savefig(f"{cfg.OUTPUT_DIR}/Figure4_Data_Efficiency.png", dpi=300)

    # 外推与低温会在各自函数内写 CSV
    run_extrapolation_test(train_df, test_df, best_params, best_feats, cfg)
    run_low_temp_focus(final_df, cfg)

    poly_rmse = np.sqrt(mean_squared_error(y_test, final_df['Pred_Poly']))
    poly_r2 = r2_score(y_test, final_df['Pred_Poly'])
    xgb_rmse = np.sqrt(mean_squared_error(y_test, final_df['Pred_XGB']))
    r2 = r2_score(y_test, final_df['Pred_XGB'])
    print("\n>>> 最终测试集指标:")
    print(f"     Poly-Ratio   RMSE = {poly_rmse:.4f} °C, R2 = {poly_r2:.5f}")
    print(f"     XGB-Optimal  RMSE = {xgb_rmse:.4f} °C, R2 = {r2:.5f}")

    final_df['Err_Poly'] = np.abs((y_test - final_df['Pred_Poly']) / y_test) * 100
    final_df['Err_XGB'] = np.abs((y_test - final_df['Pred_XGB']) / y_test) * 100
    # 如果你仍然希望有一个总汇文件，可以再保存一次（或留这一个为主）
    final_df.to_csv(os.path.join(cfg.OUTPUT_DIR, "final_predictions.csv"), index=False)
    
    print(f"所有结果已保存至 {cfg.OUTPUT_DIR}/")

if __name__ == "__main__":
    main()