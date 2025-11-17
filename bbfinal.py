import os
import glob
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import joblib
import warnings
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 忽略一些不必要的警告
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 设置绘图风格
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ==============================================================================
# 配置参数 (Configuration)
# ==============================================================================
class Config:
    RAW_DATA_DIR = 'raw_data'  # 原始数据文件夹
    OUTPUT_DIR = 'results' # 结果输出文件夹
    
    # 温度范围
    TEMP_START = 400
    TEMP_END = 1750
    TEMP_STEP = 5
    
    # 第一阶段：数据工程参数
    RESAMPLE_N = 100  # 微批次增强倍数 (从60000点压缩为100个点)
    SIGMA_THRESHOLD = 3 # 3-Sigma 清洗阈值
    
    # 数据划分
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    
    # 第四阶段：XGBoost 优化参数
    OPTUNA_TRIALS = 20  # 每个特征组合的寻优次数 (建议设为20-50)
    N_FOLDS = 3         # 交叉验证折数
    ENSEMBLE_MODELS = 10 # 集成模型的数量
    
    # 硬件加速
    # 检查是否安装了GPU版本的XGBoost
    try:
        build_info = xgb.build_info()
        DEVICE = 'cuda' if 'USE_CUDA' in build_info and build_info['USE_CUDA'] else 'cpu'
    except:
        DEVICE = 'cpu'

if not os.path.exists(Config.OUTPUT_DIR):
    os.makedirs(Config.OUTPUT_DIR)

print(f"运行设备: {Config.DEVICE}")

# ==============================================================================
# 第一阶段：数据工程 (Data Engineering)
# ==============================================================================
class DataEngineer:
    def __init__(self, config):
        self.cfg = config

    def _load_and_clean_single_file(self, filepath):
        """加载单文件并进行 3-Sigma 清洗"""
        try:
            # 前4行无用且只有一列数据
            df = pd.read_csv(filepath, header=None, skiprows=4, encoding='gbk')
            if len(df) == 0: return None
            
            raw_data = df.iloc[:, 0].values.astype(float)
            
            # 3-Sigma 过滤：剔除电路尖峰
            mean = np.mean(raw_data)
            std = np.std(raw_data)
            clean_data = raw_data[(raw_data >= mean - self.cfg.SIGMA_THRESHOLD * std) & 
                                  (raw_data <= mean + self.cfg.SIGMA_THRESHOLD * std)]
            return clean_data
        except Exception as e:
            print(f"[Warning] 读取失败 {filepath}: {e}")
            return None

    def _mini_batch_resample(self, data_v1, data_v2, temp_k):
        """微批次重采样增强：生成 '高斯云团'"""
        # 同步截断，确保长度一致
        min_len = min(len(data_v1), len(data_v2))
        v1 = data_v1[:min_len]
        v2 = data_v2[:min_len]
        
        # 确保数据量足够切分
        if min_len < self.cfg.RESAMPLE_N:
            return []

        # 随机打乱 (保留同分布特性，模拟随机采样)
        indices = np.arange(min_len)
        np.random.shuffle(indices)
        v1_shuffled = v1[indices]
        v2_shuffled = v2[indices]
        
        # 切分批次
        batch_size = min_len // self.cfg.RESAMPLE_N
        samples = []
        
        for i in range(self.cfg.RESAMPLE_N):
            start = i * batch_size
            end = (i + 1) * batch_size
            
            # 计算微批次均值
            v1_mean = np.mean(v1_shuffled[start:end])
            v2_mean = np.mean(v2_shuffled[start:end])
            
            samples.append({
                'T_true': temp_k,
                'V1': v1_mean,
                'V2': v2_mean
            })
            
        return samples

    def run_pipeline(self):
        print("\n>>> [Stage 1] 数据工程：清洗与微批次增强...")
        all_data = []
        
        temps = np.arange(self.cfg.TEMP_START, self.cfg.TEMP_END + self.cfg.TEMP_STEP, self.cfg.TEMP_STEP)
        count_files = 0
        
        for temp_c in temps:
            # 构建文件名: 400_1550.csv
            f1 = os.path.join(self.cfg.RAW_DATA_DIR, f"{temp_c}_1550.csv")
            f2 = os.path.join(self.cfg.RAW_DATA_DIR, f"{temp_c}_1650.csv")
            
            if not os.path.exists(f1) or not os.path.exists(f2):
                continue
                
            d1 = self._load_and_clean_single_file(f1)
            d2 = self._load_and_clean_single_file(f2)
            
            if d1 is None or d2 is None: continue
            
            # 转换温度单位：摄氏度 -> 开尔文
            temp_k = temp_c + 273.15
            
            # 微批次增强
            samples = self._mini_batch_resample(d1, d2, temp_k)
            all_data.extend(samples)
            count_files += 1
            
        if count_files == 0:
            print("错误：未在 raw_data 目录中找到匹配的CSV文件")
            return None, None

        df = pd.DataFrame(all_data)
        print(f"处理了 {count_files} 个温度点。")
        print(f"总生成样本量: {len(df)} (预期: {count_files * self.cfg.RESAMPLE_N})")
        
        # 严格的数据集划分
        print(">>> 划分训练集/测试集 (80/20)...")
        train_df, test_df = train_test_split(
            df, 
            test_size=self.cfg.TEST_SIZE, 
            random_state=self.cfg.RANDOM_SEED,
            shuffle=True
        )
        
        # 保存数据集，确保后续分析的一致性
        train_df.to_csv(f"{self.cfg.OUTPUT_DIR}/train_set.csv", index=False)
        test_df.to_csv(f"{self.cfg.OUTPUT_DIR}/test_set.csv", index=False)
        print(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")
        
        return train_df, test_df

# ==============================================================================
# 第二阶段：特征工程 (Feature Engineering)
# ==============================================================================
class FeatureEngineer:
    def __init__(self):
        self.base_cols = ['V1', 'V2', 'Ratio']
        self.candidate_cols = ['Log_Ratio', 'Energy_Sum', 'Norm_Diff', 'Interaction']
        
    def generate_features(self, df):
        """生成全量特征矩阵"""
        df = df.copy()
        # 2.1 基础物理特征
        df['Ratio'] = df['V1'] / df['V2']# 基础比值特征
        
        # 2.2 候选增强特征
        df['Log_Ratio'] = np.log(df['Ratio'])                 # 线性化维恩近似
        df['Energy_Sum'] = df['V1'] + df['V2']                # 总辐射能量
        df['Norm_Diff'] = (df['V1'] - df['V2']) / (df['V1'] + df['V2']) # 归一化差异
        df['Interaction'] = df['V1'] * df['V2']               # 二阶交互
        
        return df
    
    def get_feature_combinations(self):
        """生成 16 种特征组合策略 (Combinatorial Strategy)"""
        combinations = {}
        
        # 基础：{Base}
        combinations['Base'] = self.base_cols
        
        # 扩展：Base + Subset of Candidates
        # 遍历 1到4个候选特征的所有组合
        for r in range(1, len(self.candidate_cols) + 1):
            for subset in itertools.combinations(self.candidate_cols, r):
                if len(subset) == 4:
                    name = "All_Features"
                else:
                    # 简短命名，例如 Base+1_Log
                    name = f"Base+{len(subset)}_{subset[0].split('_')[0]}"
                    if len(subset) > 1: name += "..." 
                
                combinations[name] = self.base_cols + list(subset)
                
        return combinations

# ==============================================================================
# 第三阶段：基准模型构建 (The Baseline)
# ==============================================================================
class BaselineManager:
    def __init__(self):
        self.models = {}
        
    def train_baselines(self, X_train, y_train):
        print("\n>>> [Stage 3] 训练基准模型 (3次多项式)...")
        
        # 定义基准任务：单路拟合和比色法拟合
        tasks = {
            'A1_Poly1550': ['V1'],
            'A2_Poly1650': ['V2'],
            'A3_PolyRatio': ['Ratio']
        }
        
        for name, features in tasks.items():
            # 标准化 -> 多项式特征 -> 线性回归
            # 标准化对于多项式回归很重要，防止数值爆炸
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(degree=3, include_bias=False)),
                ('linear', LinearRegression())
            ])
            model.fit(X_train[features], y_train)
            self.models[name] = model
            
        return self.models
    
    def predict(self, X_test):
        """返回所有基准模型的预测结果"""
        preds = {}
        for name, model in self.models.items():
            if '1550' in name: feat = ['V1']
            elif '1650' in name: feat = ['V2']
            else: feat = ['Ratio']
            
            preds[name] = model.predict(X_test[feat])
        return pd.DataFrame(preds)

# ==============================================================================
# 第四阶段：XGBoost 极限寻优 (The Challenger)
# ==============================================================================
class XGBoostOptimizer:
    def __init__(self, config):
        self.cfg = config
        self.best_model_params = None
        self.best_features = None
        self.best_combo_name = None
        self.optimization_history = [] # 用于绘制 Figure 3
        
    def optimize(self, X_train, y_train, feature_combinations):
        print("\n>>> [Stage 4] XGBoost 极限寻优 (遍历特征组合 + 贝叶斯优化)")
        
        global_best_score = float('inf')
        
        # 4.1 外层循环：遍历特征组合
        total_combos = len(feature_combinations)
        for idx, (combo_name, feature_list) in enumerate(feature_combinations.items(), 1):
            print(f"  [{idx}/{total_combos}] Optimizing: {combo_name} ({len(feature_list)} features)")
            
            X_subset = X_train[feature_list]
            
            # 4.2 内层循环：Optuna
            study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                    'max_depth': trial.suggest_int('max_depth', 3, 9),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                    # 固定参数
                    'tree_method': 'hist',
                    'device': self.cfg.DEVICE,
                    'n_jobs': -1,
                    'verbosity': 0
                }
                
                # 交叉验证
                kf = KFold(n_splits=self.cfg.N_FOLDS, shuffle=True, random_state=42)
                scores = []
                
                # 手动CV循环以确保控制
                for t_idx, v_idx in kf.split(X_subset, y_train):
                    Xt, Xv = X_subset.iloc[t_idx], X_subset.iloc[v_idx]
                    yt, yv = y_train.iloc[t_idx], y_train.iloc[v_idx]
                    
                    model = xgb.XGBRegressor(**params)
                    model.fit(Xt, yt)
                    preds = model.predict(Xv)
                    rmse = np.sqrt(mean_squared_error(yv, preds))
                    scores.append(rmse)
                
                return np.mean(scores)

            study.optimize(objective, n_trials=self.cfg.OPTUNA_TRIALS, show_progress_bar=False)
            
            current_best_rmse = study.best_value
            
            # 记录历史 
            self.optimization_history.append({
                'Combination': combo_name,
                'Feature_Count': len(feature_list),
                'Best_CV_RMSE': current_best_rmse,
                'Best_Params': study.best_params
            })
            
            # 更新全局最优
            if current_best_rmse < global_best_score:
                global_best_score = current_best_rmse
                self.best_model_params = study.best_params
                self.best_features = feature_list
                self.best_combo_name = combo_name
                
        print(f"\n  >>> 最优模型: {self.best_combo_name}")
        print(f"  >>> 最佳 CV RMSE: {global_best_score:.4f}")
        
        # 补充必要的非搜索参数
        self.best_model_params['tree_method'] = 'hist'
        self.best_model_params['device'] = self.cfg.DEVICE
        self.best_model_params['n_jobs'] = -1
        
        return self.best_model_params, self.best_features

    def train_ensemble_and_predict(self, X_train, y_train, X_test):
        """4.3 决策与集成：训练10个模型进行不确定性预测"""
        print(f"  -> 训练集成模型 (N={self.cfg.ENSEMBLE_MODELS}, 特征={self.best_combo_name})...")
        
        ensemble_preds = []
        models = []
        
        X_train_sub = X_train[self.best_features]
        X_test_sub = X_test[self.best_features]
        
        for i in range(self.cfg.ENSEMBLE_MODELS):
            # 改变随机种子
            params = self.best_model_params.copy()
            params['random_state'] = 42 + i
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train_sub, y_train)
            
            preds = model.predict(X_test_sub)
            ensemble_preds.append(preds)
            models.append(model)
            
        ensemble_preds = np.array(ensemble_preds)
        
        # 计算统计量
        mean_pred = np.mean(ensemble_preds, axis=0)
        std_pred = np.std(ensemble_preds, axis=0)
        
        # 返回主模型(第一个)用于SHAP，以及预测结果
        return models[0], mean_pred, std_pred

# ==============================================================================
# 第五阶段：数据效率探究 (Data Efficiency)
# ==============================================================================
def run_data_efficiency_study(df_train_full, df_test_full, best_feats, best_params):
    print("\n>>> [Stage 5] 数据效率探究 (稀疏化采样对抗)")
    steps = [5, 10, 25, 50, 100]
    results = []
    
    # 测试集始终保持全量
    X_test = df_test_full
    y_test = df_test_full['T_true']
    
    for step in steps:
        # 稀疏化采样：按温度间隔降采样
        # 逻辑：取余数。将温度乘以100转为整数避免浮点误差
        train_subset = df_train_full[
            (np.round(df_train_full['T_true'] * 100).astype(int) % int(step * 100)) == 0
        ]
        
        X_train_sub = train_subset
        y_train_sub = train_subset['T_true']
        
        # 1. 重新训练 A3 (Ratio 基准)
        # 注意：Pipeline 内部会自动处理多项式特征
        model_a3 = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('linear', LinearRegression())
        ])
        model_a3.fit(X_train_sub[['Ratio']], y_train_sub)
        preds_a3 = model_a3.predict(X_test[['Ratio']])
        
        # 计算 RMSPE (Root Mean Squared Percentage Error)
        rmspe_a3 = np.sqrt(np.mean(((y_test - preds_a3) / y_test) ** 2)) * 100
        
        # 2. 重新训练 XGB-Optimal
        # 注意：需要确保参数中包含 device 设置
        model_xgb = xgb.XGBRegressor(**best_params)
        model_xgb.fit(X_train_sub[best_feats], y_train_sub)
        preds_xgb = model_xgb.predict(X_test[best_feats])
        
        rmspe_xgb = np.sqrt(np.mean(((y_test - preds_xgb) / y_test) ** 2)) * 100
        
        results.append({
            'Step_Size': step,
            'Samples': len(train_subset),
            'A3_RMSPE': rmspe_a3,
            'XGB_RMSPE': rmspe_xgb
        })
        print(f"  [Step {step}°C] 样本数: {len(train_subset)} | A3 Err: {rmspe_a3:.3f}% | XGB Err: {rmspe_xgb:.3f}%")
        
    return pd.DataFrame(results)

# ==============================================================================
# 主程序 (Main Execution)
# ==============================================================================
def main():
    cfg = Config()
    
    # --- 1. 数据工程 ---
    de = DataEngineer(cfg)
    df_train, df_test = de.run_pipeline()
    
    if df_train is None: return # 错误处理

    # --- 2. 特征工程 ---
    fe = FeatureEngineer()
    print("\n>>> [Stage 2] 特征工程...")
    df_train = fe.generate_features(df_train)
    df_test = fe.generate_features(df_test)
    
    # 提取标签
    y_train = df_train['T_true']
    y_test = df_test['T_true']
    
    # 获取所有组合
    feature_combos = fe.get_feature_combinations()
    print(f"生成了 {len(feature_combos)} 种特征组合策略。")
    
    # --- 3. 基准模型训练 ---
    bm = BaselineManager()
    bm.train_baselines(df_train, y_train)
    preds_baseline_df = bm.predict(df_test)
    
    # --- 4. XGBoost 极限寻优 ---
    xgbo = XGBoostOptimizer(cfg)
    best_params, best_features = xgbo.optimize(df_train, y_train, feature_combos)
    
    # 使用最优参数进行集成预测
    main_xgb_model, xgb_mean, xgb_std = xgbo.train_ensemble_and_predict(df_train, y_train, df_test)
    
    # 计算置信区间 (95%)
    xgb_upper = xgb_mean + 2 * xgb_std
    xgb_lower = xgb_mean - 2 * xgb_std
    
    # --- 5. 数据效率探究 ---
    efficiency_df = run_data_efficiency_study(df_train, df_test, best_features, best_params)
    
    # --- 6. 终极可视化与输出 ---
    print("\n>>> [Stage 6] 生成终极可视化报告...")
    
    # 准备数据
    final_df = df_test.copy()
    final_df['Pred_A3'] = preds_baseline_df['A3_PolyRatio']
    final_df['Pred_XGB'] = xgb_mean
    final_df['XGB_Uncertainty'] = xgb_std
    # 相对误差
    final_df['Err_A3'] = np.abs((y_test - final_df['Pred_A3']) / y_test) * 100
    final_df['Err_XGB'] = np.abs((y_test - final_df['Pred_XGB']) / y_test) * 100
    
    # 6.1 Figure 1: 拟合性能全景图
    plt.figure(figsize=(12, 7))
    # 绘制对角线
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1.5, label='Ideal')
    # 绘制 Poly-Ratio
    plt.scatter(y_test, final_df['Pred_A3'], c='blue', alpha=0.3, s=15, label='Poly-Ratio (Baseline)')
    # 绘制 XGB-Optimal
    plt.plot(y_test.sort_values(), final_df.sort_values('T_true')['Pred_XGB'], c='red', lw=2, label='XGB-Optimal')
    # 绘制置信区间 (需要排序)
    sorted_idx = y_test.argsort()
    plt.fill_between(y_test.iloc[sorted_idx], 
                     xgb_lower[sorted_idx], 
                     xgb_upper[sorted_idx], 
                     color='red', alpha=0.2, label='95% Confidence Interval')
    
    plt.xlabel('真实温度 True Temperature (K)', fontsize=12)
    plt.ylabel('预测温度 Predicted Temperature (K)', fontsize=12)
    plt.title('Figure 1: 拟合性能全景图 (Fitting Performance Panorama)', fontsize=14)
    plt.legend()
    plt.savefig(f"{cfg.OUTPUT_DIR}/Figure1_Performance.png", dpi=300)
    
    # 6.2 Figure 2: 精度/误差分布图 (关注低温)
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, final_df['Err_A3'], c='blue', alpha=0.4, label='Poly-Ratio', s=20)
    plt.scatter(y_test, final_df['Err_XGB'], c='red', alpha=0.6, label='XGB-Optimal', s=20)
    plt.axhline(1.0, color='green', linestyle='--', lw=2, label='1% 误差阈值')
    
    plt.xlabel('真实温度 (K)', fontsize=12)
    plt.ylabel('相对误差 Relative Error (%)', fontsize=12)
    plt.ylim(0, 5) # 限制Y轴以展示细节
    plt.title('Figure 2: 相对误差分布 (Focus on Low Temperature)', fontsize=14)
    plt.legend()
    plt.savefig(f"{cfg.OUTPUT_DIR}/Figure2_Error_Distribution.png", dpi=300)
    
    # 6.3 Figure 3: 特征消融路径图
    # 整理数据
    hist_df = pd.DataFrame(xgbo.optimization_history)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=hist_df, x='Feature_Count', y='Best_CV_RMSE', marker='o', linestyle='-', linewidth=2, color='purple')
    plt.xlabel('特征数量 Number of Features', fontsize=12)
    plt.ylabel('Best CV RMSE (K)', fontsize=12)
    plt.title('Figure 3: 特征消融路径 (Feature Ablation Path)', fontsize=14)
    plt.xticks(range(min(hist_df['Feature_Count']), max(hist_df['Feature_Count'])+1))
    plt.grid(True, linestyle='--')
    plt.savefig(f"{cfg.OUTPUT_DIR}/Figure3_Feature_Ablation.png", dpi=300)
    
    # 6.4 Figure 4: 数据效率对比图
    plt.figure(figsize=(10, 6))
    plt.plot(efficiency_df['Step_Size'], efficiency_df['A3_RMSPE'], 'b--o', label='Poly-Ratio', lw=2)
    plt.plot(efficiency_df['Step_Size'], efficiency_df['XGB_RMSPE'], 'r-s', label='XGB-Optimal', lw=2)
    
    plt.xlabel('采样步长 Sampling Step (°C)', fontsize=12)
    plt.ylabel('测试集 RMSPE (%)', fontsize=12)
    plt.title('Figure 4: 数据效率对比 (Data Efficiency)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig(f"{cfg.OUTPUT_DIR}/Figure4_Data_Efficiency.png", dpi=300)
    
    # 6.5 Figure 5: SHAP 可解释性图
    plt.figure()
    # 为提高速度，随机抽取部分测试集数据进行解释
    X_shap_sample = df_test[best_features].sample(n=min(2000, len(df_test)), random_state=42)
    explainer = shap.TreeExplainer(main_xgb_model)
    shap_values = explainer(X_shap_sample)
    
    shap.summary_plot(shap_values, X_shap_sample, show=False, plot_size=(10, 6))
    plt.title('Figure 5: SHAP Feature Importance', fontsize=14)
    plt.savefig(f"{cfg.OUTPUT_DIR}/Figure5_SHAP_Summary.png", bbox_inches='tight', dpi=300)
    
    # 6.6 数据输出
    # 计算最终指标
    rmse_final = np.sqrt(mean_squared_error(y_test, xgb_mean))
    r2_final = r2_score(y_test, xgb_mean)
    print(f"\n>>> 最终测试集指标: RMSE = {rmse_final:.4f} K, R2 = {r2_final:.5f}")
    
    final_df.to_csv(f"{cfg.OUTPUT_DIR}/final_report_predictions.csv", index=False)
    hist_df.to_csv(f"{cfg.OUTPUT_DIR}/optimization_history.csv", index=False)
    efficiency_df.to_csv(f"{cfg.OUTPUT_DIR}/data_efficiency_report.csv", index=False)
    
    print(f"\n所有结果已保存至: {cfg.OUTPUT_DIR}/")

if __name__ == "__main__":
    main()