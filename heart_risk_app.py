# -*- coding: utf-8 -*-
"""
====================================================================
心脏病 (CHD/MI) 风险评估系统
====================================================================
基于 2024 BRFSS 数据训练的 LightGBM 模型 (Test AUC = 0.8376)
最优阈值 = 0.486 (Accuracy ≥ 0.72 约束下最大化 Recall)

风险分层 (来自高/低风险画像分析):
  - 低风险: P <  0.0712     (测试集 <20% 分位, 实际患病率 0.35%)
  - 中风险: 0.0712 ≤ P < 0.486
  - 高风险: P ≥ 0.486       (将被模型判为阳性)

运行方式:
  streamlit run heart_risk_app.py

所需文件 (应放在同目录):
  - model_LightGBM.pkl      (训练好的 LightGBM 分类器)
  - scaler.pkl              (MinMaxScaler, fit 于 106 个 one-hot 列)
  - train_enc_columns.pkl   (训练集 one-hot 后的完整列名列表)
====================================================================
"""
import os
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 新增：导入字体管理器

# ==========================================
# 动态字体加载逻辑 (适配云端 Linux 环境)
# ==========================================
font_path = "msyh.ttc"  # 指定你上传的字体文件名

if os.path.exists(font_path):
    # 如果找到了字体文件，强制注册到系统并设置为默认
    fm.fontManager.addfont(font_path)
    prop = fm.FontProperties(fname=font_path)
    matplotlib.rcParams["font.sans-serif"] = [prop.get_name()]
else:
    # 备用方案：如果文件不存在（例如在你自己的本地电脑上运行），则使用系统默认字体
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans", "Arial Unicode MS"]

matplotlib.rcParams["axes.unicode_minus"] = False

# =========================================================
# 1. 全局常量
# =========================================================

# —— 48 个最终建模特征 (与模型训练阶段严格一致) ——
FINAL_FEATURES = [
    'SEXVAR', '_BMI5CAT', '_INCOMG1', '_SMOKER3', '_CURECI3',
    'DIABETE4_3.0', 'DIABETE4_1.0', '_IMPRACE_5.0', '_IMPRACE_2.0', '_IMPRACE_1.0',
    'EMPLOY1_8.0', 'EMPLOY1_7.0', 'EMPLOY1_6.0', 'EMPLOY1_4.0', 'EMPLOY1_2.0', 'EMPLOY1_1.0',
    'MARITAL_6.0', 'MEDCOST1', 'MARITAL_5.0', 'MARITAL_3.0', '_STATE_53.0',
    '_EDUCAG', '_RFBING6', '_AGE_G', 'CVDSTRK3', 'VETERAN3', 'DEAF', '_DRDXAR2',
    'BLIND', 'DECIDE', 'DIFFWALK', 'CHCCOPD3', 'CHCOCNC1', 'PNEUVAC4', 'ADDEPEV3',
    '_RFHLTH', '_PHYS14D', 'EXERANY2', '_MENT14D', '_HLTHPL2', '_LTASTH1',
    'CHCKDNY2', 'MARITAL_2.0', 'MARITAL_1.0', 'EMPLOY1_5.0', 'DIFFDRES', 'DIFFALON', '_IMPRACE_6.0'
]

# —— 特征中文显示名 ——
FEATURE_DISPLAY_NAMES = {
    'SEXVAR': '性别', '_BMI5CAT': 'BMI 分类', '_INCOMG1': '收入等级',
    '_SMOKER3': '吸烟状态', '_CURECI3': '电子烟使用',
    'DIABETE4_3.0': '无糖尿病', 'DIABETE4_1.0': '确诊糖尿病',
    '_IMPRACE_5.0': '西班牙裔', '_IMPRACE_2.0': '非西裔黑人',
    '_IMPRACE_1.0': '非西裔白人', '_IMPRACE_6.0': '其他种族',
    'EMPLOY1_8.0': '无法工作', 'EMPLOY1_7.0': '退休', 'EMPLOY1_6.0': '学生',
    'EMPLOY1_4.0': '失业 <1 年', 'EMPLOY1_2.0': '自雇', 'EMPLOY1_1.0': '受雇就业',
    'EMPLOY1_5.0': '全职家务',
    'MARITAL_6.0': '未婚同居', 'MARITAL_5.0': '从未结婚', 'MARITAL_3.0': '丧偶',
    'MARITAL_2.0': '离婚', 'MARITAL_1.0': '已婚',
    'MEDCOST1': '因费用未就医', '_STATE_53.0': '华盛顿州',
    '_EDUCAG': '教育程度', '_RFBING6': '暴饮行为', '_AGE_G': '年龄分组',
    'CVDSTRK3': '中风史', 'VETERAN3': '退伍军人', 'DEAF': '听力障碍',
    '_DRDXAR2': '关节炎', 'BLIND': '视力障碍', 'DECIDE': '认知困难',
    'DIFFWALK': '行走困难', 'CHCCOPD3': 'COPD / 慢性肺疾病',
    'CHCOCNC1': '癌症史', 'PNEUVAC4': '肺炎疫苗', 'ADDEPEV3': '抑郁症',
    '_RFHLTH': '自评健康良好', '_PHYS14D': '身体不适天数',
    'EXERANY2': '体育锻炼', '_MENT14D': '心理不适天数',
    '_HLTHPL2': '医疗保险', '_LTASTH1': '哮喘史', 'CHCKDNY2': '肾病',
    'DIFFDRES': '穿衣困难', 'DIFFALON': '独立外出困难',
}

# —— 决策阈值 (来自模型训练与画像分析) ——
THRESHOLD_HIGH = 0.486     # 模型最优分类阈值: Accuracy≥72% 下最大化 Recall
THRESHOLD_LOW  = 0.0712    # 低风险组门槛: 测试集下 20% 分位数

# —— 原始特征列 (34 列, 与训练阶段 X_train 完全一致) ——
RAW_COLUMNS = [
    '_STATE', 'SEXVAR', 'MEDCOST1', 'EXERANY2', 'CVDSTRK3',
    'CHCOCNC1', 'CHCCOPD3', 'ADDEPEV3', 'CHCKDNY2', 'DIABETE4',
    'MARITAL', 'VETERAN3', 'EMPLOY1', 'DEAF', 'BLIND',
    'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON', 'PNEUVAC4',
    '_IMPRACE', '_RFHLTH', '_PHYS14D', '_MENT14D', '_HLTHPL2',
    '_LTASTH1', '_DRDXAR2', '_AGE_G', '_BMI5CAT',
    '_EDUCAG', '_INCOMG1', '_SMOKER3', '_CURECI3', '_RFBING6'
]

# One-hot 列 (与训练一致)
NOMINAL_COLS = ['_STATE', 'MARITAL', 'EMPLOY1', '_IMPRACE', 'DIABETE4']


# =========================================================
# 2. 预处理管道 (与训练阶段完全一致)
# =========================================================
def apply_feature_engineering_mappings(df_mapped: pd.DataFrame) -> pd.DataFrame:
    """
    复刻训练阶段 apply_feature_engineering_mappings 函数
    将 BRFSS 原始编码映射到最终建模编码
    """
    # 1) 常规二元映射: 1=Yes, 2=No → 1=Yes, 0=No
    binary_1_yes_2_no = [
        'SEXVAR', 'MEDCOST1', 'EXERANY2', 'CVDSTRK3', 'CHCOCNC1',
        'CHCCOPD3', 'ADDEPEV3', 'CHCKDNY2', 'VETERAN3', 'DEAF',
        'BLIND', 'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON',
        'PNEUVAC4', '_HLTHPL2', '_DRDXAR2'
    ]
    for col in binary_1_yes_2_no:
        if col in df_mapped.columns:
            df_mapped[col] = df_mapped[col].map({1: 1, 2: 0})

    # 2) 特殊二元反转
    special_flips = {
        '_LTASTH1': {1: 0, 2: 1},
        '_CURECI3': {1: 0, 2: 1},
        '_RFBING6': {1: 0, 2: 1},
        '_RFHLTH':  {1: 1, 2: 0}
    }
    for col, map_dict in special_flips.items():
        if col in df_mapped.columns:
            df_mapped[col] = df_mapped[col].map(map_dict)

    # 3) 三级有序: 1→0, 2→1, 3→2
    for col in ['_PHYS14D', '_MENT14D']:
        if col in df_mapped.columns:
            df_mapped[col] = df_mapped[col].map({1: 0, 2: 1, 3: 2})

    # 4) 数值 -1 规范化
    for col in ['_AGE_G', '_BMI5CAT', '_EDUCAG', '_INCOMG1']:
        if col in df_mapped.columns:
            df_mapped[col] = df_mapped[col] - 1

    # 5) 吸烟状态反转 (4→0, 3→1, 2→2, 1→3)
    if '_SMOKER3' in df_mapped.columns:
        df_mapped['_SMOKER3'] = df_mapped['_SMOKER3'].map({4: 0, 3: 1, 2: 2, 1: 3})

    return df_mapped


def preprocess_user_input(user_raw: dict, scaler, train_enc_columns) -> pd.DataFrame:
    """
    对单条用户输入执行与训练期一致的完整预处理管道:
      (1) 构建原始 BRFSS 编码行
      (2) 应用特征工程映射
      (3) One-Hot 编码
      (4) 对齐训练期的 106 列全量字段 (缺失列填 0)
      (5) 转 float
      (6) 使用训练期拟合好的 MinMaxScaler 做归一化
      (7) 选取 48 个最终特征
    返回: pd.DataFrame, shape=(1, 48)
    """
    # (1) 原始行
    row = {col: user_raw.get(col, np.nan) for col in RAW_COLUMNS}
    df = pd.DataFrame([row], columns=RAW_COLUMNS)

    # (2) 映射
    df = apply_feature_engineering_mappings(df)

    # (2.5) 关键:训练阶段 nominal 列因经过 IterativeImputer 输出为 float64,
    #       get_dummies 生成 'MARITAL_1.0' 风格列名。此处必须保持一致,
    #       否则列对齐失败会导致所有 one-hot 列被填 0。
    for col in NOMINAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # (3) One-Hot
    df_enc = pd.get_dummies(df, columns=NOMINAL_COLS, drop_first=False)

    # (4) 对齐训练列
    df_enc = df_enc.reindex(columns=train_enc_columns, fill_value=0)

    # (5) 转 float
    df_enc = df_enc.astype(float)

    # (6) 归一化
    df_scaled = pd.DataFrame(
        scaler.transform(df_enc),
        columns=train_enc_columns
    )

    # (7) 选取 48 特征
    df_final = df_scaled[FINAL_FEATURES]
    return df_final


# =========================================================
# 3. 加载模型与预处理产物 (缓存复用)
# =========================================================
@st.cache_resource(show_spinner="正在加载模型 ...")
def load_artifacts():
    """加载模型、scaler、训练列名、SHAP 解释器"""
    required_files = {
        "model_LightGBM.pkl":    "LightGBM 分类模型",
        "scaler.pkl":            "MinMaxScaler",
        "train_enc_columns.pkl": "训练集 one-hot 列名",
    }
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        st.error(
            "❌ 缺少模型产物文件,请将以下文件放入当前目录:\n\n"
            + "\n".join(f"  - `{f}` ({required_files[f]})" for f in missing)
        )
        st.stop()

    model              = joblib.load("model_LightGBM.pkl")
    scaler             = joblib.load("scaler.pkl")
    train_enc_columns  = joblib.load("train_enc_columns.pkl")
    explainer          = shap.TreeExplainer(model)

    return model, scaler, train_enc_columns, explainer


# =========================================================
# 4. 风险分级与 SHAP 归因解释
# =========================================================
def classify_risk(prob: float):
    if prob < THRESHOLD_LOW:
        return "低风险", "#2ecc71", "🟢"
    elif prob < THRESHOLD_HIGH:
        return "中风险", "#f39c12", "🟡"
    else:
        return "高风险", "#e74c3c", "🔴"


def decode_feature_value(feat_name: str, user_raw: dict) -> str:
    """
    将用户最终特征的实际语义值翻译为可读文字,用于 SHAP 解释展示。
    例如: SEXVAR=1 → '男性'; _AGE_G=4 → '55–64 岁'
    """
    # 先处理 one-hot 列: 查询其对应原始列的原始值
    onehot_map = {
        'DIABETE4_1.0': ('DIABETE4', 1, '是'),
        'DIABETE4_3.0': ('DIABETE4', 3, '是'),
        '_IMPRACE_1.0': ('_IMPRACE', 1, '是'),
        '_IMPRACE_2.0': ('_IMPRACE', 2, '是'),
        '_IMPRACE_5.0': ('_IMPRACE', 5, '是'),
        '_IMPRACE_6.0': ('_IMPRACE', 6, '是'),
        'EMPLOY1_1.0': ('EMPLOY1', 1, '是'),
        'EMPLOY1_2.0': ('EMPLOY1', 2, '是'),
        'EMPLOY1_4.0': ('EMPLOY1', 4, '是'),
        'EMPLOY1_5.0': ('EMPLOY1', 5, '是'),
        'EMPLOY1_6.0': ('EMPLOY1', 6, '是'),
        'EMPLOY1_7.0': ('EMPLOY1', 7, '是'),
        'EMPLOY1_8.0': ('EMPLOY1', 8, '是'),
        'MARITAL_1.0': ('MARITAL', 1, '是'),
        'MARITAL_2.0': ('MARITAL', 2, '是'),
        'MARITAL_3.0': ('MARITAL', 3, '是'),
        'MARITAL_5.0': ('MARITAL', 5, '是'),
        'MARITAL_6.0': ('MARITAL', 6, '是'),
        '_STATE_53.0': ('_STATE', 53, '是'),
    }
    if feat_name in onehot_map:
        orig, code, yes_label = onehot_map[feat_name]
        return yes_label if user_raw.get(orig) == code else '否'

    v = user_raw.get(feat_name)
    if v is None or pd.isna(v):
        return "未填"

    # 二元 Yes/No (1=Yes,2=No 的原始编码)
    binary_cols = {'SEXVAR', 'MEDCOST1', 'EXERANY2', 'CVDSTRK3', 'CHCOCNC1',
                   'CHCCOPD3', 'ADDEPEV3', 'CHCKDNY2', 'VETERAN3', 'DEAF',
                   'BLIND', 'DECIDE', 'DIFFWALK', 'DIFFDRES', 'DIFFALON',
                   'PNEUVAC4', '_HLTHPL2', '_DRDXAR2'}
    if feat_name in binary_cols:
        if feat_name == 'SEXVAR':
            return '男性' if v == 1 else '女性'
        return '是' if v == 1 else '否'

    # 特殊反转二元
    flipped = {'_LTASTH1', '_CURECI3', '_RFBING6'}  # 1=No,2=Yes
    if feat_name in flipped:
        return '是' if v == 2 else '否'
    if feat_name == '_RFHLTH':
        return '良好/很好' if v == 1 else '一般/较差'

    # 有序
    if feat_name == '_AGE_G':
        return {1:'18–24 岁',2:'25–34 岁',3:'35–44 岁',4:'45–54 岁',
                5:'55–64 岁',6:'65 岁及以上'}.get(v, str(v))
    if feat_name == '_BMI5CAT':
        return {1:'偏瘦',2:'正常',3:'超重',4:'肥胖'}.get(v, str(v))
    if feat_name == '_EDUCAG':
        return {1:'未完成高中',2:'高中毕业',3:'就读大学/技校',4:'大学毕业'}.get(v, str(v))
    if feat_name == '_INCOMG1':
        return {1:'<$15K',2:'$15K–25K',3:'$25K–35K',4:'$35K–50K',
                5:'$50K–100K',6:'$100K–200K',7:'≥$200K'}.get(v, str(v))
    if feat_name == '_SMOKER3':
        return {1:'每天吸烟',2:'偶尔吸烟',3:'曾经吸烟',4:'从不吸烟'}.get(v, str(v))
    if feat_name in ('_PHYS14D', '_MENT14D'):
        return {1:'0 天',2:'1–13 天',3:'≥14 天'}.get(v, str(v))

    return str(v)


def extract_top_contributors(shap_values_row, X_row, user_raw, top_k=7):
    """
    从 SHAP 值中抽取对当前预测贡献最大的若干特征,分为风险/保护两类
    返回: DataFrame(columns=['特征','用户取值','SHAP','方向'])
    """
    abs_shap = np.abs(shap_values_row)
    order = np.argsort(abs_shap)[::-1]

    records = []
    for idx in order[:top_k]:
        feat = FINAL_FEATURES[idx]
        sv = float(shap_values_row[idx])
        records.append({
            "特征":    FEATURE_DISPLAY_NAMES.get(feat, feat),
            "用户取值": decode_feature_value(feat, user_raw),
            "SHAP":    sv,
            "方向":    "↑ 推高风险" if sv > 0 else "↓ 降低风险",
        })
    return pd.DataFrame(records)


def render_shap_bar(top_df: pd.DataFrame):
    """
    专业级 SHAP 个体归因蝴蝶图
    - 标签以"特征名 + 用户取值"两行形式独立布局于左侧,不与柱体重叠
    - 数值标签与柱端颜色一致,对称外置;正值右对齐、负值左对齐
    - 柔和珊瑚红 / 森林绿 配色,去除所有 spine,辅以淡色纵向网格
    """
    df = top_df.iloc[::-1].reset_index(drop=True)
    n = len(df)

    # —— 调色板 ——
    COLOR_RISK  = '#E66B5C'   # 柔珊瑚红 — 推高风险
    COLOR_PROT  = '#4A9D7E'   # 森林青绿 — 降低风险
    COLOR_AXIS  = '#BFBFBF'
    COLOR_GRID  = '#EFEFEF'
    COLOR_TEXT  = '#2C3E50'
    COLOR_MUTED = '#8A96A3'

    fig, ax = plt.subplots(
        figsize=(11.5, 0.78 * n + 1.9),
        dpi=120,
    )
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # —— 对称 X 轴范围(留出右/左侧数值标签空间) ——
    max_abs = max(abs(df['SHAP'].max()), abs(df['SHAP'].min()), 0.05)
    xlim = max_abs * 1.35
    label_offset = max_abs * 0.028

    # —— 条形(细白边营造分离感) ——
    colors = [COLOR_RISK if s > 0 else COLOR_PROT for s in df['SHAP']]
    ax.barh(
        range(n), df['SHAP'].values,
        color=colors,
        edgecolor='white', linewidth=2.2,
        height=0.62, alpha=0.92, zorder=3,
    )

    # —— 淡色纵向参考网格 ——
    for x in np.linspace(-max_abs, max_abs, 5):
        if abs(x) < 1e-4:
            continue
        ax.axvline(x, color=COLOR_GRID, linewidth=0.7, zorder=1)

    # —— 零轴(柔和实线) ——
    ax.axvline(0, color=COLOR_AXIS, linewidth=1.1, zorder=2)

    # —— 柱端数值标签(颜色与柱子一致,加粗) ——
    for i, sv in enumerate(df['SHAP'].values):
        if sv >= 0:
            ax.text(
                sv + label_offset, i, f"+{sv:.3f}",
                va='center', ha='left',
                fontsize=10.5, fontweight='600',
                color=COLOR_RISK,
                zorder=4,
            )
        else:
            ax.text(
                sv - label_offset, i, f"{sv:.3f}",
                va='center', ha='right',
                fontsize=10.5, fontweight='600',
                color=COLOR_PROT,
                zorder=4,
            )

    # —— 左侧独立标签列: 特征名(加粗黑) + 用户取值(灰色小字 + 方向箭头) ——
    ax.set_yticks(range(n))
    ax.set_yticklabels([''] * n)

    label_x = -xlim * 1.04
    for i, r in df.iterrows():
        direction_icon = "▲" if r['SHAP'] > 0 else "▼"
        direction_color = COLOR_RISK if r['SHAP'] > 0 else COLOR_PROT

        ax.annotate(
            r['特征'],
            xy=(label_x, i + 0.14),
            xycoords='data',
            ha='right', va='center',
            fontsize=11, fontweight='bold',
            color=COLOR_TEXT,
            annotation_clip=False,
        )
        ax.annotate(
            f"{r['用户取值']}  ",
            xy=(label_x - label_offset * 0.6, i - 0.22),
            xycoords='data',
            ha='right', va='center',
            fontsize=9.5,
            color=COLOR_MUTED,
            annotation_clip=False,
        )
        # 方向小徽标紧贴用户取值
        ax.annotate(
            direction_icon,
            xy=(label_x + label_offset * 0.4, i - 0.22),
            xycoords='data',
            ha='right', va='center',
            fontsize=8.5, fontweight='bold',
            color=direction_color,
            annotation_clip=False,
        )

    # —— 坐标范围与刻度 ——
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-0.7, n - 0.3)
    ax.tick_params(axis='x', colors=COLOR_MUTED, labelsize=9, length=0, pad=6)
    ax.tick_params(axis='y', length=0, labelleft=False, labelright=False)

    # —— X 轴说明 ——
    ax.set_xlabel(
        '←  降低风险        SHAP 贡献值        推高风险  →',
        fontsize=10.5, color=COLOR_TEXT, labelpad=14,
    )

    # —— 清除所有 spine ——
    for spine in ax.spines.values():
        spine.set_visible(False)

    # —— 顶部标题区 ——
    fig.text(
        0.03, 0.965,
        '对您个人风险贡献最大的关键因素',
        fontsize=14.5, fontweight='bold', color=COLOR_TEXT,
    )
    fig.text(
        0.03, 0.928,
        '基于 SHAP 算法的个体归因分析   ·   '
        f'展示 Top-{n} 影响特征',
        fontsize=9.5, color=COLOR_MUTED,
    )

    # —— 右上角色例 ——
    legend_y = 0.948
    fig.patches.extend([
        plt.Rectangle((0.76, legend_y), 0.018, 0.022,
                      transform=fig.transFigure,
                      color=COLOR_RISK, alpha=0.92, zorder=10),
        plt.Rectangle((0.86, legend_y), 0.018, 0.022,
                      transform=fig.transFigure,
                      color=COLOR_PROT, alpha=0.92, zorder=10),
    ])
    fig.text(0.782, legend_y + 0.01, '推高风险',
             fontsize=9.5, color=COLOR_TEXT, va='center')
    fig.text(0.882, legend_y + 0.01, '降低风险',
             fontsize=9.5, color=COLOR_TEXT, va='center')

    # —— 布局:左侧留出足够标签空间 ——
    plt.subplots_adjust(left=0.26, right=0.95, top=0.87, bottom=0.13)

    return fig


# =========================================================
# 5. UI: 问卷表单
# =========================================================
def build_questionnaire():
    """
    构建问卷界面,返回原始 BRFSS 编码字典。
    问卷分为 5 个 Tab:
      - 基本信息 (Demographics)
      - 生活方式 (Lifestyle)
      - 既往病史 (Medical History)
      - 功能状态 (Functional Status)
      - 医疗保健 (Healthcare)
    """
    tabs = st.tabs([
        "👤 基本信息",
        "🚬 生活方式",
        "🫀 既往病史",
        "🦽 功能状态",
        "🏥 医疗保健",
    ])

    user_raw = {}

    # ---------------- Tab 1: 基本信息 ----------------
    with tabs[0]:
        st.markdown("#### 基本人口学信息")
        c1, c2 = st.columns(2)
        with c1:
            sex = st.radio("性别", ["男", "女"], horizontal=True, key="sex")
            user_raw['SEXVAR'] = 1 if sex == "男" else 2

            age = st.selectbox("年龄分组",
                ["18–24 岁", "25–34 岁", "35–44 岁", "45–54 岁",
                 "55–64 岁", "65 岁及以上"], index=4, key="age")
            user_raw['_AGE_G'] = ["18–24 岁","25–34 岁","35–44 岁","45–54 岁",
                                  "55–64 岁","65 岁及以上"].index(age) + 1

            race = st.selectbox("种族 / 民族",
                ["非西班牙裔白人 (White)",
                 "非西班牙裔黑人 (Black)",
                 "西班牙裔 (Hispanic)",
                 "其他种族 / 多种族"], key="race")
            race_map = {
                "非西班牙裔白人 (White)": 1,
                "非西班牙裔黑人 (Black)": 2,
                "西班牙裔 (Hispanic)":   5,
                "其他种族 / 多种族":      6,
            }
            user_raw['_IMPRACE'] = race_map[race]

            veteran = st.radio("是否退伍军人",
                ["否", "是"], horizontal=True, key="veteran")
            user_raw['VETERAN3'] = 1 if veteran == "是" else 2

        with c2:
            education = st.selectbox("教育程度",
                ["未完成高中",
                 "高中 / 普通教育 (GED) 毕业",
                 "就读大学 / 技术学院",
                 "大学 / 技术学院毕业及以上"], index=3, key="edu")
            edu_map = {
                "未完成高中": 1,
                "高中 / 普通教育 (GED) 毕业": 2,
                "就读大学 / 技术学院": 3,
                "大学 / 技术学院毕业及以上": 4
            }
            user_raw['_EDUCAG'] = edu_map[education]

            income = st.selectbox("家庭年收入",
                ["低于 $15,000",
                 "$15,000 – $25,000",
                 "$25,000 – $35,000",
                 "$35,000 – $50,000",
                 "$50,000 – $100,000",
                 "$100,000 – $200,000",
                 "$200,000 及以上"], index=4, key="income")
            inc_map = {
                "低于 $15,000": 1, "$15,000 – $25,000": 2,
                "$25,000 – $35,000": 3, "$35,000 – $50,000": 4,
                "$50,000 – $100,000": 5, "$100,000 – $200,000": 6,
                "$200,000 及以上": 7
            }
            user_raw['_INCOMG1'] = inc_map[income]

            marital = st.selectbox("婚姻状态",
                ["已婚", "离婚", "丧偶", "分居", "从未结婚", "未婚同居"], key="marital")
            mar_map = {"已婚": 1, "离婚": 2, "丧偶": 3,
                       "分居": 4, "从未结婚": 5, "未婚同居": 6}
            user_raw['MARITAL'] = mar_map[marital]

            employ = st.selectbox("就业状态",
                ["受雇 / 领工资",
                 "自雇 (个体经营)",
                 "失业 ≥ 1 年",
                 "失业 < 1 年",
                 "全职家务 (Homemaker)",
                 "学生",
                 "退休",
                 "因健康原因无法工作"], key="employ")
            emp_map = {
                "受雇 / 领工资": 1, "自雇 (个体经营)": 2,
                "失业 ≥ 1 年": 3, "失业 < 1 年": 4,
                "全职家务 (Homemaker)": 5, "学生": 6,
                "退休": 7, "因健康原因无法工作": 8
            }
            user_raw['EMPLOY1'] = emp_map[employ]

        st.markdown("---")
        wa = st.radio(
            "您是否长期居住在美国华盛顿州 (WA)?",
            ["否", "是"], horizontal=True,
            help="模型对华盛顿州与其他地区有独立识别能力,仅 WA 为区分项",
            key="state_wa")
        user_raw['_STATE'] = 53 if wa == "是" else 99  # 99 = 任意非 WA

    # ---------------- Tab 2: 生活方式 ----------------
    with tabs[1]:
        st.markdown("#### 身高体重 & 生活方式")
        c1, c2 = st.columns(2)
        with c1:
            height_cm = st.number_input("身高 (cm)", min_value=100.0,
                max_value=230.0, value=170.0, step=0.5, key="height")
            weight_kg = st.number_input("体重 (kg)", min_value=30.0,
                max_value=250.0, value=70.0, step=0.5, key="weight")
            bmi_raw = weight_kg / ((height_cm / 100) ** 2)
            if bmi_raw < 18.5:
                bmi_cat, bmi_label = 1, "偏瘦"
            elif bmi_raw < 25.0:
                bmi_cat, bmi_label = 2, "正常"
            elif bmi_raw < 30.0:
                bmi_cat, bmi_label = 3, "超重"
            else:
                bmi_cat, bmi_label = 4, "肥胖"
            user_raw['_BMI5CAT'] = bmi_cat
            st.info(f"🧮 计算所得 BMI = **{bmi_raw:.1f}** → 分类: **{bmi_label}**")

        with c2:
            smoker = st.selectbox("当前吸烟状态",
                ["从不吸烟 (一生 < 100 支)",
                 "曾经吸烟 (已戒)",
                 "偶尔吸烟 (Some days)",
                 "每天吸烟 (Every day)"], key="smoker")
            smoke_map = {"从不吸烟 (一生 < 100 支)": 4,
                         "曾经吸烟 (已戒)": 3,
                         "偶尔吸烟 (Some days)": 2,
                         "每天吸烟 (Every day)": 1}
            user_raw['_SMOKER3'] = smoke_map[smoker]

            ecig = st.radio("近期是否使用电子烟",
                ["否", "是"], horizontal=True, key="ecig")
            user_raw['_CURECI3'] = 2 if ecig == "是" else 1

            binge = st.radio(
                "过去 30 天是否有暴饮行为 (男性 ≥5 杯/女性 ≥4 杯于一次饮酒)",
                ["否", "是"], horizontal=True, key="binge")
            user_raw['_RFBING6'] = 2 if binge == "是" else 1

            exercise = st.radio(
                "过去 30 天是否进行过体育锻炼 (工作之外)",
                ["是", "否"], horizontal=True, key="exercise")
            user_raw['EXERANY2'] = 1 if exercise == "是" else 2

    # ---------------- Tab 3: 既往病史 ----------------
    with tabs[2]:
        st.markdown("#### 是否曾被医护人员告知患有以下疾病?")
        c1, c2 = st.columns(2)
        with c1:
            stroke = st.radio("中风",      ["否", "是"], horizontal=True, key="stroke")
            cancer = st.radio("癌症 (含皮肤癌)", ["否", "是"], horizontal=True, key="cancer")
            copd   = st.radio("COPD / 肺气肿 / 慢性支气管炎",
                              ["否", "是"], horizontal=True, key="copd")
            depr   = st.radio("抑郁症",    ["否", "是"], horizontal=True, key="depr")
            kidney = st.radio("肾脏疾病",  ["否", "是"], horizontal=True, key="kidney")
            user_raw['CVDSTRK3'] = 1 if stroke == "是" else 2
            user_raw['CHCOCNC1'] = 1 if cancer == "是" else 2
            user_raw['CHCCOPD3'] = 1 if copd   == "是" else 2
            user_raw['ADDEPEV3'] = 1 if depr   == "是" else 2
            user_raw['CHCKDNY2'] = 1 if kidney == "是" else 2
        with c2:
            arth   = st.radio("关节炎 (任意类型)",
                              ["否", "是"], horizontal=True, key="arth")
            asthma = st.radio("哮喘 (一生中)",
                              ["否", "是"], horizontal=True, key="asthma")
            diab   = st.selectbox("糖尿病",
                ["无糖尿病", "确诊糖尿病",
                 "仅妊娠期糖尿病", "糖尿病前期 / 边缘性"], key="diab")
            diab_map = {"无糖尿病": 3, "确诊糖尿病": 1,
                        "仅妊娠期糖尿病": 2, "糖尿病前期 / 边缘性": 4}
            user_raw['_DRDXAR2'] = 1 if arth == "是"  else 2
            user_raw['_LTASTH1'] = 2 if asthma == "是" else 1
            user_raw['DIABETE4'] = diab_map[diab]

    # ---------------- Tab 4: 功能状态 ----------------
    with tabs[3]:
        st.markdown("#### 是否存在以下严重的功能障碍?")
        c1, c2 = st.columns(2)
        with c1:
            deaf     = st.radio("严重听力障碍",
                                ["否", "是"], horizontal=True, key="deaf")
            blind    = st.radio("严重视力障碍",
                                ["否", "是"], horizontal=True, key="blind")
            decide   = st.radio("严重认知 / 记忆困难",
                                ["否", "是"], horizontal=True, key="decide")
            user_raw['DEAF']   = 1 if deaf   == "是" else 2
            user_raw['BLIND']  = 1 if blind  == "是" else 2
            user_raw['DECIDE'] = 1 if decide == "是" else 2
        with c2:
            diffwalk = st.radio("行走或爬楼困难",
                                ["否", "是"], horizontal=True, key="diffwalk")
            diffdres = st.radio("穿衣或洗澡困难",
                                ["否", "是"], horizontal=True, key="diffdres")
            diffalon = st.radio("独立外出办事困难",
                                ["否", "是"], horizontal=True, key="diffalon")
            user_raw['DIFFWALK'] = 1 if diffwalk == "是" else 2
            user_raw['DIFFDRES'] = 1 if diffdres == "是" else 2
            user_raw['DIFFALON'] = 1 if diffalon == "是" else 2

    # ---------------- Tab 5: 医疗保健 ----------------
    with tabs[4]:
        st.markdown("#### 医疗保健与整体健康状况")
        c1, c2 = st.columns(2)
        with c1:
            insur  = st.radio("是否拥有任意类型的医疗保险",
                              ["是", "否"], horizontal=True, key="insur")
            cost   = st.radio(
                "过去 12 个月,是否曾因费用原因看不起医生",
                ["否", "是"], horizontal=True, key="cost")
            pneuvac = st.radio("是否接种过肺炎球菌疫苗",
                               ["否", "是"], horizontal=True, key="pneuvac")
            user_raw['_HLTHPL2'] = 1 if insur  == "是" else 2
            user_raw['MEDCOST1'] = 1 if cost   == "是" else 2
            user_raw['PNEUVAC4'] = 1 if pneuvac == "是" else 2
        with c2:
            rfhlth = st.selectbox("总体自评健康",
                ["非常好 (Excellent)", "很好 (Very good)", "好 (Good)",
                 "一般 (Fair)", "较差 (Poor)"], index=2, key="rfhlth")
            user_raw['_RFHLTH'] = 1 if rfhlth in ["非常好 (Excellent)",
                                                   "很好 (Very good)",
                                                   "好 (Good)"] else 2

            phys = st.selectbox(
                "过去 30 天内,身体不适 (包括身体疾病/受伤) 的天数",
                ["0 天", "1–13 天", "14 天及以上"], key="phys")
            user_raw['_PHYS14D'] = {"0 天": 1, "1–13 天": 2, "14 天及以上": 3}[phys]

            ment = st.selectbox(
                "过去 30 天内,精神状态不佳 (含压力/情绪问题) 的天数",
                ["0 天", "1–13 天", "14 天及以上"], key="ment")
            user_raw['_MENT14D'] = {"0 天": 1, "1–13 天": 2, "14 天及以上": 3}[ment]

    return user_raw


# =========================================================
# 6. 侧边栏 (说明 + 免责声明)
# =========================================================
def render_sidebar():
    with st.sidebar:
        st.markdown("## ❤️ 系统简介")
        st.markdown(
            "本系统基于 **2024 BRFSS** 美国居民健康调查数据训练的 "
            "**LightGBM** 机器学习模型,用于估计个体发生 "
            "**冠心病 (CHD) 或心肌梗死 (MI)** 的相对风险。"
        )
        st.markdown("### 📊 模型指标")
        st.markdown(
            "- **Test AUC-ROC**: 0.8376  \n"
            "- **敏感度 (Recall 患病)**: 80.68%  \n"
            "- **特异度 (Recall 健康)**: 70.94%  \n"
            "- **准确率**: 71.85%  \n"
            "- **最优阈值**: 0.486"
        )
        st.markdown("### 🎯 风险分层")
        st.markdown(
            "- 🟢 **低风险**: 预测概率 < 7.12%  \n"
            "- 🟡 **中风险**: 7.12% ≤ 概率 < 48.6%  \n"
            "- 🔴 **高风险**: 概率 ≥ 48.6%"
        )
        st.markdown("### 📋 使用步骤")
        st.markdown(
            "1. 依次填写 5 个标签页的问卷  \n"
            "2. 点击页面底部 **提交风险评估**  \n"
            "3. 查看个性化风险概率、分级和关键影响因素  \n"
            "4. 如需,可下载评估报告"
        )

        st.markdown("---")
        st.markdown(
            "<div style='background-color:#fff3cd;padding:12px;"
            "border-left:4px solid #ffc107;border-radius:4px;font-size:13px;'>"
            "<b>⚠️ 重要声明</b><br>"
            "本系统为 <b>辅助参考工具</b>,基于群体统计学习的预测概率不能取代医生诊断。"
            "本系统 <b>不是</b> 临床诊断依据,任何健康决策均应咨询具备资质的医疗专业人员。"
            "</div>",
            unsafe_allow_html=True,
        )


# =========================================================
# 7. 结果渲染
# =========================================================
def render_results(prob: float, user_raw: dict, shap_vals_row, X_row):
    level, color, icon = classify_risk(prob)

    # ---- 概率卡片 ----
    st.markdown(
        f"""
        <div style="padding:24px;border-radius:12px;background:linear-gradient(135deg,{color}22,{color}08);
                    border-left:6px solid {color};margin-bottom:20px">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap">
                <div>
                    <div style="font-size:15px;color:#666;margin-bottom:6px">您的预测风险概率</div>
                    <div style="font-size:48px;font-weight:700;color:{color};line-height:1">
                        {prob*100:.1f}<span style='font-size:24px'>%</span>
                    </div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:15px;color:#666;margin-bottom:6px">风险等级</div>
                    <div style="font-size:32px;font-weight:700;color:{color}">{icon} {level}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- 进度条 ----
    st.progress(min(prob, 1.0))
    c1, c2, c3 = st.columns(3)
    c1.caption(f"低风险门槛 {THRESHOLD_LOW*100:.2f}%")
    c2.caption(f"最优决策阈值 {THRESHOLD_HIGH*100:.1f}%")
    c3.caption(f"您的概率 {prob*100:.2f}%")

    # ---- 风险解读 ----
    if level == "低风险":
        msg = ("您的预测心脏病风险处于**较低水平**。这通常与良好的生活方式、"
               "较轻的既往病史相关。请继续保持规律运动、均衡饮食、定期体检。")
    elif level == "中风险":
        msg = ("您的预测心脏病风险处于**中等水平**。存在若干可能影响心血管健康的因素,"
               "建议针对下方显示的主要风险因素做生活方式调整,并与医生讨论是否需要进一步检查。")
    else:
        msg = ("您的预测心脏病风险处于**较高水平**。强烈建议**尽快**咨询心血管专科医生,"
               "进行系统的临床评估 (如心电图、血脂、血糖等) 并制定个体化干预计划。")
    st.info(msg)

    # ---- SHAP 个体归因 ----
    st.markdown("### 🔍 对您风险贡献最大的关键因素")
    st.markdown(
        "下方柱状图基于 **SHAP (SHapley Additive exPlanations)** 个体归因算法,"
        "展示当前各特征对**您个人**预测结果的贡献方向与幅度。"
        "<span style='color:#e74c3c'>**红色**</span>代表推高风险,"
        "<span style='color:#27ae60'>**绿色**</span>代表降低风险。",
        unsafe_allow_html=True,
    )
    top_df = extract_top_contributors(shap_vals_row, X_row, user_raw, top_k=8)
    fig = render_shap_bar(top_df)
    st.pyplot(fig)

    # ---- Top 因素明细表 ----
    st.markdown("### 📋 因素明细")
    disp = top_df.copy()
    disp['SHAP'] = disp['SHAP'].map(lambda x: f"{x:+.4f}")
    st.dataframe(disp, hide_index=True, use_container_width=True)

    # ---- 风险/保护因素分类 ----
    risk_factors = top_df[top_df['SHAP'] > 0]
    prot_factors = top_df[top_df['SHAP'] < 0]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔺 主要危险因素")
        if len(risk_factors) > 0:
            for _, r in risk_factors.head(5).iterrows():
                st.markdown(f"- **{r['特征']}** (您的情况: {r['用户取值']})")
        else:
            st.markdown("无显著危险因素")
    with c2:
        st.markdown("#### 🔻 主要保护因素")
        if len(prot_factors) > 0:
            for _, r in prot_factors.head(5).iterrows():
                st.markdown(f"- **{r['特征']}** (您的情况: {r['用户取值']})")
        else:
            st.markdown("无显著保护因素")

    # ---- 一般性建议 ----
    st.markdown("### 💡 通用生活方式建议")
    st.markdown(
        "- 🏃 **每周至少 150 分钟中等强度有氧运动** (如快走、骑行、游泳)\n"
        "- 🥗 **地中海或 DASH 饮食模式**: 多蔬果、全谷物、坚果,少加工食品与反式脂肪\n"
        "- 🚭 **完全戒烟**,避免二手烟与电子烟\n"
        "- 🍷 **限制酒精**: 男性 ≤2 标准杯/日,女性 ≤1 标准杯/日;避免暴饮\n"
        "- ⚖️ **维持健康体重**: BMI 建议 18.5–24.9\n"
        "- 😴 **规律睡眠 7–9 小时**,管理压力\n"
        "- 🩺 **定期监测血压、血脂、血糖、糖化血红蛋白**,尤其 40 岁以上人群"
    )

    # ---- 可下载报告 ----
    report_md = _build_report_markdown(prob, level, user_raw, top_df)
    st.download_button(
        "📥 下载评估报告 (Markdown)",
        data=report_md.encode("utf-8"),
        file_name=f"心脏病风险评估报告_{pd.Timestamp.now():%Y%m%d_%H%M%S}.md",
        mime="text/markdown",
    )

    # ---- 免责声明 ----
    st.markdown("---")
    st.markdown(
        "<div style='background:#f8d7da;padding:14px;border-left:4px solid #dc3545;"
        "border-radius:4px;color:#721c24;font-size:13px;line-height:1.6'>"
        "<b>⚠️ 免责声明</b><br>"
        "本风险评估系统基于美国 2024 BRFSS 调查数据训练的统计学习模型,"
        "所提供的预测概率反映<b>群体统计规律</b>,并非针对个体的临床诊断。"
        "模型未使用生化指标(如血脂、血糖、血压实测值)、心电图、影像等医学检查结果,"
        "其判断能力受限于问卷可自述的指标维度。"
        "<br><br>"
        "本系统<b>不应</b>被用作任何医疗决策、治疗方案、药物选择的唯一依据。"
        "无论结果如何,请以<b>具备资质的医师诊断为准</b>。"
        "如您有任何心血管相关的不适或担忧,请及时前往正规医疗机构就诊。"
        "</div>",
        unsafe_allow_html=True,
    )


def _build_report_markdown(prob, level, user_raw, top_df):
    """生成可下载的 Markdown 评估报告"""
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 心脏病 (CHD / MI) 风险评估报告",
        "",
        f"**报告生成时间**: {now}",
        "",
        "## 一、评估结果",
        "",
        f"- **预测风险概率**: **{prob*100:.2f}%**",
        f"- **风险等级**: **{level}**",
        f"- **模型最优决策阈值**: {THRESHOLD_HIGH*100:.1f}%",
        f"- **低风险门槛**: {THRESHOLD_LOW*100:.2f}%",
        "",
        "## 二、您本次填写的核心指标",
        "",
        f"- 性别: {'男性' if user_raw.get('SEXVAR')==1 else '女性'}",
        f"- 年龄分组 (_AGE_G): {user_raw.get('_AGE_G')}",
        f"- BMI 分类: {user_raw.get('_BMI5CAT')}",
        f"- 吸烟状态 (_SMOKER3): {user_raw.get('_SMOKER3')}",
        f"- 中风史: {'是' if user_raw.get('CVDSTRK3')==1 else '否'}",
        f"- 关节炎: {'是' if user_raw.get('_DRDXAR2')==1 else '否'}",
        f"- COPD: {'是' if user_raw.get('CHCCOPD3')==1 else '否'}",
        f"- 糖尿病状态 (DIABETE4): {user_raw.get('DIABETE4')}",
        f"- 肾脏疾病: {'是' if user_raw.get('CHCKDNY2')==1 else '否'}",
        f"- 抑郁症: {'是' if user_raw.get('ADDEPEV3')==1 else '否'}",
        "",
        "## 三、对本次预测贡献最大的因素 (基于 SHAP 个体归因)",
        "",
        "| 排名 | 特征 | 您的取值 | SHAP 值 | 方向 |",
        "|:---:|:---|:---:|:---:|:---|",
    ]
    for i, row in top_df.iterrows():
        lines.append(
            f"| {i+1} | {row['特征']} | {row['用户取值']} | "
            f"{row['SHAP']:+.4f} | {row['方向']} |"
        )
    lines += [
        "",
        "## 四、建议",
        "",
        "- 针对上述红色 (↑ 推高风险) 因素,尝试有针对性的生活方式调整。",
        "- 保持绿色 (↓ 降低风险) 因素带来的保护作用。",
        "- 定期监测血压、血脂、血糖等指标。",
        "- 若结果提示中/高风险,请及时咨询心血管专科医师。",
        "",
        "## 五、免责声明",
        "",
        "> 本报告由 **基于 2024 BRFSS 数据的机器学习模型** 自动生成,仅供参考,",
        "> **不是临床诊断依据**。具体诊疗请咨询合格医师。",
    ]
    return "\n".join(lines)


# =========================================================
# 8. 主入口
# =========================================================
def main():
    st.set_page_config(
        page_title="心脏病风险评估系统",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 微调 CSS
    st.markdown(
        """
        <style>
            .block-container { padding-top: 2rem; padding-bottom: 2rem; }
            h1 { color: #c0392b; }
            .stRadio > div { gap: 1rem; }
            .stProgress > div > div { background-color: #c0392b; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 侧边栏
    render_sidebar()

    # 主标题
    st.markdown("# ❤️ 心脏病 (CHD / MI) 风险评估系统")
    st.markdown(
        "> 基于 2024 BRFSS 调查数据与 LightGBM 模型 · "
        "集成 SHAP 个体可解释性分析 · **仅供辅助参考**"
    )

    # 加载模型
    model, scaler, train_enc_columns, explainer = load_artifacts()

    # 问卷
    with st.form("risk_form", clear_on_submit=False):
        user_raw = build_questionnaire()
        st.markdown("---")
        submitted = st.form_submit_button(
            "🚀 提交风险评估",
            use_container_width=True,
            type="primary",
        )

    # 结果
    if submitted:
        try:
            # (A) 预处理 + 预测
            X_input = preprocess_user_input(user_raw, scaler, train_enc_columns)
            prob = float(model.predict_proba(X_input)[0, 1])

            # (B) SHAP 计算
            shap_values = explainer.shap_values(X_input)
            if isinstance(shap_values, list):
                shap_values_row = shap_values[1][0]
            else:
                # LightGBM 二分类输出 (n_samples, n_features) 代表正类贡献
                shap_values_row = shap_values[0]

            st.markdown("---")
            st.markdown("## 📊 您的评估结果")
            render_results(prob, user_raw, shap_values_row, X_input.iloc[0])

        except Exception as e:
            st.error(f"❌ 预测过程出错: {e}")
            st.exception(e)

    else:
        # 占位提示
        st.markdown(
            "<div style='text-align:center;padding:40px 20px;color:#888'>"
            "📝 请在上方 5 个标签页中完成所有问题后,点击 <b>提交风险评估</b> 按钮查看结果。"
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
