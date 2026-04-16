# ❤️ 心脏病 (CHD / MI) 风险评估系统

基于 2024 BRFSS 美国居民健康调查数据训练的 **LightGBM** 机器学习模型 (Test AUC = 0.8376),集成 **SHAP 个体归因分析** 的交互式心脏病风险评估系统。

---

## 🎯 系统特点

| 项目 | 说明 |
|---|---|
| 🧠 模型 | LightGBM (Optuna 超参数贝叶斯调优) |
| 📊 测试集 AUC-ROC | **0.8376** |
| 🎯 敏感度 / 特异度 | **80.68% / 70.94%** |
| ⚖️ 最优决策阈值 | **0.486** (Accuracy ≥ 72% 下最大化 Recall) |
| 🔎 可解释性 | SHAP TreeExplainer (个体归因) |
| 🖥️ 界面 | Streamlit 问卷式交互 |
| 📝 特征数量 | 34 条用户可自报指标 → 48 个模型特征 |

### 风险分层边界 (来自高/低风险画像分析)

| 等级 | 预测概率区间 | 说明 |
|---|---|---|
| 🟢 **低风险** | `P < 0.0712` | 测试集下 20% 分位, 实际患病率 0.35% |
| 🟡 **中风险** | `0.0712 ≤ P < 0.486` | 处于决策阈值以下但非低风险区 |
| 🔴 **高风险** | `P ≥ 0.486` | 将被模型判为阳性 |

---

## 📁 文件结构

```
heart_risk_system/
├── heart_risk_app.py          # Streamlit 主应用
├── requirements.txt           # 依赖清单
├── README.md                  # 本说明
│
├── model_LightGBM.pkl         # ⚠️ 必需: 训练好的 LightGBM 模型
├── scaler.pkl                 # ⚠️ 必需: MinMaxScaler (fit 于 106 列)
└── train_enc_columns.pkl      # ⚠️ 必需: 训练集 one-hot 后完整列名
```

### 🔑 关键前置条件

请将在训练阶段生成的以下三个 `.pkl` 文件放入应用同级目录:

1. **`model_LightGBM.pkl`** — 来自可解释性分析脚本 (model_LightGBM.pkl)
2. **`scaler.pkl`** — 来自数据预处理脚本 (见步骤 6 "保存预处理转换器")
3. **`train_enc_columns.pkl`** — 同样来自数据预处理脚本

> 这三个文件是保证 **推理阶段数据口径与训练阶段完全一致** 的关键,缺一不可。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 放置模型产物

将三个 `.pkl` 文件拷贝到 `heart_risk_app.py` 同一目录。

### 3. 启动应用

```bash
streamlit run heart_risk_app.py
```

默认在浏览器中打开 `http://localhost:8501`

---

## 🔄 推理管道 (与训练阶段严格一致)

```
┌─ 用户填写 34 题问卷 (人口学 + 生活方式 + 病史 + 功能状态 + 医疗)
│
├─ (1) 构建 BRFSS 原始编码行 (34 列)
├─ (2) apply_feature_engineering_mappings()  ← 与训练函数完全一致
├─ (3) pd.get_dummies(columns=[_STATE, MARITAL, EMPLOY1, _IMPRACE, DIABETE4])
├─ (4) df.reindex(columns=train_enc_columns, fill_value=0)   ← 对齐 106 列
├─ (5) astype(float)
├─ (6) scaler.transform(X)                   ← 复用训练期 MinMaxScaler
├─ (7) X[FINAL_FEATURES]                     ← 选 48 列
│
├─ model.predict_proba(X)[0, 1]              ← 风险概率
├─ explainer.shap_values(X)                  ← SHAP 个体归因
│
└─ 输出: 概率 + 风险等级 + Top-K 贡献因素 + 可下载报告
```

---

## 🖼️ 应用界面概览

### 主界面布局
- **侧边栏**: 系统简介、模型指标、风险分层说明、免责声明
- **主区域 Tab 1**: 👤 基本信息 (性别/年龄/种族/教育/收入/婚姻/就业/退伍军人身份/居住州)
- **主区域 Tab 2**: 🚬 生活方式 (身高体重→BMI/吸烟/电子烟/暴饮/运动)
- **主区域 Tab 3**: 🫀 既往病史 (中风/癌/COPD/抑郁/肾病/关节炎/哮喘/糖尿病)
- **主区域 Tab 4**: 🦽 功能状态 (听力/视力/认知/行走/穿衣/独立外出)
- **主区域 Tab 5**: 🏥 医疗保健 (保险/费用障碍/肺炎疫苗/自评健康/身心不适天数)

### 结果展示
1. **风险概率卡片** — 大号百分数 + 彩色等级标签
2. **进度条与阈值刻度** — 直观对比您的概率位置
3. **风险解读文本** — 因等级而定的个性化建议
4. **SHAP 归因柱状图** — 红色 ↑ 推高风险 / 绿色 ↓ 降低风险
5. **因素明细表** — Top-8 特征排序
6. **危险因素 / 保护因素分栏** — 快速速览
7. **通用生活方式建议**
8. **可下载评估报告** (Markdown)
9. **强化免责声明**

---

## ⚠️ 免责声明

本系统所提供的预测概率反映 **群体统计学习的规律**,并非针对个体的临床诊断。

- 模型训练数据来自 **美国 BRFSS 2024** 调查,以美国人群为主,用于其他人群时请谨慎参考。
- 模型 **未使用** 生化检验指标 (血脂/血糖/血压数值)、心电图、影像、冠脉造影等临床证据,仅基于问卷可自报维度。
- 本系统 **不应** 被用作任何医疗决策、治疗方案、药物选择的唯一依据。
- 如有任何心血管相关的症状或担忧,请及时就诊于具备资质的医疗机构。

---

## 📚 相关文档

- BRFSS 2024 数据集: https://www.cdc.gov/brfss/annual_data/annual_2024.html
- SHAP 官方文档: https://shap.readthedocs.io
- Streamlit 官方文档: https://docs.streamlit.io
- LightGBM 官方文档: https://lightgbm.readthedocs.io
