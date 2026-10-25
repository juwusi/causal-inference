# Causal Inference（因果推断）

一个用于 WorkBuddy 的技能（Skill）：判断「A 是否真的导致 B」、估计干预的因果效应（ATE/ATT）、控制混杂因素，或从数据中发现变量间的因果方向，并输出可复用的 Python 分析脚本。

> 与影响因素挖掘（相关性/预测贡献）不同，本技能的核心是**控制混杂因素**，估计剔除混淆后的净效应。

## 功能特性

**主线 A：估计干预效应**（有明确的 treatment → outcome）

| 数据形态 | 方法 |
|---|---|
| 横截面 + 可观察混杂 | 倾向得分匹配 PSM、逆概率加权 IPW |
| 横截面 + 未观测混杂 | 工具变量 IV（含第一阶段 F 检验） |
| 横截面 + 阈值规则 | 断点回归 RDD |
| 面板 + 处理前后对照 | 双重差分 DID（平行趋势） |
| 面板 + 个体固定效应 | 固定效应模型 FE |

**主线 B：因果图结构发现**

- PC、FCI（基于条件独立性）
- GES（基于得分）
- LiNGAM（线性非高斯，可确定方向）

## 安装

```bash
# Windows
git clone https://github.com/juwusi/causal-inference.git %USERPROFILE%\.workbuddy\skills\causal-inference

# macOS / Linux
git clone https://github.com/juwusi/causal-inference.git ~/.workbuddy/skills/causal-inference
```

也可以在 WorkBuddy 对话中直接说「安装 causal-inference 技能」。

## 使用

在 WorkBuddy 对话中触发，例如：

- 「这个营销活动到底有没有效果」「这个政策是否真的提升了销量」
- 「剔除混杂后，X 对 Y 的真实影响多大」

也可以直接运行核心脚本：

```bash
# 估计因果效应（倾向得分匹配）
python scripts/causal_inference.py estimate data.csv \
  --treatment 是否参加活动 \
  --outcome 销量 \
  --method psm \
  --confounders 年龄,收入,地区

# 因果图结构发现
python scripts/causal_inference.py discover data.csv --algorithm pc --plot
```

## 依赖

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `statsmodels`
- 可选：`causal-learn`（因果发现）

```bash
pip install pandas numpy scikit-learn statsmodels causal-learn
```

## 方法说明

方法选型、识别假设（可忽略性、平行趋势、工具变量有效性等）与解读要点见 [`references/methodology.md`](references/methodology.md)。

## 注意

因果结论的强度取决于对应方法的**识别假设**是否成立，脚本会在结果中提示。混杂变量选择是核心，切勿盲目把所有变量都当混杂。

## License

[MIT](LICENSE)
