#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
causal_inference.py — 因果推断脚本
估计干预的因果效应（ATE/ATT），或从数据中发现变量间的因果方向/图结构。

用法：
    python causal_inference.py inspect <datafile>
    python causal_inference.py estimate <datafile> --treatment <T列> --outcome <Y列> [选项]
    python causal_inference.py discover <datafile> [选项]

estimate 方法：psm | ipw | iv | rdd | did | fe
discover 算法：pc | fci | ges | lingam

依赖：pandas, numpy, scikit-learn, statsmodels
可选：causal-learn（因果发现）
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 数据加载与探查
# ---------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".csv":
        for enc in ("utf-8", "gbk", "gb18030"):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, encoding="utf-8", errors="ignore")
    if ext in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t", encoding="utf-8", errors="ignore")
    raise ValueError(f"不支持的文件类型: {ext}")


def classify_columns(df):
    n_rows = len(df)
    num_cols, cat_cols, text_cols, id_cols, time_cols = [], [], [], [], []
    for c in df.columns:
        s = df[c]
        nunique = s.nunique(dropna=True)
        dtype = str(s.dtype)
        if "int" in dtype or "float" in dtype:
            if nunique <= 1:
                continue
            if nunique / max(n_rows, 1) > 0.9 and any(k in c.lower() for k in ("id", "编号", "序号", "index", "代码", "工号")):
                id_cols.append(c)
                continue
            num_cols.append(c)
        elif "datetime" in dtype:
            time_cols.append(c)
        else:
            if nunique == len(s) and s.astype(str).str.len().max() > 8:
                id_cols.append(c)
            elif s.astype(str).str.match(r"^\d{4}[-/]\d{1,2}").all() and nunique > 1:
                time_cols.append(c)
            else:
                avg_len = s.astype(str).str.len().mean()
                if avg_len > 20 or nunique > 100:
                    text_cols.append(c)
                else:
                    cat_cols.append(c)
    return num_cols, cat_cols, text_cols, id_cols, time_cols


def inspect(args):
    df = load_data(args.datafile)
    print("=" * 60)
    print(f"数据文件: {args.datafile}")
    print(f"形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print("=" * 60)
    num_cols, cat_cols, text_cols, id_cols, time_cols = classify_columns(df)
    print(f"\n数值列 ({len(num_cols)}): {num_cols}")
    print(f"分类列 ({len(cat_cols)}): {cat_cols}")
    print(f"文本列 ({len(text_cols)}): {text_cols}")
    print(f"ID 列 (建议排除) ({len(id_cols)}): {id_cols}")
    print(f"时间列 ({len(time_cols)}): {time_cols}")
    print("\n缺失率与唯一值数:")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "缺失率%": (df.isna().mean() * 100).round(2),
        "唯一值": df.nunique(),
    })
    print(info.to_string())
    print("\n提示: 使用 estimate --treatment <T> --outcome <Y> 估计因果效应，或用 discover 发现因果图")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _to_numeric_series(s, name):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        sys.exit(f"错误: 列「{name}」无法转为数值")


def _encode_categorical(s):
    """分类变量转 one-hot（保留中文列名可读）。"""
    s = s.astype(str).fillna("缺失")
    return pd.get_dummies(s, drop_first=True, prefix=s.name)


def _build_design(df, cols):
    """把指定列构建成数值设计矩阵（数值原样、分类 one-hot）。"""
    num_cols, cat_cols, _, _, _ = classify_columns(df[cols])
    parts = []
    for c in cols:
        if c in num_cols:
            parts.append(pd.to_numeric(df[c], errors="coerce"))
        else:
            parts.append(_encode_categorical(df[c]))
    return pd.concat(parts, axis=1).fillna(0.0)


def _bootstrap_ci(fn, n_boot=200, alpha=0.05):
    """用 bootstrap 估计效应点估计 + 置信区间。"""
    estimates = []
    for _ in range(n_boot):
        try:
            estimates.append(fn())
        except Exception:
            continue
    if not estimates:
        return np.nan, (np.nan, np.nan)
    estimates = np.array(estimates)
    lo, hi = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(estimates)), (float(lo), float(hi))


def _smd(treated, control):
    """标准化均值差。"""
    pooled = np.sqrt((np.var(treated) + np.var(control)) / 2)
    if pooled == 0:
        return 0.0
    return (np.mean(treated) - np.mean(control)) / pooled


# ---------------------------------------------------------------------------
# 效应估计方法
# ---------------------------------------------------------------------------

def _estimate_psm(df, treatment, outcome, confounders):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors

    t = df[treatment].astype(float).values
    y = df[outcome].astype(float).values
    X = _build_design(df, confounders).values if confounders else np.ones((len(df), 1))

    ps = LogisticRegression(max_iter=1000).fit(X, t).predict_proba(X)[:, 1]

    treated_idx = np.where(t == 1)[0]
    control_idx = np.where(t == 0)[0]

    if len(treated_idx) == 0 or len(control_idx) == 0:
        sys.exit("错误: treatment 必须是二值且两组都有样本")

    # 1:1 最近邻匹配（可放回）
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(ps[control_idx].reshape(-1, 1))
    _, matches = nn.kneighbors(ps[treated_idx].reshape(-1, 1))
    matched_control = control_idx[matches.flatten()]

    # ATT（全样本直接计算作为点估计）
    att = np.mean(y[treated_idx]) - np.mean(y[matched_control])

    def boot():
        ti = np.random.choice(treated_idx, size=len(treated_idx), replace=True)
        nn.fit(ps[control_idx].reshape(-1, 1))
        _, m = nn.kneighbors(ps[ti].reshape(-1, 1))
        mc = control_idx[m.flatten()]
        return np.mean(y[ti]) - np.mean(y[mc])

    point = att
    _, (lo, hi) = _bootstrap_ci(boot)

    # 协变量平衡（SMD）
    print("\n协变量平衡检验（匹配后 SMD，应 < 0.1）:")
    balance = {}
    for c in confounders:
        vals = pd.to_numeric(df[c], errors="coerce").fillna(0).values
        smd_before = _smd(vals[treated_idx], vals[control_idx])
        smd_after = _smd(vals[treated_idx], vals[matched_control])
        balance[c] = {"before": round(smd_before, 3), "after": round(smd_after, 3)}
        flag = "✓" if abs(smd_after) < 0.1 else "⚠"
        print(f"  {flag} {c}: SMD {smd_before:+.3f} → {smd_after:+.3f}")

    return {
        "method": "PSM (倾向得分匹配)",
        "effect": "ATT (处理组效应)",
        "point": round(float(point), 4),
        "ci": [round(float(lo), 4), round(float(hi), 4)],
        "balance": balance,
    }


def _estimate_ipw(df, treatment, outcome, confounders):
    from sklearn.linear_model import LogisticRegression

    t = df[treatment].astype(float).values
    y = df[outcome].astype(float).values
    X = _build_design(df, confounders).values if confounders else np.ones((len(df), 1))

    ps = LogisticRegression(max_iter=1000).fit(X, t).predict_proba(X)[:, 1]
    ps = np.clip(ps, 0.02, 0.98)

    def ipw_ate():
        """IPW ATE：处理组与对照组的逆概率加权均值之差。"""
        w1 = t / ps
        y1 = np.sum(w1 * y) / np.sum(w1)
        w0 = (1 - t) / (1 - ps)
        y0 = np.sum(w0 * y) / np.sum(w0)
        return y1 - y0

    def boot():
        idx = np.random.choice(len(y), size=len(y), replace=True)
        t_b, y_b, ps_b = t[idx], y[idx], ps[idx]
        w1 = t_b / ps_b
        y1 = np.sum(w1 * y_b) / np.sum(w1)
        w0 = (1 - t_b) / (1 - ps_b)
        y0 = np.sum(w0 * y_b) / np.sum(w0)
        return y1 - y0

    # 点估计用全样本，CI 用 bootstrap
    point = ipw_ate()
    _, (lo, hi) = _bootstrap_ci(boot)
    return {
        "method": "IPW (逆概率加权)",
        "effect": "ATE (平均处理效应)",
        "point": round(float(point), 4),
        "ci": [round(float(lo), 4), round(float(hi), 4)],
        "balance": {"倾向得分范围": f"[{ps.min():.3f}, {ps.max():.3f}]"},
    }


def _estimate_iv(df, treatment, outcome, instrument, confounders):
    import statsmodels.api as sm

    t = pd.to_numeric(df[treatment], errors="coerce").fillna(0).values
    y = pd.to_numeric(df[outcome], errors="coerce").fillna(0).values
    z = pd.to_numeric(df[instrument], errors="coerce").fillna(0).values

    exog = [z]
    if confounders:
        exog += [_build_design(df, confounders).values]

    # 第一阶段：检验工具变量强度
    X1 = sm.add_constant(np.column_stack(exog))
    stage1 = sm.OLS(t, X1).fit()
    f_stat = stage1.fvalue
    print(f"\n第一阶段 F 统计量: {f_stat:.2f} (应 > 10，否则为弱工具变量)")

    # 第二阶段 2SLS
    try:
        endog = np.column_stack([t])
        instr = np.column_stack(exog)
        iv_model = sm.OLS(y, sm.add_constant(np.column_stack([t] + (exog[1:] if confounders else []))))
        from linearmodels.iv import IV2SLS
        _ = IV2SLS  # 触发 import 检测
        raise ImportError
    except ImportError:
        # 用 statsmodels 手动 2SLS
        Xz = sm.add_constant(np.column_stack(exog))
        t_hat = sm.OLS(t, Xz).fit().fittedvalues
        X2 = sm.add_constant(np.column_stack([t_hat] + ([exog[1]] if confounders else [])))
        res = sm.OLS(y, X2).fit()
        coef = res.params[1]
        se = res.bse[1]
        ci = [coef - 1.96 * se, coef + 1.96 * se]

    return {
        "method": f"IV (工具变量, 工具={instrument})",
        "effect": "LATE (局部平均处理效应)",
        "point": round(float(coef), 4),
        "ci": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        "balance": {"第一阶段F统计量": round(float(f_stat), 2)},
    }


def _estimate_rdd(df, treatment, outcome, running_var, threshold):
    """断点回归：局部线性估计。running_var 为运行变量，threshold 为断点。"""
    from sklearn.linear_model import LinearRegression

    r = pd.to_numeric(df[running_var], errors="coerce").fillna(0).values
    y = pd.to_numeric(df[outcome], errors="coerce").fillna(0).values
    t = (r >= threshold).astype(float)
    centered = r - threshold

    # 带宽：使用经验法则
    h = 1.84 * np.std(centered) * (len(y) ** (-1 / 5)) if len(y) > 10 else 1.0
    h = max(h, 1e-6)

    def local_linear(side):
        mask = (centered >= -h) & (centered <= h) & ((t == side))
        if mask.sum() < 3:
            return np.nan
        X = centered[mask].reshape(-1, 1)
        return LinearRegression().fit(X, y[mask]).predict(np.array([[0.0]]))[0]

    y_left = local_linear(0)
    y_right = local_linear(1)
    if np.isnan(y_left) or np.isnan(y_right):
        sys.exit("错误: 断点两侧样本不足，无法估计")

    point = y_right - y_left

    def boot():
        idx = np.random.choice(len(y), size=len(y), replace=True)
        c2, y2, t2 = centered[idx], y[idx], t[idx]
        mask_l = (c2 >= -h) & (c2 <= h) & (t2 == 0)
        mask_r = (c2 >= -h) & (c2 <= h) & (t2 == 1)
        if mask_l.sum() < 3 or mask_r.sum() < 3:
            return np.nan
        l = LinearRegression().fit(c2[mask_l].reshape(-1, 1), y2[mask_l]).predict(np.array([[0.0]]))[0]
        rr = LinearRegression().fit(c2[mask_r].reshape(-1, 1), y2[mask_r]).predict(np.array([[0.0]]))[0]
        return rr - l

    _, (lo, hi) = _bootstrap_ci(boot)
    return {
        "method": f"RDD (断点回归, 运行变量={running_var}, 断点={threshold})",
        "effect": "断点处因果效应",
        "point": round(float(point), 4),
        "ci": [round(float(lo), 4), round(float(hi), 4)],
        "balance": {"带宽": round(float(h), 4)},
    }


def _estimate_did(df, treatment, outcome, post_col, confounders):
    """双重差分：outcome ~ treat + post + treat*post。treatment 为处理组标识，post_col 为时期标识。"""
    import statsmodels.api as sm

    y = pd.to_numeric(df[outcome], errors="coerce").fillna(0).values
    treat = pd.to_numeric(df[treatment], errors="coerce").fillna(0).values
    post = pd.to_numeric(df[post_col], errors="coerce").fillna(0).values
    did = treat * post

    cols = [treat, post, did]
    if confounders:
        cols.append(_build_design(df, confounders).values)
    X = sm.add_constant(np.column_stack(cols))
    res = sm.OLS(y, X).fit()

    coef = res.params[3]  # did 项系数
    se = res.bse[3]
    ci = [coef - 1.96 * se, coef + 1.96 * se]

    return {
        "method": f"DID (双重差分, 处理组={treatment}, 时期={post_col})",
        "effect": "平均处理效应",
        "point": round(float(coef), 4),
        "ci": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        "balance": {"p值": round(float(res.pvalues[3]), 4)},
    }


def _estimate_fe(df, treatment, outcome, unit_col, confounders):
    """固定效应：组内去均值（within estimator）。"""
    import statsmodels.api as sm

    y = pd.to_numeric(df[outcome], errors="coerce").fillna(0).values
    t = pd.to_numeric(df[treatment], errors="coerce").fillna(0).values
    unit = df[unit_col].astype(str).values

    def demean(v):
        df_tmp = pd.DataFrame({"v": v, "u": unit})
        return v - df_tmp.groupby("u")["v"].transform("mean").values

    y_dm = demean(y)
    t_dm = demean(t)

    X_cols = [t_dm]
    if confounders:
        for c in confounders:
            v = pd.to_numeric(df[c], errors="coerce").fillna(0).values
            X_cols.append(demean(v))
    X = np.column_stack(X_cols)
    res = sm.OLS(y_dm, X).fit()

    coef = res.params[0]
    se = res.bse[0]
    ci = [coef - 1.96 * se, coef + 1.96 * se]

    return {
        "method": f"固定效应 (个体标识={unit_col})",
        "effect": "组内处理效应",
        "point": round(float(coef), 4),
        "ci": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
        "balance": {"p值": round(float(res.pvalues[0]), 4)},
    }


# ---------------------------------------------------------------------------
# 因果发现
# ---------------------------------------------------------------------------

def _discover(df, algorithm, top_cols=None):
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.search.ConstraintBased.FCI import fci
        from causallearn.search.ScoreBased.GES import ges
        from causallearn.search.FCMBased import lingam
    except ImportError:
        sys.exit("错误: 因果发现需要 causal-learn 库。请运行: pip install causal-learn")

    # 只取数值列参与发现
    num_cols, _, _, id_cols, time_cols = classify_columns(df)
    use_cols = [c for c in num_cols if c not in id_cols and c not in time_cols]
    if top_cols:
        use_cols = [c for c in use_cols if c in top_cols]
    if len(use_cols) < 2:
        sys.exit("错误: 因果发现需要至少 2 个数值列")

    data = df[use_cols].fillna(df[use_cols].median()).values
    labels = use_cols
    print(f"参与因果发现的变量 ({len(use_cols)}): {use_cols}")

    if algorithm == "pc":
        cg = pc(data, alpha=0.05)
    elif algorithm == "fci":
        cg = fci(data, alpha=0.05)
    elif algorithm == "ges":
        record = ges(data)
        cg = record["G"]
    elif algorithm == "lingam":
        model = lingam.ICALiNGAM()
        model.fit(data)
        B = model.adjacency_matrix_
        edges = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                if abs(B[i, j]) > 0.05:
                    edges.append((labels[j], labels[i], round(float(B[i, j]), 3)))
        return edges, labels
    else:
        sys.exit(f"未知算法: {algorithm}")

    # 解析图：cg.G.graph 是邻接矩阵（-1 表示 →, 1 表示 ←, 0 无向边）
    edges = []
    G = cg.G.graph if hasattr(cg.G, "graph") else cg
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = G[i, j], G[j, i]
            if a == -1 and b == 1:
                edges.append((labels[i], labels[j], "→"))
            elif a == 1 and b == -1:
                edges.append((labels[j], labels[i], "→"))
            elif a == 1 and b == 1:
                edges.append((labels[i], labels[j], "—"))  # 无向（等价类）
    return edges, labels


def estimate(args):
    df = load_data(args.datafile)
    if args.treatment not in df.columns:
        sys.exit(f"错误: treatment 列「{args.treatment}」不存在。可用列: {list(df.columns)}")
    if args.outcome not in df.columns:
        sys.exit(f"错误: outcome 列「{args.outcome}」不存在。可用列: {list(df.columns)}")

    print("=" * 60)
    print(f"treatment: {args.treatment}  →  outcome: {args.outcome}")
    print(f"方法: {args.method}")
    print("=" * 60)

    # 默认混杂：其余列（排除 treatment/outcome/id/time/指定角色）
    confounders = args.confounders or []
    if not confounders:
        num_cols, cat_cols, _, id_cols, time_cols = classify_columns(df)
        confounders = [c for c in num_cols + cat_cols if c not in (args.treatment, args.outcome, args.instrument, args.post, args.unit, args.running)]
        if confounders:
            print(f"\n未指定混杂变量，默认使用: {confounders}")
            print("（提示: 混杂选择是因果推断最关键假设，请确认是否合理）")

    if args.method == "psm":
        result = _estimate_psm(df, args.treatment, args.outcome, confounders)
    elif args.method == "ipw":
        result = _estimate_ipw(df, args.treatment, args.outcome, confounders)
    elif args.method == "iv":
        if not args.instrument:
            sys.exit("错误: iv 方法需要 --instrument 工具变量列")
        result = _estimate_iv(df, args.treatment, args.outcome, args.instrument, confounders)
    elif args.method == "rdd":
        running = args.running or args.treatment
        threshold = args.threshold if args.threshold is not None else pd.to_numeric(df[running], errors="coerce").median()
        result = _estimate_rdd(df, args.treatment, args.outcome, running, threshold)
    elif args.method == "did":
        if not args.post:
            sys.exit("错误: did 方法需要 --post 时期标识列")
        result = _estimate_did(df, args.treatment, args.outcome, args.post, confounders)
    elif args.method == "fe":
        if not args.unit:
            sys.exit("错误: fe 方法需要 --unit 个体标识列")
        result = _estimate_fe(df, args.treatment, args.outcome, args.unit, confounders)
    else:
        sys.exit(f"未知方法: {args.method}")

    print("\n" + "=" * 60)
    print(f"方法: {result['method']}")
    print(f"效应类型: {result['effect']}")
    print(f"点估计: {result['point']}")
    print(f"95% 置信区间: {result['ci']}")
    print("=" * 60)
    print("\n解读提示: 该结论在对应方法的识别假设成立时才具有因果含义（见 references/methodology.md）。")

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "causal_effect.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"treatment": args.treatment, "outcome": args.outcome, **result}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_json}")


def discover(args):
    df = load_data(args.datafile)
    print("=" * 60)
    print(f"因果发现算法: {args.algorithm}")
    print("=" * 60)

    top_cols = [c.strip() for c in args.cols.split(",")] if args.cols else None
    edges, labels = _discover(df, args.algorithm, top_cols)

    print(f"\n发现的因果边 ({len(edges)} 条):")
    if not edges:
        print("  （未发现显著边）")
    for a, b, kind in edges:
        print(f"  {a} {kind} {b}")

    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "causal_graph.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"algorithm": args.algorithm, "variables": labels, "edges": [(str(a), str(b), str(k)) for a, b, k in edges]}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_json}")

    if args.plot:
        _plot_graph(edges, labels, args.out)


def _plot_graph(edges, labels, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False

        G = nx.DiGraph()
        G.add_nodes_from(labels)
        for a, b, kind in edges:
            if kind == "→":
                G.add_edge(a, b)
            else:
                G.add_edge(a, b)
                G.add_edge(b, a)
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color="#3B82F6", node_size=1800)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="SimHei")
        directed = [(a, b) for a, b, k in edges if k == "→"]
        undirected = [(a, b) for a, b, k in edges if k == "—"]
        if directed:
            nx.draw_networkx_edges(G, pos, edgelist=directed, arrows=True, arrowstyle="-|>", edge_color="#111827")
        if undirected:
            nx.draw_networkx_edges(G, pos, edgelist=undirected, arrows=False, style="dashed", edge_color="#9CA3AF")
        plt.title("因果图结构")
        plt.axis("off")
        path = os.path.join(out_dir, "causal_graph.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"图已保存: {path}")
    except ImportError:
        print("  [提示] 未安装 matplotlib/networkx，跳过绘图。")


def main():
    p = argparse.ArgumentParser(description="因果推断")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("inspect", help="探查数据结构")
    pi.add_argument("datafile")
    pi.set_defaults(func=inspect)

    pe = sub.add_parser("estimate", help="估计因果效应")
    pe.add_argument("datafile")
    pe.add_argument("--treatment", required=True)
    pe.add_argument("--outcome", required=True)
    pe.add_argument("--method", default="psm", choices=["psm", "ipw", "iv", "rdd", "did", "fe"])
    pe.add_argument("--confounders", default=None, help="混杂变量，逗号分隔")
    pe.add_argument("--instrument", default=None, help="工具变量列 (iv)")
    pe.add_argument("--running", default=None, help="运行变量列 (rdd)")
    pe.add_argument("--threshold", type=float, default=None, help="断点值 (rdd)")
    pe.add_argument("--post", default=None, help="时期标识列 (did)")
    pe.add_argument("--unit", default=None, help="个体标识列 (fe)")
    pe.add_argument("--out", default="./causal_output")
    pe.set_defaults(func=estimate)

    pd_ = sub.add_parser("discover", help="因果图结构发现")
    pd_.add_argument("datafile")
    pd_.add_argument("--algorithm", default="pc", choices=["pc", "fci", "ges", "lingam"])
    pd_.add_argument("--cols", default=None, help="参与发现的变量，逗号分隔（默认全部数值列）")
    pd_.add_argument("--plot", action="store_true")
    pd_.add_argument("--out", default="./causal_output")
    pd_.set_defaults(func=discover)

    args = p.parse_args()
    if args.cmd == "estimate" and args.confounders:
        args.confounders = [c.strip() for c in args.confounders.split(",") if c.strip()]
    args.func(args)


if __name__ == "__main__":
    main()
