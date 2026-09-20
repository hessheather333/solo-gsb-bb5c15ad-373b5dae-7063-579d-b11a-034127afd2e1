"""共线生产换线排程（临时预览）：子集动态规划求总清洁时间最短的完整生产顺序。

- 每个产品对应一条历史审查记录，过敏原集合取记录「当前配方」（标签核对
  版本：有 B 版且核对 B 版时取 B，否则取 A）的最终过敏原全集，与标签核对
  口径一致；
- 有向换线成本：前一产品含有、后一产品不含的过敏原，按用户配置的各过敏原
  清洁分钟数求和（A→B 与 B→A 成本一般不同）；
- 子集动态规划（Held-Karp）在 2^n 个子集状态上求总清洁时间最短的完整生产
  顺序（哈密顿路径，n 个产品共 n−1 段换线）；总耗时并列时按产品 ID 序列的
  字典序取最小者，结果完全确定；
- 记录缺失或处于需复核状态（旧记录无指纹 / 相关原料映射已变更）的产品一律
  禁止参与，并逐条给出原因；
- 只读预览：不写入或修改任何审查记录。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import db

MIN_PRODUCTS = 2
MAX_PRODUCTS = 10


class ChangeoverError(Exception):
    """换线排程输入校验错误（message 与 errors 均直接展示给用户）。"""

    def __init__(self, message: str, errors: Optional[List[dict]] = None):
        super().__init__(message)
        self.errors = errors or []


def current_allergens(review: dict) -> List[str]:
    """记录「当前配方」（标签核对版本）的最终过敏原集合（排序后）。

    与 db._backfill_missing_sources 选取核对版本的口径一致：结果中有 B 版
    且标签核对版本为 B 版标签时取 version_b，否则取 version_a。
    """
    result = review.get("result") or {}
    checked_label = result.get("label_check_version", review.get("label_a"))
    version = (result.get("version_b")
               if result.get("version_b") and checked_label == review.get("label_b")
               else result.get("version_a"))
    return sorted((version or {}).get("allergens", []))


def transition_allergens(prev: Sequence[str], nxt: Sequence[str]) -> List[str]:
    """前一产品含有、后一产品不含的过敏原（该段换线需清洁项，排序后）。"""
    return sorted(set(prev) - set(nxt))


def validate_minutes(clean_minutes) -> Dict[str, float]:
    """校验各过敏原清洁分钟数：必须是非负有限数字，返回 {过敏原: 分钟}。"""
    if clean_minutes is None:
        return {}
    if not isinstance(clean_minutes, dict):
        raise ChangeoverError("清洁分钟数须为「过敏原：分钟」的键值表")
    minutes: Dict[str, float] = {}
    for allergen, value in clean_minutes.items():
        key = str(allergen).strip()
        if not key:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ChangeoverError(f"过敏原「{key}」的清洁分钟数必须是数字")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ChangeoverError(f"过敏原「{key}」的清洁分钟数须为不小于 0 的数字")
        minutes[key] = number
    return minutes


def load_products(review_ids) -> List[dict]:
    """校验并加载参与排程的产品（历史审查记录）。

    数量须为 2～10 个且不重复；记录缺失或处于需复核状态的一律禁止参与，
    全部问题逐条汇总后一次抛出。
    """
    if not isinstance(review_ids, (list, tuple)) or not review_ids:
        raise ChangeoverError(
            f"请勾选 {MIN_PRODUCTS}～{MAX_PRODUCTS} 个产品参与排程")
    ids: List[int] = []
    for rid in review_ids:
        if isinstance(rid, bool) or not isinstance(rid, int):
            raise ChangeoverError(f"无效的产品 ID：{rid!r}")
        ids.append(rid)
    duplicated = sorted({rid for rid in ids if ids.count(rid) > 1})
    if duplicated:
        raise ChangeoverError(
            "产品不能重复选择：" + "、".join(f"#{rid}" for rid in duplicated))
    if not MIN_PRODUCTS <= len(ids) <= MAX_PRODUCTS:
        raise ChangeoverError(
            f"产品数量须为 {MIN_PRODUCTS}～{MAX_PRODUCTS} 个，当前 {len(ids)} 个")

    products: List[dict] = []
    errors: List[dict] = []
    for rid in ids:
        review = db.get_review(rid)
        if review is None:
            errors.append({"id": rid, "reason": "审查记录不存在（可能已被删除）"})
            continue
        if review.get("needs_recheck"):
            why = ("旧记录无字典指纹" if review.get("recheck_reason") == "legacy"
                   else "相关原料的过敏原映射已变更")
            errors.append({"id": rid, "name": review["name"],
                           "reason": f"记录处于需复核状态（{why}），"
                                     "请先在历史记录中完成复核"})
            continue
        products.append({"id": review["id"], "name": review["name"],
                         "allergens": current_allergens(review)})
    if errors:
        raise ChangeoverError("所选产品包含不可参与排程的记录", errors)
    return products


def optimal_order(products: List[dict],
                  minutes: Dict[str, float]) -> Tuple[float, List[int]]:
    """子集动态规划求总清洁时间最短的生产顺序。

    状态 ``dp[mask][j] = (最小总成本, 对应的产品 ID 序列元组)``，表示恰好覆盖
    mask 中产品且以 j 结尾的路径；转移时先比总成本、并列再比 ID 序列字典序。
    由于追加的最后一个产品相同，子问题的字典序最优前缀可直接组合，最终答案即
    全集中 (总成本, ID 序列) 字典序最小者。

    返回 (最小总清洁时间, 产品在 products 中的下标顺序)。
    """
    n = len(products)
    ids = [p["id"] for p in products]
    allergen_sets = [set(p["allergens"]) for p in products]
    # 有向换线成本矩阵：i → j 清洁 i 含有而 j 不含的过敏原
    cost = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                cost[i][j] = sum(minutes.get(a, 0.0)
                                 for a in allergen_sets[i] - allergen_sets[j])

    size = 1 << n
    dp: List[List[Optional[Tuple[float, Tuple[int, ...]]]]] = \
        [[None] * n for _ in range(size)]
    for j in range(n):
        dp[1 << j][j] = (0.0, (ids[j],))
    for mask in range(size):
        for j in range(n):
            if not (mask >> j) & 1:
                continue
            prev_mask = mask ^ (1 << j)
            if not prev_mask:
                continue
            best: Optional[Tuple[float, Tuple[int, ...]]] = None
            for i in range(n):
                prev = dp[prev_mask][i] if (prev_mask >> i) & 1 else None
                if prev is None:
                    continue
                cand = (prev[0] + cost[i][j], prev[1] + (ids[j],))
                if best is None or cand < best:
                    best = cand
            dp[mask][j] = best
    total, id_seq = min(dp[size - 1][j] for j in range(n))
    index_of = {pid: idx for idx, pid in enumerate(ids)}
    return total, [index_of[pid] for pid in id_seq]


def _num(value: float):
    """分钟数展示：整数值返回 int，其余保留小数（消除浮点尾差）。"""
    number = float(value)
    if number.is_integer():
        return int(number)
    return round(number, 6)


def build_schedule(review_ids, clean_minutes=None) -> dict:
    """共线生产换线排程主流程（只读，不改写任何审查记录）。

    返回最优生产顺序、逐段需清洁过敏原 / 耗时 / 累计耗时、总清洁时间，
    以及所选产品的过敏原并集与实际生效的清洁分钟数（供页面渲染配置项）。
    """
    products = load_products(review_ids)
    minutes = validate_minutes(clean_minutes)
    union = sorted({a for p in products for a in p["allergens"]})
    effective = {a: minutes.get(a, 0.0) for a in union}

    total, order_idx = optimal_order(products, effective)
    ordered = [products[i] for i in order_idx]

    segments: List[dict] = []
    cumulative = 0.0
    for pos in range(len(ordered) - 1):
        prev, nxt = ordered[pos], ordered[pos + 1]
        allergens = transition_allergens(prev["allergens"], nxt["allergens"])
        seg_minutes = sum(effective[a] for a in allergens)
        cumulative += seg_minutes
        segments.append({
            "from_id": prev["id"], "from_name": prev["name"],
            "to_id": nxt["id"], "to_name": nxt["name"],
            "allergens": allergens,
            "minutes": _num(seg_minutes),
            "cumulative": _num(cumulative),
        })
    return {
        "products": ordered,
        "order": [p["id"] for p in ordered],
        "allergen_union": union,
        "clean_minutes": {a: _num(v) for a, v in effective.items()},
        "segments": segments,
        "total_minutes": _num(total),
    }
