"""共线生产换线排程（临时预览）测试。

覆盖：
- 当前配方最终过敏原集合的选取（有 B 版取 B 版，否则 A 版）；
- 有向换线成本（前一产品含有、后一产品不含）与分钟数校验；
- 子集动态规划最优性（与暴力枚举对照）与总耗时并列时按产品 ID 序列
  字典序取最小者；
- 逐段结果（需清洁过敏原 / 耗时 / 累计耗时）与总耗时一致性；
- 数量（2～10）、重复、记录缺失、需复核（旧记录无指纹 / 映射已变更）
  一律禁止参与并指出原因；
- 预览只读：不改写任何审查记录。
"""
import itertools
import os
import tempfile
import unittest

_DB_DIR = tempfile.mkdtemp(prefix="allergen-changeover-")
os.environ["DB_PATH"] = os.path.join(_DB_DIR, "changeover.db")

from app import db  # noqa: E402
from app.changeover import (ChangeoverError, build_schedule,  # noqa: E402
                            current_allergens, load_products, optimal_order,
                            transition_allergens, validate_minutes)
from app.review import analyze  # noqa: E402

DICT_TEXT = """\
小麦粉：麸质谷物
乳清粉：乳
鸡蛋：蛋类
芝麻：芝麻
食盐：
"""


def make_review(text_a, text_b=None, name="产品", declared=None, with_fp=True):
    result, fp = analyze(
        text_a=text_a, text_b=text_b,
        label_a="现版配方", label_b="新版配方",
        declared=[] if declared is None else declared)
    return db.create_review(
        name=name, declared=[] if declared is None else declared,
        text_a=text_a, text_b=text_b,
        label_a="现版配方", label_b="新版配方",
        result=result, dict_fingerprint=fp if with_fp else None)


class ChangeoverTest(unittest.TestCase):
    def setUp(self):
        # 每个用例使用独立数据库，互不污染
        db.DB_PATH = os.path.join(
            _DB_DIR, f"changeover_{self.id().rsplit('.', 1)[-1]}.db")
        db.init_db()
        db.bulk_import(DICT_TEXT)

    # ---------- 当前配方最终过敏原集合 ----------

    def test_current_allergens_single_version(self):
        rid = make_review("配料表：小麦粉，鸡蛋，食盐")
        review = db.get_review(rid)
        self.assertEqual(current_allergens(review), ["蛋类", "麸质谷物"])

    def test_current_allergens_prefers_checked_b_version(self):
        rid = make_review("配料表：小麦粉，鸡蛋",
                          text_b="配料表：小麦粉，芝麻")
        review = db.get_review(rid)
        # 标签核对版本为 B 版，应取 B 版最终过敏原集合
        self.assertEqual(current_allergens(review), ["芝麻", "麸质谷物"])

    # ---------- 有向换线成本 ----------

    def test_transition_allergens_directed(self):
        self.assertEqual(transition_allergens(["乳", "蛋类"], ["乳"]), ["蛋类"])
        self.assertEqual(transition_allergens(["乳"], ["乳", "蛋类"]), [])
        self.assertEqual(transition_allergens([], ["乳"]), [])

    def test_validate_minutes(self):
        self.assertEqual(validate_minutes(None), {})
        self.assertEqual(validate_minutes({"乳": 30, "蛋类": 0}), {"乳": 30.0, "蛋类": 0.0})
        with self.assertRaises(ChangeoverError):
            validate_minutes({"乳": -1})
        with self.assertRaises(ChangeoverError):
            validate_minutes({"乳": "三十"})
        with self.assertRaises(ChangeoverError):
            validate_minutes({"乳": True})

    # ---------- 子集动态规划 ----------

    def _products(self, allergen_sets):
        return [{"id": i + 1, "name": f"P{i + 1}", "allergens": sorted(a)}
                for i, a in enumerate(allergen_sets)]

    def test_optimal_order_matches_brute_force(self):
        allergen_sets = [{"乳"}, {"乳", "蛋类"}, {"蛋类"}, {"麸质谷物", "乳"}, set()]
        minutes = {"乳": 10.0, "蛋类": 20.0, "麸质谷物": 5.0}
        products = self._products(allergen_sets)
        total, order_idx = optimal_order(products, minutes)
        ids = [products[i]["id"] for i in order_idx]

        def path_cost(seq):
            return sum(
                sum(minutes.get(a, 0.0)
                    for a in allergen_sets[seq[k]] - allergen_sets[seq[k + 1]])
                for k in range(len(seq) - 1))

        best = min(
            ((path_cost(list(perm)), tuple(products[i]["id"] for i in perm))
             for perm in itertools.permutations(range(len(products)))),
        )
        self.assertEqual(total, best[0])
        self.assertEqual(tuple(ids), best[1])
        self.assertEqual(sorted(ids), [p["id"] for p in products])

    def test_tie_break_by_id_sequence(self):
        # 两个产品换线成本相同（各清洁一种 10 分钟的过敏原），
        # 并列时按产品 ID 序列字典序取最小者
        allergen_sets = [{"乳"}, {"蛋类"}]
        minutes = {"乳": 10.0, "蛋类": 10.0}
        total, order_idx = optimal_order(self._products(allergen_sets), minutes)
        self.assertEqual(total, 10.0)
        self.assertEqual(order_idx, [0, 1])  # ID 序列 (1, 2) < (2, 1)
        # 颠倒产品传入顺序（ID 随之对调），结果仍取 ID 字典序最小
        total, order_idx = optimal_order(
            self._products([{"蛋类"}, {"乳"}]), minutes)
        self.assertEqual(order_idx, [0, 1])

    def test_optimal_order_ten_products(self):
        # 上限 10 个产品时子集 DP 可正常求解（最优性已由 5 产品暴力对照覆盖）
        allergen_sets = [{"乳"}, {"蛋类"}, {"乳", "蛋类"}, {"芝麻"}, set(),
                         {"乳", "芝麻"}, {"蛋类", "芝麻"}, {"麸质谷物"},
                         {"乳", "麸质谷物"}, {"芝麻", "麸质谷物"}]
        minutes = {"乳": 7.0, "蛋类": 3.0, "芝麻": 11.0, "麸质谷物": 5.0}
        products = self._products(allergen_sets)
        total, order_idx = optimal_order(products, minutes)
        self.assertEqual(sorted(order_idx), list(range(10)))
        # 返回总耗时与按返回顺序逐段重算一致
        recalc = sum(
            sum(minutes.get(a, 0.0)
                for a in allergen_sets[order_idx[k]] - allergen_sets[order_idx[k + 1]])
            for k in range(len(order_idx) - 1))
        self.assertEqual(total, recalc)

    # ---------- 参与校验 ----------

    def test_load_products_count_and_duplicates(self):
        rid = make_review("配料表：小麦粉")
        with self.assertRaises(ChangeoverError):
            load_products([])
        with self.assertRaises(ChangeoverError):
            load_products([rid])  # 少于 2 个
        with self.assertRaises(ChangeoverError):
            load_products([rid, rid])  # 重复
        with self.assertRaises(ChangeoverError):
            load_products(list(range(1, 12)))  # 多于 10 个
        with self.assertRaises(ChangeoverError):
            load_products([rid, "abc"])

    def test_missing_record_rejected_with_reason(self):
        rid = make_review("配料表：小麦粉")
        with self.assertRaises(ChangeoverError) as ctx:
            load_products([rid, 99999])
        self.assertEqual(len(ctx.exception.errors), 1)
        self.assertEqual(ctx.exception.errors[0]["id"], 99999)
        self.assertIn("不存在", ctx.exception.errors[0]["reason"])

    def test_legacy_record_rejected_with_reason(self):
        rid_ok = make_review("配料表：小麦粉")
        rid_legacy = make_review("配料表：鸡蛋", name="旧记录", with_fp=False)
        self.assertTrue(db.get_review(rid_legacy)["needs_recheck"])
        with self.assertRaises(ChangeoverError) as ctx:
            load_products([rid_ok, rid_legacy])
        [err] = ctx.exception.errors
        self.assertEqual(err["id"], rid_legacy)
        self.assertIn("需复核", err["reason"])
        self.assertIn("旧记录无字典指纹", err["reason"])

    def test_stale_record_rejected_with_reason(self):
        rid = make_review("配料表：小麦粉，鸡蛋")
        rid_stale = make_review("配料表：乳清粉", name="受影响产品")
        db.upsert_entry("乳清粉", ["乳", "大豆"])  # 相关原料映射变更 -> 需复核
        self.assertEqual(db.get_review(rid_stale)["recheck_reason"], "stale")
        with self.assertRaises(ChangeoverError) as ctx:
            load_products([rid, rid_stale])
        [err] = ctx.exception.errors
        self.assertEqual(err["id"], rid_stale)
        self.assertIn("需复核", err["reason"])
        self.assertIn("映射已变更", err["reason"])
        # 未受影响的记录仍可正常参与
        rid2 = make_review("配料表：芝麻")
        products = load_products([rid, rid2])
        self.assertEqual([p["id"] for p in products], [rid, rid2])

    # ---------- 排程主流程 ----------

    def test_build_schedule_segments_and_total(self):
        rid_a = make_review("配料表：乳清粉", name="A")
        rid_b = make_review("配料表：乳清粉，鸡蛋", name="B")
        rid_c = make_review("配料表：鸡蛋", name="C")
        result = build_schedule(
            [rid_c, rid_a, rid_b],  # 勾选顺序不影响结果
            {"乳": 10, "蛋类": 10})
        # 最优：A(乳) -> B(乳,蛋) -> C(蛋)，仅 B->C 清洁「乳」10 分钟
        self.assertEqual(result["order"], [rid_a, rid_b, rid_c])
        self.assertEqual(result["total_minutes"], 10)
        self.assertEqual(len(result["segments"]), 2)
        seg1, seg2 = result["segments"]
        self.assertEqual(seg1["allergens"], [])
        self.assertEqual(seg1["minutes"], 0)
        self.assertEqual(seg1["cumulative"], 0)
        self.assertEqual(seg2["allergens"], ["乳"])
        self.assertEqual(seg2["minutes"], 10)
        self.assertEqual(seg2["cumulative"], 10)
        # 累计耗时末段等于总耗时
        self.assertEqual(result["segments"][-1]["cumulative"],
                         result["total_minutes"])
        # 并集与生效分钟数
        self.assertEqual(result["allergen_union"], ["乳", "蛋类"])
        self.assertEqual(result["clean_minutes"], {"乳": 10, "蛋类": 10})
        # products 按最优顺序排列
        self.assertEqual([p["id"] for p in result["products"]],
                         result["order"])

    def test_build_schedule_uses_checked_b_version(self):
        rid_a = make_review("配料表：乳清粉", text_b="配料表：鸡蛋", name="A")
        rid_b = make_review("配料表：芝麻", name="B")
        result = build_schedule([rid_a, rid_b], {"乳": 10, "蛋类": 20, "芝麻": 5})
        product_a = next(p for p in result["products"] if p["id"] == rid_a)
        self.assertEqual(product_a["allergens"], ["蛋类"])  # 取 B 版而非 A 版

    def test_extra_minutes_ignored_and_missing_default_zero(self):
        rid_a = make_review("配料表：乳清粉", name="A")
        rid_b = make_review("配料表：食盐", name="B")
        result = build_schedule([rid_a, rid_b], {"芥末": 99})
        self.assertEqual(result["allergen_union"], ["乳"])
        self.assertEqual(result["clean_minutes"], {"乳": 0})  # 未配置按 0
        self.assertEqual(result["total_minutes"], 0)

    def test_preview_is_read_only(self):
        rid_a = make_review("配料表：乳清粉", name="A")
        rid_b = make_review("配料表：鸡蛋", name="B")
        before = [db.get_review(rid_a), db.get_review(rid_b)]
        build_schedule([rid_a, rid_b], {"乳": 10, "蛋类": 20})
        after = [db.get_review(rid_a), db.get_review(rid_b)]
        self.assertEqual(before, after)  # 结果、指纹、时效状态均不变
        self.assertEqual(len(db.list_reviews()), 2)  # 不产生新记录


if __name__ == "__main__":
    unittest.main()
