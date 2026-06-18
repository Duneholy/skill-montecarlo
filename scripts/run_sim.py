#!/usr/bin/env python3
"""Universal Monte Carlo runner for montecarlo skill."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_ITERATIONS = 100_000
_ALLOWED_FORMULA = re.compile(r"^[0-9a-zA-Z_+\-*/().\s]+$")
_ALLOWED_SUCCESS = re.compile(r"^[0-9a-zA-Z_<>=!+\-*/().\s]+$")


def _load_config(path: Path | None, json_arg: str | None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if json_arg:
        return json.loads(json_arg)
    raise ValueError("Provide --config <file.json> or a JSON string argument")


def _triangular(params: dict[str, Any], default_spread: float = 1.5) -> float:
    vmin = float(params.get("min", 0))
    vmode = float(params.get("mode", vmin))
    vmax = float(params.get("max", vmode))
    conf = float(params.get("confidence", 1.0))
    spread = float(params.get("confidence_spread", default_spread))
    w = 1.0 + (1.0 - max(0.0, min(1.0, conf))) * spread
    adj_min = vmode - (vmode - vmin) * w
    adj_max = vmode + (vmax - vmode) * w
    if adj_min > adj_max:
        adj_min, adj_max = adj_max, adj_min
    if adj_min == adj_max == vmode:
        return vmode
    return random.triangular(adj_min, adj_max, vmode)


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, (centre - margin) * 100), min(100.0, (centre + margin) * 100)


def _wide_range_warning(params: dict[str, Any]) -> bool:
    vmin = float(params.get("min", 0))
    vmode = float(params.get("mode", 1))
    vmax = float(params.get("max", vmode))
    if vmode == 0:
        return abs(vmax - vmin) > 0 and abs(vmax - vmin) > 3
    span = abs(vmax - vmin) / abs(vmode)
    return span > 3 or (vmin != 0 and abs(vmax / vmin) > 10)


def _compile_expr(expr: str, label: str) -> Any:
    pattern = _ALLOWED_SUCCESS if label == "success_condition" else _ALLOWED_FORMULA
    if not pattern.match(expr):
        raise ValueError(f"{label}: disallowed characters")
    return compile(expr, "<string>", "eval")


def _sample_risks(
    risks: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, bool]]:
    values: dict[str, float] = {}
    fired: dict[str, bool] = {}
    for i, r in enumerate(risks):
        rid = str(r.get("id") or f"risk_{i}")
        prob = float(r.get("prob", 0))
        hit = random.random() < prob
        fired[rid] = hit
        if "multiplier" in r:
            values[rid] = float(r["multiplier"]) if hit else 1.0
        else:
            values[rid] = float(r.get("impact", 0)) if hit else 0.0
    return values, fired


def run_simulation(config: dict[str, Any]) -> dict[str, Any]:
    iterations = int(config.get("iterations", DEFAULT_ITERATIONS))
    iterations = max(10_000, min(100_000, iterations))

    # Global confidence spread factor (default 1.5), overridable per variable
    default_spread = float(config.get("confidence_spread", 1.5))

    vars_config_raw = config.get("variables") or {}
    if isinstance(vars_config_raw, list):
        vars_config: dict[str, Any] = {str(v.get("id", f"var_{i}")): v for i, v in enumerate(vars_config_raw)}
    else:
        vars_config: dict[str, Any] = vars_config_raw
    risks: list[dict[str, Any]] = list(config.get("risks") or [])
    formula = str(config.get("formula", "0"))
    success_condition = str(config.get("success_condition", "result > 0"))

    try:
        compiled_formula = _compile_expr(formula, "formula")
        compiled_success = _compile_expr(success_condition, "success_condition")
    except Exception as e:
        return {"error": str(e)}

    warnings: list[str] = []
    for name, params in vars_config.items():
        if _wide_range_warning(params):
            warnings.append(f"wide range on '{name}'")

    success_count = 0
    tail_count = 0
    results: list[float] = []
    tail_cfg = config.get("tail") or {}

    factor_keys = set(vars_config.keys()) | {str(r.get("id") or f"risk_{i}") for i, r in enumerate(risks)}
    sum_success = {k: 0.0 for k in factor_keys}
    sum_fail = {k: 0.0 for k in factor_keys}
    risk_fires_fail: dict[str, int] = {
        str(r.get("id") or f"risk_{i}"): 0 for i, r in enumerate(risks)
    }
    risk_fires_success: dict[str, int] = {
        str(r.get("id") or f"risk_{i}"): 0 for i, r in enumerate(risks)
    }

    for _ in range(iterations):
        env: dict[str, float] = {}
        for v_name, v_params in vars_config.items():
            env[v_name] = _triangular(v_params, default_spread)

        risk_vals, fired = _sample_risks(risks)
        env.update(risk_vals)

        try:
            res = float(eval(compiled_formula, {"__builtins__": None}, env))
            env["result"] = res
            ok = bool(eval(compiled_success, {"__builtins__": None}, env))
        except Exception as e:
            return {"error": f"Evaluation error: {e}"}

        results.append(res)
        if ok:
            success_count += 1
            for k in sum_success:
                sum_success[k] += env.get(k, 0.0)
            for rid, h in fired.items():
                if h:
                    risk_fires_success[rid] = risk_fires_success.get(rid, 0) + 1
        else:
            for k in sum_fail:
                sum_fail[k] += env.get(k, 0.0)
            for rid, h in fired.items():
                if h:
                    risk_fires_fail[rid] = risk_fires_fail.get(rid, 0) + 1

        if "threshold" in tail_cfg:
            t = float(tail_cfg["threshold"])
            op = tail_cfg.get("op", ">")
            if op == "<" and res < t:
                tail_count += 1
            elif op == "<=" and res <= t:
                tail_count += 1
            elif op == ">=" and res >= t:
                tail_count += 1
            elif op == ">" and res > t:
                tail_count += 1

    fail_count = iterations - success_count
    ci_low, ci_high = _wilson_ci(success_count, iterations)

    critical_factors: list[dict[str, Any]] = []
    if success_count > 0 and fail_count > 0:
        for k in factor_keys:
            avg_s = sum_success[k] / success_count
            avg_f = sum_fail[k] / fail_count
            denom = (abs(avg_s) + abs(avg_f)) / 2
            if denom <= 0:
                continue
            diff_pct = abs(avg_s - avg_f) / denom
            if diff_pct > 0.1:
                critical_factors.append({
                    "name": k,
                    "avg_when_success": round(avg_s, 2),
                    "avg_when_fail": round(avg_f, 2),
                })
        for rid, fail_n in risk_fires_fail.items():
            fail_rate = fail_n / fail_count
            succ_rate = risk_fires_success.get(rid, 0) / success_count
            if fail_rate - succ_rate > 0.05:
                critical_factors.append({
                    "name": rid,
                    "fire_rate_on_failure_pct": round(fail_rate * 100, 1),
                    "fire_rate_on_success_pct": round(succ_rate * 100, 1),
                })
        critical_factors.sort(
            key=lambda x: x.get("fire_rate_on_failure_pct", x.get("avg_when_fail", 0)),
            reverse=True,
        )
        critical_factors = critical_factors[:3]

    out: dict[str, Any] = {
        "success_pct": round(success_count / iterations * 100, 1),
        "ci_95_low_pct": round(ci_low, 1),
        "ci_95_high_pct": round(ci_high, 1),
        "iterations": iterations,
        "critical_factors": critical_factors,
        "warnings": warnings,
    }
    if tail_count and tail_cfg:
        out["tail_fraction_pct"] = round(tail_count / iterations * 100, 1)
        out["tail_label"] = tail_cfg.get("label", "tail event")
    if results:
        sr = sorted(results)
        out["result_median"] = round(sr[len(sr) // 2], 2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo skill runner")
    parser.add_argument("--config", type=Path, help="Path to JSON config file")
    parser.add_argument("json", nargs="?", help="JSON config string (optional)")
    args = parser.parse_args()

    try:
        cfg = _load_config(args.config, args.json)
        res = run_simulation(cfg)
        print(json.dumps(res, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
