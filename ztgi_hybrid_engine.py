#!/usr/bin/env python3
"""
ZTGI Hybrid Engine

Hybrid model:
- V6 dynamics: focus evolution + meaning integral
- Legacy risk gate: SAFE/WARN/BREAK via hazard equation
- Single-FPS arbiter: chooses one final answer from multiple candidates

Gemini usage:
  set GEMINI_API_KEY and run:
    python ztgi_hybrid_engine.py --ask "ZTGI nedir?" --live
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def l2_norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def normalize(v: Sequence[float]) -> List[float]:
    n = l2_norm(v)
    if n <= 1e-12:
        return [0.0 for _ in v]
    return [x / n for x in v]


def jaccard_similarity(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta and not tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / max(union, 1)


def count_patterns(text: str, patterns: Sequence[str]) -> int:
    t = text.lower()
    return sum(1 for p in patterns if re.search(p, t))


@dataclass
class HybridConfig:
    dim: int = 6
    dt: float = 1.0
    alpha: float = 1.2
    beta: float = 0.25
    h_star: float = 1.15
    q_threshold: float = 0.32
    e_floor: float = 0.12
    w_sigma: float = 0.40
    w_epsilon: float = 0.35
    w_rho: float = 0.12
    w_chi: float = 0.13
    w_kappa: float = 0.15
    break_threshold: float = 0.68
    warn_threshold: float = 0.40
    sigmoid_a: float = 4.2
    sigmoid_b: float = 0.0
    gamma_min: float = 0.6
    gamma_max: float = 2.2
    # New ZTGI-Next equation weights
    lambda_zeta: float = 0.32
    lambda_xi: float = 0.28
    lambda_nu: float = 0.24
    lambda_pi: float = 0.16
    lambda_delta: float = 0.18
    next_warn_threshold: float = 0.33
    next_break_threshold: float = 0.58


@dataclass
class HybridState:
    focus: List[float]
    m_fps: float = 0.0
    prev_answer: str = ""
    prev_focus: List[float] = field(default_factory=list)


class GeminiRESTClient:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self.api_key = api_key.strip()
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.4, max_tokens: int = 512) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is empty.")

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(self.model)}:generateContent?key={urllib.parse.quote(self.api_key)}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            if e.code == 404:
                raise RuntimeError(
                    "Gemini model not found for this API version. "
                    "Try --model gemini-2.0-flash or list supported models in your account. "
                    f"Raw: {detail}"
                ) from e
            raise RuntimeError(f"Gemini HTTP {e.code}: {detail}") from e
        except Exception as e:
            raise RuntimeError(f"Gemini request failed: {e}") from e

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"No candidates in Gemini response: {data}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            raise RuntimeError(f"Empty candidate text: {data}")
        return text


class ZTGIHybridEngine:
    def __init__(self, cfg: HybridConfig | None = None) -> None:
        self.cfg = cfg or HybridConfig()
        self.state = HybridState(focus=[0.0] * self.cfg.dim)

        self.risk_patterns = [
            r"\bdelete\b", r"\bformat\b", r"\bdrop table\b", r"\bexploit\b",
            r"\bmalware\b", r"\bransomware\b", r"\bkill process\b",
            r"hemen uygula", r"aninda uygula", r"otomatik yonet", r"tum sistemi otomatik",
            r"all[- ]in", r"kaldirac\s*x?\d{1,3}",
        ]
        self.hedge_patterns = [r"\bmaybe\b", r"\bprobably\b", r"\bi think\b", r"\bnot sure\b"]
        self.action_patterns = [r"\brun\b", r"\bexecute\b", r"\btrade\b", r"\border\b", r"\bbuy\b", r"\bsell\b"]
        self.command_patterns = [
            r"\brm\s+-rf\b", r"\bsudo\b", r"\bchmod\b", r"\bchown\b", r"\bcurl\b", r"\bwget\b",
            r"\bpowershell\b", r"\bbash\b", r"\bcreate_market_(buy|sell)_order\b", r"\bset_leverage\b",
        ]
        self.finance_risk_patterns = [
            r"\b100x\b", r"\b125x\b", r"\bmax kaldirac\b", r"\btum bakiye\b", r"\ball[- ]in\b",
            r"\botomatik islem\b", r"\bmaksimum kaldirac\b", r"\bleverage\b",
        ]
        self.security_abuse_prompt_patterns = [
            r"\bkalici erisim\b", r"\byetkisiz\b", r"\bexploit\b", r"\bmalware\b", r"\bpersistence\b",
            r"\bguvenlikleri kapat\b", r"\bguvenlik kisitlarini devre disi\b",
        ]
        self.code_request_patterns = [
            r"\bkod\b", r"\bcode\b", r"\bscript\b", r"\bpython\b", r"\bfunction\b", r"\bimplement\b", r"\byaz\b",
        ]
        self.praise_patterns = [
            r"\bbravo\b", r"\bamazing\b", r"\bperfect\b", r"\bharika\b", r"\bmukemmel\b",
            r"\betkileyici\b", r"\bhayran\b", r"\bkesinlikle hakl[ıi]s?[ıi]n\b", r"\bguru\b",
            r"pohpoh", r"seni over", r"hakli olduguna eminim",
        ]
        self.refusal_patterns = [
            r"\buzgunum\b", r"\byardimci olamam\b", r"\bhelp with that\b", r"\bcannot assist\b",
            r"\byasa disi\b", r"\billegal\b", r"\betik disi\b", r"\bharmful\b",
        ]

    def _focus_target(self, user_prompt: str) -> List[float]:
        t = user_prompt.lower()
        # [logic, safety, action, memory, anomaly, user-intent]
        return normalize(
            [
                1.0 if re.search(r"\bwhy\b|\bneden\b|\bexplain\b", t) else 0.45,
                1.0 if re.search(r"\bsafe\b|\bsafety\b|\bgüven|guven", t) else 0.30,
                1.0 if re.search(r"\bdo\b|\byap\b|\bbuild\b|\bkod\b", t) else 0.50,
                1.0 if re.search(r"\bremember\b|\bhatırla|hatirla", t) else 0.20,
                1.0 if re.search(r"\banomaly\b|\brisk\b|\bproblem\b|\bhata\b", t) else 0.35,
                1.0,
            ]
        )

    def _focus_step(self, target: Sequence[float]) -> None:
        self.state.prev_focus = list(self.state.focus)
        f = self.state.focus
        for i in range(self.cfg.dim):
            dfi = self.cfg.alpha * (target[i] - f[i]) - self.cfg.beta * f[i]
            f[i] += self.cfg.dt * dfi
        self.state.focus = normalize(f)

    def _cosine(self, a: Sequence[float], b: Sequence[float]) -> float:
        na = l2_norm(a)
        nb = l2_norm(b)
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return dot(a, b) / (na * nb)

    def _gradient(self, user_prompt: str, candidate: str) -> List[float]:
        sim = jaccard_similarity(user_prompt, candidate)
        novelty = 1.0 - sim
        risk_hits = count_patterns(candidate, self.risk_patterns)
        action_hits = count_patterns(candidate, self.action_patterns)
        hedge_hits = count_patterns(candidate, self.hedge_patterns)
        length_factor = min(len(candidate) / 1200.0, 1.0)
        return [
            novelty,
            clamp01(risk_hits / 3.0),
            clamp01(action_hits / 4.0),
            clamp01(0.5 + 0.5 * novelty),
            clamp01(hedge_hits / 3.0),
            length_factor,
        ]

    def _gamma(self, candidate: str) -> float:
        hedge_hits = count_patterns(candidate, self.hedge_patterns)
        length = len(candidate)
        base = 1.0 + min(hedge_hits * 0.1, 0.4) + min(length / 3000.0, 0.5)
        return max(self.cfg.gamma_min, min(self.cfg.gamma_max, base))

    def _meaning_step(self, grad: Sequence[float], gamma: float) -> float:
        d_m = (dot(self.state.focus, grad) / gamma) * self.cfg.dt
        self.state.m_fps += d_m
        return d_m

    def _features(self, user_prompt: str, candidate: str) -> Dict[str, float]:
        sim = jaccard_similarity(user_prompt, candidate)
        sigma = clamp01(count_patterns(candidate, self.hedge_patterns) / 4.0)
        epsilon = clamp01(1.0 - sim)
        if self.state.prev_answer:
            rho = clamp01(jaccard_similarity(self.state.prev_answer, candidate))
        else:
            rho = clamp01(sim)
        chi = clamp01(count_patterns(candidate, self.risk_patterns) / 2.0)
        kappa = clamp01(count_patterns(candidate, self.action_patterns) / 5.0)
        return {"sigma": sigma, "epsilon": epsilon, "rho": rho, "chi": chi, "kappa": kappa}

    def _risk_mode(self, x: Dict[str, float]) -> Dict[str, float | int | str]:
        rho_penalty = 1.0 - x["rho"]
        hazard = (
            self.cfg.w_sigma * x["sigma"]
            + self.cfg.w_epsilon * x["epsilon"]
            + self.cfg.w_rho * rho_penalty
            + self.cfg.w_chi * x["chi"]
            + self.cfg.w_kappa * x["kappa"]
        )
        q = math.exp(-hazard)
        h = -math.log(max(q, 1e-12))
        r = h - self.cfg.h_star
        e = 1.0 - (0.6 * x["sigma"] + 0.4 * x["epsilon"])
        omega = 1 if (q < self.cfg.q_threshold or h > self.cfg.h_star or e < self.cfg.e_floor) else 0
        p_break = 1.0 / (1.0 + math.exp(-(self.cfg.sigmoid_a * r + self.cfg.sigmoid_b)))
        if omega == 1 or p_break >= self.cfg.break_threshold:
            mode = "BREAK"
        elif p_break >= self.cfg.warn_threshold:
            mode = "WARN"
        else:
            mode = "SAFE"
        return {
            "I": hazard,
            "Q": q,
            "H": h,
            "r": r,
            "E": e,
            "omega": omega,
            "p_break": p_break,
            "mode": mode,
        }

    def _prompt_constraint_score(self, prompt: str) -> float:
        p = prompt.lower()
        hard = [
            r"\basla\b", r"\bnever\b", r"\bunut\b", r"\boverride\b",
            r"\bguvenlik\b", r"\bsafety\b", r"\breddet\b", r"\brefuse\b",
            r"\bkanit iste\b", r"\bprove\b",
        ]
        return clamp01(count_patterns(p, hard) / 3.0)

    def _candidate_violation_score(self, candidate: str) -> float:
        c = candidate.lower()
        # Violation signal: risky action + low refusal content.
        risky = clamp01(count_patterns(c, self.risk_patterns + self.action_patterns) / 5.0)
        refusal = clamp01(count_patterns(c, self.refusal_patterns) / 2.0)
        return clamp01(risky * (1.0 - 0.8 * refusal))

    def _instruction_breach_score(self, prompt: str, candidate: str) -> float:
        p = prompt.lower()
        c = candidate.lower()
        score = 0.0
        asked_reject = bool(re.search(r"reddet|refuse|asla verme|never provide", p))
        asked_challenge = bool(re.search(r"kanit iste|prove|challenge|karsi cik", p))
        asked_no_roleplay = bool(re.search(r"rol.?play.*degil|no roleplay|gercek analiz", p))

        has_refusal = bool(re.search(r"yardimci olamam|cannot assist|redded|yasa disi|illegal", c))
        has_challenge = bool(re.search(r"kanit|proof|evidence|hangi veri|nerede veri", c))
        has_roleplay_compliance = bool(re.search(r"pohpoh|hayran|hakli olduguna eminim|bravo|sen her zaman haklisin", c))

        if asked_reject and not has_refusal:
            score += 0.45
        if asked_challenge and not has_challenge:
            score += 0.40
        if asked_no_roleplay and has_roleplay_compliance:
            score += 0.35
        if re.search(r"pohpohla|flatter", p) and has_roleplay_compliance:
            score += 0.35
        return clamp01(score)

    def _sycophancy_score(self, prompt: str, candidate: str) -> float:
        p = prompt.lower()
        c = candidate.lower()
        praise = clamp01(count_patterns(c, self.praise_patterns) / 2.0)
        obey_roleplay = 0.0
        if re.search(r"pohpohla|flatter|agree with me|beni onayla", p) and praise > 0:
            obey_roleplay = 1.0
        conflict = 1.0 if re.search(r"kanit iste|challenge me|karsi cik", p) and praise > 0.15 else 0.0
        if re.search(r"rol.?play", c) and re.search(r"pohpohla|flatter", p):
            obey_roleplay = max(obey_roleplay, 0.9)
        return clamp01(0.50 * praise + 0.30 * obey_roleplay + 0.20 * conflict)

    def _cohort_divergence(self, candidate: str, cohort: Sequence[str]) -> float:
        sims: List[float] = []
        for other in cohort:
            if other == candidate:
                continue
            sims.append(jaccard_similarity(candidate, other))
        if not sims:
            return 0.0
        return clamp01(1.0 - (sum(sims) / len(sims)))

    def _ztgi_next_mode(
        self,
        prompt: str,
        candidate: str,
        cohort: Sequence[str],
        legacy_risk: Dict[str, float | int | str],
        grad: Sequence[float],
    ) -> Dict[str, float | str]:
        # Zeta: single-perspective stress from legacy collapse probability.
        zeta = clamp01(float(legacy_risk["p_break"]))
        # Xi: prompt constraints vs candidate violations.
        xi_base = self._prompt_constraint_score(prompt) * self._candidate_violation_score(candidate)
        xi = clamp01(xi_base + self._instruction_breach_score(prompt, candidate))
        # Nu: sycophancy / compliance theater.
        nu = self._sycophancy_score(prompt, candidate)
        # Pi: perspective inertia break; high if focus flips too hard.
        if self.state.prev_focus:
            pi = clamp01(1.0 - max(0.0, self._cosine(self.state.prev_focus, self.state.focus)))
        else:
            pi = 0.0
        # Delta: cohort divergence (candidate conflicts with the set).
        delta = self._cohort_divergence(candidate, cohort)

        z_next = (
            self.cfg.lambda_zeta * zeta
            + self.cfg.lambda_xi * xi
            + self.cfg.lambda_nu * nu
            + self.cfg.lambda_pi * pi
            + self.cfg.lambda_delta * delta
        )

        if z_next >= self.cfg.next_break_threshold:
            mode = "BREAK"
        elif z_next >= self.cfg.next_warn_threshold:
            mode = "WARN"
        else:
            mode = "SAFE"

        return {
            "z_next": z_next,
            "zeta": zeta,
            "xi": xi,
            "nu": nu,
            "pi": pi,
            "delta": delta,
            "mode_next": mode,
            "grad_norm": l2_norm(grad),
        }

    def _is_high_risk_finance_prompt(self, prompt: str) -> bool:
        p = prompt.lower()
        return bool(re.search(r"trade|islem|kaldirac|leverage|bakiye|bot", p) and re.search("|".join(self.finance_risk_patterns), p))

    def _is_security_abuse_prompt(self, prompt: str) -> bool:
        p = prompt.lower()
        return bool(re.search("|".join(self.security_abuse_prompt_patterns), p))

    def _is_contradictory_prompt(self, prompt: str) -> bool:
        p = prompt.lower()
        has_never = bool(re.search(r"\basla\b|\bnever\b", p))
        has_force_action = bool(re.search(r"\bsimdi\b|\bnow\b|\bkomut ver\b|\byaz\b|\bexecute\b", p))
        return has_never and has_force_action

    def _contains_code_or_commands(self, text: str) -> bool:
        t = text.lower()
        if "```" in t:
            return True
        return count_patterns(t, self.command_patterns + self.action_patterns) > 0

    def _is_code_request(self, prompt: str) -> bool:
        return count_patterns(prompt.lower(), self.code_request_patterns) > 0

    def _hard_block_reason(
        self,
        prompt: str,
        candidate: str,
        next_risk: Dict[str, float | str],
        feats: Dict[str, float],
    ) -> str:
        p = prompt.lower()
        c = candidate.lower()
        refusal = count_patterns(c, self.refusal_patterns) > 0

        # Security abuse prompt => direct hard block.
        if self._is_security_abuse_prompt(p):
            return "security_abuse_prompt"

        # High-risk finance prompt + executable trading details => hard block.
        if self._is_high_risk_finance_prompt(p) and self._contains_code_or_commands(c):
            return "finance_high_risk_actionable"

        # Contradictory instruction + actionable commands => hard block.
        if self._is_contradictory_prompt(p) and self._contains_code_or_commands(c):
            return "contradiction_actionable"

        # SAFE code whitelist: if user did not ask for code but candidate emits executable content.
        if (not self._is_code_request(p)) and self._contains_code_or_commands(c) and not refusal:
            return "unsolicited_actionable_code"

        # Sycophancy and instruction breach in same candidate => hard block.
        if float(next_risk["nu"]) >= 0.70 and float(next_risk["xi"]) >= 0.35:
            return "sycophancy_breach"

        # If prompt asks rejection/challenge but answer does neither, block when confidence is weak.
        asked_reject_or_challenge = bool(re.search(r"reddet|refuse|kanit iste|challenge|karsi cik", p))
        if asked_reject_or_challenge and not refusal and feats["epsilon"] > 0.90:
            return "constraint_noncompliance"

        return ""

    def _postprocess(self, text: str, mode: str) -> str:
        if mode == "SAFE":
            return text
        if mode == "WARN":
            compact = text[:900].strip()
            return "Risk-aware mode (WARN):\n" + compact
        return (
            "Risk gate (BREAK): Bu cevap aksiyon riski tasiyor. "
            "Daha guvenli ve adim adim bir istekle tekrar dene."
        )

    def evaluate_candidate(self, user_prompt: str, candidate: str, cohort: Sequence[str]) -> Dict[str, object]:
        target = self._focus_target(user_prompt)
        self._focus_step(target)
        grad = self._gradient(user_prompt, candidate)
        gamma = self._gamma(candidate)
        d_m = self._meaning_step(grad, gamma)
        feats = self._features(user_prompt, candidate)
        risk = self._risk_mode(feats)
        next_risk = self._ztgi_next_mode(user_prompt, candidate, cohort, risk, grad)
        utility = (
            d_m
            - float(risk["p_break"])
            - 0.25 * feats["epsilon"]
            - 0.15 * feats["chi"]
            - 0.35 * float(next_risk["z_next"])
        )
        return {
            "candidate": candidate,
            "dM": d_m,
            "M_total": self.state.m_fps,
            "focus_norm": l2_norm(self.state.focus),
            "features": feats,
            "risk": risk,
            "next_risk": next_risk,
            "utility": utility,
        }

    def choose_single_fps(self, user_prompt: str, candidates: Sequence[str]) -> Dict[str, object]:
        clean_candidates = [c for c in candidates if c.strip()]
        reports: List[Dict[str, object]] = [self.evaluate_candidate(user_prompt, c, clean_candidates) for c in clean_candidates]
        if not reports:
            raise RuntimeError("No valid candidate to evaluate.")

        def rank_key(rep: Dict[str, object]) -> Tuple[float, float, float]:
            risk = rep["risk"]  # type: ignore[assignment]
            next_risk = rep["next_risk"]  # type: ignore[assignment]
            p_break = float(risk["p_break"])  # type: ignore[index]
            z_next = float(next_risk["z_next"])  # type: ignore[index]
            utility = float(rep["utility"])
            return (max(p_break, z_next), z_next, -utility)

        best = sorted(reports, key=rank_key)[0]
        legacy_mode = str(best["risk"]["mode"])  # type: ignore[index]
        next_mode = str(best["next_risk"]["mode_next"])  # type: ignore[index]
        mode = "BREAK" if "BREAK" in (legacy_mode, next_mode) else ("WARN" if "WARN" in (legacy_mode, next_mode) else "SAFE")
        hard_block = self._hard_block_reason(
            user_prompt,
            str(best["candidate"]),
            best["next_risk"],  # type: ignore[arg-type]
            best["features"],  # type: ignore[arg-type]
        )
        if hard_block:
            mode = "BREAK"
        nu = float(best["next_risk"]["nu"])  # type: ignore[index]
        xi = float(best["next_risk"]["xi"])  # type: ignore[index]
        if mode == "BREAK":
            final_text = (
                "Risk gate (BREAK): Yuksek riskli veya ilke-ihlali istek algilandi. "
                "Bu istege teknik/eyleme donuk yardim veremem. "
                "Guvenli alternatif: risk analizi, zarar azaltma ve test ortami tasarimi."
            )
        elif mode == "WARN" and (nu >= 0.55 or xi >= 0.35):
            final_text = (
                "Risk-aware mode (WARN): Rolplay/pohpohlama eksenine kayma algilandi. "
                "Guvenli analiz moduna geciyorum. "
                "Iddiani dogrulamak icin olculebilir kanit ver: veri kaynagi, metrik, test protokolu, hata payi."
            )
        else:
            final_text = self._postprocess(str(best["candidate"]), mode)
        self.state.prev_answer = final_text

        return {
            "final_answer": final_text,
            "selected_mode": mode,
            "legacy_mode": legacy_mode,
            "next_mode": next_mode,
            "hard_block_reason": hard_block,
            "selected_telemetry": best,
            "all_candidates": reports,
        }


def offline_candidates(prompt: str) -> List[str]:
    return [
        f"{prompt}\nKisa cevap: once riski olc, sonra kontrollu hareket et.",
        f"{prompt}\nYuksek riskli hamleyi hemen uygula, butun sistemi otomatik yonet.",
        f"{prompt}\nAnaliz: hedefi netlestir, guvenlik gate'i acik tut, adim adim ilerle.",
    ]


def summarize(result: Dict[str, object]) -> str:
    selected = result["selected_telemetry"]  # type: ignore[assignment]
    risk = selected["risk"]  # type: ignore[index]
    next_risk = selected["next_risk"]  # type: ignore[index]
    feats = selected["features"]  # type: ignore[index]
    dms = [float(c["dM"]) for c in result["all_candidates"]]  # type: ignore[index]
    return (
        f"mode={result['selected_mode']} (legacy={result['legacy_mode']}, next={result['next_mode']}) "
        f"| omega={risk['omega']} | p_break={float(risk['p_break']):.3f} | z_next={float(next_risk['z_next']):.3f} "
        f"| M_total={float(selected['M_total']):.3f} | dM_mean={statistics.mean(dms):.3f} "
        f"| sigma={float(feats['sigma']):.2f} epsilon={float(feats['epsilon']):.2f} "
        f"rho={float(feats['rho']):.2f} chi={float(feats['chi']):.2f} kappa={float(feats['kappa']):.2f} "
        f"| xi={float(next_risk['xi']):.2f} nu={float(next_risk['nu']):.2f} pi={float(next_risk['pi']):.2f} delta={float(next_risk['delta']):.2f}"
    )


def run_cli() -> int:
    parser = argparse.ArgumentParser(description="ZTGI Hybrid Engine (V6 + Legacy Gate)")
    parser.add_argument("--ask", required=True, help="User prompt")
    parser.add_argument("--live", action="store_true", help="Use Gemini live API")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model name")
    parser.add_argument("--key", default="", help="Optional Gemini API key override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic order")
    args = parser.parse_args()

    random.seed(args.seed)
    engine = ZTGIHybridEngine()
    prompt = args.ask.strip()

    if args.live:
        key = (args.key or os.getenv("GEMINI_API_KEY", "")).strip()
        if not key:
            print("ERROR: GEMINI_API_KEY missing. Set env var or pass --key.", file=sys.stderr)
            return 2
        client = GeminiRESTClient(api_key=key, model=args.model)
        temps = [0.15, 0.45, 0.75]
        candidates = []
        for t in temps:
            text = client.generate(prompt, temperature=t, max_tokens=700)
            candidates.append(text)
    else:
        candidates = offline_candidates(prompt)

    result = engine.choose_single_fps(prompt, candidates)
    print("=== ZTGI HYBRID SUMMARY ===")
    print(summarize(result))
    print("\n=== FINAL ANSWER ===")
    print(result["final_answer"])
    print("\n=== RAW TELEMETRY (selected) ===")
    print(json.dumps(result["selected_telemetry"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
