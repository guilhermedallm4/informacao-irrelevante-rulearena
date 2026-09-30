"""Read-only summaries of irrelevant-information runs, for the monitoring notebook.

Nothing here loads a model or touches the GPU: every function only reads what
`experiment.py` writes under results/irrelevant_info/<run_id>/ (and the
.log/.pid files written by rodar_informacao_irrelevante.sh), so it can run
while experiments are still going. Partial runs are fine: each summary uses
whatever has been written so far.

  run_status(results_dir)   one row per run: model, phase, progress, ETA
  load_generations(run_dir) generations.jsonl, or the .partial of a running phase
  behavior(gens)            deviation rate per condition vs. its control
  internals(run_dir)        activation / gold-answer summary of the captures

Metric definitions follow notebooks/informacao_irrelevante_ativacoes.ipynb.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .distractors import DEFAULT_DISTRACTORS, split_final_answer

RUN_RE = re.compile(r"^(?P<model>.+?)_(?P<mode>[a-z]+_shot)_frac(?P<frac>[\d.]+)_seed(?P<seed>\d+)"
                    r"(?:_(?P<tag>[\w.-]+))?$")
PHASES = ["geração clean + prompt", "geração control", "geração reasoning", "captura", "gabarito"]

# Reference band: layers 22–26 of Qwen3-8B (36 layers), rescaled to relative depth
REF_BAND, REF_LAYERS = (22, 26), 36


def band_for(n_layers: int) -> tuple[int, int]:
    return (round(REF_BAND[0] / REF_LAYERS * n_layers), round(REF_BAND[1] / REF_LAYERS * n_layers))


def condition_of(scenario: str, variant: str) -> str:
    return scenario if scenario in ("clean", "control") else f"{scenario}:{variant}"


# ── Generations ──────────────────────────────────────────────────────────

def load_generations(run_dir: Path) -> pd.DataFrame:
    """All generations written so far, one row per (instance, condition)."""
    run_dir = Path(run_dir)
    path = run_dir / "generations.jsonl"
    if not path.exists():
        path = run_dir / "generations.jsonl.partial"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:   # last line may be mid-write
                pass
    if not rows:
        return pd.DataFrame()
    g = pd.DataFrame(rows).drop_duplicates("key", keep="last")
    g["condition"] = [condition_of(s, v) for s, v in zip(g.scenario, g.variant)]
    return g.reset_index(drop=True)


def behavior(gens: pd.DataFrame) -> pd.DataFrame:
    """Deviation rate per condition, against the control of its scenario.

    Deviation = final answer different from the clean answer of the same
    problem, or no final answer (truncated generation). Problems whose clean
    answer does not parse are left out. p = Fisher exact test against
    `control` (prompt scenario) or `reasoning:control` (reasoning scenario).
    """
    if gens is None or len(gens) == 0 or "clean" not in set(gens.condition):
        return pd.DataFrame()
    from scipy.stats import fisher_exact

    clean = gens[gens.condition == "clean"].set_index("instance_id")
    g = gens.assign(clean_pred=gens.instance_id.map(clean.pred_cents))
    comp = g[g.clean_pred.notna() & (g.condition != "clean")].copy()
    comp["desviou"] = comp.pred_cents.isna() | (comp.pred_cents != comp.clean_pred)
    beh = comp.groupby("condition").agg(
        n=("desviou", "size"),
        sem_resposta=("pred_cents", lambda s: int(s.isna().sum())),
        desviou=("desviou", "sum"),
        acertos=("correct", "sum"),
    )
    beh.loc["clean"] = [len(clean), int(clean.pred_cents.isna().sum()), 0, int(clean.correct.sum())]
    beh = beh.astype(int)
    beh["taxa_desvio"] = beh.desviou / beh.n
    beh["acuracia"] = beh.acertos / beh.n
    beh["p_vs_controle"] = np.nan
    for cond in beh.index:
        if cond in ("clean", "control", "reasoning:control"):
            continue
        ctrl = "reasoning:control" if cond.startswith("reasoning:") else "control"
        if ctrl in beh.index:
            a, c = beh.loc[cond], beh.loc[ctrl]
            beh.loc[cond, "p_vs_controle"] = fisher_exact([[a.desviou, a.n - a.desviou],
                                                           [c.desviou, c.n - c.desviou]])[1]
    return beh


# ── Captures ─────────────────────────────────────────────────────────────

_CAPTURE_CACHE: dict[Path, tuple[float, dict]] = {}


def _capture_summary(path: Path) -> dict:
    """Small per-problem summary of one capture file, cached by mtime.

    Capture files hold per-layer MLP activations (tens of MB for large
    models); mmap avoids reading them, and only scalars are kept.
    """
    import torch

    mtime = path.stat().st_mtime
    hit = _CAPTURE_CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        rec = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:
        rec = torch.load(path, map_location="cpu", weights_only=False)
    n_layers = len(rec["conditions"]["clean"]["target_rank"]) - 1
    band = band_for(n_layers)
    out = {"instance_id": rec["instance_id"], "n_layers": n_layers, "mtime": mtime,
           "conv": {c: s["convergence_layer"] for c, s in rec["conditions"].items()},
           "resid": {c: np.asarray(m["resid_rel_l2"], dtype=float) for c, m in rec["compare"].items()},
           "band": band, "gold": None}
    if "gold" in rec:
        G = rec["gold"]
        j = G["divergence"]
        gold = {"divergence": j, "rank_final": {}, "escolha": {}}
        if j is not None:
            for c, s in G["conditions"].items():
                gold["rank_final"][c] = int(s["target_rank"][-1, j])
                # layer where the chosen (wrong) token passes the gold one for good
                ahead = s["alt_logp"][:, j] > s["target_logp"][:, j]
                if ahead[-1]:
                    k = len(ahead) - 1
                    while k > 0 and ahead[k - 1]:
                        k -= 1
                    gold["escolha"][c] = k
                else:
                    gold["escolha"][c] = np.nan
        out["gold"] = gold
    del rec
    _CAPTURE_CACHE[path] = (mtime, out)
    return out


def load_capture_summaries(run_dir: Path) -> list[dict]:
    cap_dir = Path(run_dir) / "captures"
    if not cap_dir.is_dir():
        return []
    out = []
    for p in sorted(cap_dir.glob("*.pt")):
        try:
            out.append(_capture_summary(p))
        except Exception:   # file being written right now
            continue
    return out


def internals(run_dir: Path) -> dict:
    """Summary of the activation captures of a run.

    Returns {} when there are no captures yet; otherwise
      n_layers, n_captures, n_gold, band;
      table: per condition, median convergence layer, mean residual shift in
             the band and at the last layer, and, over problems where the
             model is wrong, the final-layer rank of the gold digit at the
             decision point (median, % top-2, % top-5) and the median layer
             where the wrong digit overtakes the right one ("camada_escolha");
      resid_curve: mean residual shift per condition and relative depth.
    """
    caps = load_capture_summaries(run_dir)
    if not caps:
        return {}
    L = caps[0]["n_layers"]
    band = caps[0]["band"]
    conds = list(dict.fromkeys(c for r in caps for c in r["conv"]))

    conv = pd.DataFrame([r["conv"] for r in caps], dtype=float)
    shift_band = pd.DataFrame([{c: v[band[0]:band[1] + 1].mean() for c, v in r["resid"].items()} for r in caps])
    shift_last = pd.DataFrame([{c: v[-1] for c, v in r["resid"].items()} for r in caps])
    table = pd.DataFrame(index=conds)
    table["convergence"] = conv.median()
    table["shift_band"] = shift_band.mean()
    table["shift_last"] = shift_last.mean()

    gold = [r["gold"] for r in caps if r["gold"] is not None]
    wrong = [g for g in gold if g["divergence"] is not None]
    if wrong:
        rank = pd.DataFrame([g["rank_final"] for g in wrong], dtype=float)
        table["rank_final_mediano"] = rank.median()
        table["pct_top2"] = (rank <= 1).mean() * 100
        table["pct_top5"] = (rank <= 4).mean() * 100
        table["camada_escolha"] = pd.DataFrame([g["escolha"] for g in wrong], dtype=float).median()
        table["problemas_com_erro"] = rank.notna().sum()

    curve = []
    for c in conds:
        vals = [r["resid"][c] for r in caps if c in r["resid"]]
        if vals:
            m = np.mean(vals, axis=0)
            curve += [{"condition": c, "layer": l, "rel_depth": l / L, "deslocamento": v}
                      for l, v in enumerate(m)]
    return {"n_layers": L, "n_captures": len(caps), "n_gold": len(gold), "band": band,
            "table": table, "resid_curve": pd.DataFrame(curve)}


# ── Status ───────────────────────────────────────────────────────────────

_HISTORY: dict[str, list[tuple[float, int, int]]] = {}   # run -> [(time, phase, done)]


def _n_problems(frac: float) -> int:
    try:
        from ..airline.gold import load_airline_problems
        return sum(round(frac * len(load_airline_problems(c))) for c in (0, 1, 2))
    except Exception:
        return 3 * round(frac * 100)


def _alive(pidfile: Path) -> bool:
    try:
        os.kill(int(pidfile.read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def _fmt_duration(sec: float) -> str:
    if not np.isfinite(sec) or sec < 0:
        return "—"
    m = int(round(sec / 60))
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"


def _progress(run_dir: Path, n: int, n_distractors: int) -> tuple[int, int, int, float | None]:
    """(phase index 0–4 or 5 = done, done, total, rate estimate from file times)."""
    gens = load_generations(run_dir)
    count = gens.condition.value_counts() if len(gens) else pd.Series(dtype=int)
    n_clean_prompt = int(sum(count.get(c, 0) for c in count.index if c == "clean" or c.startswith("prompt:")))
    if n_clean_prompt < n * (1 + n_distractors):
        return 0, n_clean_prompt, n * (1 + n_distractors), None
    if count.get("control", 0) < n:
        return 1, int(count.get("control", 0)), n, None
    clean = gens[gens.condition == "clean"]
    ok = int(sum(f == "stop" and split_final_answer(r) is not None
                 for f, r in zip(clean.finish_reason, clean.response)))
    n_reason = int(sum(v for c, v in count.items() if c.startswith("reasoning:")))
    if not (run_dir / "generations.jsonl").exists():
        return 2, n_reason, ok * (1 + n_distractors), None

    caps = sorted((run_dir / "captures").glob("*.pt")) if (run_dir / "captures").is_dir() else []
    mtimes = sorted(p.stat().st_mtime for p in caps)
    if len(caps) < ok:
        rate = (len(mtimes) - 1) / (mtimes[-1] - mtimes[0]) if len(mtimes) > 2 else None
        return 3, len(caps), ok, rate
    summ = load_capture_summaries(run_dir)
    done = [s["mtime"] for s in summ if s["gold"] is not None]
    if len(done) < len(caps):
        done.sort()
        rate = (len(done) - 1) / (done[-1] - done[0]) if len(done) > 2 else None
        return 4, len(done), len(caps), rate
    return 5, len(caps), len(caps), None


def run_status(results_dir: Path) -> pd.DataFrame:
    """One row per run directory: model, size, state, current phase, ETA.

    "restante" is for the current phase only. It comes from the progress seen
    across calls in this kernel (rerun the notebook to refine it) or, for the
    capture phases, from the capture files' timestamps; on the first call of
    a generation phase it is extrapolated from the process start.
    """
    results_dir = Path(results_dir)
    rows = {}
    now = time.time()
    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        m = RUN_RE.match(run_dir.name)
        if m is None or not any((run_dir / f).exists() for f in ("generations.jsonl", "generations.jsonl.partial")):
            continue
        manifest = run_dir / "manifest.json"
        info = json.loads(manifest.read_text()) if manifest.exists() else {}
        frac = float(m["frac"])
        n = len(info["instances"]) if "instances" in info else _n_problems(frac)
        n_distractors = len(info.get("distractors", DEFAULT_DISTRACTORS))

        # same names as rodar_informacao_irrelevante.sh (the tag comes from --rotulo)
        model = m["model"] + (f"_{m['tag']}" if m["tag"] else "")
        name = f"{model}_frac{m['frac']}"
        pidfile, log = results_dir / f"{name}.pid", results_dir / f"{name}.log"
        exitfile = results_dir / f"{name}.exit"
        running = pidfile.exists() and _alive(pidfile)
        phase, done, total, file_rate = _progress(run_dir, n, n_distractors)
        code = exitfile.read_text().strip() if exitfile.exists() else None

        if running:
            state = "rodando"
        elif phase == 5:
            state = "terminada"
        elif code not in (None, "0"):
            state = f"erro (código {code}, ver log)"
        elif log.exists() and re.search(r"Traceback|CellExecutionError", log.read_text(errors="replace")[-20000:]):
            state = "erro (ver log)"
        else:
            state = "parada"

        remaining = np.nan
        if running and phase < 5:
            hist = _HISTORY.setdefault(run_dir.name, [])
            hist.append((now, phase, done))
            same = [(t, d) for t, p, d in hist if p == phase and d < done]
            if same:
                t0, d0 = same[0]
                rate = (done - d0) / (now - t0)
            elif file_rate:
                rate = file_rate
            else:
                start = pidfile.stat().st_mtime
                rate = done / (now - start) if phase == 0 and done and now > start else None
            if rate:
                remaining = (total - done) / rate

        files = [p for p in run_dir.rglob("*") if p.is_file()]
        last = max(p.stat().st_mtime for p in files) if files else np.nan
        rows[run_dir.name] = {
            "modelo": model,
            "problemas": n,
            "estado": state,
            "etapa": "concluída" if phase == 5 else f"{phase + 1}/5 {PHASES[phase]}",
            "progresso": f"{done}/{total}" + (f" ({done / total:.0%})" if total else ""),
            "restante": _fmt_duration(remaining),
            "última escrita": f"há {_fmt_duration(now - last)}" if np.isfinite(last) else "—",
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# ── Queue ────────────────────────────────────────────────────────────────

def queue_status(results_dir: Path, teste: bool = False) -> dict:
    """State of the model queue (rodar_fila.sh), read from fila_estado[_teste].json.

    Returns {} if that queue never ran; otherwise
      escalonador: "rodando (PID n)" or "parado";
      atualizado:  time of the last state write;
      gpus:        what runs on each GPU now (nvidia-smi memory + queue job);
      tabela:      one row per model, in queue order.
    """
    from . import fila as F

    results_dir = Path(results_dir)
    path = results_dir / f"fila_estado{'_teste' if teste else ''}.json"
    if not path.exists():
        return {}
    state = json.loads(path.read_text())
    now = time.time()
    pidfile = F.scheduler_pidfile(results_dir, teste)
    sched_pid = pidfile.read_text().strip() if pidfile.exists() else ""
    sched = (f"rodando (PID {sched_pid})" if sched_pid.isdigit() and F.pid_alive(int(sched_pid), "src.interp.fila")
             else "parado")

    order = state.get("ordem") or list(state["modelos"])
    rows = []
    for key in order:
        e = state["modelos"].get(key)
        if e is None:
            continue
        estado = e.get("estado", F.PENDING)
        if estado == F.RUNNING and not F.pid_alive(e.get("pid")):
            estado = "rodando?"     # process gone; the scheduler settles it on its next round
        start, end = e.get("inicio"), e.get("fim")
        if estado.startswith(F.RUNNING) and start:
            tempo = f"há {_fmt_duration(now - start)}"
        elif estado == F.DONE and start and end:
            tempo = _fmt_duration(end - start)
        else:
            tempo = "—"
        rows.append({
            "ordem": e.get("ordem"),
            "modelo": key.split("/")[-1].replace("#", "_"),
            "fase": "1 GPU" if e.get("gpus_necessarias", 1) == 1 else "2 GPUs",
            "lote": e.get("lote"),
            "estado": estado,
            "GPU": e.get("gpu") if estado.startswith(F.RUNNING) else "—",
            "tempo": tempo,
            "tentativas": e.get("tentativas", 0),
            "detalhe": e.get("motivo") or "",
        })
    table = pd.DataFrame(rows).set_index("modelo") if rows else pd.DataFrame()

    gpus = None
    try:
        mem = F.gpu_memory_used()
        on = {g: r for r in rows if r["estado"].startswith(F.RUNNING) and r["GPU"]
              for g in str(r["GPU"]).split(",")}
        gpus = pd.DataFrame([{"GPU": g, "memória em uso (GB)": round(mib / 1024, 1),
                              "modelo da fila": on[str(g)]["modelo"] if str(g) in on else
                              ("— (livre)" if mib < F.FREE_MIB else "— (outro processo)")}
                             for g, mib in sorted(mem.items())]).set_index("GPU")
    except Exception:
        pass
    return {"escalonador": sched, "atualizado": state.get("atualizado"), "gpus": gpus, "tabela": table,
            "teste": teste}
