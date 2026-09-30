"""Model queue for the irrelevant-information study: one model per GPU, big ones last.

Reads fila_modelos.yaml and runs rodar_informacao_irrelevante.sh for each
model, in two phases:

  1. single-GPU models: both GPUs stay busy; whenever one frees up, the next
     single-GPU model (by `ordem`) starts on it (CUDA_VISIBLE_DEVICES=0 or 1);
  2. two-GPU models: only once no single-GPU model *ordered before them* is
     pending or running and both GPUs are free, one at a time, on
     CUDA_VISIBLE_DEVICES=0,1.

A pending two-GPU model is a barrier: single-GPU models with a larger `ordem`
wait for it. If such later single-GPU runs are what keeps the GPUs busy, they
are paused (preempted) right after their next save (a generation batch or a
capture file), so almost no work is lost, and resumed after the barrier.

Before launching on a GPU, nvidia-smi must show it idle, so the queue never
collides with other processes on the machine.

A run succeeds only if its exit code is 0 *and* it is complete on disk
(generations.jsonl plus one capture per problem, each with the gold readout).
A failed run is retried once (generation resumes from what was saved), then
marked as failed with the end of its log, and the queue moves on.

State lives in results/irrelevant_info/fila_estado[_teste].json. Runs are
started in their own session and record their exit code in <name>.exit, so
they survive the scheduler: a restarted scheduler adopts runs that are still
going, resumes interrupted ones and skips finished ones.

Usage: see rodar_fila.sh, or `python -m src.interp.fila --help`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "irrelevant_info"
QUEUE_FILE = ROOT / "fila_modelos.yaml"
RUN_SCRIPT = "rodar_informacao_irrelevante.sh"

FRACTION = {False: "0.30", True: "0.02"}   # must match rodar_informacao_irrelevante.sh
PROMPT_MODE, SEED = "one_shot", 42          # must match the study notebook
N_GPUS = 2
FREE_MIB = 2000        # a GPU using less memory than this counts as idle
MAX_ATTEMPTS = 2       # first run + one retry
PREEMPT_MAX_WAIT = 4 * 3600   # pause a run anyway if it saves nothing for this long

PENDING, RUNNING, DONE, FAILED, NO_ACCESS, REFUSED = (
    "pendente", "rodando", "concluído", "falhou", "sem acesso", "recusado")


# ── Queue file ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Model:
    id: str
    gpus: int
    lote: int
    ordem: int
    rotulo: str = ""
    max_tokens: int = 0     # 0 = padrão do notebook (4096)

    @property
    def key(self) -> str:
        return f"{self.id}#{self.rotulo}" if self.rotulo else self.id

    @property
    def short(self) -> str:
        return self.id.split("/")[-1]

    def name(self, teste: bool) -> str:
        """Base name of the .log/.pid/.exit files (as in the run script)."""
        return f"{self.short}{'_' + self.rotulo if self.rotulo else ''}_frac{FRACTION[teste]}"

    def run_dir(self, results: Path, teste: bool) -> Path:
        """Results directory (as in the study notebook)."""
        tag = f"_{self.rotulo}" if self.rotulo else ""
        return results / f"{self.short}_{PROMPT_MODE}_frac{FRACTION[teste]}_seed{SEED}{tag}"


def load_queue(path: Path = QUEUE_FILE) -> list[Model]:
    import yaml
    data = yaml.safe_load(Path(path).read_text()) or {}
    models = []
    for i, m in enumerate(data.get("modelos") or []):
        model = Model(id=str(m["id"]), gpus=int(m.get("gpus", 1)), lote=int(m.get("lote", 8)),
                      ordem=int(m.get("ordem", i + 1)), rotulo=str(m.get("rotulo") or ""),
                      max_tokens=int(m.get("max_tokens") or 0))
        if model.gpus not in (1, 2):
            raise ValueError(f"{model.id}: gpus deve ser 1 ou 2, não {model.gpus}")
        if model.lote < 1:
            raise ValueError(f"{model.id}: lote deve ser positivo")
        if model.max_tokens < 0:
            raise ValueError(f"{model.id}: max_tokens deve ser positivo")
        models.append(model)
    keys = [m.key for m in models]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        raise ValueError(f"modelos repetidos na fila: {sorted(dup)}")
    return sorted(models, key=lambda m: (m.ordem, m.id))


# ── Checks (hub access, architecture, completion, GPUs, processes) ───────

def check_model(model_id: str) -> tuple[str | None, str]:
    """(None, "") if the model can run; else (NO_ACCESS or REFUSED, reason).

    Downloads config.json, which tells both whether the logged-in token can
    read a gated repo and whether the model is a mixture of experts.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
    try:
        path = hf_hub_download(model_id, "config.json")
    except GatedRepoError:
        return NO_ACCESS, "repositório restrito: o token logado não tem acesso (hf auth login e aceitar a licença)"
    except RepositoryNotFoundError:
        return NO_ACCESS, "repositório não encontrado (ou restrito sem login)"
    except Exception as e:   # offline, rate limit...: use the cache if there is one
        try:
            path = hf_hub_download(model_id, "config.json", local_files_only=True)
        except Exception:
            return None, f"não foi possível verificar ({type(e).__name__}); tentando mesmo assim"
    cfg = json.loads(Path(path).read_text())
    if is_moe(cfg):
        return REFUSED, "modelo MoE: os hooks exigem MLP densa com mlp.down_proj"
    return None, ""


def is_moe(cfg: dict) -> bool:
    cfgs = [cfg] + [v for v in cfg.values() if isinstance(v, dict)]   # e.g. text_config
    for c in cfgs:
        if "moe" in str(c.get("model_type", "")).lower():
            return True
        if any("moe" in a.lower() for a in c.get("architectures") or []):
            return True
        for k in ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts"):
            if (c.get(k) or 0) > 1:
                return True
    return False


_GOLD_CACHE: dict[Path, tuple[float, bool]] = {}


def capture_has_gold(path: Path) -> bool:
    import torch
    mtime = path.stat().st_mtime
    hit = _GOLD_CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        rec = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:
        rec = torch.load(path, map_location="cpu", weights_only=False)
    ok = isinstance(rec, dict) and "gold" in rec
    del rec
    _GOLD_CACHE[path] = (mtime, ok)
    return ok


def capturable(gen_path: Path) -> set[str]:
    """Problems that get a capture: clean answer finished and parseable (as in experiment.run_capture)."""
    from .distractors import split_final_answer
    out = set()
    with open(gen_path) as f:
        for line in f:
            r = json.loads(line)
            if (r["scenario"] == "clean" and r["finish_reason"] == "stop"
                    and split_final_answer(r["response"]) is not None):
                out.add(r["instance_id"])
    return out


def run_complete(run_dir: Path) -> tuple[bool, str]:
    """Completion criterion: generations.jsonl + one capture with "gold" per capturable problem.

    Problems whose clean answer is truncated or lacks "The total cost is $X"
    have no capture by design, so they are not required.
    """
    manifest, gen_path = run_dir / "manifest.json", run_dir / "generations.jsonl"
    if not gen_path.exists() or not manifest.exists():
        return False, "gerações incompletas (sem generations.jsonl)"
    n = len(json.loads(manifest.read_text())["instances"])
    expected = capturable(gen_path)
    caps = {p.stem: p for p in (run_dir / "captures").glob("*.pt")} if (run_dir / "captures").is_dir() else {}
    missing = expected - set(caps)
    if missing:
        return False, f"{len(expected) - len(missing)} de {len(expected)} capturas"
    no_gold = sum(not capture_has_gold(caps[i]) for i in expected)
    if no_gold:
        return False, f"{no_gold} de {len(expected)} capturas sem gabarito"
    skipped = n - len(expected)
    return True, f"{n} problemas completos" + (f" ({skipped} sem resposta legível, sem captura)" if skipped else "")


def gpu_memory_used() -> dict[int, int]:
    """MiB in use per GPU, by nvidia-smi index."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True, timeout=60).stdout
    return {int(i): int(m) for i, m in (line.split(",") for line in out.strip().splitlines())}


def pid_alive(pid: int | None, marker: str = RUN_SCRIPT) -> bool:
    """True if `pid` runs and its command line contains `marker` (guards against PID reuse)."""
    if not pid:
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        if stat.rsplit(")", 1)[1].split()[0] == "Z":        # zombie: already finished
            return False
        return marker in Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, IndexError):
        return False


def descendants(pid: int) -> list[int]:
    """All descendant PIDs (the Jupyter kernel runs in its own session, so killpg misses it)."""
    children: dict[int, list[int]] = {}
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                ppid = int(p.joinpath("stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            children.setdefault(ppid, []).append(int(p.name))
    out, todo = [], [pid]
    while todo:
        for c in children.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


def log_tail(path: Path, lines: int = 12, width: int = 600) -> str:
    if not path.exists():
        return "(sem log)"
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace"))
    tail = [l for l in text.strip().splitlines() if l.strip()][-lines:]
    return " | ".join(tail)[-width:]


def progress_signature(run_dir: Path) -> tuple:
    """Changes whenever a run saves something (a generation batch or a capture)."""
    partial = run_dir / "generations.jsonl.partial"
    caps = list((run_dir / "captures").glob("*.pt")) if (run_dir / "captures").is_dir() else []
    return (partial.stat().st_size if partial.exists() else 0, len(caps),
            max((p.stat().st_mtime for p in caps), default=0))


def kill_tree(pid: int) -> None:
    """SIGKILL a run and its children (no .exit is written, so it counts as an interruption)."""
    for p in [pid] + descendants(pid):
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass


def launch_run(model: Model, gpus: str, teste: bool) -> subprocess.Popen:
    """Start the run script in the foreground mode, in its own session."""
    cmd = ["bash", RUN_SCRIPT, model.id, "--gpu", gpus, "--lote", str(model.lote), "--primeiro-plano"]
    if teste:
        cmd.append("--teste")
    if model.rotulo:
        cmd += ["--rotulo", model.rotulo]
    if model.max_tokens:
        cmd += ["--max-tokens", str(model.max_tokens)]
    return subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=sys.stdout, stderr=sys.stderr,
                            start_new_session=True)


# ── Scheduler ────────────────────────────────────────────────────────────

@dataclass
class Job:
    model: Model
    gpus: str
    pid: int
    popen: subprocess.Popen | None = None   # None: adopted after a scheduler restart

    def alive(self, pid_check: Callable[[int], bool]) -> bool:
        if self.popen is not None:
            return self.popen.poll() is None
        return pid_check(self.pid)


@dataclass
class Scheduler:
    models: list[Model]
    teste: bool = False
    results: Path = RESULTS
    state_path: Path | None = None
    launch: Callable = launch_run
    gpu_memory: Callable[[], dict[int, int]] = gpu_memory_used
    is_complete: Callable[[Path], tuple[bool, str]] = run_complete
    check: Callable[[str], tuple[str | None, str]] = check_model
    pid_check: Callable[[int], bool] = pid_alive
    now: Callable[[], float] = time.time
    log: Callable[[str], None] = print
    n_gpus: int = N_GPUS
    free_mib: int = FREE_MIB
    progress: Callable[[Path], tuple] = progress_signature
    kill: Callable[[int], None] = kill_tree
    jobs: dict[str, Job] = field(default_factory=dict)
    preempting: dict[str, tuple] = field(default_factory=dict)   # key -> (signature, since)

    def __post_init__(self):
        self.results = Path(self.results)
        if self.state_path is None:
            self.state_path = self.results / f"fila_estado{'_teste' if self.teste else ''}.json"
        self.by_key = {m.key: m for m in self.models}
        self.state = self._load()

    # state file
    def _load(self) -> dict:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
        else:
            state = {"modelos": {}}
        state["teste"] = self.teste
        state["fracao"] = FRACTION[self.teste]
        return state

    def save(self):
        self.state["atualizado"] = self.now()
        self.state["ordem"] = [m.key for m in self.models]
        for m in self.models:
            self.state["modelos"][m.key].update(id=m.id, gpus_necessarias=m.gpus, lote=m.lote,
                                                 ordem=m.ordem, rotulo=m.rotulo,
                                                 execucao=m.run_dir(self.results, self.teste).name,
                                                 log=str(self._log_path(m)))
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False))
        tmp.replace(self.state_path)

    def entry(self, m: Model) -> dict:
        return self.state["modelos"].setdefault(m.key, {"estado": PENDING, "tentativas": 0, "historico": []})

    def _log_path(self, m: Model) -> Path:
        return self.results / f"{m.name(self.teste)}.log"

    def _exit_path(self, m: Model) -> Path:
        return self.results / f"{m.name(self.teste)}.exit"

    # startup
    def prepare(self, refazer_falhas: bool = False):
        """Access/architecture checks and completion on disk, for every model not done."""
        for m in self.models:
            e = self.entry(m)
            if e["estado"] == RUNNING:
                continue
            if refazer_falhas and e["estado"] == FAILED:
                e.update(estado=PENDING, tentativas=0)
            if e["estado"] == FAILED:
                continue
            complete, detail = self.is_complete(m.run_dir(self.results, self.teste))
            if complete:
                if e["estado"] != DONE:
                    self.log(f"{m.key}: já completo no disco ({detail}); marcado como concluído")
                e.update(estado=DONE, motivo=detail)
                continue
            if e["estado"] == DONE:     # results removed since: run it again
                e["estado"] = PENDING
            status, reason = self.check(m.id)
            if status is not None:
                if e["estado"] != status:
                    self.log(f"{m.key}: {status} ({reason})")
                e.update(estado=status, motivo=reason)
            elif e["estado"] in (NO_ACCESS, REFUSED):   # access granted since last time
                e.update(estado=PENDING, motivo=reason)
            elif reason:
                e["motivo"] = reason
        self.save()

    def recover(self):
        """Adopt runs still alive; settle the ones that ended while nobody watched."""
        for m in self.models:
            e = self.entry(m)
            if e["estado"] != RUNNING:
                continue
            job = Job(m, e["gpu"], e["pid"])
            if self.pid_check(e["pid"]):
                self.log(f"{m.key}: ainda rodando (PID {e['pid']}, GPU {e['gpu']}); acompanhando")
                self.jobs[m.key] = job
                continue
            exit_path = self._exit_path(m)
            if exit_path.exists() and exit_path.stat().st_mtime >= e.get("inicio", 0):
                self._finish(job)
            else:
                self.log(f"{m.key}: execução interrompida sem registrar o fim; volta para a fila")
                e.update(estado=PENDING, pid=None, gpu=None,
                         motivo="interrompida (escalonador ou processo encerrado); retoma do que foi salvo")
        self.save()

    # main loop
    def _exit_code(self, job: Job) -> int | None:
        path = self._exit_path(job.model)
        if path.exists():
            try:
                return int(path.read_text().strip())
            except ValueError:
                pass
        return job.popen.returncode if job.popen is not None else None

    def _finish(self, job: Job):
        m, e = job.model, self.entry(job.model)
        self.jobs.pop(m.key, None)
        code = self._exit_code(job)
        complete, detail = self.is_complete(m.run_dir(self.results, self.teste))
        record = {"inicio": e.get("inicio"), "fim": self.now(), "gpu": job.gpus, "codigo": code}
        if code == 0 and complete:
            e.update(estado=DONE, fim=self.now(), pid=None, motivo=detail)
            e["historico"].append(record | {"resultado": DONE})
            self.log(f"{m.key}: concluído na GPU {job.gpus} ({detail})")
            return
        reason = (f"código de saída {code}" if code != 0 else "código 0, mas incompleta") + f"; {detail}"
        reason += f"; fim do log: {log_tail(self._log_path(m))}"
        e["tentativas"] = e.get("tentativas", 0) + 1
        e["historico"].append(record | {"resultado": FAILED, "motivo": reason})
        if e["tentativas"] < MAX_ATTEMPTS:
            e.update(estado=PENDING, pid=None, gpu=None, motivo=f"tentativa {e['tentativas']} falhou: {reason}")
            self.log(f"{m.key}: falhou ({reason}); vai tentar de novo")
        else:
            e.update(estado=FAILED, fim=self.now(), pid=None, motivo=reason)
            self.log(f"{m.key}: falhou {e['tentativas']} vezes; desistindo ({reason})")

    def _start(self, m: Model, gpus: str):
        self._exit_path(m).unlink(missing_ok=True)
        popen = self.launch(m, gpus, self.teste)
        self.jobs[m.key] = Job(m, gpus, popen.pid, popen)
        e = self.entry(m)
        e.update(estado=RUNNING, gpu=gpus, pid=popen.pid, inicio=self.now(), fim=None)
        self.log(f"{m.key}: iniciado na GPU {gpus} (lote {m.lote}, PID {popen.pid}, tentativa {e.get('tentativas', 0) + 1})")

    def pending(self, gpus: int) -> list[Model]:
        return [m for m in self.models if m.gpus == gpus and self.entry(m)["estado"] == PENDING]

    def step(self) -> bool:
        """One scheduling round. Returns False when there is nothing left to do."""
        for key, job in list(self.jobs.items()):
            if not job.alive(self.pid_check):
                self._finish(job)

        double = self.pending(2)
        barrier = min((m.ordem for m in self.models if m.gpus == 2
                       and self.entry(m)["estado"] in (PENDING, RUNNING)), default=None)
        before = lambda m: barrier is None or m.ordem < barrier
        single = [m for m in self.pending(1) if before(m)]
        # Runs ordered after a waiting two-GPU model: pause them at their next save
        later = [j for j in self.jobs.values() if j.model.gpus == 1 and not before(j.model)]
        if double and not single and later and len(later) == len(self.jobs):
            self._preempt(later, double[0])
        else:
            self.preempting.clear()
        if single or double:
            try:
                mem = self.gpu_memory()
            except Exception as e:     # nvidia-smi hiccup: try again next round
                self.log(f"nvidia-smi falhou ({type(e).__name__}: {e}); nada iniciado nesta rodada")
                mem = {}
            busy = {int(g) for job in self.jobs.values() for g in job.gpus.split(",")}
            idle = [g for g in range(self.n_gpus) if g not in busy and mem.get(g, 1 << 30) < self.free_mib]
            if single:
                for g, m in zip(idle, single):
                    self._start(m, str(g))
            elif double and not self.jobs and len(idle) == self.n_gpus:
                self._start(double[0], ",".join(str(g) for g in range(self.n_gpus)))
            elif not double:          # no barrier left: later single-GPU models may run
                for g, m in zip(idle, self.pending(1)):
                    self._start(m, str(g))
        self.save()
        return bool(self.jobs or self.pending(1) or self.pending(2))

    def _preempt(self, jobs: list, target: Model):
        for job in jobs:
            key = job.model.key
            sig = self.progress(job.model.run_dir(self.results, self.teste))
            if key not in self.preempting:
                self.preempting[key] = (sig, self.now())
                self.log(f"{key}: será pausado no próximo salvamento, para liberar as GPUs para {target.key}")
                continue
            sig0, since = self.preempting[key]
            if sig != sig0 or self.now() - since > PREEMPT_MAX_WAIT:
                self.kill(job.pid)
                if job.popen is not None:
                    try:
                        job.popen.wait(timeout=60)     # reap it, so it is not left as a zombie
                    except subprocess.TimeoutExpired:
                        pass
                self.jobs.pop(key, None)
                self.preempting.pop(key, None)
                self.entry(job.model).update(estado=PENDING, pid=None, gpu=None,
                                             motivo=f"pausado para {target.key}; retoma do que foi salvo")
                self.log(f"{key}: pausado para {target.key} (retoma do que foi salvo)")

    def run(self, poll: float = 30, refazer_falhas: bool = False, should_stop: Callable[[], bool] = lambda: False):
        self.prepare(refazer_falhas)
        self.recover()
        waiting_logged = None
        while not should_stop():
            if not self.step():
                break
            blocked = not self.jobs and (self.pending(1) or self.pending(2))
            if blocked and waiting_logged is None:
                waiting_logged = self.now()
                try:
                    usage = self.gpu_memory()
                except Exception:
                    usage = "?"
                self.log(f"aguardando GPU livre (uso atual, MiB: {usage})")
            elif not blocked:
                waiting_logged = None
            time.sleep(poll)
        self.save()
        counts = {}
        for m in self.models:
            counts[self.entry(m)["estado"]] = counts.get(self.entry(m)["estado"], 0) + 1
        self.log(("escalonador parado (ele não encerra as execuções em andamento; --parar encerra): "
                  if self.jobs else "fila encerrada: ")
                 + ", ".join(f"{v} {k}" for k, v in counts.items()))


# ── Command line ─────────────────────────────────────────────────────────

def scheduler_pidfile(results: Path, teste: bool) -> Path:
    return results / f"fila{'_teste' if teste else ''}.pid"


def listar(models: list[Model], teste: bool, results: Path, verificar: bool = True) -> str:
    sched = Scheduler(models, teste=teste, results=results)
    rows = [f"Fila: {len(models)} modelos | {'TESTE (6 problemas)' if teste else '90 problemas'} | "
            f"estado em {sched.state_path}"]
    pidfile = scheduler_pidfile(results, teste)
    alive = pidfile.exists() and pid_alive(int(pidfile.read_text().strip() or 0), "src.interp.fila")
    rows.append(f"Escalonador: {'rodando (PID ' + pidfile.read_text().strip() + ')' if alive else 'parado'}")
    rows.append(f"{'ordem':>5}  {'modelo':45s} {'gpus':>4} {'lote':>4}  {'estado':12s} detalhe")
    for m in models:
        e = sched.entry(m)
        state, detail = e["estado"], e.get("motivo") or ""
        if verificar and state in (PENDING, NO_ACCESS, REFUSED):
            status, reason = sched.check(m.id)
            state, detail = (status, reason) if status else (PENDING, reason)
        if state == RUNNING:
            detail = f"GPU {e.get('gpu')}, PID {e.get('pid')}"
        fase = "fase 1" if m.gpus == 1 else "fase 2"
        rows.append(f"{m.ordem:>5}  {m.key:45s} {m.gpus:>4} {m.lote:>4}  {state:12s} {fase}; {detail[:110]}")
    return "\n".join(rows)


def parar(teste: bool, results: Path, timeout: float = 60) -> None:
    """Stop the scheduler and every run it started; runs go back to pending."""
    pidfile = scheduler_pidfile(results, teste)
    if pidfile.exists():
        pid = int(pidfile.read_text().strip() or 0)
        if pid_alive(pid, "src.interp.fila"):
            os.kill(pid, signal.SIGTERM)
            for _ in range(int(timeout)):
                if not pid_alive(pid, "src.interp.fila"):
                    break
                time.sleep(1)
            print(f"escalonador (PID {pid}) encerrado")
    state_path = results / f"fila_estado{'_teste' if teste else ''}.json"
    if not state_path.exists():
        return
    state = json.loads(state_path.read_text())
    victims = []
    for key, e in state["modelos"].items():
        if e.get("estado") == RUNNING and pid_alive(e.get("pid")):
            pids = [e["pid"]] + descendants(e["pid"])
            victims += pids
            for p in pids:
                try:
                    os.kill(p, signal.SIGTERM)
                except OSError:
                    pass
            print(f"{key}: encerrando execução (PIDs {pids})")
        if e.get("estado") == RUNNING:
            e.update(estado=PENDING, pid=None, gpu=None, motivo="interrompida por --parar; retoma do que foi salvo")
    for _ in range(int(timeout)):
        if not any(Path(f"/proc/{p}").exists() and pid_alive(p, "") for p in victims):
            break
        time.sleep(1)
    for p in victims:
        if pid_alive(p, ""):
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fila de modelos do estudo de informação irrelevante.")
    ap.add_argument("--fila", type=Path, default=QUEUE_FILE, help="arquivo YAML da fila")
    ap.add_argument("--teste", action="store_true", help="6 problemas por modelo (estado separado)")
    ap.add_argument("--listar", action="store_true", help="mostra a fila e o estado, sem executar nada")
    ap.add_argument("--parar", action="store_true", help="encerra o escalonador e as execuções dele")
    ap.add_argument("--refazer-falhas", action="store_true", help="devolve à fila os modelos que falharam")
    ap.add_argument("--intervalo", type=float, default=30, help="segundos entre verificações")
    args = ap.parse_args(argv)
    teste = args.teste

    if args.parar:
        parar(teste, RESULTS)
        return
    models = load_queue(args.fila)
    if args.listar:
        print(listar(models, teste, RESULTS))
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    pidfile = scheduler_pidfile(RESULTS, teste)
    if pidfile.exists():
        other = int(pidfile.read_text().strip() or 0)
        if other != os.getpid() and pid_alive(other, "src.interp.fila"):
            sys.exit(f"já existe um escalonador rodando (PID {other})")
    pidfile.write_text(str(os.getpid()))

    stop = {"flag": False}

    def on_signal(signum, _frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def log(msg):
        print(f"[{time.strftime('%F %T')}] {msg}", flush=True)

    log(f"escalonador iniciado (PID {os.getpid()}) | fila {args.fila.name} | "
        f"{'teste' if teste else 'completa'} | {len(models)} modelos")
    sched = Scheduler(models, teste=teste, log=log)
    sched.run(poll=args.intervalo, refazer_falhas=args.refazer_falhas, should_stop=lambda: stop["flag"])


if __name__ == "__main__":
    main()
