"""Tests for the model queue scheduler (src/interp/fila.py). No GPU needed:
processes, nvidia-smi and the HuggingFace Hub are simulated."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.interp import fila as F  # noqa: E402


class FakeProc:
    """Stands in for subprocess.Popen; the test decides when it ends."""
    _next_pid = 1000

    def __init__(self):
        FakeProc._next_pid += 1
        self.pid = FakeProc._next_pid
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class World:
    """Simulated machine: GPUs, launched runs, completed results, hub access."""

    def __init__(self, tmp_path, n_gpus=2):
        self.results = tmp_path / "results"
        self.results.mkdir()
        self.mem = {g: 15 for g in range(n_gpus)}
        self.procs = {}          # model key -> FakeProc
        self.launches = []       # (key, gpus)
        self.complete = set()    # model keys complete on disk
        self.alive_pids = set()  # pids that pid_check reports alive (adopted runs)
        self.access = {}         # model id -> (status, reason)
        self.clock = 0.0
        self.saves = {}          # model short name -> save counter (progress signature)
        self.killed = []

    def launch(self, model, gpus, teste):
        p = FakeProc()
        self.procs[model.key] = p
        self.launches.append((model.key, gpus))
        return p

    def finish(self, sched, key, code=0, complete=True, log="tudo certo"):
        """End a run: write its .exit file and log, as the run script does."""
        m = sched.by_key[key]
        sched._exit_path(m).write_text(str(code))
        sched._log_path(m).write_text(log)
        if complete:
            self.complete.add(key)
        self.procs[key].returncode = code

    def progress(self, run_dir):
        return self.saves.get(run_dir.name.split("_")[0], 0)

    def kill(self, pid):
        self.killed.append(pid)
        for p in self.procs.values():
            if p.pid == pid:
                p.returncode = -9

    def is_complete(self, run_dir):
        keys = {k for k in self.complete if run_dir.name.startswith(k.split("/")[-1].split("#")[0] + "_")}
        return (bool(keys), "90 problemas completos" if keys else "0 de 90 capturas")

    def scheduler(self, models, **kw):
        def now():
            self.clock += 1
            return self.clock
        return F.Scheduler(models, results=self.results, launch=self.launch,
                           gpu_memory=lambda: dict(self.mem), is_complete=self.is_complete,
                           check=lambda mid: self.access.get(mid, (None, "")),
                           pid_check=lambda pid: pid in self.alive_pids, now=now,
                           progress=self.progress, kill=self.kill, log=lambda msg: None, **kw)


def M(name, gpus=1, ordem=1, lote=8, rotulo=""):
    return F.Model(id=f"org/{name}", gpus=gpus, lote=lote, ordem=ordem, rotulo=rotulo)


QUEUE = [M("a", ordem=1), M("b", ordem=2), M("c", ordem=3), M("big", gpus=2, ordem=10)]


# ── Queue file ───────────────────────────────────────────────────────────

def test_load_queue_sorts_and_validates(tmp_path):
    p = tmp_path / "fila.yaml"
    p.write_text("modelos:\n"
                 "  - {id: org/x, gpus: 1, lote: 16, ordem: 2}\n"
                 "  - {id: org/y, gpus: 2, lote: 4, ordem: 1, rotulo: r}\n")
    models = F.load_queue(p)
    assert [m.id for m in models] == ["org/y", "org/x"]
    assert models[0].key == "org/y#r" and models[0].lote == 4
    p.write_text("modelos:\n  - {id: org/z, gpus: 1, max_tokens: 32768}\n")
    assert F.load_queue(p)[0].max_tokens == 32768
    p.write_text("modelos:\n  - {id: org/x, gpus: 3}\n")
    with pytest.raises(ValueError):
        F.load_queue(p)
    p.write_text("modelos:\n  - {id: org/x}\n  - {id: org/x}\n")
    with pytest.raises(ValueError, match="repetidos"):
        F.load_queue(p)


def test_repository_queue_file_is_valid():
    models = F.load_queue(F.QUEUE_FILE)
    assert len(models) >= 11 and len({m.key for m in models}) == len(models)
    assert {m.gpus for m in models if "72B" in m.id or "70B" in m.id} == {2}
    ref = next(m for m in models if m.id == "Qwen/Qwen3-8B")
    assert ref.rotulo, "o Qwen3-8B da fila precisa de rótulo para não misturar com a referência"


def test_names_match_run_script_and_notebook(tmp_path):
    m = M("Qwen3-8B", rotulo="rtxpro6000")
    assert m.name(False) == "Qwen3-8B_rtxpro6000_frac0.30"
    assert m.run_dir(tmp_path, True).name == "Qwen3-8B_one_shot_frac0.02_seed42_rtxpro6000"
    assert M("x").run_dir(tmp_path, False).name == "x_one_shot_frac0.30_seed42"


# ── Phases and GPU choice ────────────────────────────────────────────────

def test_single_gpu_models_fill_both_gpus_and_big_model_waits(tmp_path):
    w = World(tmp_path)
    s = w.scheduler(QUEUE)
    s.prepare()
    s.step()
    # "big" needs 2 GPUs and comes after a, b, c: it waits for them
    assert w.launches == [("org/a", "0"), ("org/b", "1")]

    w.finish(s, "org/a")
    s.step()
    assert w.launches[-1] == ("org/c", "0")      # freed GPU gets the next single-GPU model

    w.finish(s, "org/b")
    s.step()
    assert len(w.launches) == 3                  # c still running on GPU 0: big must not start
    w.finish(s, "org/c")
    s.step()
    assert w.launches[-1] == ("org/big", "0,1")
    w.finish(s, "org/big")
    assert s.step() is False
    assert all(s.entry(m)["estado"] == F.DONE for m in QUEUE)


def test_busy_gpu_from_another_process_is_skipped(tmp_path):
    w = World(tmp_path)
    w.mem[0] = 40_000                            # someone else is using GPU 0
    s = w.scheduler(QUEUE)
    s.prepare()
    s.step()
    assert w.launches == [("org/a", "1")]


def test_two_gpu_model_needs_both_gpus_idle(tmp_path):
    w = World(tmp_path)
    s = w.scheduler([M("big", gpus=2)])
    s.prepare()
    w.mem[1] = 30_000
    s.step()
    assert w.launches == []
    w.mem[1] = 15
    s.step()
    assert w.launches == [("org/big", "0,1")]


def test_nvidia_smi_failure_does_not_launch_or_crash(tmp_path):
    w = World(tmp_path)
    s = w.scheduler(QUEUE)
    s.prepare()
    s.gpu_memory = lambda: (_ for _ in ()).throw(RuntimeError("nvidia-smi"))
    assert s.step() is True
    assert w.launches == []


# ── Success, failure and retry ───────────────────────────────────────────

def test_exit_zero_but_incomplete_counts_as_failure_and_retries_once(tmp_path):
    w = World(tmp_path)
    s = w.scheduler([M("a")])
    s.prepare()
    s.step()
    w.finish(s, "org/a", code=0, complete=False, log="linha 1\nlinha final do log")
    s.step()                                     # first failure -> relaunched
    e = s.entry(s.by_key["org/a"])
    assert e["tentativas"] == 1 and len(w.launches) == 2 and e["estado"] == F.RUNNING
    assert "incompleta" in e["historico"][0]["motivo"] and "linha final do log" in e["historico"][0]["motivo"]

    w.finish(s, "org/a", code=1, complete=False, log="Traceback ...\nCellExecutionError: OOM")
    assert s.step() is False                     # second failure -> gives up, queue ends
    e = s.entry(s.by_key["org/a"])
    assert e["estado"] == F.FAILED and "código de saída 1" in e["motivo"] and "OOM" in e["motivo"]
    assert len(w.launches) == 2


def test_nonzero_exit_even_if_complete_is_failure(tmp_path):
    w = World(tmp_path)
    s = w.scheduler([M("a")])
    s.prepare()
    s.step()
    w.finish(s, "org/a", code=1, complete=True)
    s.step()
    assert s.entry(s.by_key["org/a"])["tentativas"] == 1


def test_failed_model_does_not_block_the_queue(tmp_path):
    w = World(tmp_path)
    s = w.scheduler([M("a", ordem=1), M("b", ordem=2)], n_gpus=1)
    w.mem = {0: 15}
    s.prepare()
    for _ in range(2):
        s.step()
        w.finish(s, "org/a", code=1, complete=False)
    s.step()
    assert s.entry(s.by_key["org/a"])["estado"] == F.FAILED
    assert w.launches[-1] == ("org/b", "0")


# ── Access and architecture checks ───────────────────────────────────────

def test_no_access_and_moe_are_skipped_without_blocking(tmp_path):
    w = World(tmp_path)
    w.access = {"org/a": (F.NO_ACCESS, "restrito"), "org/b": (F.REFUSED, "MoE")}
    s = w.scheduler(QUEUE)
    s.prepare()
    s.step()
    assert w.launches == [("org/c", "0")]
    assert s.entry(s.by_key["org/a"])["estado"] == F.NO_ACCESS
    assert s.entry(s.by_key["org/b"])["estado"] == F.REFUSED


def test_access_granted_later_puts_model_back_in_queue(tmp_path):
    w = World(tmp_path)
    w.access = {"org/a": (F.NO_ACCESS, "restrito")}
    w.scheduler([M("a")]).prepare()
    w.access = {}
    s = w.scheduler([M("a")])
    s.prepare()
    assert s.entry(s.by_key["org/a"])["estado"] == F.PENDING


@pytest.mark.parametrize("cfg, moe", [
    ({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}, False),
    ({"model_type": "qwen3_moe", "num_experts": 128}, True),
    ({"model_type": "mixtral", "num_local_experts": 8}, True),
    ({"model_type": "llama4", "text_config": {"num_local_experts": 16}}, True),
    ({"model_type": "phi3", "architectures": ["Phi3ForCausalLM"]}, False),
])
def test_is_moe(cfg, moe):
    assert F.is_moe(cfg) is moe


# ── Resume from the state file ───────────────────────────────────────────

def test_restart_adopts_running_and_skips_done(tmp_path):
    w = World(tmp_path)
    s = w.scheduler(QUEUE)
    s.prepare()
    s.step()                                     # a on GPU 0, b on GPU 1
    w.finish(s, "org/a")
    s.step()                                     # a done, c on GPU 0
    pid_b, pid_c = w.procs["org/b"].pid, w.procs["org/c"].pid

    # scheduler dies; b and c keep running in their own sessions
    w.alive_pids = {pid_b, pid_c}
    s2 = w.scheduler(QUEUE)
    s2.prepare()
    s2.recover()
    assert set(s2.jobs) == {"org/b", "org/c"}
    s2.step()
    assert len(w.launches) == 3                  # nothing relaunched, a not redone

    # c ends while nobody watches: its .exit file settles it
    w.finish(s2, "org/c")
    w.alive_pids.discard(pid_c)
    s2.step()
    assert s2.entry(s2.by_key["org/c"])["estado"] == F.DONE
    assert s2.entry(s2.by_key["org/a"])["estado"] == F.DONE


def test_restart_resumes_run_killed_without_exit_file(tmp_path):
    w = World(tmp_path)
    s = w.scheduler([M("a")])
    s.prepare()
    s.step()
    # everything killed (e.g. power loss): no .exit, pid gone
    s2 = w.scheduler([M("a")])
    s2.prepare()
    s2.recover()
    e = s2.entry(s2.by_key["org/a"])
    assert e["estado"] == F.PENDING and e["tentativas"] == 0   # interruption is not a failure
    s2.step()
    assert len(w.launches) == 2


def test_restart_settles_run_that_failed_while_scheduler_was_down(tmp_path):
    w = World(tmp_path)
    s = w.scheduler([M("a")])
    s.prepare()
    s.step()
    w.finish(s, "org/a", code=1, complete=False)    # fails while nobody watches
    s2 = w.scheduler([M("a")])
    s2.prepare()
    s2.recover()
    assert s2.entry(s2.by_key["org/a"])["tentativas"] == 1


def test_results_already_on_disk_are_marked_done(tmp_path):
    w = World(tmp_path)
    w.complete.add("org/a")
    s = w.scheduler([M("a"), M("b", ordem=2)])
    s.prepare()
    s.step()
    assert s.entry(s.by_key["org/a"])["estado"] == F.DONE
    assert w.launches == [("org/b", "0")]


def test_state_file_roundtrip_and_separate_test_state(tmp_path):
    w = World(tmp_path)
    s = w.scheduler([M("a")])
    s.prepare()
    s.step()
    data = json.loads((w.results / "fila_estado.json").read_text())
    e = data["modelos"]["org/a"]
    assert e["estado"] == F.RUNNING and e["gpu"] == "0" and e["pid"] and e["inicio"]
    assert e["execucao"] == "a_one_shot_frac0.30_seed42"
    t = w.scheduler([M("a")], teste=True)
    assert t.state_path.name == "fila_estado_teste.json"
    assert t.entry(t.by_key["org/a"])["estado"] == F.PENDING


def test_refazer_falhas(tmp_path):
    w = World(tmp_path)
    s = w.scheduler([M("a")])
    s.prepare()
    s.entry(s.by_key["org/a"]).update(estado=F.FAILED, tentativas=2)
    s.save()
    s2 = w.scheduler([M("a")])
    s2.prepare()
    assert s2.entry(s2.by_key["org/a"])["estado"] == F.FAILED
    s2.prepare(refazer_falhas=True)
    assert s2.entry(s2.by_key["org/a"])["estado"] == F.PENDING


# ── Completion criterion on real files ───────────────────────────────────

def test_run_complete_checks_generations_captures_and_gold(tmp_path):
    torch = pytest.importorskip("torch")
    d = tmp_path / "run"
    (d / "captures").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"instances": ["c0_000", "c0_001", "c0_002"]}))
    assert F.run_complete(d)[0] is False                         # no generations.jsonl

    def gen(iid, response, finish="stop"):
        return json.dumps({"instance_id": iid, "scenario": "clean", "finish_reason": finish, "response": response})
    (d / "generations.jsonl").write_text("\n".join([
        gen("c0_000", "... The total cost is $1,365."),
        gen("c0_001", "... The total cost is $200."),
        gen("c0_002", "sem frase final"),                        # unparseable: no capture expected
        json.dumps({"instance_id": "c0_000", "scenario": "prompt", "finish_reason": "stop", "response": "x"}),
    ]) + "\n")
    torch.save({"instance_id": "c0_000", "gold": {}}, d / "captures" / "c0_000.pt")
    ok, detail = F.run_complete(d)
    assert not ok and "1 de 2" in detail
    torch.save({"instance_id": "c0_001"}, d / "captures" / "c0_001.pt")
    ok, detail = F.run_complete(d)
    assert not ok and "sem gabarito" in detail
    torch.save({"instance_id": "c0_001", "gold": {}}, d / "captures" / "c0_001.pt")
    ok, detail = F.run_complete(d)
    assert ok and "1 sem resposta legível" in detail


# ── Ordering across phases and preemption ────────────────────────────────

def test_two_gpu_model_is_a_barrier_for_later_single_gpu_models(tmp_path):
    w = World(tmp_path)
    q = [M("a", ordem=1), M("big", gpus=2, ordem=5), M("z", ordem=9)]
    s = w.scheduler(q)
    s.prepare()
    s.step()
    assert w.launches == [("org/a", "0")]          # z waits although GPU 1 is idle
    w.finish(s, "org/a")
    s.step()
    assert w.launches[-1] == ("org/big", "0,1")
    s.step()
    assert len(w.launches) == 2                     # z still waits while big runs
    w.finish(s, "org/big")
    s.step()
    assert w.launches[-1] == ("org/z", "0")


def test_later_run_is_paused_at_its_next_save_then_resumed(tmp_path):
    w = World(tmp_path)
    q = [M("a", ordem=1), M("big", gpus=2, ordem=5), M("z", ordem=9)]
    w.access = {"org/big": (F.NO_ACCESS, "ainda sem acesso")}
    s = w.scheduler(q)
    s.prepare()
    s.step()
    assert w.launches == [("org/a", "0"), ("org/z", "1")]   # no barrier yet: z runs
    s.entry(s.by_key["org/big"]).update(estado=F.PENDING)   # big becomes available
    s.step()
    assert w.killed == []                           # a (before big) still runs: no pause yet
    w.finish(s, "org/a")
    s.step()                                        # only z (after big) is left: marked for pause
    s.step()
    assert w.killed == []                           # nothing saved yet: keeps running
    w.saves["z"] = 1                                # z saves a batch
    s.step()
    z = s.entry(s.by_key["org/z"])
    assert w.killed == [w.procs["org/z"].pid]
    assert z["estado"] == F.PENDING and z["tentativas"] == 0 and "pausado" in z["motivo"]
    s.step()
    assert w.launches[-1] == ("org/big", "0,1")
    w.finish(s, "org/big")
    s.step()
    assert w.launches[-1] == ("org/z", "0")        # resumed after the barrier


def test_no_pause_when_a_single_gpu_model_before_the_barrier_is_pending(tmp_path):
    w = World(tmp_path)
    q = [M("a", ordem=1), M("b", ordem=2), M("big", gpus=2, ordem=5), M("z", ordem=9)]
    s = w.scheduler(q, n_gpus=1)
    w.mem = {0: 15}
    s.prepare()
    s.entry(s.by_key["org/big"]).update(estado=F.NO_ACCESS)
    s.step()                                        # a on GPU 0
    s.entry(s.by_key["org/big"]).update(estado=F.PENDING)
    w.saves["a"] = 3
    s.step()
    assert w.killed == []                           # a is before the barrier: never paused
