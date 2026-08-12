# DFlash async verify — mental model

Disaggregated speculative decode with `disagg_dflash_remote_only` + `disagg_dflash_async_verify`.
No ping-pong HS slots. One shared verify HS buffer, gated by staging release; socket is 1-deep; FREE is deferred off execute start when busy.

Sections **1–9** describe the **TP=1** happy path (verify GPU0 + sink GPU1).
Section **10** extends the same nomenclature to **verify TP>1** (e.g. PD2S1 / PD2S2 with `--hs-nixl-sink --async-verify`).

---

## 1. Naming convention

| Prefix | Meaning |
|--------|---------|
| `GPU:0_*` | Buffer on verify **TP0** GPU (physical device 0 when TP=1) |
| `GPU:0rK_*` | Same logical buffer on verify TP rank K (TP>1 only; see §10) |
| `GPU:1_*` | Buffer on draft/sink GPU (device 1 when TP=1; first GPU after verify TP set when TP>1) |
| `CPU_*` | Host / CPU memory (pinned or ordinary) |
| `T_*` | Thread or process actor |
| Shared | Same object touched by more than one `T_*` |
| Private | Owned by one actor; others only see copies / messages |

---

## 2. Buffer inventory

### 2.1 Verify GPU (`GPU:0`)

| Name | Code symbol | Shared? | Role |
|------|-------------|---------|------|
| **GPU:0_1** | `DFlashSpeculator.hidden_states` | **Yes** — `T_exec` writes, `T_xfer` stages | Single source HS. Next write must wait until staging releases it. |
| **GPU:0_2** | `DFlashHsNixlProbe._local_buf` | Private to probe/`T_xfer` after stage | NIXL-registered staging. After DtoD from GPU:0_1, NIXL WRITEs this to GPU:1. |
| **GPU:0_3** | per-reply `remote_gpu` in `_async_ready_queue` | Handed from `T_xfer` → install | Owned GPU draft-token tensor for one batch. |
| **GPU:0_4** | `req_states.draft_tokens` | **Yes** — install vs later execute | Durable per-req draft slots used on the next verify step. |
| **GPU:0_5** | target forward `hidden_states` (execute state) | Private to `T_exec` for that step | Source copied into GPU:0_1 at propose. |

CUDA events (not buffers, but part of the HS lifetime):

- `_hs_ready_event` — recorded on default stream after GPU:0_1 write; `T_xfer` waits before staging.
- `_staging_done_event` / `_meta_done_event` — copy-stream completion on `T_xfer`.

### 2.2 Draft / sink GPU (`GPU:1`)

| Name | Code symbol | Shared? | Role |
|------|-------------|---------|------|
| **GPU:1_1** | `DFlashDraftRunner.hidden_states` (= sink `_staging`) | **Yes** — NIXL write from verify, then draft kernels | Destination of PtoP WRITE. Sink must not read until HS_READY. |
| **GPU:1_2** | draft KV caches / block tables | Private to sink | Per-req KV; freed by FREE. |
| **GPU:1_3** | durable meta GPU bufs (`_last_sampled`, positions, …) | Private to sink | H2D-patched each SPECulate. |
| **GPU:1_4** | sink `draft_tokens` sample output | Private to sink → serialized to ZMQ | Becomes the SPECulate reply payload. |

### 2.3 CPU / host (`CPU_*`)

| Name | Code symbol | Shared? | Role |
|------|-------------|---------|------|
| **CPU_1** | `deferred_pack` host snapshots (`qsl`, `nst`, `seq_ub`, …) | Kick thread creates; `T_xfer` consumes | Host-only snapshot at kick (no CUDA on propose thread). |
| **CPU_2** | probe `_meta_pin_*` pinned tensors | Private to `T_xfer` during DtoH | Pinned landing pads for GPU meta → encode SPECulate frames. |
| **CPU_3** | ZMQ SPECulate request frames | Cross-process message | DEALER → ROUTER: meta for draft. |
| **CPU_4** | ZMQ `HS_READY` frame | Cross-process message | “GPU:1_1 is filled; safe to run draft.” |
| **CPU_5** | ZMQ SPECulate reply (`draft_tokens` on host) | Cross-process message | Sink → verify draft ids. |
| **CPU_6** | `_async_draft_pin` + `remote_cpu` in ready queue | Speculator under `_async_lock` | Pin for H2D into GPU:0_3; CPU copy for scheduler/side channel. |
| **CPU_7** | `_async_draft_side_queue` payload `(req_ids, draft_ids)` | **Yes** — `T_xfer` put, `T_engine` get | `multiprocessing.Queue` so engine learns drafts without waiting on worker RPC. |
| **CPU_8** | `request.spec_token_ids` | Private to scheduler / `T_engine` | Gate for wait-for-drafts (empty ⇒ skip decode). |

**Are the CPUs shared?**

- **CPU_1–CPU_6**: process-local. Not “shared memory” across processes; only one thread at a time should mutate under locks / ownership rules.
- **CPU_7**: **shared across processes** (verify worker ↔ EngineCore) via `mp.Queue`.
- **CPU_3/4/5**: **shared as a message channel** (ZMQ DEALER/ROUTER), not as a mutable buffer both sides write in place.
- **GPU:0_1** is the important *shared GPU* buffer: execute thread and xfer thread both touch it (write vs stage).

---

## 3. Actors (threads / processes)

| Name | What it is | Main job |
|------|------------|----------|
| **T_engine** | EngineCore process / busy loop | Poll CPU_7 → update CPU_8 → schedule → submit execute/sample. |
| **T_exec** | Worker execute / RPC thread | `execute_model` (target forward), `finish_requests`, `sample_tokens` → `propose` kick. |
| **T_xfer** | `dflash-hs-nixl-xfer` daemon thread | Meta DtoH → SPECulate send → stage GPU:0_1→GPU:0_2 → NIXL → HS_READY → recv reply → stash + put CPU_7 (no GPU:0_4 install). |
| **T_sink** | Sink process (`DFlashHsNixlSink`) | ROUTER: SPECulate prep ∥ wait HS_READY → draft on GPU:1 → reply. |
| **T_nixl** | NIXL progress thread(s) inside agent | Drive PtoP completion (internal). |

---

## 4. Who blocks, and on what

| Actor | Blocks on | When | Notes |
|-------|-----------|------|-------|
| **T_exec** | `wait_hs_src_released` (`threading.Event`) | Before writing GPU:0_1 | Short if staging already done; **not** remote draft. |
| **T_exec** | Catchup: `Thread.join` / ZMQ recv via `finish_speculate` | Before next kick if socket still busy | Can align with remote draft ⇒ long `sem_wait` (known limit without ping-pong). |
| **T_exec** | CUDA kernels / graph launch | Target forward | Normal GPU work. |
| **T_exec** | Does **not** join on FREE at execute start when busy | Deferred FREE | Avoids FREE-induced `sem_wait` hole. |
| **T_xfer** | `_meta_done_event.synchronize` | After meta DtoH | Local copy-stream sync. |
| **T_xfer** | `_staging_done_event.synchronize` | After HS stage | Then sets `_hs_src_released` (GPU:0_1 free). |
| **T_xfer** | NIXL wait | PtoP WRITE | Until GPU:1_1 filled. |
| **T_xfer** | ZMQ `recv_multipart` | SPECulate reply | Overlaps sink generate; this is where draft latency lives. |
| **T_sink** | ZMQ recv SPECulate, then HS_READY | Before touching GPU:1_1 for generate | Prep can overlap PtoP. |
| **T_engine** | Worker futures (`future.result`) | Batch queue / step | Side channel itself is non-blocking `get_nowait`. |
| **T_engine** | Logical wait-for-drafts | Decode with empty CPU_8 | Skips scheduling decode; prefills still OK. |

---

## 5. One loop (happy path)

```text
T_engine:  poll CPU_7 → fill CPU_8 → schedule (prefills OK if decode draft-blocked)
    │
    ▼
T_exec:    execute_model  →  target forward produces GPU:0_5
           finish_requests → if FREE needed & socket busy: queue ids (no join)
    │
    ▼
T_exec:    sample_tokens / propose
           (1) wait Event: GPU:0_1 released by prior stage
           (2) write GPU:0_1  ←── THIS is the HS barrier placement
           (3) if prior SPECulate still on socket: catchup (may join/recv)
           (4) flush deferred FREE if idle
           (5) record _hs_ready_event; clear Event; start T_xfer
           (6) return None  (no drafts this RPC)
    │
    ├────────────────────────────────────────────────────────────┐
    ▼                                                            ▼
T_xfer (overlaps next T_exec / T_engine work):              T_sink:
  CPU_1 → meta DtoH into CPU_2                                 recv CPU_3 (SPECulate)
  send CPU_3 (SPECulate)                                       CPU prep / H2D meta
  stage GPU:0_1 → GPU:0_2; set Event (GPU:0_1 free)            wait CPU_4 (HS_READY)
  NIXL WRITE GPU:0_2 → GPU:1_1                                 generate on GPU:1_*
  send CPU_4 (HS_READY)                                        send CPU_5 (drafts)
  recv CPU_5
  stash → GPU:0_3 + CPU_6; put CPU_7 (peek only — leave queue)
    │
    ▼
T_engine:  get CPU_7 → CPU_8
           RPC poll → T_exec installs GPU:0_4 on execute stream
           → schedule decode with real drafts
```

---

## 6. Ownership diagram (shared vs private)

```text
                    ┌──────────── T_engine ────────────┐
                    │  CPU_8 (spec_token_ids)          │
                    │  reads CPU_7 (mp.Queue)          │
                    └──────────────▲───────────────────┘
                                   │ CPU_7 (shared channel)
┌──────── T_exec ──────────┐       │      ┌──────── T_xfer ─────────┐
│ writes GPU:0_1           │       │      │ stages GPU:0_1→GPU:0_2  │
│ waits Event before write │◄──────┼──────┤ sets Event after stage  │
│ catchup may join T_xfer  │       │      │ NIXL, ZMQ, callback     │
│ deferred FREE bookkeep   │       │      │ puts CPU_7 (CPU only)   │
│ poll installs GPU:0_4    │       │      │                         │
└────────────┬─────────────┘       │      └───────────┬─────────────┘
             │ kick / host meta    │                  │ CPU_3/4/5 (ZMQ)
             │                     │                  ▼
             │                     │      ┌──────── T_sink ─────────┐
             │                     │      │ GPU:1_1 ← NIXL          │
             │                     │      │ GPU:1_2 KV, generate    │
             │                     │      │ replies CPU_5           │
             │                     │      └─────────────────────────┘
             ▼
        GPU:0_1  ◄══ shared between T_exec (write) and T_xfer (read/stage)
        GPU:0_2  ─── private staging for NIXL
        GPU:1_1  ◄══ shared across machines: NIXL writer then T_sink kernels
```

---

## 7. Lifetime rules (correctness)

1. **GPU:0_1** may be overwritten by `T_exec` only after `T_xfer` has staged into **GPU:0_2** and set `_hs_src_released`. That wait is **before** the write — not after a full SPECulate recv.
2. **GPU:1_1** may be read by draft kernels only after **CPU_4** (HS_READY).
3. ZMQ DEALER is **1-deep**: no new SPECulate until prior reply is finished (or drained). Catchup-before-kick enforces this on `T_exec` when needed.
4. **FREE** must not interleave with an in-flight SPECulate on the socket. If busy, ids go to `_async_deferred_free_ids` and flush when idle (poll/catchup) — not via `Thread.join` at execute start.
5. **CPU_8** empty ⇒ do not schedule decode for that req (wait-for-drafts). No decode-1 fallback.
6. **GPU:0_4** is installed in **prepare_inputs** from scheduled real draft ids (execute stream), not on the engine poll RPC and not on T_xfer. Poll/side-channel stay CPU-only so next-batch schedule can overlap prior GPU execute.
7. Every stash publishes **CPU_7**. Engine drains side channel only (no CUDA poll) when the side channel exists.

---

## 8. What still causes a long `T_exec` stall

| Stall | Barrier | Avoidable without ping-pong? |
|-------|---------|------------------------------|
| FREE drain/`join` at execute start | Was joining remote recv | **Yes — deferred FREE** |
| HS overwrite vs staging | Event before write | **Yes — short local wait** |
| Catchup-before-kick while reply still in flight | `join` / ZMQ recv on `T_exec` | **No** (needs second HS+meta slot / arm) |

So: remote draft on `T_sink` / `T_xfer` can overlap other `T_engine` / prefill work **unless** `T_exec` must kick a new SPECulate while the previous reply is still pending — then catchup blocks `T_exec` by design of the 1-deep socket.

---

## 9. Glossary (code ↔ nomenclature)

| Nomenclature | Code |
|--------------|------|
| GPU:0_1 | `speculator.hidden_states` |
| GPU:0_2 | `probe._local_buf` |
| GPU:0_3 | ready-queue `remote_gpu` |
| GPU:0_4 | `req_states.draft_tokens` |
| GPU:1_1 | sink/draft `hidden_states` / `_staging` |
| CPU_1 | `deferred_pack` host fields |
| CPU_2 | `probe._meta_pin_*` |
| CPU_3 / CPU_4 / CPU_5 | ZMQ SPECulate / HS_READY / reply |
| CPU_6 | `_async_draft_pin` / `remote_cpu` |
| CPU_7 | `_async_draft_side_queue` |
| CPU_8 | `request.spec_token_ids` |
| T_engine | EngineCore |
| T_exec | worker execute / `sample_tokens` / `propose` |
| T_xfer | `dflash-hs-nixl-xfer` |
| T_sink | `dflash_hs_nixl_sink` process |

---

## 10. Verify TP>1 (same protocol, TP0-owned sink path)

### 10.1 What stays the same

- Sink is still **one** process, **TP=1** draft (`GPU:1_*`, ZMQ 1-deep, NIXL one writer).
- Protocol frames **CPU_3 / CPU_4 / CPU_5** unchanged.
- Engine wait-for-drafts (**CPU_8**), side channel (**CPU_7**), and install-in-`prepare_inputs` (**GPU:0_4**) unchanged in spirit.
- Target / aux hidden states are **full `[N,H]` on every TP rank** after RowParallel all-reduce — no HS gather for the kick.

### 10.2 What changes

| Concern | TP=1 | TP>1 |
|---------|------|------|
| Who creates `DFlashHsNixlProbe` | The only verify worker | **Only `get_tp_group().rank == 0`** |
| Who runs `T_xfer` / NIXL / ZMQ | That worker | **TP0 only** |
| Who publishes **CPU_7** | That worker | **TP0 only** (side queue wired to driver worker) |
| Who kicks SPECulate | That worker | **TP0 only** |
| Non-TP0 `propose` (async) | n/a | Combine HS locally, **`return None`** (no probe) |
| Who installs **GPU:0_4** | That worker | **Every TP rank** from the same scheduled draft ids |
| Device layout (launch) | verify `0`, sink `1` | verify `0..TP-1`, sink first GPU after (e.g. PD2 → sink `2`) |

Sync `remote_only` without async on non-TP0 is rejected (no probe to wait on). Use `disagg_dflash_async_verify=true` for TP>1.

### 10.3 Extra naming (TP ranks)

| Name | Meaning |
|------|---------|
| **T_exec_0** | Verify TP0 worker execute thread — owns probe, kick, catchup, FREE, CPU_7 publish |
| **T_exec_k** | Verify TP rank k>0 execute thread — target forward + combine; no probe |
| **GPU:0r0_*** | TP0 copies of §2.1 buffers (the ones NIXL/ZMQ actually use) |
| **GPU:0rk_4** | Rank k `req_states.draft_tokens` — installed from schedule like TP0 |

`GPU:0_1` / `GPU:0_2` in §§2–8 always mean **GPU:0r0_*** under TP>1. Non-TP0 may still have a local `hidden_states` tensor after combine; it is **not** staged or NIXL’d.

### 10.4 Actors under TP>1

| Name | Rank | Main job |
|------|------|----------|
| **T_engine** | — | Same: drain CPU_7 → CPU_8 → schedule (broadcast to all ranks) |
| **T_exec_0** | TP0 | Target forward; HS wait/write; kick SPECulate; deferred FREE; catchup |
| **T_exec_k** | TP>0 | Target forward + combine; async `propose` returns None; no NIXL/ZMQ |
| **T_xfer** | TP0 only | Same as §3 (lives in TP0 probe) |
| **T_sink** | sink GPU | Unchanged |

Side channel: multiproc executor passes `_async_draft_side_queue` only to the **driver** worker (`rank % tp_size == 0` → TP0 when PP=1). DFlash rejects PP, so that is TP0.

### 10.5 One loop (TP=2 sketch)

```text
T_engine:  poll CPU_7 → CPU_8 → schedule (same SchedulerOutput to all ranks)
    │
    ├──────────────────────────────┬─────────────────────────────┐
    ▼                              ▼                             ▼
T_exec_0:                     T_exec_1:                      (more ranks…)
  execute_model                 execute_model
  target forward (TP shard)     target forward (TP shard)
  all_reduce → full HS          all_reduce → full HS
  combine → write GPU:0r0_1     combine (local only)
  wait HS released (probe)      (no probe)
  kick T_xfer / SPECulate       propose → return None
  return None
    │
    ▼
T_xfer (TP0) + T_sink:  same as §5 (CPU_3/4/5, GPU:0r0_2 → GPU:1_1)
    │
    ▼
T_xfer: put CPU_7
T_engine: CPU_7 → CPU_8 → schedule decode with real draft ids
    │
    ├──────────────────────────────┐
    ▼                              ▼
T_exec_0 prepare_inputs:        T_exec_1 prepare_inputs:
  install GPU:0r0_4               install GPU:0r1_4
  (pinned H2D, no CPU setitem)    (same scheduled ids)
```

### 10.6 Ownership diagram (TP=2)

```text
                         ┌──────────── T_engine ────────────┐
                         │  CPU_8 ← CPU_7 (from TP0 only)   │
                         │  schedule broadcast to all ranks │
                         └──────────────▲───────────────────┘
                                        │ CPU_7
        ┌──────── T_exec_0 (TP0) ───────┤      ┌──── T_xfer (TP0) ────┐
        │ GPU:0r0_1 write + HS wait     │      │ stage / NIXL / ZMQ   │
        │ kick SPECulate                │◄─────┤ put CPU_7            │
        │ install GPU:0r0_4 from sched  │      └──────────┬───────────┘
        └───────────────────────────────┘                 │ CPU_3/4/5
                                                          ▼
        ┌──────── T_exec_1 (TP1) ───────┐      ┌──────── T_sink ──────┐
        │ full HS locally (no NIXL)     │      │ GPU:1_1 ← NIXL       │
        │ propose returns None          │      │ draft + reply CPU_5  │
        │ install GPU:0r1_4 from sched  │      └──────────────────────┘
        └───────────────────────────────┘
```

### 10.7 Lifetime / correctness rules (TP>1 addenda)

1. **At most one DEALER** against the sink — only TP0 constructs the probe. A second rank opening NIXL/ZMQ would fight the 1-deep socket.
2. **CPU_7 has a single producer** (TP0 `T_xfer`) and a single consumer (`T_engine`). Non-TP0 never publishes.
3. **Draft ids must land on every rank’s GPU:0rk_4** before verify. That happens via scheduled `scheduled_spec_decode_tokens` + `prepare_inputs` H2D on each rank — not via per-rank ZMQ.
4. **Collective `propose` / `sample_tokens` still run on all ranks.** Non-TP0 must not block on remote draft; async early-out keeps the collective aligned with TP0’s non-blocking kick.
5. **FREE / deferred FREE** only matter on TP0 (probe present). Non-TP0 `free_remote_kv` is a no-op when probe is None.
6. **Device packing:** verify occupies `0..TP-1`; sink defaults to the next GPU index (launch script). Do not put sink on a verify TP device.

### 10.8 Who blocks (TP>1 deltas)

| Actor | Blocks on | TP>1 note |
|-------|-----------|-----------|
| **T_exec_0** | Same as §4 `T_exec` | Catchup / HS wait / deferred FREE only here |
| **T_exec_k** | Target kernels only | No `wait_hs_src_released`, no catchup join, no ZMQ |
| **T_engine** | Same | Still soft-waits collect when draft-blocked; prefills OK |
| **T_xfer / T_sink** | Same | Only one of each, attached to TP0 |

### 10.9 Glossary addenda

| Nomenclature | Code / meaning |
|--------------|----------------|
| T_exec_0 | TP0 worker `sample_tokens` / `propose` (probe owner) |
| T_exec_k | Non-TP0 worker same entrypoints, `_hs_nixl_probe is None` |
| GPU:0r0_1 | TP0 `speculator.hidden_states` (NIXL source) |
| GPU:0rk_4 | Rank k `req_states.draft_tokens` |
| Probe gate | `get_tp_group().rank == 0` in `DFlashSpeculator.__init__` |
| Side-channel gate | `async_draft_side_queue` only if `is_driver_worker` |

---

## 11. NVTX map (post-forward gap attribution)

Pure instrumentation — no logic change. `execute_context` wraps the whole
`execute_model` RPC (prep + target forward), **not** sampling or draft.

Each numbered span is nested under a **generic** outer span with a stable name
so nsys can group stats (`execute_context`, `sample`, `propose`,
`dflash_draft_forward`, …) while the inner name keeps `i{N}` / `vi{N}` detail.

**Colocated (local draft on verify GPU):**

```
execute_context
  i{N}_execute_context_*        # prepare_inputs + target forward
  (engine / sample_tokens RPC)  # CPU gap if unlabeled
sample_tokens
  i{N}_sample_tokens
    sample / i{N}_sample        # logits + sample / rejection sample
    postprocess_sampled / i{N}_postprocess_sampled
    propose / i{N}_propose
      dflash_hs_prep / …_vi{N}           # combine HS + DtoD into draft HS
      dflash_draft_prepare / …_vi{N}     # prepare + context KV + CG meta
      dflash_draft_forward / …_vi{N}     # draft forward (FULL CG may include sample)
      dflash_draft_sample / …_vi{N}      # sample_draft (eager / piecewise only)
```

**Disagg / remote_only (draft on sink):**

```
execute_context / i{N}_execute_context_*
sample_tokens / i{N}_sample_tokens
  sample / i{N}_sample
  propose / i{N}_propose
    dflash_hs_prep / …_vi{N}
    dflash_hs_nixl_*            # existing kick / meta_pack / stage / send / wait
                                # (no dflash_draft_forward on verify GPU)
```

Draft forward on the sink remains under existing `dflash_hs_nixl_speculate*` /
`dflash_sink_*` spans. Remaining hole between `execute_context` end and
`sample_tokens` start is engine/RPC/grammar, not missing GPU work.

---

## 12. Perf note — mid-run GPU-idle `execute_context` gaps

Long idle gaps inside verify `execute_context` (nsys: `posix_spawn` → `waitpid` → `cuModuleLoadData`) are usually **not** draft-wait (`blocked_0`). On gpt-oss they come from late Triton JIT of MoE routing:

- `_topk_forward`, `_sum_bitmatrix_rows`, `_combined_routing_memset` / `_combined_routing_compute(_pow2)`
- Path: `triton_kernel_moe_forward` / `make_routing_data` in `gpt_oss_triton_kernels_moe.py`
- Compile keys include bitmatrix strides `cdiv(n_tokens, 32)*32`, so each 32-token pad bucket can re-JIT

### Type-A mixed-step (TP>1)

Worst holes (~0.5–1.2s on TP=4) are **mixed prefill+decode** steps whose total token count first-touches a new pad bucket:

1. Attention / early kernels finish (~20ms into `execute_context`)
2. GPU idle while CPU `posix_spawn`/`waitpid` compiles `_topk_forward` (then bitmatrix / combined routing)
3. Resume MoE → fat `nccl AllReduce` / `multimem_all_reduce` while other ranks catch up

Decode-only is healthy once routing buckets are warm. Long PV `hs_src_wait` outliers are a **symptom** of Type-A (TP0 busy in the prior step’s AllReduce / meta_pack), not a separate NIXL bug.

Init warmup: `gpt_oss_triton_moe_warmup` runs from `kernel_warmup` (**pre_cudagraph**) and again at the end of V2 dummy warms (**post_dummy_warms**), covering every 32-token stride bucket (plus `pad-1`) up to `max_num_batched_tokens`. Dense `_dummy_run` sweeps alone are not enough when live mixed sizes differ from CUDA-graph-padded shapes. If discovery fails, logs a **warning** (silent no-op used to leave Type-A fully exposed).
