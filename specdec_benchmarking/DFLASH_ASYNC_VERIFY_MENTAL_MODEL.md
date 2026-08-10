# DFlash async verify — mental model

Disaggregated speculative decode with `disagg_dflash_remote_only` + `disagg_dflash_async_verify`.
No ping-pong HS slots. One shared verify HS buffer, gated by staging release; socket is 1-deep; FREE is deferred off execute start when busy.

---

## 1. Naming convention

| Prefix | Meaning |
|--------|---------|
| `GPU:0_*` | Buffer on verify GPU (device 0) |
| `GPU:1_*` | Buffer on draft/sink GPU (device 1) |
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
