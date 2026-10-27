"""Mooncake draft KV transfer between different prefill/decode attention TP sizes.

An MLA target's KV is replicated over attention TP, but an appended MHA draft
pool (DFLASH, EAGLE3) is head-sharded, so its per-rank page size follows the TP
size. The cases register the decode rank and drive the real transfer worker
over bounds-checked byte buffers.
"""

import concurrent.futures
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.srt.disaggregation.ascend.conn import AscendKVManager
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.utils import FastQueue, TransferKVChunk
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
    TransferInfo,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOM = 7
SESSION = "decode"
PAGE_SIZE = 4
NUM_PAGES = 8
HEAD_BYTES = 2
TARGET_WIDTH = 6  # bytes per token of one replicated MLA layer
NUM_TARGET = 2
NUM_DRAFT = 2  # K and V of one draft layer
# Global layer ids for the hybrid pool; the draft takes the band above layer 8.
HYBRID_LAYER_IDS = [3, 7, 8, 8]
SRC_PAGES = np.array([6, 1, 3], dtype=np.int32)
# Ends on the last page, so a write at the prefill stride runs past the pool.
DST_PAGES = np.array([2, 7, 5], dtype=np.int32)
DEVICE_DST_PAGES = np.array([0, 4, 6], dtype=np.int32)
NUM_TOKENS = len(SRC_PAGES) * PAGE_SIZE
GUARD = 0xEE


def _token_rows(pages):
    return (pages.astype(np.int64)[:, None] * PAGE_SIZE + np.arange(PAGE_SIZE)).ravel()


def _heads(rank, tp_size, num_heads):
    """Heads one rank holds; consecutive ranks share a head when heads < TP."""
    per_rank = max(1, num_heads // tp_size)
    start = (rank // max(1, tp_size // num_heads)) * per_rank
    return slice(start, start + per_rank)


def _draft_width(tp_size, num_heads):
    return max(1, num_heads // tp_size) * HEAD_BYTES


def _kv_senders(src_tp, dst_tp, dst_rank, pure_mla):
    """Prefill ranks that write KV to one decode rank, per the real rank mapping."""
    decode = SimpleNamespace(
        is_mla_backend=pure_mla,
        is_hybrid_mla_backend=not pure_mla,
        attn_tp_size=dst_tp,
        attn_cp_size=1,
        attn_cp_rank=0,
        pp_size=1,
        pp_rank=0,
        enable_all_cp_ranks_for_transfer=False,
        kv_args=SimpleNamespace(engine_rank=dst_rank),
    )
    info = SimpleNamespace(
        attn_tp_size=src_tp,
        attn_cp_size=1,
        pp_size=1,
        enable_kv_cache_sharding=False,
        enable_dsa_cache_layer_split=False,
    )
    CommonKVManager._resolve_rank_mapping(decode, info)
    # Pure MLA marks every target rank but the selected one as a dummy.
    return [r for r in info.target_tp_ranks if not pure_mla or r == info.target_tp_rank]


class _FakeMemory:
    """Byte buffers at fake addresses. A batch touching memory outside every
    buffer fails before writing, as an RDMA write outside registered memory
    does."""

    def __init__(self):
        self.buffers = {}
        self.out_of_range = []
        self.num_batches = 0
        self._next_base = 1 << 24

    def alloc(self, nbytes, fill):
        base = self._next_base
        self.buffers[base] = np.full(nbytes, fill, dtype=np.uint8)
        # Leave a gap so an overrun never lands in the next buffer.
        self._next_base += nbytes + (1 << 20)
        return base

    def _locate(self, addr, length):
        for base, buf in self.buffers.items():
            if base <= addr and addr + length <= base + buf.size:
                return buf, addr - base
        return None

    def batch_transfer_sync(self, session, src_addrs, dst_addrs, lengths):
        self.num_batches += 1
        located = []
        for src, dst, length in zip(src_addrs, dst_addrs, lengths, strict=True):
            src_loc, dst_loc = self._locate(src, length), self._locate(dst, length)
            if src_loc is None or dst_loc is None:
                self.out_of_range.append((src, dst, length))
                return -1
            located.append((src_loc, dst_loc, length))
        for (src_buf, src_off), (dst_buf, dst_off), length in located:
            dst_buf[dst_off : dst_off + length] = src_buf[src_off : src_off + length]
        return 0


def _prefill_manager(
    memory,
    *,
    src_tp,
    src_rank,
    src_ptrs,
    src_lens,
    layer_ids,
    pure_mla,
    manager_class=MooncakeKVManager,
    **kw,
):
    manager = manager_class.__new__(manager_class)
    vars(manager).update(
        engine=memory,
        kv_args=SimpleNamespace(
            kv_data_ptrs=src_ptrs,
            kv_item_lens=src_lens,
            kv_layer_ids=layer_ids,
            num_draft_entries=kw.get("num_draft", NUM_DRAFT),
            draft_kv_cache_layout=kw.get("layout", "nhd"),
            page_size=PAGE_SIZE,
            engine_rank=src_rank,
            mla_compression_ratios=None,
            prefill_start_layer=kw.get("start_layer", 0),
            aux_data_ptrs=[],
            state_types=[],
        ),
        is_mla_backend=pure_mla,
        is_hybrid_mla_backend=not pure_mla,
        attn_tp_size=src_tp,
        state_tp_size=kw.get("state_tp", src_tp),
        attn_cp_size=kw.get("cp_size", 1),
        pp_size=kw.get("pp_size", 1),
        enable_custom_mem_pool=kw.get("custom_pool", False),
        custom_mem_pool_type="NVLINK" if kw.get("custom_pool", False) else None,
        max_transfer_batch_indices=kw.get("batch", 0),
        enable_deferred_decode_kv_release=False,
        enable_trace=False,
        enable_staging=False,
        bootstrap_port=8998,
        request_status={ROOM: KVPoll.WaitingForInput},
        session_lock=threading.Lock(),
        state_layout_rejections={},
        state_strides_validated=set(),
        failed_sessions=set(),
        failed_session_probes={},
        session_failures=defaultdict(int),
        decode_kv_args_table={},
        _staging_outstanding=defaultdict(int),
        conclude_failure=MagicMock(),
    )
    return manager


def _registration(*, dst_rank, dst_tp, pure_mla, dst_ptrs, dst_lens, **kw):
    return KVArgsRegisterInfo(
        room="None",
        endpoint="127.0.0.1",
        dst_port=9000 + dst_rank,
        mooncake_session_id=SESSION,
        dst_kv_ptrs=dst_ptrs,
        dst_aux_ptrs=[],
        dst_state_data_ptrs=[],
        dst_tp_rank=dst_rank,
        dst_attn_tp_size=dst_tp,
        dst_kv_item_len=dst_lens[0],
        dst_state_item_lens=[],
        dst_state_dim_per_tensor=[],
        dst_kv_layer_ids=kw.get(
            "dst_layer_ids", [] if pure_mla else list(HYBRID_LAYER_IDS)
        ),
        dst_state_layer_ids=[],
        dst_kv_item_lens=dst_lens if kw.get("publish_lens", True) else [],
        requires_dcp_relayout=kw.get("dcp", False),
    )


def _register(manager, registration):
    """Register the decode rank as the bootstrap thread does; returns the rejection."""
    with patch(
        "sglang.srt.disaggregation.mooncake.conn.get_spec",
        return_value=SimpleNamespace(speculative_algorithm="DFLASH"),
    ):
        return manager._publish_peer_registration(SESSION, registration)


def _run_worker(manager, registration, executor, dst_device_pages=None):
    manager.transfer_infos = {
        ROOM: {
            SESSION: TransferInfo(
                room=ROOM,
                endpoint=registration.endpoint,
                dst_port=registration.dst_port,
                mooncake_session_id=SESSION,
                dst_kv_indices=DST_PAGES,
                dst_aux_index=0,
                dst_state_indices=[],
                required_dst_info_num=1,
                is_dummy=False,
                dst_device_kv_indices=dst_device_pages,
            )
        }
    }
    queue = FastQueue()
    queue.put(
        TransferKVChunk(
            room=ROOM,
            prefill_kv_indices=SRC_PAGES.copy(),
            index_slice=slice(0, len(SRC_PAGES)),
            is_last_chunk=False,
            prefill_aux_index=None,
            state_indices=None,
        )
    )
    queue.put(None)
    with patch(
        "sglang.srt.disaggregation.mooncake.conn.get_memory",
        return_value=SimpleNamespace(enable_unified_memory=False),
    ):
        manager.transfer_worker(queue, executor)


class TestAsymmetricTpDraftKvTransfer(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.addCleanup(self.executor.shutdown)

    def test_each_decode_rank_receives_its_draft_heads(self):
        """Prefill TP < decode TP: draft pages land at the decode stride, with
        each decode rank's head slice, and the target pages are unchanged."""
        for src_tp, dst_tp, num_heads in (
            (1, 2, 8),
            (2, 4, 8),
            (1, 4, 8),
            (2, 8, 8),
            (2, 8, 4),  # four heads over eight decode ranks: pairs share one
        ):
            for pure_mla in (True, False):
                for custom_pool in (False, True):
                    for batch in (0, 2):
                        with self.subTest(
                            src_tp=src_tp,
                            dst_tp=dst_tp,
                            heads=num_heads,
                            pure_mla=pure_mla,
                            custom_pool=custom_pool,
                            batch=batch,
                        ):
                            self._check(
                                src_tp,
                                dst_tp,
                                num_heads,
                                pure_mla=pure_mla,
                                custom_pool=custom_pool,
                                batch=batch,
                            )

    def test_pipelined_prefill_writes_the_paired_decode_entries(self):
        """Prefill PP 2 to decode PP 1: the last stage holds the second target
        layer and the draft, so its entries pair with decode entries 1 to 3."""
        for src_tp, dst_tp in ((1, 2), (2, 4)):
            for pure_mla in (True, False):
                for custom_pool in (False, True):
                    for batch in (0, 2):
                        with self.subTest(
                            src_tp=src_tp,
                            dst_tp=dst_tp,
                            pure_mla=pure_mla,
                            custom_pool=custom_pool,
                            batch=batch,
                        ):
                            self._check(
                                src_tp,
                                dst_tp,
                                8,
                                pure_mla=pure_mla,
                                custom_pool=custom_pool,
                                batch=batch,
                                prefill_pp=2,
                            )

    def test_hisparse_draft_uses_device_pages(self):
        """The device-only draft suffix is addressed by the device page ids."""
        for pure_mla in (True, False):
            with self.subTest(pure_mla=pure_mla):
                self._check(2, 4, 8, pure_mla=pure_mla, device_pages=True)

    def test_equal_draft_widths_keep_the_page_copy(self):
        """Equal TP, or heads replicated on both sides, needs no head slicing;
        each page moves as one block per contiguous run, as before."""
        for src_tp, dst_tp, num_heads in ((2, 2, 8), (4, 4, 8), (4, 8, 2)):
            for pure_mla in (True, False):
                with self.subTest(
                    src_tp=src_tp, dst_tp=dst_tp, heads=num_heads, pure_mla=pure_mla
                ):
                    num_batches = self._check(
                        src_tp, dst_tp, num_heads, pure_mla=pure_mla
                    )
                    # One legacy batch per decode rank: target and draft together.
                    self.assertEqual(num_batches, dst_tp)

    def _check(
        self,
        src_tp,
        dst_tp,
        num_heads,
        *,
        pure_mla,
        custom_pool=False,
        batch=0,
        device_pages=False,
        prefill_pp=1,
    ):
        rng = np.random.default_rng(41038)
        target = rng.integers(
            0, 256, (NUM_TARGET, NUM_TOKENS, TARGET_WIDTH), dtype=np.uint8
        )
        draft = rng.integers(
            0, 256, (NUM_DRAFT, NUM_TOKENS, num_heads, HEAD_BYTES), dtype=np.uint8
        )
        src_width = _draft_width(src_tp, num_heads)
        dst_width = _draft_width(dst_tp, num_heads)
        src_rows = _token_rows(SRC_PAGES)
        dst_rows = _token_rows(DST_PAGES)
        draft_dst_rows = _token_rows(DEVICE_DST_PAGES if device_pages else DST_PAGES)
        num_slots = NUM_PAGES * PAGE_SIZE
        # Entries each prefill PP stage owns, in decode order [target..., draft...];
        # the draft lives on the last stage.
        all_entries = list(range(NUM_TARGET + NUM_DRAFT))
        stages = [all_entries] if prefill_pp == 1 else [[0], all_entries[1:]]
        memory = _FakeMemory()
        num_batches = 0
        for dst_rank in range(dst_tp):
            dst_ptrs = [
                memory.alloc(num_slots * TARGET_WIDTH, GUARD) for _ in range(NUM_TARGET)
            ] + [memory.alloc(num_slots * dst_width, GUARD) for _ in range(NUM_DRAFT)]
            dst_lens = [PAGE_SIZE * TARGET_WIDTH] * NUM_TARGET + [
                PAGE_SIZE * dst_width
            ] * NUM_DRAFT
            for src_rank in _kv_senders(src_tp, dst_tp, dst_rank, pure_mla):
                for entries in stages:
                    src_ptrs, src_lens = [], []
                    for entry in entries:
                        if entry < NUM_TARGET:
                            ptr = memory.alloc(num_slots * TARGET_WIDTH, 0x11)
                            rows = target[entry]
                            src_lens.append(PAGE_SIZE * TARGET_WIDTH)
                        else:
                            ptr = memory.alloc(num_slots * src_width, 0x22)
                            shard = draft[entry - NUM_TARGET][
                                :, _heads(src_rank, src_tp, num_heads)
                            ]
                            rows = shard.reshape(NUM_TOKENS, src_width)
                            src_lens.append(PAGE_SIZE * src_width)
                        memory.buffers[ptr].reshape(num_slots, -1)[src_rows] = rows
                        src_ptrs.append(ptr)
                    manager = _prefill_manager(
                        memory,
                        src_tp=src_tp,
                        src_rank=src_rank,
                        src_ptrs=src_ptrs,
                        src_lens=src_lens,
                        layer_ids=(
                            [] if pure_mla else [HYBRID_LAYER_IDS[e] for e in entries]
                        ),
                        pure_mla=pure_mla,
                        num_draft=sum(e >= NUM_TARGET for e in entries),
                        pp_size=prefill_pp,
                        start_layer=entries[0],
                        custom_pool=custom_pool,
                        batch=batch,
                    )
                    # Every prefill rank parses its own copy of the registration.
                    registration = _registration(
                        dst_rank=dst_rank,
                        dst_tp=dst_tp,
                        pure_mla=pure_mla,
                        dst_ptrs=dst_ptrs,
                        dst_lens=dst_lens,
                    )
                    self.assertIsNone(_register(manager, registration))
                    batches_before = memory.num_batches
                    _run_worker(
                        manager,
                        registration,
                        self.executor,
                        dst_device_pages=DEVICE_DST_PAGES if device_pages else None,
                    )
                    num_batches += memory.num_batches - batches_before
                    manager.conclude_failure.assert_not_called()
                    self.assertEqual(memory.out_of_range, [])

            for entry in range(NUM_TARGET):
                expected = np.full((num_slots, TARGET_WIDTH), GUARD, dtype=np.uint8)
                expected[dst_rows] = target[entry]
                np.testing.assert_array_equal(
                    memory.buffers[dst_ptrs[entry]].reshape(num_slots, -1), expected
                )
            for entry in range(NUM_DRAFT):
                expected = np.full((num_slots, dst_width), GUARD, dtype=np.uint8)
                expected[draft_dst_rows] = draft[entry][
                    :, _heads(dst_rank, dst_tp, num_heads)
                ].reshape(NUM_TOKENS, dst_width)
                np.testing.assert_array_equal(
                    memory.buffers[dst_ptrs[NUM_TARGET + entry]].reshape(num_slots, -1),
                    expected,
                )
        return num_batches


class TestUnsupportedDraftLayoutsFailExplicitly(CustomTestCase):
    @staticmethod
    def _peer(src_tp, dst_tp, pure_mla, src_width, dst_width, **kw):
        memory = _FakeMemory()
        num_slots = NUM_PAGES * PAGE_SIZE
        manager = _prefill_manager(
            memory,
            src_tp=src_tp,
            src_rank=0,
            src_ptrs=[memory.alloc(num_slots * TARGET_WIDTH, 0x11)] * NUM_TARGET
            + [memory.alloc(num_slots * src_width, 0x22)] * NUM_DRAFT,
            src_lens=[PAGE_SIZE * TARGET_WIDTH] * NUM_TARGET
            + [PAGE_SIZE * src_width] * NUM_DRAFT,
            layer_ids=[] if pure_mla else list(HYBRID_LAYER_IDS),
            pure_mla=pure_mla,
            **kw,
        )
        if not kw.get("mla", True):
            manager.is_mla_backend = manager.is_hybrid_mla_backend = False
        registration = _registration(
            dst_rank=0,
            dst_tp=dst_tp,
            pure_mla=pure_mla,
            dst_ptrs=[memory.alloc(num_slots * TARGET_WIDTH, GUARD)] * NUM_TARGET
            + [memory.alloc(num_slots * dst_width, GUARD)] * NUM_DRAFT,
            dst_lens=[PAGE_SIZE * TARGET_WIDTH] * NUM_TARGET
            + [PAGE_SIZE * dst_width] * NUM_DRAFT,
            **kw,
        )
        return memory, manager, registration

    def test_registration_rejects_draft_heads_it_cannot_reslice(self):
        for src_tp, dst_tp, pure_mla, src_width, dst_width, kw, reason in (
            # Prefill TP > decode TP: dummy (pure MLA) or KV-skipping (hybrid)
            # prefill ranks hold draft heads the decode rank never receives.
            (4, 2, True, 4, 8, {}, "multiple of a smaller prefill"),
            (4, 2, False, 4, 8, {}, "multiple of a smaller prefill"),
            # The same TP size cannot disagree on the draft head split.
            (2, 2, True, 8, 4, {}, "multiple of a smaller prefill"),
            # Widths that do not split into whole decode shards.
            (2, 4, True, 6, 4, {}, "decode head shards"),
            (1, 2, False, 16, 4, {}, "decode head shards"),
            # Under prefill CP the draft may not be sharded over attention TP.
            (2, 4, True, 8, 4, {"cp_size": 2}, "prefill attention CP"),
            # Head-outer pages (ROCm AITER) or a pool that declares no layout.
            (2, 4, True, 8, 4, {"layout": "vectorized_5d"}, "NHD KV cache layout"),
            (2, 4, False, 8, 4, {"layout": None}, "NHD KV cache layout"),
            # A decode missing a target layer cannot be paired at all.
            (2, 4, False, 8, 4, {"dst_layer_ids": [3, 9, 8, 8]}, "Could not plan"),
        ):
            with self.subTest(
                src_tp=src_tp,
                dst_tp=dst_tp,
                pure_mla=pure_mla,
                widths=(src_width, dst_width),
                **kw,
            ):
                _, manager, registration = self._peer(
                    src_tp, dst_tp, pure_mla, src_width, dst_width, **kw
                )
                mismatch = _register(manager, registration)
                self.assertIsNotNone(mismatch)
                self.assertIn(reason, mismatch)
                self.assertEqual(manager.state_layout_rejections, {SESSION: mismatch})
                self.assertIsNone(registration.draft_kv_head_slices)

    def test_registration_accepts_layouts_the_transfer_handles(self):
        for src_tp, dst_tp, src_width, dst_width, kw, sliced in (
            (2, 4, 8, 4, {}, True),  # resliced draft heads
            (2, 2, 8, 8, {}, False),  # equal widths
            # Equal widths keep the page copy in any layout both sides share.
            (2, 2, 8, 8, {"layout": "vectorized_5d"}, False),
            # CP/TP group sharing: attention TP 1, draft on the physical TP.
            (1, 4, 4, 4, {"cp_size": 4, "state_tp": 4}, False),
            # Paths that do not use send_kvcache's draft slicing.
            (2, 4, 8, 4, {"dcp": True}, False),
            (4, 2, 4, 8, {"mla": False}, False),
            # A decode without per-entry item lengths keeps the legacy copy.
            (4, 2, 4, 8, {"publish_lens": False}, False),
        ):
            for pure_mla in (True, False):
                with self.subTest(
                    src_tp=src_tp,
                    dst_tp=dst_tp,
                    widths=(src_width, dst_width),
                    pure_mla=pure_mla,
                    **kw,
                ):
                    _, manager, registration = self._peer(
                        src_tp, dst_tp, pure_mla, src_width, dst_width, **kw
                    )
                    self.assertIsNone(_register(manager, registration))
                    self.assertEqual(
                        registration.draft_kv_head_slices is not None, sliced
                    )
                    self.assertIs(manager.decode_kv_args_table[SESSION], registration)

    def test_ascend_accepts_only_unsliced_draft_pages(self):
        """Ascend's own copy moves whole draft pages, so it rejects a decode
        rank that needs a head slice of them."""
        for dst_tp, dst_width, reason in ((4, 4, "Ascend PD cannot"), (2, 8, None)):
            for pure_mla in (True, False):
                with self.subTest(dst_tp=dst_tp, pure_mla=pure_mla):
                    _, manager, registration = self._peer(
                        2, dst_tp, pure_mla, 8, dst_width, manager_class=AscendKVManager
                    )
                    mismatch = _register(manager, registration)
                    if reason is None:
                        self.assertIsNone(mismatch)
                    else:
                        self.assertIn(reason, mismatch)
                    self.assertIsNone(registration.draft_kv_head_slices)

    def test_rejected_peer_fails_requests_without_writing(self):
        for src_tp, dst_tp, src_width, dst_width, kw in (
            (4, 2, 4, 8, {}),
            (2, 4, 8, 4, {"layout": "vectorized_5d"}),
        ):
            with self.subTest(src_tp=src_tp, dst_tp=dst_tp, **kw):
                memory, manager, registration = self._peer(
                    src_tp, dst_tp, True, src_width, dst_width, **kw
                )
                mismatch = _register(manager, registration)
                self.assertIsNotNone(mismatch)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    _run_worker(manager, registration, executor)
                manager.conclude_failure.assert_called_once_with(
                    bootstrap_room=ROOM, failure_reason=mismatch
                )
                self.assertEqual(memory.num_batches, 0)


if __name__ == "__main__":
    unittest.main()
