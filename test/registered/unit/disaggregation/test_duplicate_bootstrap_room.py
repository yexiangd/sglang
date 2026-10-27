"""A bootstrap_room that is already in use is rejected, not shared."""

import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
    KVTransferError,
)
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROOM = 424242
BOOTSTRAP_ADDR = "127.0.0.1:8998"


def make_manager(mode, *, defer_decode_allocation=False):
    """Room bookkeeping of a CommonKVManager, without its engine or threads."""
    mgr = CommonKVManager.__new__(CommonKVManager)
    mgr.disaggregation_mode = mode
    mgr.defer_decode_allocation = defer_decode_allocation
    mgr.request_status = {}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr._staging_outstanding = {}
    mgr._deferred_ack_targets = {}
    mgr._deferred_ack_poisoned_rooms = set()
    if mode == DisaggregationMode.PREFILL:
        mgr.is_dummy_cp_rank = False
        mgr.bootstrap_timeout = 300
        mgr.sender_rooms = set()
        mgr.transfer_infos = {}
        mgr.req_to_decode_prefix_len = {}
    else:
        mgr.waiting_timeout = 300
        mgr.addr_to_rooms_tracker = defaultdict(set)
        mgr.prefill_response_tracker = {}
        mgr.required_prefill_response_num_table = {}
    return mgr


def make_sender(mgr):
    with get_context().override_server_args(dp_size=1, enable_trace=False):
        return CommonKVSender(mgr, BOOTSTRAP_ADDR, ROOM)


class TestDuplicateDecodeRoom(unittest.TestCase):
    def setUp(self):
        self.mgr = make_manager(DisaggregationMode.DECODE)
        self.owner = CommonKVReceiver(self.mgr, BOOTSTRAP_ADDR, ROOM)

    def assert_room_state_intact(self, status=KVPoll.Bootstrapping):
        self.assertEqual(self.mgr.request_status[ROOM], status)
        self.assertIn(ROOM, self.mgr.prefill_response_tracker)
        self.assertIn(ROOM, self.mgr.addr_to_rooms_tracker[BOOTSTRAP_ADDR])
        self.assertNotIn(ROOM, self.mgr.failure_records)

    def test_duplicate_fails_alone_with_its_own_reason(self):
        duplicate = CommonKVReceiver(self.mgr, BOOTSTRAP_ADDR, ROOM)
        self.assertEqual(duplicate.poll(), KVPoll.Failed)
        with self.assertRaises(KVTransferError) as ctx:
            duplicate.failure_exception()
        self.assertIn("Duplicate decode bootstrap room", str(ctx.exception))
        self.assertFalse(ctx.exception.is_from_another_rank)
        duplicate.abort()
        self.assert_room_state_intact()
        self.mgr.update_status(ROOM, KVPoll.Success)
        self.assertEqual(self.owner.poll(), KVPoll.Success)

    def test_duplicate_in_handshake_queue_is_aborted_instead_of_raising(self):
        # The owner left the queue and its transfer finished while the
        # duplicate was still waiting for preallocation.
        duplicate = CommonKVReceiver(self.mgr, BOOTSTRAP_ADDR, ROOM)
        self.mgr.update_status(ROOM, KVPoll.Success)
        req = SimpleNamespace(
            rid="duplicate",
            bootstrap_room=ROOM,
            return_logprob=False,
            finished_reason=None,
        )
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.queue = [DecodeRequest(req=req, kv_receiver=duplicate)]
        queue.pp_size = 1
        queue.tp_rank = 0
        queue.gloo_group = None
        queue.scheduler = MagicMock()
        queue.scheduler.metrics_reporter.enable_metrics = False
        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            side_effect=lambda pollers, group: [int(p.poll()) for p in pollers],
        ):
            queue._update_handshake_waiters()
        self.assertIn("Duplicate decode bootstrap room", req.finished_reason.message)
        self.assertEqual(self.owner.poll(), KVPoll.Success)
        self.assertEqual(self.mgr.request_status[ROOM], KVPoll.Success)

    def test_duplicate_poll_after_owner_clear_does_not_raise(self):
        duplicate = CommonKVReceiver(self.mgr, BOOTSTRAP_ADDR, ROOM)
        self.mgr.update_status(ROOM, KVPoll.Success)
        self.assertEqual(self.owner.poll(), KVPoll.Success)
        self.owner.clear()
        self.assertEqual(duplicate.poll(), KVPoll.Failed)

    def test_cleared_receiver_leaves_next_owner_alone(self):
        # Deferred KV release clears a failed receiver again later, when a new
        # request may already own the room.
        self.mgr.update_status(ROOM, KVPoll.Failed)
        with self.assertRaises(KVTransferError):
            self.owner.failure_exception()
        successor = CommonKVReceiver(self.mgr, BOOTSTRAP_ADDR, ROOM)
        self.assertEqual(successor.poll(), KVPoll.Bootstrapping)
        self.owner.clear()
        self.owner.abort()
        self.assertEqual(successor.poll(), KVPoll.Bootstrapping)
        self.assert_room_state_intact()

    def test_prefill_complete_fails_both_requests(self):
        mgr = make_manager(DisaggregationMode.DECODE, defer_decode_allocation=True)
        owner = CommonKVReceiver(mgr, BOOTSTRAP_ADDR, ROOM)
        duplicate = CommonKVReceiver(mgr, BOOTSTRAP_ADDR, ROOM)
        for receiver in (duplicate, owner):
            self.assertEqual(receiver.poll(), KVPoll.Failed)
            with self.assertRaisesRegex(
                KVTransferError, "Duplicate decode bootstrap room"
            ):
                receiver.failure_exception()

    def test_prefill_complete_duplicate_after_owner_success_leaves_no_record(self):
        # The metadata gate can keep an owner that already saw Success in the
        # transfer queue when the duplicate arrives.
        mgr = make_manager(DisaggregationMode.DECODE, defer_decode_allocation=True)
        owner = CommonKVReceiver(mgr, BOOTSTRAP_ADDR, ROOM)
        mgr.update_status(ROOM, KVPoll.Success)
        self.assertEqual(owner.poll(), KVPoll.Success)
        duplicate = CommonKVReceiver(mgr, BOOTSTRAP_ADDR, ROOM)
        with self.assertRaisesRegex(KVTransferError, "Duplicate decode bootstrap room"):
            duplicate.failure_exception()
        self.assertEqual(owner.poll(), KVPoll.Success)
        owner.clear()
        self.assertNotIn(ROOM, mgr.failure_records)
        # A later request on this room that fails on another rank must not
        # report the duplicate's reason as its own.
        successor = CommonKVReceiver(mgr, BOOTSTRAP_ADDR, ROOM)
        mgr.update_status(ROOM, KVPoll.Failed)
        with self.assertRaises(KVTransferError) as ctx:
            successor.failure_exception()
        self.assertTrue(ctx.exception.is_from_another_rank)


class TestDuplicatePrefillRoom(unittest.TestCase):
    def setUp(self):
        self.mgr = make_manager(DisaggregationMode.PREFILL)

    def register_decode(self):
        # What a backend bootstrap thread records when decode registers first.
        self.mgr.transfer_infos[ROOM] = {"decode-session": object()}
        self.mgr.req_to_decode_prefix_len[ROOM] = 16
        self.mgr.update_status(ROOM, KVPoll.WaitingForInput)

    def assert_room_state_intact(self, status=KVPoll.WaitingForInput):
        self.assertEqual(self.mgr.request_status[ROOM], status)
        self.assertIn(ROOM, self.mgr.transfer_infos)
        self.assertEqual(self.mgr.req_to_decode_prefix_len[ROOM], 16)
        self.assertNotIn(ROOM, self.mgr.failure_records)

    def test_first_sender_after_early_decode_registration_owns_room(self):
        self.register_decode()
        owner = make_sender(self.mgr)
        self.assertEqual(owner.poll(), KVPoll.WaitingForInput)

    def test_duplicate_fails_alone_with_its_own_reason(self):
        for is_dummy_cp_rank in (False, True):
            with self.subTest(is_dummy_cp_rank=is_dummy_cp_rank):
                self.mgr = make_manager(DisaggregationMode.PREFILL)
                self.mgr.is_dummy_cp_rank = is_dummy_cp_rank
                self.register_decode()
                owner = make_sender(self.mgr)
                duplicate = make_sender(self.mgr)
                self.assertEqual(duplicate.poll(), KVPoll.Failed)
                with self.assertRaises(KVTransferError) as ctx:
                    duplicate.failure_exception()
                self.assertIn("Duplicate prefill bootstrap room", str(ctx.exception))
                self.assertFalse(ctx.exception.is_from_another_rank)
                duplicate.abort()
                self.assert_room_state_intact()
                self.assertEqual(owner.poll(), KVPoll.WaitingForInput)

    def test_cleared_sender_leaves_next_owner_alone(self):
        self.register_decode()
        owner = make_sender(self.mgr)
        self.mgr.update_status(ROOM, KVPoll.Success)
        self.assertEqual(owner.poll(), KVPoll.Success)
        owner.clear()
        self.register_decode()
        successor = make_sender(self.mgr)
        self.assertEqual(successor.poll(), KVPoll.WaitingForInput)
        owner.clear()
        owner.abort()
        self.assertEqual(successor.poll(), KVPoll.WaitingForInput)
        self.assert_room_state_intact()


if __name__ == "__main__":
    unittest.main()
