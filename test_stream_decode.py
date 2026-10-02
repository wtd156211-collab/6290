#!/usr/bin/env python3
"""增量 UTF-8 解码的 unittest 测试。

运行：python -m unittest test_stream_decode -v
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from incremental_utf8 import (  # noqa: E402
    CharEvent,
    CoverageChecker,
    IllegalEvent,
    IncrementalUTF8Decoder,
    REPLACEMENT,
)
from stream_decode import json_escape_char, list_chunk_files, main as cli_main  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
CHUNKS_DIR = os.path.join(ROOT, "samples", "chunks")
EXPECTED_DIR = os.path.join(ROOT, "samples", "expected")


def read_stream(case):
    """拼接样例字节流（仅测试内允许）。"""
    case_dir = os.path.join(CHUNKS_DIR, case)
    parts = []
    for name in list_chunk_files(case_dir):
        with open(os.path.join(case_dir, name), "rb") as fh:
            parts.append(fh.read())
    return b"".join(parts)


def read_expected(case):
    decoded, illegal = [], []
    with open(os.path.join(EXPECTED_DIR, case + ".decoded.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                decoded.append(json.loads(line))
    with open(os.path.join(EXPECTED_DIR, case + ".illegal.jsonl"),
              encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                illegal.append(json.loads(line))
    return decoded, illegal


def decode_chunks(chunks):
    """用引擎按给定分块解码，返回 (chars, illegals, decoder)。"""
    dec = IncrementalUTF8Decoder()
    chars, illegals = [], []
    coverage = CoverageChecker()
    for chunk in chunks:
        for ev in dec.feed(chunk):
            if isinstance(ev, CharEvent):
                coverage.note(ev.start, ev.end)
                chars.append(ev)
            else:
                illegals.append(ev)
    for ev in dec.finish():
        if isinstance(ev, CharEvent):
            coverage.note(ev.start, ev.end)
            chars.append(ev)
        else:
            illegals.append(ev)
    coverage.complete(sum(len(c) for c in chunks))
    return chars, illegals, dec


def chars_to_records(chars):
    return [
        {"char": ev.char, "span": [ev.start, ev.end], "complete": ev.complete}
        for ev in chars
    ]


def illegals_to_records(illegals):
    return [
        {"span": [ev.start, ev.end], "reason": ev.reason} for ev in illegals
    ]


CASES = sorted(os.listdir(CHUNKS_DIR))


class TestSamples(unittest.TestCase):
    """验收口径 1：逐用例与期望做字段级比较。"""

    def test_all_cases_match_expected(self):
        for case in CASES:
            with self.subTest(case=case):
                case_dir = os.path.join(CHUNKS_DIR, case)
                chunks = []
                for name in list_chunk_files(case_dir):
                    with open(os.path.join(case_dir, name), "rb") as fh:
                        chunks.append(fh.read())
                chars, illegals, _ = decode_chunks(chunks)
                exp_dec, exp_ill = read_expected(case)
                self.assertEqual(chars_to_records(chars), exp_dec)
                self.assertEqual(illegals_to_records(illegals), exp_ill)

    def test_cli_outputs_byte_identical(self):
        for case in CASES:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as out_dir:
                    rc = cli_main(
                        ["stream_decode.py",
                         os.path.join(CHUNKS_DIR, case), out_dir])
                    self.assertEqual(rc, 0)
                    for kind in ("decoded", "illegal"):
                        with open(os.path.join(
                                EXPECTED_DIR,
                                "%s.%s.jsonl" % (case, kind)), "rb") as fh:
                            expected = fh.read()
                        with open(os.path.join(out_dir, kind + ".jsonl"),
                                  "rb") as fh:
                            actual = fh.read()
                        self.assertEqual(actual, expected)


class TestChunkingInvariance(unittest.TestCase):
    """验收口径 2：同一字节流任意切分，输出必须完全一致。"""

    def test_rechunking(self):
        rng = random.Random(20260928)
        for case in CASES:
            stream = read_stream(case)
            exp_dec, exp_ill = read_expected(case)
            splits = [
                [stream[i:i + 1] for i in range(len(stream))],
                [stream[i:i + 2] for i in range(0, len(stream), 2)],
                [stream[i:i + 3] for i in range(0, len(stream), 3)],
                [stream[i:i + 7] for i in range(0, len(stream), 7)],
                [stream],
                [b"", stream, b""],
            ]
            random_split, pos = [], 0
            while pos < len(stream):
                step = rng.randint(1, 5)
                random_split.append(stream[pos:pos + step])
                pos += step
            splits.append(random_split)
            for chunks in splits:
                with self.subTest(case=case, chunks=len(chunks)):
                    chars, illegals, _ = decode_chunks(chunks)
                    self.assertEqual(chars_to_records(chars), exp_dec)
                    self.assertEqual(illegals_to_records(illegals), exp_ill)

    def test_07_and_08_are_same_stream(self):
        self.assertEqual(read_stream("07-log-scene-mixed"),
                         read_stream("08-same-stream-one-byte-chunks"))


class TestStateMachine(unittest.TestCase):
    """针对状态机与原因码的单元测试。"""

    def decode(self, data, chunk_size=None):
        if chunk_size is None:
            chunks = [data]
        else:
            chunks = [data[i:i + chunk_size]
                      for i in range(0, len(data), chunk_size)] or [b""]
        return decode_chunks(chunks)

    def test_ascii(self):
        chars, illegals, _ = self.decode(b"ab\x00c")
        self.assertEqual([c.char for c in chars], ["a", "b", "\x00", "c"])
        self.assertEqual([(c.start, c.end) for c in chars],
                         [(0, 1), (1, 2), (2, 3), (3, 4)])
        self.assertTrue(all(c.complete for c in chars))
        self.assertEqual(illegals, [])

    def test_two_three_four_byte(self):
        chars, _, _ = self.decode("½汉😀".encode("utf-8"))
        self.assertEqual([c.char for c in chars], ["½", "汉", "😀"])
        self.assertEqual([(c.start, c.end) for c in chars],
                         [(0, 2), (2, 5), (5, 9)])

    def test_bom_is_plain_character(self):
        chars, illegals, _ = self.decode(b"\xef\xbb\xbf")
        self.assertEqual(len(chars), 1)
        self.assertEqual(chars[0].char, "﻿")  # U+FEFF 原样输出
        self.assertTrue(chars[0].complete)
        self.assertEqual(illegals, [])

    def test_real_replacement_char_is_complete(self):
        chars, illegals, _ = self.decode(b"\xef\xbf\xbd")
        self.assertEqual(chars[0].char, REPLACEMENT)
        self.assertTrue(chars[0].complete)
        self.assertEqual(illegals, [])

    def test_invalid_lead_bytes(self):
        for b in (0x80, 0xBF, 0xC0, 0xC1, 0xF5, 0xFF):
            with self.subTest(byte=b):
                chars, illegals, _ = self.decode(bytes([b]))
                self.assertEqual(chars[0].char, REPLACEMENT)
                self.assertFalse(chars[0].complete)
                self.assertEqual(illegals_to_records(illegals),
                                 [{"span": [0, 1], "reason": "invalid_lead"}])

    def test_overlong_e0(self):
        chars, illegals, _ = self.decode(b"\xe0\x80\x80")
        self.assertEqual([c.complete for c in chars], [False] * 3)
        self.assertEqual(illegals_to_records(illegals), [
            {"span": [0, 1], "reason": "invalid_continuation"},
            {"span": [1, 2], "reason": "invalid_lead"},
            {"span": [2, 3], "reason": "invalid_lead"},
        ])

    def test_surrogate_ed(self):
        _, illegals, _ = self.decode(b"\xed\xa0\x80")
        self.assertEqual(illegals_to_records(illegals), [
            {"span": [0, 1], "reason": "invalid_continuation"},
            {"span": [1, 2], "reason": "invalid_lead"},
            {"span": [2, 3], "reason": "invalid_lead"},
        ])

    def test_out_of_range_f4(self):
        _, illegals, _ = self.decode(b"\xf4\x90\x80\x80")
        self.assertEqual(illegals_to_records(illegals), [
            {"span": [0, 1], "reason": "invalid_continuation"},
            {"span": [1, 2], "reason": "invalid_lead"},
            {"span": [2, 3], "reason": "invalid_lead"},
            {"span": [3, 4], "reason": "invalid_lead"},
        ])

    def test_interrupted_by_ascii(self):
        chars, illegals, _ = self.decode(b"\xe2\x81(")
        self.assertEqual(chars[-1].char, "(")
        self.assertEqual((chars[-1].start, chars[-1].end), (2, 3))
        self.assertEqual(illegals_to_records(illegals),
                         [{"span": [0, 2], "reason": "invalid_continuation"}])

    def test_truncated_at_eof(self):
        _, illegals, _ = self.decode(b"\xe4\xb8")
        self.assertEqual(illegals_to_records(illegals),
                         [{"span": [0, 2], "reason": "truncated_at_eof"}])

    def test_invalid_byte_rejudged_as_lead(self):
        # 非法续字节之后，当前字节要按起始字节重新判定
        chars, _, _ = self.decode(b"\xe4\x41")  # E2 81 类：E4 被 'A' 打断
        self.assertEqual(chars[1].char, "A")
        self.assertTrue(chars[1].complete)

    def test_split_everywhere(self):
        # 每个可能的切点都试一遍，结果必须一致
        data = "收到：中😀".encode("utf-8") + b"\x80" + b"\xef\xbf\xbd"
        chars1, illegals1, _ = self.decode(data)
        for cut in range(len(data) + 1):
            with self.subTest(cut=cut):
                chars2, illegals2, _ = decode_chunks(
                    [data[:cut], data[cut:]])
                self.assertEqual(chars1, chars2)
                self.assertEqual(illegals1, illegals2)

    def test_empty_stream_and_empty_chunks(self):
        chars, illegals, dec = decode_chunks([b"", b""])
        self.assertEqual(chars, [])
        self.assertEqual(illegals, [])
        self.assertEqual(dec.position, 0)

    def test_offsets_are_absolute_across_chunks(self):
        chars, _, _ = decode_chunks([b"ab", "汉".encode("utf-8"), b"cd"])
        self.assertEqual([(c.start, c.end) for c in chars],
                         [(0, 1), (1, 2), (2, 5), (5, 6), (6, 7)])


class TestResumeCounter(unittest.TestCase):
    """跨块续上次数：与用 span/块边界独立推算的结果一致。"""

    def expected_resumes(self, chunks, chars):
        boundaries = set()
        pos = 0
        for chunk in chunks[:-1]:
            pos += len(chunk)
            boundaries.add(pos)
        boundaries.discard(0)
        count = 0
        for ev in chars:
            # 已收字节区间 (start, end) 内每跨过一个块边界即续上一次；
            # 之后序列是否被打断不影响「续上」这一事实
            for b in boundaries:
                if ev.start < b < ev.end:
                    count += 1
        return count

    def test_resume_counter(self):
        rng = random.Random(7)
        for case in CASES:
            stream = read_stream(case)
            for chunks in (
                [stream[i:i + 1] for i in range(len(stream))],
                [stream[i:i + 3] for i in range(0, len(stream), 3)],
                [stream],
            ):
                chars, _, dec = decode_chunks(chunks)
                self.assertEqual(dec.continuation_resumes,
                                 self.expected_resumes(chunks, chars),
                                 msg=case)
        # 随机切分也来几轮
        stream = read_stream("07-log-scene-mixed")
        for _ in range(20):
            chunks, pos = [], 0
            while pos < len(stream):
                step = rng.randint(1, 4)
                chunks.append(stream[pos:pos + step])
                pos += step
            chars, _, dec = decode_chunks(chunks)
            self.assertEqual(dec.continuation_resumes,
                             self.expected_resumes(chunks, chars))


class TestJsonEscape(unittest.TestCase):
    def test_matches_json_module(self):
        for cp in list(range(0x80)) + [0x4E2D, 0x1F600, 0xFFFD, 0xFEFF]:
            ch = chr(cp)
            dumped = json.dumps(ch, ensure_ascii=False)[1:-1]
            self.assertEqual(json_escape_char(ch), dumped, msg=hex(cp))


class TestCli(unittest.TestCase):
    def test_bad_args_exit_nonzero(self):
        self.assertEqual(cli_main(["stream_decode.py"]), 2)
        self.assertEqual(
            cli_main(["stream_decode.py", "/no/such/dir", "/tmp/x"]), 2)

    def test_empty_chunk_dir_produces_empty_outputs(self):
        with tempfile.TemporaryDirectory() as chunk_dir, \
                tempfile.TemporaryDirectory() as out_dir:
            rc = cli_main(["stream_decode.py", chunk_dir, out_dir])
            self.assertEqual(rc, 0)
            for kind in ("decoded", "illegal"):
                with open(os.path.join(out_dir, kind + ".jsonl"), "rb") as fh:
                    self.assertEqual(fh.read(), b"")


class TestAgainstReference(unittest.TestCase):
    """随机合法流：与 Python 内置解码交叉验证字符与偏移。"""

    def test_random_valid_streams(self):
        rng = random.Random(99)
        for _ in range(30):
            text = "".join(
                chr(rng.choice(
                    [rng.randint(0x20, 0x7E),
                     rng.randint(0xA0, 0x2FFF),
                     rng.randint(0x4E00, 0x9FFF),
                     rng.randint(0x10000, 0x10FFF)]))
                for _ in range(rng.randint(0, 200)))
            data = text.encode("utf-8")
            size = rng.randint(1, 5)
            chunks = [data[i:i + size] for i in range(0, len(data), size)]
            chars, illegals, dec = decode_chunks(chunks or [b""])
            self.assertEqual("".join(c.char for c in chars), text)
            self.assertEqual(illegals, [])
            # 用偏移切回原字节，重新编码必须一致
            for c in chars:
                self.assertEqual(data[c.start:c.end],
                                 c.char.encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
