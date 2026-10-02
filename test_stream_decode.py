#!/usr/bin/env python3
"""stream_decode 的 unittest 测试。"""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from stream_decode import (  # noqa: E402
    INVALID_CONTINUATION,
    INVALID_LEAD,
    REPLACEMENT_CHAR,
    TRUNCATED_AT_EOF,
    StreamDecoder,
    run,
)

SAMPLES_CHUNKS = os.path.join(HERE, "samples", "chunks")
SAMPLES_EXPECTED = os.path.join(HERE, "samples", "expected")


def decode_bytes(stream, chunk_sizes=None):
    """用引擎解码整条字节流，返回 (decoded, illegal) 记录列表。"""
    if chunk_sizes is None:
        chunks = [stream]
    else:
        chunks = []
        pos = 0
        idx = 0
        while pos < len(stream):
            size = chunk_sizes[idx % len(chunk_sizes)]
            chunks.append(stream[pos:pos + size])
            pos += size
            idx += 1
    decoder = StreamDecoder()
    decoded, illegal = [], []

    def emit(char, start, end, complete, reason):
        decoded.append({"char": char, "span": [start, end],
                        "complete": complete})
        if not complete:
            illegal.append({"span": [start, end], "reason": reason})

    for chunk in chunks:
        decoder.feed(chunk, emit)
    decoder.finish(emit)
    return decoded, illegal


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_names():
    return sorted(os.listdir(SAMPLES_CHUNKS))


def sample_stream(name):
    parts = []
    dirpath = os.path.join(SAMPLES_CHUNKS, name)
    for fname in sorted(os.listdir(dirpath)):
        with open(os.path.join(dirpath, fname), "rb") as f:
            parts.append(f.read())
    return b"".join(parts)


class TestSamples(unittest.TestCase):
    """逐用例与 samples/expected 字段级比较。"""

    def test_all_samples(self):
        for name in sample_names():
            with self.subTest(sample=name), \
                    tempfile.TemporaryDirectory() as out:
                rc = run(os.path.join(SAMPLES_CHUNKS, name), out)
                self.assertEqual(rc, 0)
                for kind in ("decoded", "illegal"):
                    got = read_jsonl(os.path.join(out, f"{kind}.jsonl"))
                    exp = read_jsonl(os.path.join(
                        SAMPLES_EXPECTED, f"{name}.{kind}.jsonl"))
                    self.assertEqual(got, exp, f"{name}.{kind}")


class TestChunkingInvariance(unittest.TestCase):
    """同一字节流的任意切分必须得到完全相同的输出。"""

    def test_random_rechunking(self):
        rng = random.Random(20261002)
        for name in sample_names():
            stream = sample_stream(name)
            exp_decoded = read_jsonl(os.path.join(
                SAMPLES_EXPECTED, f"{name}.decoded.jsonl"))
            exp_illegal = read_jsonl(os.path.join(
                SAMPLES_EXPECTED, f"{name}.illegal.jsonl"))
            for trial in range(5):
                with self.subTest(sample=name, trial=trial):
                    sizes = []
                    remaining = len(stream)
                    while remaining > 0:
                        size = rng.randint(0, 7)
                        sizes.append(size)
                        remaining -= size
                    got_decoded, got_illegal = decode_bytes(stream, sizes)
                    self.assertEqual(got_decoded, exp_decoded)
                    self.assertEqual(got_illegal, exp_illegal)

    def test_byte_per_byte(self):
        for name in sample_names():
            with self.subTest(sample=name):
                stream = sample_stream(name)
                got_decoded, got_illegal = decode_bytes(stream, [1])
                exp_decoded = read_jsonl(os.path.join(
                    SAMPLES_EXPECTED, f"{name}.decoded.jsonl"))
                exp_illegal = read_jsonl(os.path.join(
                    SAMPLES_EXPECTED, f"{name}.illegal.jsonl"))
                self.assertEqual(got_decoded, exp_decoded)
                self.assertEqual(got_illegal, exp_illegal)


class TestStateMachine(unittest.TestCase):
    """状态机与非法序列口径的单元测试。"""

    def test_invalid_lead_bytes(self):
        decoded, illegal = decode_bytes(b"\x80\xc0\xc1\xf5\xff")
        self.assertEqual(len(decoded), 5)
        self.assertTrue(all(not r["complete"] for r in decoded))
        self.assertEqual([r["reason"] for r in illegal],
                         [INVALID_LEAD] * 5)
        self.assertEqual([r["span"] for r in illegal],
                         [[i, i + 1] for i in range(5)])

    def test_invalid_continuation_longest_prefix(self):
        # E4 B8 是合法前缀，0x21 打断 -> 替换字符覆盖 [0,2)，0x21 重新判定
        decoded, illegal = decode_bytes(b"\xe4\xb8!")
        self.assertEqual(decoded[0], {"char": REPLACEMENT_CHAR,
                                      "span": [0, 2], "complete": False})
        self.assertEqual(decoded[1], {"char": "!", "span": [2, 3],
                                      "complete": True})
        self.assertEqual(illegal, [{"span": [0, 2],
                                    "reason": INVALID_CONTINUATION}])

    def test_invalid_continuation_reprocess_as_lead(self):
        # E4 B8 后紧跟合法起始字节 E5：E5 应作为新序列起点重新判定
        decoded, _ = decode_bytes("中".encode("utf-8")[:2] +
                                  "汉".encode("utf-8"))
        self.assertEqual(decoded[0]["complete"], False)
        self.assertEqual(decoded[0]["span"], [0, 2])
        self.assertEqual(decoded[1], {"char": "汉", "span": [2, 5],
                                      "complete": True})

    def test_truncated_at_eof(self):
        decoded, illegal = decode_bytes(b"ab\xf0\x9f\x98")
        self.assertEqual(decoded[-1], {"char": REPLACEMENT_CHAR,
                                       "span": [2, 5], "complete": False})
        self.assertEqual(illegal, [{"span": [2, 5],
                                    "reason": TRUNCATED_AT_EOF}])

    def test_real_replacement_char_is_complete(self):
        # 流中真实编码的 EF BF BD 是 complete=true
        decoded, illegal = decode_bytes(b"\xef\xbf\xbd")
        self.assertEqual(decoded, [{"char": REPLACEMENT_CHAR,
                                    "span": [0, 3], "complete": True}])
        self.assertEqual(illegal, [])

    def test_overlong_and_surrogate_and_out_of_range(self):
        cases = [
            b"\xe0\x80\x80",      # 过长编码
            b"\xed\xa0\x80",      # 代理区
            b"\xf4\x90\x80\x80",  # 超出 U+10FFFF
            b"\xc1\xbf",          # C1 不能作起始字节
        ]
        for stream in cases:
            with self.subTest(stream=stream):
                decoded, illegal = decode_bytes(stream)
                self.assertTrue(all(not r["complete"] for r in decoded))
                self.assertTrue(len(illegal) >= 1)

    def test_split_everywhere_equivalent(self):
        # 一个汉字被劈在任意位置，结果与整体喂入一致
        stream = "日志：中文😀".encode("utf-8")
        whole, _ = decode_bytes(stream)
        for cut in range(len(stream) + 1):
            with self.subTest(cut=cut):
                got, _ = decode_bytes(stream, [cut, len(stream)])
                self.assertEqual(got, whole)


class TestCoverageInvariant(unittest.TestCase):
    """span 严格递增、互不重叠、并集恰好是 [0, N)。"""

    def test_coverage(self):
        rng = random.Random(7)
        stream = bytes(rng.randrange(256) for _ in range(4096))
        for trial in range(3):
            with self.subTest(trial=trial):
                sizes = [rng.randint(0, 5) for _ in range(2000)]
                decoded, illegal = decode_bytes(stream, sizes)
                pos = 0
                for record in decoded:
                    start, end = record["span"]
                    self.assertEqual(start, pos)
                    self.assertGreater(end, start)
                    pos = end
                self.assertEqual(pos, len(stream))
                # decoded 中 complete=false 与 illegal 一一对应
                false_spans = [r["span"] for r in decoded
                               if not r["complete"]]
                self.assertEqual(false_spans,
                                 [r["span"] for r in illegal])


class TestDeterminismAndReport(unittest.TestCase):
    """同样本跑两遍输出逐字节相同；报告数字与引擎输出一致。"""

    def test_deterministic_and_report(self):
        for name in ("07-log-scene-mixed", "06-illegal-multibyte-forms"):
            with self.subTest(sample=name):
                outputs = []
                for _ in range(2):
                    out = tempfile.mkdtemp()
                    self.assertEqual(
                        run(os.path.join(SAMPLES_CHUNKS, name), out), 0)
                    outputs.append(out)
                files = {}
                for out in outputs:
                    for fname in ("decoded.jsonl", "illegal.jsonl",
                                  "report.txt"):
                        with open(os.path.join(out, fname), "rb") as f:
                            files.setdefault(fname, []).append(f.read())
                for fname, pair in files.items():
                    self.assertEqual(pair[0], pair[1], fname)

                # 报告数字与引擎输出核对
                out = outputs[0]
                decoded = read_jsonl(os.path.join(out, "decoded.jsonl"))
                illegal = read_jsonl(os.path.join(out, "illegal.jsonl"))
                with open(os.path.join(out, "report.txt"),
                          encoding="utf-8") as f:
                    report = f.read()
                complete = sum(1 for r in decoded if r["complete"])
                self.assertIn(f"解出字符数: {len(decoded)} "
                              f"(完整 {complete}, 替换 {len(illegal)})",
                              report)
                self.assertIn(f"非法序列数: {len(illegal)}", report)
                stream = sample_stream(name)
                self.assertIn(f"总字节数: {len(stream)}", report)
                for rec in illegal:
                    self.assertIn(
                        f"[{rec['span'][0]},{rec['span'][1]}) "
                        f"{rec['reason']}", report)


class TestStreaming(unittest.TestCase):
    """内存不随流长增长：状态机只保留常数状态，引擎可处理 MiB 级流。"""

    def test_large_stream(self):
        rng = random.Random(11)
        unit = "log 中文 😀 ok\n".encode("utf-8")
        total = 4 * 1024 * 1024
        stream = (unit * (total // len(unit) + 1))[:total]
        decoder = StreamDecoder()
        count = [0]

        def emit(char, start, end, complete, reason):
            count[0] += 1
            self.assertTrue(complete)

        fed = 0
        while fed < total:
            n = min(rng.randint(1, 64), total - fed)
            decoder.feed(stream[fed:fed + n], emit)
            fed += n
        decoder.finish(emit)
        # 状态机不缓存已处理内容：没有随流长增长的字段
        self.assertLessEqual(decoder.seq_have, 3)
        self.assertGreater(count[0], 0)


if __name__ == "__main__":
    unittest.main()
