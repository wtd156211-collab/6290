#!/usr/bin/env python3
"""按块对字节流做增量式严格 UTF-8 解码。

用法：python stream_decode.py <块目录> <输出目录>

输出目录下生成：
    decoded.jsonl  {"char": ..., "span": [s, e], "complete": bool}
    illegal.jsonl  {"span": [s, e], "reason": "..."}

块目录内的 chunk-NNN.bin 按文件名字典序读取，不递归。不拼接整条流，
不把合法性判定交给 bytes.decode / codecs / TextIOWrapper 等现成设施。
输出行是手写的最小 JSON 序列化（仅字符串转义），与 json 规范一致。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from incremental_utf8 import (  # noqa: E402
    CharEvent,
    CoverageChecker,
    IncrementalUTF8Decoder,
)

CHUNK_PREFIX = "chunk-"
CHUNK_SUFFIX = ".bin"

# ASCII 控制字符的 JSON 短转义（其余控制字符用 \u00xx）。
_ESCAPES = {
    0x08: "\\b", 0x09: "\\t", 0x0A: "\\n", 0x0C: "\\f",
    0x0D: "\\r", 0x22: "\\\"", 0x5C: "\\\\",
}


def json_escape_char(ch):
    """把单个已解码字符转成 JSON 字符串内容（不含外层引号）。"""
    o = ord(ch)
    short = _ESCAPES.get(o)
    if short is not None:
        return short
    if o < 0x20:
        return "\\u%04x" % o
    return ch


def list_chunk_files(chunk_dir):
    """列出 chunk-NNN.bin，按文件名字典序；忽略子目录与其他文件。"""
    names = [
        name for name in os.listdir(chunk_dir)
        if name.startswith(CHUNK_PREFIX) and name.endswith(CHUNK_SUFFIX)
        and os.path.isfile(os.path.join(chunk_dir, name))
    ]
    names.sort()
    return names


def _handle_event(ev, coverage, decode_buf, illegal_buf, unpaired,
                  event_sink, chunk_index):
    """处理一个引擎事件：做覆盖/配对校验并生成输出行。"""
    if event_sink is not None:
        event_sink(chunk_index, ev)
    if isinstance(ev, CharEvent):
        coverage.note(ev.start, ev.end)
        decode_buf.append(
            '{"char":"%s","span":[%d,%d],"complete":%s}\n'
            % (json_escape_char(ev.char), ev.start, ev.end,
               "true" if ev.complete else "false"))
        if not ev.complete:
            unpaired[ev.start] = ev.end
    else:
        end = unpaired.pop(ev.start, None)
        if end is None or end != ev.end:
            raise AssertionError(
                "illegal 记录与替换字符不一致：[%d, %d)" % (ev.start, ev.end))
        illegal_buf.append(
            '{"span":[%d,%d],"reason":"%s"}\n' % (ev.start, ev.end, ev.reason))


def decode_directory(chunk_dir, decoded_out, illegal_out, event_sink=None):
    """读块目录并写出两个 jsonl 文件，返回 (decoder, names, total_bytes)。

    event_sink(chunk_index, event) 可选，每产出一个事件调用一次：块内事件
    chunk_index 为块编号，finish 阶段（truncated_at_eof）为 len(names)。
    供报告脚本直接取引擎数字，不二次扫描输出文件。
    """
    names = list_chunk_files(chunk_dir)
    decoder = IncrementalUTF8Decoder()
    coverage = CoverageChecker()
    unpaired = {}

    decode_buf = []
    illegal_buf = []
    total_bytes = 0
    out_d = open(decoded_out, "w", encoding="utf-8", newline="")
    out_i = open(illegal_out, "w", encoding="utf-8", newline="")
    try:
        for chunk_index, name in enumerate(names):
            with open(os.path.join(chunk_dir, name), "rb") as fh:
                data = fh.read()
            total_bytes += len(data)
            for ev in decoder.feed(data):
                _handle_event(ev, coverage, decode_buf, illegal_buf,
                              unpaired, event_sink, chunk_index)
            out_d.write("".join(decode_buf))
            out_i.write("".join(illegal_buf))
            decode_buf.clear()
            illegal_buf.clear()

        for ev in decoder.finish():
            _handle_event(ev, coverage, decode_buf, illegal_buf,
                          unpaired, event_sink, len(names))
        out_d.write("".join(decode_buf))
        out_i.write("".join(illegal_buf))

        coverage.complete(total_bytes)
        if unpaired:
            raise AssertionError("存在未配对的替换字符记录")
    finally:
        out_d.close()
        out_i.close()

    return decoder, names, total_bytes


def main(argv):
    if len(argv) != 3:
        sys.stderr.write("用法: python stream_decode.py <块目录> <输出目录>\n")
        return 2
    chunk_dir, out_dir = argv[1], argv[2]
    if not os.path.isdir(chunk_dir):
        sys.stderr.write("错误: 块目录不存在或不是目录: %s\n" % chunk_dir)
        return 2
    os.makedirs(out_dir, exist_ok=True)
    try:
        decode_directory(
            chunk_dir,
            os.path.join(out_dir, "decoded.jsonl"),
            os.path.join(out_dir, "illegal.jsonl"),
        )
    except OSError as exc:
        sys.stderr.write("IO 错误: %s\n" % exc)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
