#!/usr/bin/env python3
"""生成逐块解码报告（数字全部来自解码引擎，不做二次推算）。

用法：
    python decode_report.py <块目录>                 # 报告写到 stdout
    python decode_report.py <块目录> <报告文件路径>   # 报告写到文件

报告内容：逐块列出解出字符数、跨块续上次数、非法位置（span 与原因），
末尾给整条流的整体统计。用于定位「半个汉字」一类事故的精确字节位置。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from incremental_utf8 import (  # noqa: E402
    CharEvent,
    CoverageChecker,
    IllegalEvent,
    IncrementalUTF8Decoder,
)
from stream_decode import list_chunk_files  # noqa: E402


def run_engine(chunk_dir):
    """跑一遍引擎，返回 (decoder, names, chunk_sizes, per_chunk, total_bytes)。

    per_chunk[i] = {"chars": int, "resumed": int,
                    "illegals": [(start, end, reason), ...]}
    finish 阶段的事件（truncated_at_eof）计入最后一块。
    """
    names = list_chunk_files(chunk_dir)
    decoder = IncrementalUTF8Decoder()
    coverage = CoverageChecker()
    per_chunk = [
        {"chars": 0, "resumed": 0, "illegals": []} for _ in names
    ]
    chunk_sizes = []
    total_bytes = 0

    def record(index, ev):
        row = per_chunk[index]
        if isinstance(ev, CharEvent):
            coverage.note(ev.start, ev.end)
            row["chars"] += 1
        elif isinstance(ev, IllegalEvent):
            row["illegals"].append((ev.start, ev.end, ev.reason))

    for index, name in enumerate(names):
        path = os.path.join(chunk_dir, name)
        with open(path, "rb") as fh:
            data = fh.read()
        chunk_sizes.append(len(data))
        total_bytes += len(data)
        before = decoder.continuation_resumes
        for ev in decoder.feed(data):
            record(index, ev)
        per_chunk[index]["resumed"] = decoder.continuation_resumes - before

    if names:
        for ev in decoder.finish():
            record(len(names) - 1, ev)
    else:
        for ev in decoder.finish():
            pass
    coverage.complete(total_bytes)
    return decoder, names, chunk_sizes, per_chunk, total_bytes


def format_report(chunk_dir, decoder, names, chunk_sizes, per_chunk,
                  total_bytes):
    lines = []
    lines.append("UTF-8 增量解码报告")
    lines.append("块目录: %s" % chunk_dir)
    lines.append("块数: %d" % len(names))
    lines.append("")
    lines.append("逐块统计")
    if not names:
        lines.append("- (空目录，没有块)")
    for index, name in enumerate(names):
        row = per_chunk[index]
        lines.append(
            "- %s  字节=%d  字符=%d  跨块续上=%d  非法=%d"
            % (name, chunk_sizes[index], row["chars"], row["resumed"],
               len(row["illegals"])))
        for start, end, reason in row["illegals"]:
            lines.append("    非法 [%d, %d) %s" % (start, end, reason))
    lines.append("")
    lines.append("整体统计")
    lines.append("- 总字节数: %d" % total_bytes)
    lines.append("- 总字符数: %d" % decoder.chars_emitted)
    lines.append("- 跨块续上总次数: %d" % decoder.continuation_resumes)
    lines.append("- 非法序列数: %d" % decoder.illegal_count)
    truncated = any(
        reason == "truncated_at_eof"
        for row in per_chunk for _, _, reason in row["illegals"])
    lines.append("- 流结束时存在未完成序列: %s" % ("是" if truncated else "否"))
    return "\n".join(lines) + "\n"


def build_report(chunk_dir):
    """跑引擎并返回报告文本（UTF-8）。"""
    decoder, names, chunk_sizes, per_chunk, total_bytes = run_engine(chunk_dir)
    return format_report(chunk_dir, decoder, names, chunk_sizes,
                         per_chunk, total_bytes)


def main(argv):
    if len(argv) not in (2, 3):
        sys.stderr.write("用法: python decode_report.py <块目录> [报告文件]\n")
        return 2
    chunk_dir = argv[1]
    if not os.path.isdir(chunk_dir):
        sys.stderr.write("错误: 块目录不存在或不是目录: %s\n" % chunk_dir)
        return 2
    text = build_report(chunk_dir)
    if len(argv) == 2:
        sys.stdout.buffer.write(text.encode("utf-8"))
    else:
        parent = os.path.dirname(os.path.abspath(argv[2]))
        os.makedirs(parent, exist_ok=True)
        with open(argv[2], "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
