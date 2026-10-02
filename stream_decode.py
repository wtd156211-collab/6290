#!/usr/bin/env python3
"""按块增量解码 UTF-8 字节流，输出逐字符记录与非法位置清单。

用法: python stream_decode.py <块目录> <输出目录>

块目录内是 chunk-NNN.bin，按文件名字典序读取拼接为字节流 S。
输出目录下写三个文件:
  decoded.jsonl  每行 {"char":..., "span":[s,e), "complete":bool}
  illegal.jsonl  每行 {"span":[s,e), "reason":...}
  report.txt     逐块统计与整体汇总（数字全部来自解码引擎）

判定逻辑从 0 实现，不使用 bytes.decode / codecs 等现成解码设施。
"""

import json
import os
import sys

REPLACEMENT_CHAR = "�"  # U+FFFD

# 原因码
INVALID_LEAD = "invalid_lead"
INVALID_CONTINUATION = "invalid_continuation"
TRUNCATED_AT_EOF = "truncated_at_eof"


def _lead_length(byte):
    """起始字节 -> 序列总长；非起始字节返回 0。"""
    if byte <= 0x7F:
        return 1
    if 0xC2 <= byte <= 0xDF:
        return 2
    if 0xE0 <= byte <= 0xEF:
        return 3
    if 0xF0 <= byte <= 0xF4:
        return 4
    return 0  # 80-C1、F5-FF 不能作为起始字节


def _second_byte_ok(first, second):
    """多字节序列第 2 个字节的允许范围（随首字节不同而不同）。"""
    if first == 0xE0:
        return 0xA0 <= second <= 0xBF
    if first == 0xED:
        return 0x80 <= second <= 0x9F
    if first == 0xF0:
        return 0x90 <= second <= 0xBF
    if first == 0xF4:
        return 0x80 <= second <= 0x8F
    return 0x80 <= second <= 0xBF


class StreamDecoder:
    """增量 UTF-8 解码状态机。

    跨块状态（README 第 3 节）:
      seq_len   序列总长 n，0 表示空闲
      seq_have  已收字节数 i
      seq_start 序列起始偏移 s（绝对字节偏移）
      first     首字节（判定第 2 字节范围用）
      codepoint 已累积的码点位
    最多保留 3 个待续字节的等价信息，内存占用与流长无关。
    """

    def __init__(self):
        self.pos = 0        # 下一个待处理字节的绝对偏移
        self.seq_len = 0
        self.seq_have = 0
        self.seq_start = 0
        self.first = 0
        self.codepoint = 0

    @property
    def pending(self):
        """是否处于待续状态（有未完成的序列）。"""
        return self.seq_len != 0

    def feed(self, data, emit):
        """喂入一块字节；emit(char, start, end, complete, reason) 回调输出记录。"""
        base = self.pos
        size = len(data)
        self.pos = base + size
        seq_len = self.seq_len
        seq_have = self.seq_have
        seq_start = self.seq_start
        first = self.first
        codepoint = self.codepoint
        idx = 0
        while idx < size:
            byte = data[idx]
            if seq_len == 0:
                if byte <= 0x7F:
                    emit(ASCII_CHARS[byte], base + idx, base + idx + 1,
                         True, None)
                    idx += 1
                    continue
                length = _lead_length(byte)
                if length == 0:
                    emit(REPLACEMENT_CHAR, base + idx, base + idx + 1,
                         False, INVALID_LEAD)
                    idx += 1
                else:
                    seq_len = length
                    seq_have = 1
                    seq_start = base + idx
                    first = byte
                    codepoint = byte & (0x7F >> length)
                    idx += 1
            else:
                if seq_have == 1:
                    ok = _second_byte_ok(first, byte)
                else:
                    ok = 0x80 <= byte <= 0xBF
                if ok:
                    codepoint = (codepoint << 6) | (byte & 0x3F)
                    seq_have += 1
                    idx += 1
                    if seq_have == seq_len:
                        emit(chr(codepoint), seq_start,
                             seq_start + seq_len, True, None)
                        seq_len = 0
                        seq_have = 0
                else:
                    # 最长合法前缀 = 已收的 seq_have 个字节
                    emit(REPLACEMENT_CHAR, seq_start,
                         seq_start + seq_have, False,
                         INVALID_CONTINUATION)
                    seq_len = 0
                    # 当前字节不消费，回到空闲重新按起始字节判定
        self.seq_len = seq_len
        self.seq_have = seq_have
        self.seq_start = seq_start
        self.first = first
        self.codepoint = codepoint

    def finish(self, emit):
        """流结束；若仍在待续状态，按 truncated_at_eof 收尾。"""
        if self.seq_len != 0:
            emit(REPLACEMENT_CHAR, self.seq_start,
                 self.seq_start + self.seq_have, False, TRUNCATED_AT_EOF)
            self.seq_len = 0
            self.seq_have = 0


def _list_chunks(chunk_dir):
    try:
        names = os.listdir(chunk_dir)
    except OSError as exc:
        raise SystemExit(f"无法读取块目录 {chunk_dir}: {exc}")
    chunks = []
    for name in names:
        path = os.path.join(chunk_dir, name)
        if name.startswith("chunk-") and name.endswith(".bin") \
                and os.path.isfile(path):
            chunks.append(name)
    chunks.sort()
    if not chunks:
        raise SystemExit(f"块目录 {chunk_dir} 中没有 chunk-*.bin 文件")
    return chunks


# 0..127 每个 ASCII 字符的 JSON 字符串形式（含引号），预先算好。
# 多字节字符（码点 >= 0x80）在 JSON 中无需转义，直接包引号即可。
_ASCII_JSON = tuple(json.dumps(chr(b), ensure_ascii=False) for b in range(128))
ASCII_CHARS = tuple(chr(b) for b in range(128))
_REPLACEMENT_JSON = '"' + REPLACEMENT_CHAR + '"'


def _decoded_line(char, start, end, complete):
    if len(char) == 1 and ord(char) < 128:
        char_json = _ASCII_JSON[ord(char)]
    else:
        char_json = '"' + char + '"'
    return ('{"char":' + char_json + ',"span":[' + str(start) + ',' +
            str(end) + '],"complete":' + ('true' if complete else 'false') +
            '}\n')


def _illegal_line(start, end, reason):
    return ('{"span":[' + str(start) + ',' + str(end) + '],"reason":"' +
            reason + '"}\n')


def run(chunk_dir, out_dir):
    chunks = _list_chunks(chunk_dir)
    os.makedirs(out_dir, exist_ok=True)
    decoded_path = os.path.join(out_dir, "decoded.jsonl")
    illegal_path = os.path.join(out_dir, "illegal.jsonl")
    report_path = os.path.join(out_dir, "report.txt")

    decoder = StreamDecoder()
    stats = {"complete": 0, "illegal": 0}
    cur = None  # 当前块的统计容器，由 emit 闭包填充
    chunk_reports = []
    eof_illegal = []  # finish() 产生、无法归到任何块的非法记录

    def make_cur():
        return {"chars": 0, "illegal": []}

    with open(decoded_path, "w", encoding="utf-8", newline="\n") as f_dec, \
            open(illegal_path, "w", encoding="utf-8", newline="\n") as f_ill:

        dec_lines = []
        ill_lines = []

        def emit(char, start, end, complete, reason):
            dec_lines.append(_decoded_line(char, start, end, complete))
            cur["chars"] += 1
            if complete:
                stats["complete"] += 1
            else:
                record = {"span": [start, end], "reason": reason}
                ill_lines.append(_illegal_line(start, end, reason))
                cur["illegal"].append(record)
                stats["illegal"] += 1

        for name in chunks:
            path = os.path.join(chunk_dir, name)
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError as exc:
                raise SystemExit(f"无法读取块 {path}: {exc}")
            start = decoder.pos
            continued = 1 if (decoder.pending and data) else 0
            cur = make_cur()
            decoder.feed(data, emit)
            if dec_lines:
                f_dec.write("".join(dec_lines))
                dec_lines.clear()
            if ill_lines:
                f_ill.write("".join(ill_lines))
                ill_lines.clear()
            chunk_reports.append({
                "name": name,
                "span": [start, start + len(data)],
                "chars": cur["chars"],
                "continued": continued,
                "illegal": cur["illegal"],
            })
        cur = make_cur()
        decoder.finish(emit)
        if dec_lines:
            f_dec.write("".join(dec_lines))
        if ill_lines:
            f_ill.write("".join(ill_lines))
        eof_illegal = cur["illegal"]

    total_bytes = decoder.pos
    total_chars = stats["complete"] + stats["illegal"]
    continued_total = sum(r["continued"] for r in chunk_reports)

    lines = []
    lines.append("# 解码报告")
    lines.append(f"块目录: {chunk_dir}")
    lines.append("")
    for r in chunk_reports:
        lines.append(
            "块 {name}: 字节区间 [{s},{e}) 字符 {chars} 跨块续上 {cont} "
            "非法 {ill}".format(
                name=r["name"], s=r["span"][0], e=r["span"][1],
                chars=r["chars"], cont=r["continued"],
                ill=len(r["illegal"])))
        for rec in r["illegal"]:
            span = rec["span"]
            lines.append("    非法 [{s},{e}) {reason}".format(
                s=span[0], e=span[1], reason=rec["reason"]))
    if eof_illegal:
        lines.append("流结束收尾:")
        for rec in eof_illegal:
            span = rec["span"]
            lines.append("    非法 [{s},{e}) {reason}".format(
                s=span[0], e=span[1], reason=rec["reason"]))
    lines.append("")
    lines.append("# 整体统计")
    lines.append(f"总字节数: {total_bytes}")
    lines.append(f"总块数: {len(chunk_reports)}")
    lines.append(f"解出字符数: {total_chars} (完整 {stats['complete']}, "
                 f"替换 {stats['illegal']})")
    lines.append(f"跨块续上次数: {continued_total}")
    lines.append(f"非法序列数: {stats['illegal']}")
    lines.append("")
    with open(report_path, "w", encoding="utf-8", newline="\n") as f_rep:
        f_rep.write("\n".join(lines))
    return 0


def main(argv):
    if len(argv) != 3:
        print("用法: python stream_decode.py <块目录> <输出目录>",
              file=sys.stderr)
        return 2
    return run(argv[1], argv[2])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
