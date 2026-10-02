"""增量式严格 UTF-8 解码器（仅标准库，判定逻辑从零实现）。

核心是一个状态机，逐字节消费分块输入：

    空闲                  -- 等待起始字节
    待续(n, i, s, lead)   -- 序列总长 n、已收 i 个字节、起始偏移 s、首字节 lead

跨块只延续待续状态本身（最多缓存 3 个续字节），已输出的内容不缓存，
因此常驻内存与流长度无关（上界为单个块 + 少量计数）。

事件按流顺序产出：
    CharEvent(char, span, complete)   每解出一个字符一条（含替换字符）
    IllegalEvent(span, reason)        每个非法序列一条，与
                                      complete=False 的 CharEvent 一一对应

原因码：invalid_lead / invalid_continuation / truncated_at_eof。
"""

from collections import namedtuple

REPLACEMENT = "\ufffd"

CharEvent = namedtuple("CharEvent", ("char", "start", "end", "complete"))
IllegalEvent = namedtuple("IllegalEvent", ("start", "end", "reason"))


def encode_codepoint(codepoint):
    """把码点重新编码回 UTF-8 字节串（仅用于输出字符，不用于合法性判定）。"""
    if codepoint <= 0x7F:
        return bytes((codepoint,))
    if codepoint <= 0x7FF:
        return bytes((0xC0 | (codepoint >> 6),
                      0x80 | (codepoint & 0x3F)))
    if codepoint <= 0xFFFF:
        return bytes((0xE0 | (codepoint >> 12),
                      0x80 | ((codepoint >> 6) & 0x3F),
                      0x80 | (codepoint & 0x3F)))
    return bytes((0xF0 | (codepoint >> 18),
                  0x80 | ((codepoint >> 12) & 0x3F),
                  0x80 | ((codepoint >> 6) & 0x3F),
                  0x80 | (codepoint & 0x3F)))


class IncrementalUTF8Decoder:
    """按块喂字节的严格 UTF-8 解码器。

    用法：

        dec = IncrementalUTF8Decoder()
        for chunk in chunks:
            for ev in dec.feed(chunk):
                ...
        for ev in dec.finish():
            ...

    feed/finish 返回的是生成器，调用方必须耗尽后再喂下一块（CLI 与报告
    脚本都按这个约定使用）。

    统计口径（全部来自引擎实时计数，不缓存历史）：
        position               已消费的字节总数（下一个字节的绝对偏移）
        chars_emitted          已输出字符数（含替换字符）
        illegal_count          非法序列数
        continuation_resumes   跨块续上的次数：上一块结束时序列未完成，
                               本块第 1 个字节作为合法续字节被收下即记 1；
                               若该字节非法、导致在块边界处直接判坏，不计。
    """

    def __init__(self):
        self.position = 0
        self.chars_emitted = 0
        self.illegal_count = 0
        self.continuation_resumes = 0
        self._pending_len = 0       # n：序列总长，0 表示空闲
        self._pending_got = 0       # i：已收字节数
        self._pending_start = 0     # s：起始字节在流中的绝对偏移
        self._pending_lead = 0      # b0：起始字节
        self._codepoint = 0         # 已收字节还原出的码点中间值

    @property
    def in_progress(self):
        """当前是否有未完成的多字节序列（跨块状态）。"""
        return self._pending_len != 0

    def feed(self, data):
        """消费一块字节，按流顺序 yield CharEvent / IllegalEvent。"""
        pos = self.position
        resume_pending = self._pending_len
        idx = 0
        size = len(data)
        pending_len = self._pending_len
        pending_got = self._pending_got
        pending_start = self._pending_start
        pending_lead = self._pending_lead
        codepoint = self._codepoint

        while idx < size:
            b = data[idx]

            if pending_len == 0:
                # 空闲：按起始字节判定（README 2.2 表）
                if b <= 0x7F:
                    idx += 1
                    pos += 1
                    self.chars_emitted += 1
                    yield CharEvent(chr(b), pos - 1, pos, True)
                elif b <= 0xC1:          # 80..BF 续字节，C0/C1 过长首字节
                    idx += 1
                    pos += 1
                    self.chars_emitted += 1
                    self.illegal_count += 1
                    yield CharEvent(REPLACEMENT, pos - 1, pos, False)
                    yield IllegalEvent(pos - 1, pos, "invalid_lead")
                elif b <= 0xDF:
                    pending_len, pending_got = 2, 1
                    pending_start, pending_lead = pos, b
                    codepoint = b & 0x1F
                    idx += 1
                    pos += 1
                elif b <= 0xEF:
                    pending_len, pending_got = 3, 1
                    pending_start, pending_lead = pos, b
                    codepoint = b & 0x0F
                    idx += 1
                    pos += 1
                elif b <= 0xF4:
                    pending_len, pending_got = 4, 1
                    pending_start, pending_lead = pos, b
                    codepoint = b & 0x07
                    idx += 1
                    pos += 1
                else:                    # F5..FF
                    idx += 1
                    pos += 1
                    self.chars_emitted += 1
                    self.illegal_count += 1
                    yield CharEvent(REPLACEMENT, pos - 1, pos, False)
                    yield IllegalEvent(pos - 1, pos, "invalid_lead")
                continue

            # 待续：检查第 got+1 位的续字节是否合法
            at_boundary = (resume_pending != 0 and idx == 0)
            good = 0x80 <= b <= 0xBF
            if good:
                if pending_got == 1:
                    if pending_lead == 0xE0:
                        good = b >= 0xA0
                    elif pending_lead == 0xED:
                        good = b <= 0x9F
                    elif pending_lead == 0xF0:
                        good = b >= 0x90
                    elif pending_lead == 0xF4:
                        good = b <= 0x8F
                    # C0/C1、F5+ 不可能进入待续状态（空闲时已判 invalid_lead）

            if not good:
                # 越界：已收的 i 个字节作为一条非法序列，当前字节重新判定
                end = pending_start + pending_got
                self.chars_emitted += 1
                self.illegal_count += 1
                yield CharEvent(REPLACEMENT, pending_start, end, False)
                yield IllegalEvent(pending_start, end, "invalid_continuation")
                pending_len = 0
                if at_boundary:
                    # 跨块后的第 1 个字节就把序列打断，不算“续上”
                    resume_pending = 0
                # 不消费 b：下一轮按空闲状态重新执行起始字节判定
                continue

            if at_boundary:
                self.continuation_resumes += 1
                resume_pending = 0

            codepoint = (codepoint << 6) | (b & 0x3F)
            pending_got += 1
            idx += 1
            pos += 1

            if pending_got == pending_len:
                self.chars_emitted += 1
                yield CharEvent(chr(codepoint), pending_start, pos, True)
                pending_len = 0
                codepoint = 0

        self.position = pos
        self._pending_len = pending_len
        self._pending_got = pending_got
        self._pending_start = pending_start
        self._pending_lead = pending_lead
        self._codepoint = codepoint

    def finish(self):
        """流结束：仍有待续序列则产出 truncated_at_eof。"""
        if self._pending_len != 0:
            end = self._pending_start + self._pending_got
            self.chars_emitted += 1
            self.illegal_count += 1
            yield CharEvent(REPLACEMENT, self._pending_start, end, False)
            yield IllegalEvent(self._pending_start, end, "truncated_at_eof")
            self._pending_len = 0
            self._codepoint = 0


class CoverageChecker:
    """校验覆盖不变式：span 严格递增、互不重叠、并集恰好 [0, N)。

    只保留一个游标语，O(1) 内存。流结束时调 complete(total_bytes)。
    """

    def __init__(self):
        self._cursor = 0

    def note(self, start, end):
        if start != self._cursor or end <= start:
            raise AssertionError(
                "span 不连续/不合法：期望起点 %d，收到 [%d, %d)"
                % (self._cursor, start, end))
        self._cursor = end

    def complete(self, total_bytes):
        if self._cursor != total_bytes:
            raise AssertionError(
                "覆盖不完整：已到偏移 %d，流总长 %d" % (self._cursor, total_bytes))
