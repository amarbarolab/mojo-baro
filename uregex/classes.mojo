# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-uregex/src/uregex/classes.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
# Copyright 2026 amarbaro.com. Licensed under the Apache License, Version 2.0.
from .unicode_tables import category_table, escape_table, lower_table, upper_table


def _hex(s: StringSlice) -> Int:
    var v = 0
    for b in s.as_bytes():
        var c = Int(b)
        if c >= 48 and c <= 57:
            v = v * 16 + (c - 48)
        elif c >= 97 and c <= 102:
            v = v * 16 + (c - 87)
        elif c >= 65 and c <= 70:
            v = v * 16 + (c - 55)
    return v


def parse_ranges(table: String) -> List[Int]:
    var out = List[Int]()
    if table.byte_length() == 0:
        return out^
    for item in table.split(";"):
        var ab = item.split("-")
        out.append(_hex(ab[0]))
        out.append(_hex(ab[1]))
    return out^


struct CharClass(Copyable, Movable):
    var ranges: List[Int]
    var negate: Bool

    def __init__(out self):
        self.ranges = List[Int]()
        self.negate = False

    def add(mut self, lo: Int, hi: Int):
        self.ranges.append(lo)
        self.ranges.append(hi)

    def add_ranges(mut self, rs: List[Int]):
        for i in range(0, len(rs), 2):
            self.ranges.append(rs[i])
            self.ranges.append(rs[i + 1])

    def normalize(mut self):
        var n = len(self.ranges) // 2
        var idx = List[Int](capacity=n)
        for i in range(n):
            idx.append(i)
        for i in range(1, n):
            var j = i
            while j > 0 and self.ranges[2 * idx[j - 1]] > self.ranges[2 * idx[j]]:
                var t = idx[j]
                idx[j] = idx[j - 1]
                idx[j - 1] = t
                j -= 1
        var merged = List[Int]()
        for k in range(n):
            var lo = self.ranges[2 * idx[k]]
            var hi = self.ranges[2 * idx[k] + 1]
            if len(merged) > 0 and lo <= merged[len(merged) - 1] + 1:
                if hi > merged[len(merged) - 1]:
                    merged[len(merged) - 1] = hi
            else:
                merged.append(lo)
                merged.append(hi)
        self.ranges = merged^

    def _in(self, cp: Int) -> Bool:
        var lo = 0
        var hi = len(self.ranges) // 2
        while lo < hi:
            var mid = (lo + hi) // 2
            if cp < self.ranges[2 * mid]:
                hi = mid
            elif cp > self.ranges[2 * mid + 1]:
                lo = mid + 1
            else:
                return True
        return False

    def contains(self, cp: Int) -> Bool:
        return self._in(cp) != self.negate


struct CaseMap(Copyable, Movable):
    var lower: Dict[Int, Int]
    var upper: Dict[Int, Int]

    def __init__(out self):
        self.lower = Dict[Int, Int]()
        self.upper = Dict[Int, Int]()
        var lo = parse_ranges(lower_table())
        for i in range(0, len(lo), 2):
            self.lower[lo[i]] = lo[i + 1]
        var up = parse_ranges(upper_table())
        for i in range(0, len(up), 2):
            self.upper[up[i]] = up[i + 1]

    def fold(self, cp: Int) -> Int:
        return self.lower.get(cp, cp)

    def up(self, cp: Int) -> Int:
        return self.upper.get(cp, cp)


def category_class(name: String) raises -> List[Int]:
    var t = category_table(name)
    if t.byte_length() == 0:
        raise Error("uregex: unknown Unicode category \\p{" + name + "}")
    return parse_ranges(t)


def escape_class(c: String) raises -> List[Int]:
    var t = escape_table(c)
    if t.byte_length() == 0:
        raise Error("uregex: unknown class escape \\" + c)
    return parse_ranges(t)
