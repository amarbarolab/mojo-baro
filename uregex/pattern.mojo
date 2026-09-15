# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-uregex/src/uregex/pattern.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
# Copyright 2026 amarbaro.com. Licensed under the Apache License, Version 2.0.
from .parser import Parser
from .vm import Program


@fieldwise_init
struct Match(Copyable, Movable, ImplicitlyCopyable):
    var start: Int
    var end: Int


def to_codepoints(s: StringSlice) -> List[Int]:
    var out = List[Int]()
    for c in s.codepoints():
        out.append(Int(c.to_u32()))
    return out^


def from_codepoints(cps: List[Int], start: Int, end: Int) -> String:
    var s = String("")
    for i in range(start, end):
        s += String(Codepoint.from_u32(UInt32(cps[i])).value())
    return s^


struct Pattern(Copyable, Movable):
    var prog: Program
    var source: String

    def __init__(out self, pattern: String) raises:
        var p = Parser(pattern)
        var ast = p.parse()
        self.prog = Program(ast)
        self.source = pattern

    def match_at(self, text: List[Int], pos: Int) -> Int:
        return self.prog.exec(text, 0, pos, False)

    def search_from(self, text: List[Int], start: Int) -> Optional[Match]:
        for s in range(start, len(text) + 1):
            var e = self.match_at(text, s)
            if e >= 0:
                return Match(s, e)
        return None

    def search(self, text: String) -> Optional[Match]:
        return self.search_from(to_codepoints(text), 0)

    def findall_cps(self, text: List[Int]) -> List[Match]:
        var out = List[Match]()
        var pos = 0
        while pos <= len(text):
            var m = self.search_from(text, pos)
            if not m:
                break
            var mm = m.value()
            out.append(mm)
            pos = mm.end if mm.end > mm.start else mm.end + 1
        return out^

    def findall(self, text: String) -> List[String]:
        var cps = to_codepoints(text)
        var out = List[String]()
        for m in self.findall_cps(cps):
            out.append(from_codepoints(cps, m.start, m.end))
        return out^

    def split_keep(self, text: String) -> List[String]:
        var cps = to_codepoints(text)
        var out = List[String]()
        var last = 0
        for m in self.findall_cps(cps):
            if m.start > last:
                out.append(from_codepoints(cps, last, m.start))
            if m.end > m.start:
                out.append(from_codepoints(cps, m.start, m.end))
            last = m.end
        if last < len(cps):
            out.append(from_codepoints(cps, last, len(cps)))
        return out^


def compile(pattern: String) raises -> Pattern:
    return Pattern(pattern)


def search(pattern: String, text: String) raises -> Optional[Match]:
    return Pattern(pattern).search(text)


def findall(pattern: String, text: String) raises -> List[String]:
    return Pattern(pattern).findall(text)


def split_keep(pattern: String, text: String) raises -> List[String]:
    return Pattern(pattern).split_keep(text)
