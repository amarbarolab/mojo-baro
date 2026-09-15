# VENDORED COPY. Upstream is ~/Projects/mojo/mojo-uregex/src/uregex/vm.mojo;
# this repo keeps a real file rather than a symlink or an -I path outside the
# tree, because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
# Copyright 2026 amarbaro.com. Licensed under the Apache License, Version 2.0.
from .parser import (
    Ast, Node, N_EMPTY, N_LIT, N_CLASS, N_ANY, N_CAT, N_ALT, N_REPEAT, N_LOOK, N_BOL, N_EOL,
)
from .classes import CharClass, CaseMap

comptime I_CHAR = 0
comptime I_CLASS = 1
comptime I_ANY = 2
comptime I_SPLIT = 3
comptime I_JMP = 4
comptime I_MATCH = 5
comptime I_LOOK = 6
comptime I_LOOK_END = 7
comptime I_BOL = 8
comptime I_EOL = 9


@fieldwise_init
struct Inst(Copyable, Movable):
    var op: Int
    var a: Int
    var b: Int
    var ci: Bool
    var neg: Bool


struct Program(Copyable, Movable):
    var code: List[Inst]
    var classes: List[CharClass]
    var cases: CaseMap

    def __init__(out self, ast: Ast):
        self.code = List[Inst]()
        self.classes = ast.classes.copy()
        self.cases = CaseMap()
        self.emit_node(ast, ast.root)
        self.code.append(Inst(I_MATCH, 0, 0, False, False))

    def emit(mut self, op: Int, a: Int = 0, b: Int = 0, ci: Bool = False, neg: Bool = False) -> Int:
        self.code.append(Inst(op, a, b, ci, neg))
        return len(self.code) - 1

    def emit_node(mut self, ast: Ast, i: Int):
        var n = ast.nodes[i].copy()
        if n.kind == N_EMPTY:
            return
        if n.kind == N_LIT:
            _ = self.emit(I_CHAR, n.cp, 0, n.ci)
        elif n.kind == N_CLASS:
            _ = self.emit(I_CLASS, n.cls, 0, n.ci)
        elif n.kind == N_ANY:
            _ = self.emit(I_ANY)
        elif n.kind == N_BOL:
            _ = self.emit(I_BOL)
        elif n.kind == N_EOL:
            _ = self.emit(I_EOL)
        elif n.kind == N_CAT:
            for k in n.kids:
                self.emit_node(ast, k)
        elif n.kind == N_ALT:
            var jumps = List[Int]()
            for j in range(len(n.kids)):
                if j < len(n.kids) - 1:
                    var split = self.emit(I_SPLIT)
                    self.code[split].a = split + 1
                    self.emit_node(ast, n.kids[j])
                    jumps.append(self.emit(I_JMP))
                    self.code[split].b = len(self.code)
                else:
                    self.emit_node(ast, n.kids[j])
            for jp in jumps:
                self.code[jp].a = len(self.code)
        elif n.kind == N_REPEAT:
            for _ in range(n.lo):
                self.emit_node(ast, n.kids[0])
            if n.hi == -1:
                var split = self.emit(I_SPLIT)
                self.code[split].a = split + 1
                self.emit_node(ast, n.kids[0])
                _ = self.emit(I_JMP, split)
                self.code[split].b = len(self.code)
            else:
                var splits = List[Int]()
                for _ in range(n.hi - n.lo):
                    var split = self.emit(I_SPLIT)
                    self.code[split].a = split + 1
                    splits.append(split)
                    self.emit_node(ast, n.kids[0])
                for sp in splits:
                    self.code[sp].b = len(self.code)
        elif n.kind == N_LOOK:
            var look = self.emit(I_LOOK, 0, 0, False, n.neg)
            self.emit_node(ast, n.kids[0])
            _ = self.emit(I_LOOK_END)
            self.code[look].a = len(self.code)

    def char_eq(self, a: Int, b: Int, ci: Bool) -> Bool:
        if a == b:
            return True
        if not ci:
            return False
        return self.cases.fold(a) == self.cases.fold(b) or self.cases.up(a) == self.cases.up(b)

    def class_has(self, cls: Int, cp: Int, ci: Bool) -> Bool:
        if self.classes[cls].contains(cp):
            return True
        if not ci:
            return False
        return self.classes[cls].contains(self.cases.fold(cp)) or self.classes[cls].contains(self.cases.up(cp))

    def exec(self, text: List[Int], start_pc: Int, start_pos: Int, sub: Bool) -> Int:
        var n = len(text)
        var stack = List[Int]()
        var pc = start_pc
        var pos = start_pos
        while True:
            var ins = self.code[pc].copy()
            var ok = True
            if ins.op == I_MATCH:
                if not sub:
                    return pos
                ok = False
            elif ins.op == I_LOOK_END:
                if sub:
                    return pos
                pc += 1
                continue
            elif ins.op == I_CHAR:
                if pos < n and self.char_eq(text[pos], ins.a, ins.ci):
                    pos += 1
                    pc += 1
                else:
                    ok = False
            elif ins.op == I_CLASS:
                if pos < n and self.class_has(ins.a, text[pos], ins.ci):
                    pos += 1
                    pc += 1
                else:
                    ok = False
            elif ins.op == I_ANY:
                if pos < n and text[pos] != 10:
                    pos += 1
                    pc += 1
                else:
                    ok = False
            elif ins.op == I_BOL:
                if pos == 0:
                    pc += 1
                else:
                    ok = False
            elif ins.op == I_EOL:
                if pos == n:
                    pc += 1
                else:
                    ok = False
            elif ins.op == I_JMP:
                pc = ins.a
            elif ins.op == I_SPLIT:
                stack.append(ins.b)
                stack.append(pos)
                pc = ins.a
            elif ins.op == I_LOOK:
                var r = self.exec(text, pc + 1, pos, True)
                if (r >= 0) == ins.neg:
                    ok = False
                else:
                    pc = ins.a
            if not ok:
                if len(stack) == 0:
                    return -1
                pos = stack.pop()
                pc = stack.pop()
