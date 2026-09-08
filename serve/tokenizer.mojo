"""Byte-level BPE tokenizer read straight from a GGUF header (no tensors touched).

Mirrors tools/gguf-tokenizer.py: gpt2-model vocab + merges + token_type from the
tokenizer.ggml.* keys, pre-tokenizer regexes per tokenizer.ggml.pre (llama.cpp's
tables), control/user_defined tokens matched in the text before BPE. Regexes run
on mojo-uregex (~/Projects/mojo-uregex/src, build with -I).
"""
from std.collections import Dict
from uregex import Pattern
from uregex.pattern import to_codepoints, from_codepoints

comptime T_NORMAL = 1
comptime T_UNKNOWN = 2
comptime T_CONTROL = 3
comptime T_USER = 4
comptime T_UNUSED = 5
comptime T_BYTE = 6

comptime HEADER_MAX = 96 << 20


struct Reader:
    var buf: List[UInt8]
    var pos: Int

    def __init__(out self, var buf: List[UInt8]):
        self.buf = buf^
        self.pos = 0

    def need(self, n: Int) raises:
        if self.pos + n > len(self.buf):
            raise Error("gguf: header longer than " + String(HEADER_MAX) + " bytes")

    def u32(mut self) raises -> Int:
        self.need(4)
        var v = 0
        for i in range(4):
            v |= Int(self.buf[self.pos + i]) << (8 * i)
        self.pos += 4
        return v

    def u64(mut self) raises -> Int:
        self.need(8)
        var v = 0
        for i in range(8):
            v |= Int(self.buf[self.pos + i]) << (8 * i)
        self.pos += 8
        return v

    def string(mut self) raises -> String:
        var n = self.u64()
        self.need(n)
        var s = String(StringSlice(unsafe_from_utf8=Span(self.buf)[self.pos : self.pos + n]))
        self.pos += n
        return s^

    def skip_scalar(mut self, vtype: Int) raises:
        var n = 0
        if vtype == 0 or vtype == 1 or vtype == 7:
            n = 1
        elif vtype == 2 or vtype == 3:
            n = 2
        elif vtype == 4 or vtype == 5 or vtype == 6:
            n = 4
        elif vtype == 10 or vtype == 11 or vtype == 12:
            n = 8
        else:
            raise Error("gguf: unknown scalar type " + String(vtype))
        self.need(n)
        self.pos += n

    def scalar_int(mut self, vtype: Int) raises -> Int:
        var n = 0
        if vtype == 0 or vtype == 1 or vtype == 7:
            n = 1
        elif vtype == 2 or vtype == 3:
            n = 2
        elif vtype == 4 or vtype == 5:
            n = 4
        elif vtype == 10 or vtype == 11:
            n = 8
        else:
            self.skip_scalar(vtype)
            return -1
        self.need(n)
        var v = 0
        for i in range(n):
            v |= Int(self.buf[self.pos + i]) << (8 * i)
        self.pos += n
        return v


struct Tokenizer(Movable):
    var tokens: List[String]
    var types: List[Int]
    var tok2id: Dict[String, Int]
    var ranks: Dict[String, Int]
    var pre: String
    var patterns: List[Pattern]
    var specials: List[List[Int]]
    var special_ids: List[Int]
    var bos_id: Int
    var eos_id: Int
    var pad_id: Int
    var add_bos: Bool
    var chat_template: String
    var b2u: List[Int]
    var u2b: Dict[Int, Int]

    def __init__(out self, gguf_path: String) raises:
        self.tokens = List[String]()
        self.types = List[Int]()
        self.tok2id = Dict[String, Int]()
        self.ranks = Dict[String, Int]()
        self.pre = String("default")
        self.patterns = List[Pattern]()
        self.specials = List[List[Int]]()
        self.special_ids = List[Int]()
        self.bos_id = -1
        self.eos_id = -1
        self.pad_id = -1
        self.add_bos = False
        self.chat_template = String("")
        self.b2u = List[Int]()
        self.u2b = Dict[Int, Int]()
        self._byte_maps()
        self._read_gguf(gguf_path)
        self._compile_pre()
        for i in range(len(self.tokens)):
            self.tok2id[self.tokens[i]] = i
            if self.types[i] == T_CONTROL or self.types[i] == T_USER:
                self.specials.append(to_codepoints(self.tokens[i]))
                self.special_ids.append(i)

    def _byte_maps(mut self):
        var n = 0
        for b in range(256):
            var printable = (b >= 33 and b <= 126) or (b >= 161 and b <= 172) or (b >= 174 and b <= 255)
            var u = b
            if not printable:
                u = 256 + n
                n += 1
            self.b2u.append(u)
            self.u2b[u] = b

    def _read_gguf(mut self, path: String) raises:
        var buf: List[UInt8]
        with open(path, "r") as f:
            buf = f.read_bytes(HEADER_MAX)
        var r = Reader(buf^)
        var magic = r.u32()
        if magic != 0x46554747:
            raise Error("gguf: bad magic")
        var version = r.u32()
        if version != 3:
            raise Error("gguf: unsupported version " + String(version))
        _ = r.u64()
        var n_kv = r.u64()
        var model = String("")
        for _ in range(n_kv):
            var key = r.string()
            var vtype = r.u32()
            if vtype == 8:
                var s = r.string()
                if key == "tokenizer.ggml.model":
                    model = s
                elif key == "tokenizer.ggml.pre":
                    self.pre = s
                elif key == "tokenizer.chat_template":
                    self.chat_template = s
            elif vtype == 9:
                var etype = r.u32()
                var n = r.u64()
                if key == "tokenizer.ggml.tokens" and etype == 8:
                    self.tokens.reserve(n)
                    for _ in range(n):
                        self.tokens.append(r.string())
                elif key == "tokenizer.ggml.merges" and etype == 8:
                    for i in range(n):
                        self.ranks[r.string()] = i
                elif key == "tokenizer.ggml.token_type":
                    self.types.reserve(n)
                    for _ in range(n):
                        self.types.append(r.scalar_int(etype))
                elif etype == 8:
                    for _ in range(n):
                        _ = r.string()
                elif etype == 9:
                    raise Error("gguf: nested arrays unsupported")
                else:
                    for _ in range(n):
                        r.skip_scalar(etype)
            else:
                var v = r.scalar_int(vtype)
                if key == "tokenizer.ggml.bos_token_id":
                    self.bos_id = v
                elif key == "tokenizer.ggml.eos_token_id":
                    self.eos_id = v
                elif key == "tokenizer.ggml.padding_token_id":
                    self.pad_id = v
                elif key == "tokenizer.ggml.add_bos_token":
                    self.add_bos = v == 1
        if model != "gpt2":
            raise Error("tokenizer.ggml.model=" + model + ": only byte-level BPE (gpt2) is supported")
        if len(self.tokens) == 0 or len(self.tokens) != len(self.types):
            raise Error("gguf: tokens/token_type missing or mismatched")

    def _compile_pre(mut self) raises:
        comptime CONTR = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        if self.pre == "qwen2" or self.pre == "deepseek-r1-qwen":
            self.patterns.append(Pattern(CONTR + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"))
        elif self.pre == "qwen35":
            self.patterns.append(Pattern(CONTR + r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"))
        elif self.pre == "llama3" or self.pre == "llama-bpe":
            self.patterns.append(Pattern(CONTR + r"|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"))
        elif self.pre == "spark2_5":
            self.patterns.append(Pattern(r"\p{N}{1,3}"))
            self.patterns.append(Pattern(r"[一-龥぀-ゟ゠-ヿ]+"))
            self.patterns.append(Pattern(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+|[\r\n]|\s+(?!\S)|\s+"))
            self.patterns.append(Pattern(r"\p{N}"))
        elif self.pre == "gpt-2" or self.pre == "default":
            self.patterns.append(Pattern(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"))
        else:
            raise Error("tokenizer.ggml.pre=" + self.pre + ": no pre-tokenizer regex on file")

    def _pieces(self, chunk: String) -> List[String]:
        var pieces = List[String]()
        pieces.append(chunk)
        for p in self.patterns:
            var next = List[String]()
            for piece in pieces:
                for q in p.split_keep(piece):
                    next.append(q)
            pieces = next^
        return pieces^

    def _bpe(self, piece: String, mut out: List[Int]) raises:
        var syms = List[String]()
        for b in piece.as_bytes():
            syms.append(String(Codepoint.from_u32(UInt32(self.b2u[Int(b)])).value()))
        while len(syms) > 1:
            var best = -1
            var best_rank = 1 << 60
            for i in range(len(syms) - 1):
                var key = String(syms[i])
                key += " "
                key += syms[i + 1]
                var r = self.ranks.get(key, -1)
                if r >= 0 and r < best_rank:
                    best_rank = r
                    best = i
            if best < 0:
                break
            var merged = String(syms[best])
            merged += syms[best + 1]
            syms[best] = merged^
            _ = syms.pop(best + 1)
        for s in syms:
            var id = self.tok2id.get(s, -1)
            if id < 0:
                raise Error("tokenizer: symbol not in vocab: " + s)
            out.append(id)

    def _match_special(self, cps: List[Int], at: Int) -> Int:
        var best = -1
        var best_len = 0
        for k in range(len(self.specials)):
            var n = len(self.specials[k])
            if n <= best_len or at + n > len(cps):
                continue
            var ok = True
            for j in range(n):
                if cps[at + j] != self.specials[k][j]:
                    ok = False
                    break
            if ok:
                best = k
                best_len = n
        return best

    def token_str(self, id: Int) -> String:
        return self.tokens[id] if id >= 0 and id < len(self.tokens) else String("")

    def encode(self, text: String, add_special: Bool = True) raises -> List[Int]:
        var out = List[Int]()
        if add_special and self.add_bos and self.bos_id >= 0:
            out.append(self.bos_id)
        var cps = to_codepoints(text)
        var start = 0
        var i = 0
        while i <= len(cps):
            var k = -1 if i == len(cps) else self._match_special(cps, i)
            if k >= 0 or i == len(cps):
                if i > start:
                    for piece in self._pieces(from_codepoints(cps, start, i)):
                        self._bpe(piece, out)
                if k >= 0:
                    out.append(self.special_ids[k])
                    i += len(self.specials[k])
                    start = i
                else:
                    i += 1
            else:
                i += 1
        return out^

    def decode(self, ids: List[Int], keep_special: Bool = False) -> String:
        var bytes = List[UInt8]()
        for id in ids:
            if id < 0 or id >= len(self.tokens):
                continue
            var ty = self.types[id]
            if ty == T_CONTROL and not keep_special:
                continue
            if ty == T_CONTROL or ty == T_USER:
                for b in self.tokens[id].as_bytes():
                    bytes.append(b)
                continue
            for c in self.tokens[id].codepoints():
                var b = self.u2b.get(Int(c.to_u32()), -1)
                if b >= 0:
                    bytes.append(UInt8(b))
        return String(StringSlice(unsafe_from_utf8=Span(bytes)))
