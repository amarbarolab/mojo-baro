"""Generic line protocol helpers, shared between every engine binary that
speaks serve/PROTOCOL.md over stdin/stdout (BARO_SERVE=1): the byte-scanner
JSON reader, the request line parser, and the sampler-field struct it fills.

Extracted verbatim out of serve/engine.mojo (M4, briefs/2026-09-15-wiring-lane.md):
this file carried none of qwen35's model logic, so moving it here and having
engine.mojo import it instead of defining it locally is a refactor, not a
rewrite -- qwen35/qwen35moe behavior is unchanged. serve/spark.mojo (the
dense-family engine) imports the same functions rather than re-implementing
the wire format, per serve/PROTOCOL.md's own instruction to reuse the
existing protocol.
"""
from std.ffi import c_ssize_t, external_call


def read_line(fd: Int) raises -> Optional[String]:
    # One line from fd, newline stripped; None at EOF with nothing buffered.
    var buf = List[UInt8]()
    var b = List[UInt8](unsafe_uninit_length=1)
    while True:
        var n = external_call["read", c_ssize_t](fd, b.unsafe_ptr(), 1)
        if n <= 0:
            if len(buf) == 0:
                return None
            break
        if b[0] == 10:
            break
        buf.append(b[0])
    return String(from_utf8=Span[UInt8](buf))


def cancel_pending(fd: Int, req_id: Int, mut pending: List[String]) raises -> Bool:
    # Non-blocking check for a "{"cancel":ID}" line on fd, once per
    # step_window call. poll(fd, POLLIN, 0) never blocks; a hit means the
    # writer's single write() of a short line is already in the pipe, so the
    # blocking read_line below will not stall on a torn line.
    var pfd = List[UInt8](unsafe_uninit_length=8)
    var p = pfd.unsafe_ptr()
    p.unsafe_bitcast[Int32]().unsafe_offset(0)[] = Int32(fd)
    p.unsafe_bitcast[Int16]().unsafe_offset(2)[] = Int16(1)  # events = POLLIN
    p.unsafe_bitcast[Int16]().unsafe_offset(3)[] = Int16(0)  # revents
    var r = external_call["poll", Int32](p, UInt64(1), Int32(0))
    if r <= 0 or (Int(p.unsafe_bitcast[Int16]().unsafe_offset(3)[]) & 1) == 0:
        return False
    var line_in = read_line(fd)
    if not line_in:
        return False
    var line = line_in.value()
    var i = json_key(line, "cancel")
    var cid = 0
    if i < 0 or not json_int(line, i, cid):
        # Not a cancel: this is the NEXT REQUEST, already in the pipe because
        # the client queued it while this one was still decoding. Hand it back
        # to the request loop. Dropping it here silently ate 19 of 20 requests
        # the first time bench/force-ab-serve.sh was ever run (2026-09-12).
        pending.append(line)
        return False
    return cid == req_id


def json_key(line: String, key: String) -> Int:
    # Index of the first byte of the value for "key": ..., or -1.
    var i = line.find(String("\"") + key + "\"")
    if i < 0:
        return -1
    var b = line.as_bytes()
    i += key.byte_length() + 2
    while i < len(b) and (b[i] == 32 or b[i] == 9):
        i += 1
    if i >= len(b) or b[i] != 58:
        return -1
    i += 1
    while i < len(b) and (b[i] == 32 or b[i] == 9):
        i += 1
    return i


def json_int(line: String, mut i: Int, mut v: Int) -> Bool:
    var b = line.as_bytes()
    var neg = False
    if i < len(b) and b[i] == 45:
        neg = True
        i += 1
    var have = False
    v = 0
    while i < len(b) and b[i] >= 48 and b[i] <= 57:
        v = v * 10 + Int(b[i] - 48)
        i += 1
        have = True
    if neg:
        v = -v
    return have


def json_float(line: String, mut i: Int, mut v: Float64) -> Bool:
    # Plain decimal (sign, digits, optional '.', digits); no exponent form --
    # none of the sampler fields need one.
    var b = line.as_bytes()
    var neg = False
    if i < len(b) and b[i] == 45:
        neg = True
        i += 1
    var have = False
    var ip: Float64 = 0
    while i < len(b) and b[i] >= 48 and b[i] <= 57:
        ip = ip * 10 + Float64(Int(b[i] - 48))
        i += 1
        have = True
    var frac: Float64 = 0
    if i < len(b) and b[i] == 46:
        i += 1
        var scale: Float64 = 1
        while i < len(b) and b[i] >= 48 and b[i] <= 57:
            scale /= 10
            frac += Float64(Int(b[i] - 48)) * scale
            i += 1
            have = True
    v = ip + frac
    if neg:
        v = -v
    return have


@fieldwise_init
struct SampleParams(Copyable, Movable):
    # C3 (bench/chat-protocol.md): parsed from the wire, not yet acted on --
    # the live decode loop still always takes the greedy/MTP path. Ready for
    # serve/sample_ref.mojo (host reference) or kernels/sample.mojo (device,
    # lane-KSAMP) to read once either is wired in. temperature <= 0 means
    # "off" throughout, matching both references' own convention.
    var temperature: Float64
    var top_p: Float64
    var top_k: Int
    var min_p: Float64
    var seed: UInt64
    var presence_penalty: Float64
    var frequency_penalty: Float64


def default_sample_params() -> SampleParams:
    return SampleParams(temperature=0, top_p=1.0, top_k=0, min_p=0, seed=0, presence_penalty=0, frequency_penalty=0)


def parse_request(
    line: String, mut id: Int, mut prompt: List[Int], mut n: Int, mut spec: Bool, mut has_spec: Bool, mut stop: List[List[Int]], mut ckpt: List[Int], mut sample: SampleParams
) -> String:
    # {"id":INT,"prompt":[INT,...],"n":INT,"spec":BOOL,"stop":[[INT,...],...],
    #  "ckpt":[INT,...],"temperature":FLOAT,"top_p":FLOAT,"top_k":INT,
    #  "min_p":FLOAT,"seed":INT,"presence_penalty":FLOAT,
    #  "frequency_penalty":FLOAT}; everything past prompt/n optional. Returns
    # "" on success, else the error text (id is set when it parsed).
    id = 0
    var i = json_key(line, "id")
    if i < 0 or not json_int(line, i, id):
        return "missing or non-integer id"
    i = json_key(line, "n")
    if i < 0 or not json_int(line, i, n):
        return "missing or non-integer n"
    i = json_key(line, "prompt")
    var b = line.as_bytes()
    if i < 0 or i >= len(b) or b[i] != 91:
        return "missing prompt array"
    i += 1
    while True:
        while i < len(b) and (b[i] == 32 or b[i] == 44):
            i += 1
        if i >= len(b):
            return "unterminated prompt array"
        if b[i] == 93:
            break
        var v = 0
        if not json_int(line, i, v) or v < 0:
            return "prompt must hold non-negative integers"
        prompt.append(v)
    has_spec = False
    i = json_key(line, "spec")
    if i >= 0:
        if line.as_bytes()[i] == 116:
            spec = True
            has_spec = True
        elif line.as_bytes()[i] == 102:
            spec = False
            has_spec = True
        else:
            return "spec must be true or false"
    var si = json_key(line, "stop")
    if si >= 0:
        if si >= len(b) or b[si] != 91:
            return "stop must be an array of arrays"
        si += 1
        while True:
            while si < len(b) and (b[si] == 32 or b[si] == 44):
                si += 1
            if si >= len(b):
                return "unterminated stop array"
            if b[si] == 93:
                break
            if b[si] != 91:
                return "stop entries must be arrays of token ids"
            si += 1
            var seq = List[Int]()
            while True:
                while si < len(b) and (b[si] == 32 or b[si] == 44):
                    si += 1
                if si >= len(b):
                    return "unterminated stop sequence"
                if b[si] == 93:
                    si += 1
                    break
                var v2 = 0
                if not json_int(line, si, v2) or v2 < 0:
                    return "stop sequence must hold non-negative integers"
                seq.append(v2)
            stop.append(seq^)
    var ci = json_key(line, "ckpt")
    if ci >= 0:
        if ci >= len(b) or b[ci] != 91:
            return "ckpt must be an array of integers"
        ci += 1
        while True:
            while ci < len(b) and (b[ci] == 32 or b[ci] == 44):
                ci += 1
            if ci >= len(b):
                return "unterminated ckpt array"
            if b[ci] == 93:
                break
            var v3 = 0
            if not json_int(line, ci, v3) or v3 < 0:
                return "ckpt must hold non-negative integers"
            ckpt.append(v3)
    var fi = json_key(line, "temperature")
    if fi >= 0:
        var fv: Float64 = 0
        if not json_float(line, fi, fv):
            return "temperature must be a number"
        sample.temperature = fv
    fi = json_key(line, "top_p")
    if fi >= 0:
        var fv2: Float64 = 0
        if not json_float(line, fi, fv2):
            return "top_p must be a number"
        sample.top_p = fv2
    fi = json_key(line, "top_k")
    if fi >= 0:
        var iv = 0
        if not json_int(line, fi, iv):
            return "top_k must be an integer"
        sample.top_k = iv
    fi = json_key(line, "min_p")
    if fi >= 0:
        var fv3: Float64 = 0
        if not json_float(line, fi, fv3):
            return "min_p must be a number"
        sample.min_p = fv3
    fi = json_key(line, "seed")
    if fi >= 0:
        var iv2 = 0
        if not json_int(line, fi, iv2) or iv2 < 0:
            return "seed must be a non-negative integer"
        sample.seed = UInt64(iv2)
    fi = json_key(line, "presence_penalty")
    if fi >= 0:
        var fv4: Float64 = 0
        if not json_float(line, fi, fv4):
            return "presence_penalty must be a number"
        sample.presence_penalty = fv4
    fi = json_key(line, "frequency_penalty")
    if fi >= 0:
        var fv5: Float64 = 0
        if not json_float(line, fi, fv5):
            return "frequency_penalty must be a number"
        sample.frequency_penalty = fv5
    return ""
