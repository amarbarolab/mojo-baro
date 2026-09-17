import re, sys
src, name = sys.argv[1], sys.argv[2]
lines = open(src).read().splitlines()
out = ['target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"',
       'target triple = "spirv64-unknown-unknown"',
       'declare spir_func i64 @_Z12get_group_idEj(i32)', 'declare spir_func i64 @_Z12get_local_idEj(i32)',
       'declare spir_func i64 @_Z14get_local_sizeEj(i32)', 'declare spir_func float @_Z4exp2f(float)']
alias = {}
body = False
for l in lines:
    if l.startswith("define "):
        m = re.match(r'define void @(\S+?)\((.*)\) local_unnamed_addr #0 \{', l)
        args = [a.strip() for a in re.split(r', (?=ptr|<3|i32)', m.group(2))]
        keep = [re.sub(r' "[^"]*"', '', a).replace("noundef ", "").replace("nonnull ", "") for a in args if a.startswith("ptr")]
        out.append(f"define spir_kernel void @{name}({', '.join(keep)}) {{")
        out.append("entry:")
        for var, fn in [("threadgroup_position_in_grid", "get_group_id"), ("thread_position_in_threadgroup", "get_local_id"), ("threads_per_threadgroup", "get_local_size")]:
            n = {"get_group_id": "_Z12get_group_idEj", "get_local_id": "_Z12get_local_idEj", "get_local_size": "_Z14get_local_sizeEj"}[fn]
            prev = "undef"
            for d in range(3):
                out.append(f"  %{var}.w{d} = call spir_func i64 @{n}(i32 {d})")
                out.append(f"  %{var}.t{d} = trunc i64 %{var}.w{d} to i32")
                nxt = f"%{var}" if d == 2 else f"%{var}.v{d}"
                out.append(f"  {nxt} = insertelement <3 x i32> {prev}, i32 %{var}.t{d}, i64 {d}")
                prev = nxt
        out.append("  br label %4")
        out.append("4:")
        body = True
        continue
    if not body:
        continue
    if l == "}":
        out.append(l); body = False; continue
    m = re.match(r'\s*(%\S+) = load \{ ptr addrspace\(1\).*\}, ptr addrspace\(1\) (%\S+), align 8', l)
    if m:
        alias[m.group(1)] = m.group(2); continue
    m = re.match(r'\s*(%\S+) = extractvalue \{ ptr addrspace\(1\).*\} (%\S+), 0', l)
    if m:
        out.append(f"  {m.group(1)} = getelementptr i8, ptr addrspace(1) {alias[m.group(2)]}, i64 0"); continue
    l = l.replace("call float @air.exp2.f32", "call spir_func float @_Z4exp2f")
    l = re.sub(r"f0x([0-9A-F]{8})", lambda h: "0x" + "%016X" % __import__("struct").unpack("<Q", __import__("struct").pack("<d", __import__("struct").unpack("<f", bytes.fromhex(h.group(1))[::-1])[0]))[0], l)
    out.append(l)
open(sys.argv[3], "w").write("\n".join(out) + "\n")
