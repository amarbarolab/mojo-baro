import re, sys
src = open(sys.argv[1]).read()
fn = re.search(r"^define void @\S+\((.*?)\) local_unnamed_addr #\d+ \{\n(.*?)^\}", src, re.S | re.M)
body = fn.group(2).splitlines()
body = [re.sub(r"%(\d+)\b", r"%v\1", l) for l in body]
body = [re.sub(r"^(\d+):", r"v\1:", l) for l in body]
body = [re.sub(r"%\.loaded", "%loaded", l) for l in body]
glob = re.search(r"@(\S+\._gpu_shared_mem)", src).group(1)
alias, out_lines = {}, []
for l in body:
    m = re.match(r"\s*(%v\d+) = load \{ ptr addrspace\(1\).*\}, ptr addrspace\(1\) (%v\d+), align 8", l)
    if m:
        alias[m.group(1)] = m.group(2); continue
    m = re.match(r"\s*(%v\d+) = extractvalue \{ ptr addrspace\(1\).*\} (%v\d+), 0", l)
    if m:
        out_lines.append(f"  {m.group(1)} = getelementptr i8, ptr addrspace(1) {alias[m.group(2)]}, i64 0"); continue
    l = re.sub(r"(%v\d+) = call float @air\.convert\.f\.f32\.s\.i64\(i64 (%v\d+)\)", r"\1 = sitofp i64 \2 to float", l)
    l = l.replace("call float @air.rsqrt.f32", "call spir_func float @_Z5rsqrtf")
    l = l.replace(f"ptr addrspace(3) @{glob}", "ptr addrspace(1) %shared").replace("addrspace(3)", "addrspace(1)")
    out_lines.append(l)
body = out_lines
first_br = next(i for i, l in enumerate(body) if l.strip().startswith("br "))
prelude = body[:first_br]
ids = """  %grp.w = call spir_func i64 @_Z12get_group_idEj(i32 0)
  %lid.w = call spir_func i64 @_Z12get_local_idEj(i32 0)
  %lsz.w = call spir_func i64 @_Z14get_local_sizeEj(i32 0)
  %threadgroup_position_in_grid = insertelement <3 x i32> zeroinitializer, i32 %grp.t, i64 0
  %thread_position_in_threadgroup = insertelement <3 x i32> zeroinitializer, i32 %lid.t, i64 0
  %grp.t = trunc i64 %grp.w to i32
  %lid.t = trunc i64 %lid.w to i32""".splitlines()
ids = [ids[0], ids[1], ids[2], ids[5], ids[6], ids[3], ids[4]]
args = "ptr addrspace(1) %v0, ptr addrspace(1) %v1, ptr addrspace(1) %v2, ptr addrspace(2) %v3, ptr addrspace(2) %v4, ptr addrspace(1) %P, ptr addrspace(1) %S"
def head(name):
    return [f"define spir_kernel void @{name}({args}) {{", "v5:"] + ids + prelude
def label_idx(lbl):
    return next(i for i, l in enumerate(body) if l.startswith(lbl + ":"))
shuf = next(i for i, l in enumerate(body) if "air.simd_shuffle_xor" in l)
a = head("rms_a") + body[first_br:shuf] + [
    "  %pidx.r = mul i64 %grp.w, 256", "  %pidx = add i64 %pidx.r, %lid.w",
    "  %pp = getelementptr float, ptr addrspace(1) %P, i64 %pidx", "  store float %v41, ptr addrspace(1) %pp, align 4", "  ret void", "}"]
b = head("rms_b") + [
    "  %lane = urem i64 %lid.w, 32", "  %isl = icmp eq i64 %lane, 0", "  br i1 %isl, label %go, label %done",
    "go:", "  %base.r = mul i64 %grp.w, 256", "  %base = add i64 %base.r, %lid.w", "  br label %loop",
    "loop:", "  %j = phi i64 [ 0, %go ], [ %j1, %loop ]", "  %acc = phi float [ 0.0, %go ], [ %acc1, %loop ]",
    "  %k = add i64 %base, %j", "  %pk = getelementptr float, ptr addrspace(1) %P, i64 %k", "  %x = load float, ptr addrspace(1) %pk, align 4",
    "  %acc1 = fadd float %acc, %x", "  %j1 = add i64 %j, 1", "  %c = icmp ult i64 %j1, 32", "  br i1 %c, label %loop, label %put",
    "put:", "  %w = udiv i64 %lid.w, 32", "  %s.r = mul i64 %grp.w, 8", "  %s = add i64 %s.r, %w",
    "  %ps = getelementptr float, ptr addrspace(1) %S, i64 %s", "  store float %acc1, ptr addrspace(1) %ps, align 4", "  br label %done",
    "done:", "  ret void", "}"]
bar = next(i for i, l in enumerate(body) if "air.wg.barrier" in l)
c = head("rms_c") + ["  %sh.o = mul i64 %grp.w, 8", "  %shared = getelementptr float, ptr addrspace(1) %S, i64 %sh.o", "  br label %v70", "v70:"] + body[bar + 1:] + ["}"]
hdr = ['target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"',
       'target triple = "spirv64-unknown-unknown"',
       "declare spir_func i64 @_Z12get_group_idEj(i32)", "declare spir_func i64 @_Z12get_local_idEj(i32)",
       "declare spir_func i64 @_Z14get_local_sizeEj(i32)", "declare spir_func float @_Z5rsqrtf(float)"]
open(sys.argv[2], "w").write("\n".join(hdr + a + b + c) + "\n")
