# P6-aarch64 report

## Result

Step 1 PASS. Step 2 PASS (coordinator-approved option 1, 2026-09-17: an
aarch64 lab image, `briefs/2026-09-17-team-c-aarch64-labimage.md`): a real
Arch Linux ARM guest built and booted, `baro-serve` builds natively (not
cross-compiled) as a real aarch64 ELF binary, `pip install "max[all]==26.5.0"`
installs Mojo 1.0.0 natively on aarch64, and `tools/ci-checks.sh` passes
clean (EXIT=0) inside the guest. One sub-check is BLOCKED with an exact
reason (full model-based tokenizer parity needs a GGUF pack that is out of
this S probe's authorized download scope) rather than substituted. No GPU
work was performed; downloads were limited to the authorized rootfs plus its
`.md5`/`.sig`.

## Item template

- **Files:** NEW `exchange/lane-P6A64-report.md`; receipts under
  `.work/team-C/{codex,sonnet}/p6-aarch64/`; NEW `~/AMDHQ/labiso/build-aarch64.sh`
  and `~/AMDHQ/labiso/vm/aarch64/boot.sh` (separate repo, own commit `abc4ff6`).
- **Build command:** Step 1 (cross):
  `cd serve && CARGO_TARGET_DIR=$PWD/../.work/team-C/codex/aarch64-target CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc cargo build --release --target aarch64-unknown-linux-gnu`.
  Step 2 (native, in the guest): `cd serve && cargo build --release`.
- **Preflight:** `rustup target add aarch64-unknown-linux-gnu` succeeded, and
  `rustup target list --installed` confirms the target. The target release
  build passed after the approved linker installation. For Step 2, the
  guest's `pip install "max[all]==26.5.0"` and `mojo --version` are the
  preflight read-back before any build.
- **Gate:** build `baro-serve` and CPU-only Mojo tools on an aarch64 guest,
  run `tools/ci-checks.sh` and tokenizer parity -- PASS, see Step 2 below.
  The router is not built anywhere in the platform yet (P0b unbuilt), so it
  is out of scope for this gate, not silently skipped.
- **Receipts:** `.work/team-C/codex/p6-aarch64/{environment.md,
  step1-cross-build.txt, vm-rig.md}`; `.work/team-C/sonnet/p6-aarch64/{build.log,
  boot-console.log}` and `.work/team-C/sonnet/p6-aarch64/step2/*` (ssh-uname,
  firstboot, harvest-active, mojo-version, mojo-tools-listing, rust-build-native,
  baro-serve-native-file, ci-checks-stale-index-EXIT1, ci-checks-clean-EXIT0,
  baro-tokenize-native-file, baro-tokenize-run).
- **Kill line:** no usable aarch64 guest, or `tools/ci-checks.sh` failing for
  a real (non-environment-artifact) reason, would void Step 2. Neither
  happened: the guest is real and `ci-checks.sh` is EXIT=0.
- **GPU budget:** none anywhere in this item. CPU and VM tooling only.
- **Dependencies:** Step 2 depended on the coordinator's option-1 approval
  (`briefs/2026-09-17-team-c-aarch64-labimage.md`) and the ~500 MB rootfs
  download authorization; both given 2026-09-17.
- **Owner:** codex led Step 1 and the original Step 2 rig investigation;
  codex ran out of usage mid-Step-2 (2026-09-17), Sonnet finished Step 2
  solo (build, boot, guest gates, report) per the coordinator's handoff.
- **Size:** S probe; grew past the original estimate once Step 2 became a
  real image build, still report+receipts+two small scripts, no engine work.

## Step 1: Rust

The installed-target read-back is `aarch64-unknown-linux-gnu`,
`aarch64-unknown-linux-musl`, `x86_64-unknown-linux-gnu`, and
`x86_64-unknown-linux-musl`. With the approved `/usr/bin/aarch64-linux-gnu-gcc`
16.1.0, the target release build exited 0 and produced an ELF aarch64 binary.
The exact command and `file` read-back are in
`.work/team-C/codex/p6-aarch64/step1-cross-build.txt`.

The earlier linker block is resolved. No Rust runtime execution is claimed on
the x86 host because the output is aarch64.

## Step 2: aarch64 lab image (coordinator option 1, 2026-09-17)

The original `~/AMDHQ/labiso/vm/boot-test.sh` rig was confirmed x86_64-only
and unrelated (see below); the coordinator authorized building a real
aarch64 guest instead of stopping there.

**Build.** NEW `~/AMDHQ/labiso/build-aarch64.sh` (committed `abc4ff6`):
downloads `http://os.archlinuxarm.org/os/ArchLinuxARM-aarch64-latest.tar.gz`
(829,367,415 bytes, bigger than the ~500 MB estimate but the exact authorized
URL) plus its `.md5` and `.sig`; verifies both (md5 exact match; gpg
`GOODSIG`/`VALIDSIG` from fingerprint `68B3537F39A313B3E574D06777193F152BDBE6A6`,
"Arch Linux ARM Build System <builder@archlinuxarm.org>", fetched from
`hkps://keyserver.ubuntu.com`); creates a 24 GB raw disk, one ext4 partition
via loop+parted+mkfs, `bsdtar -xpf` extraction (no chroot, no binfmt); copies
the same `overlay/airootfs` as the x86_64 lab ISO (hostname, root ssh key,
sshd drop-in, mDNS, `lab-harvest` + unit, correct modes); adds a one-shot
`lab-firstboot` unit (pacman-key init/populate, `pacman -Syu`, then
`packages.extra` filtered to what `pacman -Si` resolves on aarch64 --
everything installed except `hwinfo`, logged, never silently dropped); an
EXIT trap disables the unit and stamps a done-marker on every exit path, not
just success, so a bad package name can't loop the unit forever; extracts
`/boot/Image` + `/boot/initramfs-linux.img`, converts the disk to qcow2.
NEW `vm/aarch64/boot.sh`: direct-kernel `qemu-system-aarch64 -M virt -cpu max
-smp 8 -m 16G`, SSH on `localhost:2222`, UEFI fallback noted but not needed.

**Guest gate.**
- `ssh -p 2222 root@localhost uname -m` -> `aarch64`
  (`.work/team-C/sonnet/p6-aarch64/step2/ssh-uname.txt`).
- Firstboot log shows pacman finished and disabled itself, rc=0
  (`.../step2/firstboot.log`); `lab-harvest.service` active
  (`.../step2/harvest-active.txt`).
- `pip install "max[all]==26.5.0"` in a venv: EXIT=0, installs `mojo-1.0.0`,
  `mojo-compiler-1.0.0`, `max-26.5.0`, `msgspec-0.21.1` (the one dependency
  with no wheel found in an earlier *host*-side dry-run probe -- it has a
  source distribution and built fine natively on the guest, confirming the
  host probe measured host/venv-python-version mismatch, not an aarch64
  gap). `mojo --version` -> `Mojo 1.0.0 (ed45d567)`
  (`.../step2/mojo-version.txt`).
- `cd serve && cargo build --release` natively (rust installed via pacman):
  EXIT=0 in 8m09s under TCG emulation; `file` on the binary reads
  `ELF 64-bit LSB pie executable, ARM aarch64, ... dynamically linked`
  (`.../step2/rust-build-native.log`, `.../step2/baro-serve-native-file.txt`)
  -- a native build, not the Step 1 cross-build.
- `tools/ci-checks.sh` (repo's 21 MB of tracked files rsynced to
  `/root/mojo-baro` with a fresh local-only git init, no remote; `PATH`
  pointed at the venv's `mojo`): **EXIT=0, "all non-GPU checks passed"**
  (`.../step2/ci-checks-clean-EXIT0.log`) -- kernel census (104 kernels, 58
  registered, 0 orphans), docs/KERNELS.md current, pre-tokenizer regex parity
  (10/10 match llama.cpp, see tokenizer parity below), 88 Python files parse,
  91 shell scripts parse, issue-template YAML, protocol rules P1-P6, vendored
  file checks (upstream absent on this machine is reported as a pass-with-
  note, not a failure), 697 referenced doc paths resolve, hardware receipts
  well-formed, and 33/35 bench sources compile to objects with the GPU-free
  path landed on main between the two guest runs (`1d77f37`, credited to
  codex: benches no longer need a live GPU device to resolve their compile
  target -- see the first run below for what that fixed).
  An earlier run against the same guest checkout, before this fix landed on
  `main`, failed all bench compiles with `function instantiation failed`
  (Mojo's GPU target resolves from the device at compile time, and no GPU
  device exists on this or any aarch64 QEMU virt guest); after merging main
  to `eac30a4` and resyncing, that class of failure is gone.
  A second run against the resynced checkout still showed one FAIL
  (`.work/team-C/sonnet/p6-aarch64/step2/ci-checks-stale-index-EXIT1.log`,
  EXIT=1): `tools/spirv-probe/rms2spv.py` reported missing by the Python
  parse step. Diagnosis (confirmed by `git status --short` inside the
  guest's snapshot repo showing unstaged deletions): the guest's own
  throwaway git index, taken once at the first rsync, was never refreshed
  after a second rsync updated the working tree to a later host commit that
  had genuinely removed the file upstream -- not a real aarch64 problem.
  Fix: `git add -A && git commit` inside the guest's snapshot repo, then
  rerun; the clean rerun above is EXIT=0.
- `tools/baro-tokenize.mojo` (the CPU-only tokenizer CLI) builds
  (`mojo build ... -I . -I serve`) and runs natively: `file` reports
  `ELF 64-bit ... ARM aarch64`, and it prints its usage line correctly
  (`.../step2/baro-tokenize-native-file.txt`, `.../step2/baro-tokenize-run.txt`).

**Tokenizer parity -- partial, one piece BLOCKED.** `tools/ci-checks.sh`'s
own pre-tokenizer regex check (`tools/pretok-check.py` against
`serve/pretok-table.json`) ran on the guest and PASSED: 10 implemented, 0
differ. The deeper, model-based gate (`tools/test_tokenizer_mojo.py`,
`decode(encode(x)) == x` plus agreement with `llama-tokenize` on a real GGUF)
is **BLOCKED**: it requires `--gguf` (a real model pack, hundreds of MB to
GB), a built `~/llama.cpp/build/bin/llama-tokenize`, and
`~/llama.cpp/gguf-py` on `sys.path`, none of which exist on this guest.
Copying a model pack onto the guest is outside this S probe's authorized
download scope (the rootfs plus its `.md5`/`.sig` only); not substituted
with a weaker check.

**Original x86_64 rig, for the record.** `qemu-system-aarch64` is present at
version 11.1.1 on the host. The pre-existing `~/AMDHQ/labiso/vm/boot-test.sh`
rig is unrelated: it invokes `qemu-system-x86_64`, UEFI boots
`kairos-core-amd64-v4.2.0.iso` (314,179,584 bytes) for the AMDHQ latent-os
boot-test lane, and uses x86_64-specific firmware. It is real (Brain recall's
first pass missed it, scoped to `~/Brain` and `~/iTools/INDEX.md` only) but
has zero aarch64 relevance, which is why the coordinator authorized a fresh
aarch64-native image (this section) instead.

## Commits and suite

- P6-pwa prerequisite commits: `2c2a90a`, `e7817a6`, `d241573`.
- `./run-tests.sh` (host, x86_64): exit 0, 47 `PASS` lines, 104 kernels, 58
  registered, 0 orphans. Receipt: `.work/team-C/codex/p6-pwa/run-tests.log`.
- `tools/ci-checks.sh` (aarch64 guest): exit 0, "all non-GPU checks passed".
  Receipt: `.work/team-C/sonnet/p6-aarch64/step2/ci-checks-clean-EXIT0.log`.
- `~/AMDHQ` commit `abc4ff6`: `labiso/build-aarch64.sh` +
  `labiso/vm/aarch64/boot.sh`.
- This report's commit on `lane-team-c`: see the report commit itself.
