# P6-aarch64 report

## Result

Step 1 PASS. Overall item remains blocked at Step 2 because no usable
aarch64 guest rig is available. No download or GPU work was performed.

## Item template

- **Files:** NEW `exchange/lane-P6A64-report.md`; receipts under
  `.work/team-C/codex/p6-aarch64/`.
- **Build command:** `cd serve && CARGO_TARGET_DIR=$PWD/../.work/team-C/codex/aarch64-target CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc cargo build --release --target aarch64-unknown-linux-gnu`.
- **Preflight:** `rustup target add aarch64-unknown-linux-gnu` succeeded, and
  `rustup target list --installed` confirms the target. The target release
  build passed after the approved linker installation.
- **Gate:** On an aarch64 guest, build `baro-serve`, the router, and CPU-only
  Mojo tools, then run `tools/ci-checks.sh` and tokenizer parity. This gate is
  UNVERIFIED because the available guest rig is x86_64-only and unrelated.
- **Receipts:** `.work/team-C/codex/p6-aarch64/environment.md`,
  `.work/team-C/codex/p6-aarch64/step1-cross-build.txt`, and
  `.work/team-C/codex/p6-aarch64/vm-rig.md`.
- **Kill line:** missing guest rig stops Step 2. Step 1 is now a verified
  cross-build, not a substituted host build.
- **GPU budget:** none. The probe is CPU and VM tooling only.
- **Dependencies:** an aarch64 Linux guest with the Modular aarch64 package for
  native builds; no downloads were authorized.
- **Owner:** codex leads the probe and report; Sonnet verified the linker result.
- **Size:** S probe, report and receipts only.

## Step 1: Rust

The installed-target read-back is `aarch64-unknown-linux-gnu`,
`aarch64-unknown-linux-musl`, `x86_64-unknown-linux-gnu`, and
`x86_64-unknown-linux-musl`. With the approved `/usr/bin/aarch64-linux-gnu-gcc`
16.1.0, the target release build exited 0 and produced an ELF aarch64 binary.
The exact command and `file` read-back are in
`.work/team-C/codex/p6-aarch64/step1-cross-build.txt`.

The earlier linker block is resolved. No Rust runtime execution is claimed on
the x86 host because the output is aarch64.

## Step 2: VM rig

`qemu-system-aarch64` is present at version 11.1.1. The available
`~/AMDHQ/labiso/vm/boot-test.sh` is an unrelated latent-os rig: it invokes
`qemu-system-x86_64`, UEFI boots `kairos-core-amd64-v4.2.0.iso` (314,179,584
bytes), and uses x86_64-specific firmware. Brain recall, targeted
`iTools/INDEX.md` and `Brain/OS` search, and direct inspection found no
aarch64 guest or launcher. The labiso path is real, but has zero aarch64
relevance for this item.

An aarch64 guest would need a Linux aarch64 image, matching firmware, the
Modular Linux aarch64 package, the repo checkout, and enough storage for the
CPU-only tool builds. The exact image and package download sizes are not known
locally. Do not download until the coordinator names the image, URL, and size.

## Commits and suite

- P6-pwa prerequisite commits: `2c2a90a`, `e7817a6`, `d241573`.
- `./run-tests.sh`: exit 0, 47 `PASS` lines, 104 kernels, 58 registered,
  0 orphans. Receipt: `.work/team-C/codex/p6-pwa/run-tests.log`.
- No aarch64 build or aarch64 `tools/ci-checks.sh` run is claimed.
