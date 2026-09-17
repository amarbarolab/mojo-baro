# P6-aarch64 report

## Result

BLOCKED at the host cross-link step, with the requested evidence collected.
No download or GPU work was performed.

## Item template

- **Files:** NEW `exchange/lane-P6A64-report.md`; receipts under
  `.work/team-C/codex/p6-aarch64/`.
- **Build command:** `cd serve && CARGO_TARGET_DIR=$PWD/../.work/team-C/codex/target cargo build --release`
  is the native Rust baseline. The target command would be
  `cd serve && CARGO_TARGET_DIR=$PWD/../.work/team-C/codex/aarch64-target cargo build --release --target aarch64-unknown-linux-gnu`.
- **Preflight:** `rustup target add aarch64-unknown-linux-gnu` succeeded, and
  `rustup target list --installed` confirms the target. The target build was
  not started because no `aarch64-linux-gnu-gcc` linker is installed.
- **Gate:** On an aarch64 guest, build `baro-serve`, the router, and CPU-only
  Mojo tools, then run `tools/ci-checks.sh` and tokenizer parity. This gate is
  UNVERIFIED because the available guest rig is x86_64-only and unrelated.
- **Receipts:** `.work/team-C/codex/p6-aarch64/environment.md` and
  `.work/team-C/codex/p6-aarch64/vm-rig.md`.
- **Kill line:** missing linker stops Step 1; missing guest rig stops Step 2.
  The item is reported blocked rather than substituted with a host build.
- **GPU budget:** none. The probe is CPU and VM tooling only.
- **Dependencies:** aarch64 GNU linker for host cross-linking, or an aarch64
  Linux guest with the Modular aarch64 package for native builds; no downloads
  were authorized.
- **Owner:** codex leads the probe and report; Sonnet verified the linker result.
- **Size:** S probe, report and receipts only.

## Step 1: Rust

The installed-target read-back is `aarch64-unknown-linux-gnu`,
`aarch64-unknown-linux-musl`, `x86_64-unknown-linux-gnu`, and
`x86_64-unknown-linux-musl`. `aarch64-linux-gnu-gcc` is absent, so the target
cross-build was not run.

Sonnet identified the exact unrun package request as
`pacman -S aarch64-linux-gnu-gcc`, with a 90.49 MiB download and 410.87 MiB
installed footprint, including its cross-toolchain dependencies. Installing
system packages requires coordinator approval.

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
