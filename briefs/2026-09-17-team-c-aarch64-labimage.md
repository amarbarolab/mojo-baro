# Team C: P6-aarch64 step 2, an aarch64 lab image (the maintainer chose option 1, 2026-09-17)

Build rules `briefs/2026-09-17-team-build-rules.md` still bind (room C, pathspec commits, no em
dashes, loud failures, reports). No GPU. Sonnet leads (writes outside the worktree and needs sudo);
codex reviews the script and the receipts.

## Goal

An aarch64 guest that behaves like the x86_64 lab ISO (`~/AMDHQ/labiso/build.sh`): same
`overlay/airootfs` (hostname, root ssh key, sshd drop-in, mDNS, `lab-harvest` + unit) and the
`packages.extra` set where aarch64 packages exist. Then run the original step 2 inside it.

## Authorized

- Download `http://os.archlinuxarm.org/os/ArchLinuxARM-aarch64-latest.tar.gz` (~500 MB) plus its
  `.md5`/`.sig`; verify before use. Store under `~/AMDHQ/labiso/vm/aarch64/` (disk, not /tmp).
- `sudo` for loop mounts and extraction with ownership (`bsdtar -xpf`).
- No new pacman packages on the host. No binfmt/qemu-user-static: packages are installed on the
  guest's first boot over qemu user networking, not in a host chroot.

## Build (NEW `~/AMDHQ/labiso/build-aarch64.sh`, `set -euo pipefail`, FAIL lines)

1. `qemu-img create -f raw` 24 GB, one ext4 partition, loop-mount, `bsdtar -xpf` the rootfs.
2. Copy `overlay/airootfs/.` over it; set modes like build.sh's `file_permissions` (700 `.ssh`, 600
   `authorized_keys`, 755 `lab-harvest`); enable sshd + `lab-harvest.service` via symlinks.
3. Add a one-shot `lab-firstboot.service`: `pacman-key --init`, `pacman-key --populate
   archlinuxarm`, `pacman -Syu --noconfirm` + the `packages.extra` names that exist for aarch64
   (log the missing ones, never silently drop), then disable itself.
4. Copy `/boot/Image` and `/boot/initramfs-linux.img` out of the rootfs; convert to qcow2.
5. `vm/aarch64/boot.sh`: `qemu-system-aarch64 -M virt -cpu max -smp 8 -m 16G -kernel Image
   -initrd initramfs-linux.img -append "root=/dev/vda1 rw console=ttyAMA0" -drive
   if=virtio,file=lab-aarch64.qcow2 -nic user,hostfwd=tcp::2222-:22 -nographic`
   (UEFI firmware `/usr/share/edk2/aarch64/QEMU_EFI.fd` only if direct kernel boot fails).

## Gate (each a receipt under `.work/team-C/*/p6-aarch64/step2/`)

- `ssh -p 2222 root@localhost uname -m` prints `aarch64` with the lab key; firstboot log shows
  pacman finished; `lab-harvest` ran.
- Original step 2 inside the guest: build `baro-serve` natively, install Modular's aarch64 Mojo
  package, build the CPU-only Mojo tools, run `tools/ci-checks.sh` and tokenizer parity. Emulation is
  10 to 20x slow: run long builds detached in the guest (tmux), poll, never block a turn on them.
- Any step that cannot pass (e.g. no aarch64 Mojo package) is reported BLOCKED with the exact error,
  never substituted.

## Deliverables

`~/AMDHQ/labiso/build-aarch64.sh` + `vm/aarch64/boot.sh` committed in `~/AMDHQ` (pathspec);
`exchange/lane-P6A64-report.md` step 2 section updated and committed on `lane-team-c`; the Brain
note `~/Brain/AMDHQ/` or the labiso README gets the image recipe. Reply in room C only with the
report path.
