# VENDORED COPY. Upstream is ~/AMDHQ/src/latentos/__init__.mojo; this repo keeps
# a real file rather than a symlink or an -I path outside the tree,
# because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
# __init__.mojo — liblatentos package root

from .sys import (
    sys_memfd_create,
    sys_ftruncate,
    sys_fcntl_add_seals,
    sys_fcntl_get_seals,
    sys_mmap,
    sys_munmap,
    sys_mlock2,
    sys_munlock,
    sys_mincore,
    sys_close,
    sys_clock_monotonic_ns,
    sys_clock_monotonic_us,
    sys_clock_monotonic_s,
    sys_posix_spawn,
    sys_waitpid,
    sys_kill,
    sys_sd_notify,
    sys_read_meminfo_hugepages,
)

from .proto import (
    LATENT_MAGIC,
    LATENT_VERSION,
    LATENT_HEADER_SIZE,
    KIND_KV_PAGES,
    KIND_SSM_CKPT,
    KIND_HIDDEN,
    KIND_LOGITS_TOPK,
    KIND_TEXT,
    DTYPE_F32,
    DTYPE_BF16,
    BATCH_M1,
    BATCH_FIXED_M,
    BATCH_DYN,
    QWYTHOS_KV_BYTES_PER_TOK,
    QWYTHOS_SSM_CKPT_BYTES,
    QWYTHOS_HIDDEN_BYTES_PER_STEP,
    LatentHeader,
    byte_to_hex,
    u64_to_hex,
)

from .ipc import (
    create_ipc_pair,
    mint_memfd_rw,
    seal_and_finalize,
    map_readonly,
    send_handle,
    recv_handle,
    LatentStore,
)
