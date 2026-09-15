# VENDORED COPY. Upstream is ~/AMDHQ/src/latentos/ipc.mojo; this repo keeps
# a real file rather than a symlink or an -I path outside the tree,
# because a clone must build without anything outside it. Sync by hand
# if the upstream changes; tools/ci-checks.sh compares the two when the
# upstream is present and says so when they drift.
#
# ipc.mojo — LatentOS IPC handle transport & memfd lifecycle (03 §4, §5).
# Provides sealed memfd creation, pure Mojo SCM_RIGHTS descriptor passing, and the agent handle table.

from std.memory import Pointer, unsafe_memset
from std.memory.alloc import unsafe_alloc
from std.ffi import external_call

import latentos.sys as sys
import latentos.proto as proto

comptime BytePtr = Pointer[UInt8, MutUntrackedOrigin]

def create_ipc_pair() raises -> Tuple[Int32, Int32]:
    """Creates a connected full-duplex UNIX stream socketpair for agent<->engine IPC."""
    var sv = unsafe_alloc[Int32](2)
    var rc = external_call["socketpair", Int32](sys.AF_UNIX, sys.SOCK_STREAM, 0, sv)
    var fd0 = sv[unsafe_offset=0]
    var fd1 = sv[unsafe_offset=1]
    sv.unsafe_free()
    if rc != 0:
        raise Error("socketpair failed with errno " + String(sys.errno_now()))
    return (fd0, fd1)

def create_unix_listener(path: String) raises -> Int32:
    """Binds and listens on a UNIX domain socket at path."""
    _ = sys.sys_unlink(path)
    var fd = sys.sys_socket(sys.AF_UNIX, sys.SOCK_STREAM, 0)
    if fd < 0:
        raise Error("socket() failed with errno " + String(sys.errno_now()))
    var bind_rc = sys.sys_bind_unix(fd, path)
    if bind_rc != 0:
        _ = sys.sys_close(fd)
        raise Error("bind() failed for " + path + " with errno " + String(sys.errno_now()))
    var listen_rc = sys.sys_listen(fd, 16)
    if listen_rc != 0:
        _ = sys.sys_close(fd)
        raise Error("listen() failed with errno " + String(sys.errno_now()))
    return fd

def connect_unix_socket(path: String) raises -> Int32:
    """Connects to a UNIX domain socket at path."""
    var fd = sys.sys_socket(sys.AF_UNIX, sys.SOCK_STREAM, 0)
    if fd < 0:
        raise Error("socket() failed with errno " + String(sys.errno_now()))
    var rc = sys.sys_connect_unix(fd, path)
    if rc != 0:
        _ = sys.sys_close(fd)
        raise Error("connect() failed to " + path + " with errno " + String(sys.errno_now()))
    return fd

def mint_memfd_rw(name: String, size_bytes: Int) raises -> Tuple[Int32, BytePtr]:
    """Mints a sealable memfd and maps it writable.
    Caller fills the buffer, then calls seal_and_finalize().
    """
    var fd = sys.sys_memfd_create(name, sys.MFD_CLOEXEC | sys.MFD_ALLOW_SEALING)
    if fd < 0:
        raise Error("memfd_create failed with errno " + String(sys.errno_now()))
    var rc = sys.sys_ftruncate(fd, Int64(size_bytes))
    if rc != 0:
        _ = sys.sys_close(fd)
        raise Error("ftruncate failed with errno " + String(sys.errno_now()))
    var p = sys.sys_mmap(size_bytes, sys.PROT_READ | sys.PROT_WRITE, sys.MAP_SHARED, fd, 0)
    if Int(p) == -1:
        _ = sys.sys_close(fd)
        raise Error("mmap failed with errno " + String(sys.errno_now()))
    # 2 MiB shmem folios where the THP policy allows (shmem_enabled=advise): each fresh
    # 4 KiB page costs an allocation plus zeroing, about 9 ms per 20 MiB. Best effort.
    _ = sys.sys_madvise(p, size_bytes, sys.MADV_HUGEPAGE)
    return (fd, p)

def seal_and_finalize(fd: Int32, ptr: BytePtr, size_bytes: Int) -> Bool:
    """Unmaps the writable pointer and applies immutable seals.
    Mint order (03 §4): write -> munmap -> fcntl(F_ADD_SEALS) -> export.
    """
    var unmap_rc = sys.sys_munmap(ptr, size_bytes)
    if unmap_rc != 0:
        return False
    var seals = sys.F_SEAL_GROW | sys.F_SEAL_SHRINK | sys.F_SEAL_WRITE | sys.F_SEAL_SEAL
    var seal_rc = sys.sys_fcntl_add_seals(fd, Int32(seals))
    return seal_rc == 0

def map_readonly(fd: Int32, size_bytes: Int) raises -> BytePtr:
    """Maps a sealed memfd read-only (zero-copy ingest)."""
    var p = sys.sys_mmap(size_bytes, sys.PROT_READ, sys.MAP_SHARED, fd, 0)
    if Int(p) == -1:
        raise Error("map_readonly failed with errno " + String(sys.errno_now()))
    _ = sys.sys_madvise(p, size_bytes, sys.MADV_HUGEPAGE)
    return p

# --- SCM_RIGHTS Handle Passing
def _build_msghdr(msg: BytePtr, iov: BytePtr, payload: BytePtr, payload_len: Int, ctrl: BytePtr, ctrl_len: Int):
    unsafe_memset(msg, 0, 56)
    iov.unsafe_offset(0).unsafe_bitcast[Int64]()[] = Int64(Int(payload))
    iov.unsafe_offset(8).unsafe_bitcast[Int64]()[] = Int64(payload_len)
    msg.unsafe_offset(16).unsafe_bitcast[Int64]()[] = Int64(Int(iov))
    msg.unsafe_offset(24).unsafe_bitcast[Int64]()[] = 1
    msg.unsafe_offset(32).unsafe_bitcast[Int64]()[] = Int64(Int(ctrl))
    msg.unsafe_offset(40).unsafe_bitcast[Int64]()[] = Int64(ctrl_len)

def send_handle(sock: Int32, header: proto.LatentHeader, fd: Int32) -> Bool:
    """Sends a 256-byte LatentHeader and open memfd atomically over a Unix socket via SCM_RIGHTS."""
    var hbuf = unsafe_alloc[UInt8](proto.LATENT_HEADER_SIZE)
    header.serialize(hbuf)

    var msg = unsafe_alloc[UInt8](56)
    var iov = unsafe_alloc[UInt8](16)
    var ctrl = unsafe_alloc[UInt8](24)
    unsafe_memset(ctrl, 0, 24)

    # cmsghdr layout (x86-64 glibc):
    #   0..7: cmsg_len = 20 (CMSG_LEN(sizeof(int)))
    #   8..11: cmsg_level = SOL_SOCKET (1)
    #   12..15: cmsg_type = SCM_RIGHTS (1)
    #   16..19: fd
    ctrl.unsafe_offset(0).unsafe_bitcast[Int64]()[] = 20
    ctrl.unsafe_offset(8).unsafe_bitcast[Int32]()[] = sys.SOL_SOCKET
    ctrl.unsafe_offset(12).unsafe_bitcast[Int32]()[] = sys.SCM_RIGHTS
    ctrl.unsafe_offset(16).unsafe_bitcast[Int32]()[] = fd

    _build_msghdr(msg, iov, hbuf, proto.LATENT_HEADER_SIZE, ctrl, 24)
    var n = external_call["sendmsg", Int64](sock, msg, 0)

    hbuf.unsafe_free()
    msg.unsafe_free()
    iov.unsafe_free()
    ctrl.unsafe_free()

    return Int(n) == proto.LATENT_HEADER_SIZE

def recv_handle(sock: Int32) -> Tuple[proto.LatentHeader, Int32]:
    """Receives a 256-byte LatentHeader and its passed memfd atomically over a Unix socket."""
    var hbuf = unsafe_alloc[UInt8](proto.LATENT_HEADER_SIZE)
    var msg = unsafe_alloc[UInt8](56)
    var iov = unsafe_alloc[UInt8](16)
    var ctrl = unsafe_alloc[UInt8](24)
    unsafe_memset(ctrl, 0, 24)

    _build_msghdr(msg, iov, hbuf, proto.LATENT_HEADER_SIZE, ctrl, 24)
    var n = external_call["recvmsg", Int64](sock, msg, 0)

    var header = proto.LatentHeader()
    var received_fd: Int32 = -1

    if Int(n) == proto.LATENT_HEADER_SIZE:
        header = proto.LatentHeader.deserialize(hbuf)
        var sol = ctrl.unsafe_offset(8).unsafe_bitcast[Int32]()[]
        var scm = ctrl.unsafe_offset(12).unsafe_bitcast[Int32]()[]
        if sol == sys.SOL_SOCKET and scm == sys.SCM_RIGHTS:
            received_fd = ctrl.unsafe_offset(16).unsafe_bitcast[Int32]()[]

    hbuf.unsafe_free()
    msg.unsafe_free()
    iov.unsafe_free()
    ctrl.unsafe_free()

    return (header^, received_fd)

# --- TCP transport (03 sec.5, cross-host tier: raw payload, no shared kernel to
# hand an fd to, so this always copies bytes, never SCM_RIGHTS).
comptime FNV_OFFSET: UInt64 = 0xcbf29ce484222325
comptime FNV_PRIME: UInt64 = 0x100000001b3
comptime TCP_CHUNK = 4 * 1024 * 1024

def fnv1a64(ptr: BytePtr, n: Int) -> UInt64:
    """FNV-1a 64 over n bytes at ptr -- the same function latent_ipc.h names for prefix_hash,
    reused here as the wire integrity check so cross-host framing stays self-contained (no
    crypto library to ship into a guest)."""
    var h: UInt64 = FNV_OFFSET
    for i in range(n):
        h = h ^ UInt64(ptr[unsafe_offset=i])
        h = h * FNV_PRIME
    return h

def tcp_listen(bind_ip: String, port: UInt16) raises -> Int32:
    """Creates, binds (SO_REUSEADDR) and listens on a TCP socket at bind_ip:port."""
    var fd = sys.sys_socket(sys.AF_INET, sys.SOCK_STREAM, 0)
    if fd < 0:
        raise Error("socket() failed with errno " + String(sys.errno_now()))
    _ = sys.sys_setsockopt_reuseaddr(fd)
    var bind_rc = sys.sys_bind_inet(fd, bind_ip, port)
    if bind_rc != 0:
        _ = sys.sys_close(fd)
        raise Error("bind() failed for " + bind_ip + ":" + String(port) + " with errno " + String(sys.errno_now()))
    var listen_rc = sys.sys_listen(fd, 4)
    if listen_rc != 0:
        _ = sys.sys_close(fd)
        raise Error("listen() failed with errno " + String(sys.errno_now()))
    return fd

def tcp_accept(listen_fd: Int32) raises -> Int32:
    var fd = sys.sys_accept(listen_fd)
    if fd < 0:
        raise Error("accept() failed with errno " + String(sys.errno_now()))
    return fd

def tcp_connect(ip: String, port: UInt16) raises -> Int32:
    var fd = sys.sys_socket(sys.AF_INET, sys.SOCK_STREAM, 0)
    if fd < 0:
        raise Error("socket() failed with errno " + String(sys.errno_now()))
    var rc = sys.sys_connect_inet(fd, ip, port)
    if rc != 0:
        _ = sys.sys_close(fd)
        raise Error("connect() failed to " + ip + ":" + String(port) + " with errno " + String(sys.errno_now()))
    return fd

def _tcp_write_all(fd: Int32, ptr: BytePtr, n: Int) raises:
    var sent = 0
    while sent < n:
        var rc = sys.sys_write(fd, ptr.unsafe_offset(sent), n - sent)
        if rc <= 0:
            raise Error("write() failed with errno " + String(sys.errno_now()))
        sent += Int(rc)

def _tcp_read_all(fd: Int32, ptr: BytePtr, n: Int) raises:
    var got = 0
    while got < n:
        var rc = sys.sys_read(fd, ptr.unsafe_offset(got), n - got)
        if rc <= 0:
            raise Error("read() failed or peer closed early with errno " + String(sys.errno_now()))
        got += Int(rc)

def send_tcp_latent(sock: Int32, header: proto.LatentHeader, payload: BytePtr, payload_len: Int) raises -> UInt64:
    """Sends a 256-byte LatentHeader, an 8-byte little-endian payload length, the payload itself
    (chunked), and a trailing 8-byte FNV-1a64 hash, over a connected TCP socket. Returns the hash
    sent, for the caller's own logging."""
    var hbuf = unsafe_alloc[UInt8](proto.LATENT_HEADER_SIZE)
    header.serialize(hbuf)
    _tcp_write_all(sock, hbuf, proto.LATENT_HEADER_SIZE)
    hbuf.unsafe_free()

    var lenbuf = unsafe_alloc[UInt8](8)
    lenbuf.unsafe_bitcast[UInt64]()[] = UInt64(payload_len)
    _tcp_write_all(sock, lenbuf, 8)
    lenbuf.unsafe_free()

    var sent = 0
    while sent < payload_len:
        var n = payload_len - sent
        if n > TCP_CHUNK:
            n = TCP_CHUNK
        _tcp_write_all(sock, payload.unsafe_offset(sent), n)
        sent += n

    var hash = fnv1a64(payload, payload_len)
    var hashbuf = unsafe_alloc[UInt8](8)
    hashbuf.unsafe_bitcast[UInt64]()[] = hash
    _tcp_write_all(sock, hashbuf, 8)
    hashbuf.unsafe_free()
    return hash

def recv_tcp_latent(sock: Int32, dst: BytePtr, dst_cap: Int) raises -> Tuple[proto.LatentHeader, Int, UInt64, Bool]:
    """Receives a LatentHeader + length-prefixed payload (must fit in dst_cap) + trailing sender
    hash over a connected TCP socket. Returns (header, payload_len, local_hash, hash_matches)."""
    var hbuf = unsafe_alloc[UInt8](proto.LATENT_HEADER_SIZE)
    _tcp_read_all(sock, hbuf, proto.LATENT_HEADER_SIZE)
    var header = proto.LatentHeader.deserialize(hbuf)
    hbuf.unsafe_free()

    var lenbuf = unsafe_alloc[UInt8](8)
    _tcp_read_all(sock, lenbuf, 8)
    var payload_len = Int(lenbuf.unsafe_bitcast[UInt64]()[])
    lenbuf.unsafe_free()

    if payload_len > dst_cap:
        raise Error("payload_len " + String(payload_len) + " exceeds destination capacity " + String(dst_cap))

    var got = 0
    while got < payload_len:
        var n = payload_len - got
        if n > TCP_CHUNK:
            n = TCP_CHUNK
        _tcp_read_all(sock, dst.unsafe_offset(got), n)
        got += n

    var hashbuf = unsafe_alloc[UInt8](8)
    _tcp_read_all(sock, hashbuf, 8)
    var sender_hash = hashbuf.unsafe_bitcast[UInt64]()[]
    hashbuf.unsafe_free()

    var local_hash = fnv1a64(dst, payload_len)
    return (header^, payload_len, local_hash, sender_hash == local_hash)

# --- Latent Store Handle Entry
struct HandleEntry(Copyable, Movable):
    var address: String
    var header: proto.LatentHeader
    var fd: Int32
    var minted_at_s: Int64
    var expires_at_s: Int64
    var refcount: Int

    def __init__(out self, address: String, header: proto.LatentHeader, fd: Int32, now_s: Int64):
        self.address = address
        self.header = header.copy()
        self.fd = fd
        self.minted_at_s = now_s
        self.expires_at_s = now_s + Int64(header.ttl_s)
        self.refcount = 1

    def is_expired(self, now_s: Int64) -> Bool:
        return now_s >= self.expires_at_s

# --- Local Agent Handle Registry (03 §4)
struct LatentStore:
    """In-agent handle table that survives engine crashes. Holds open memfds and manages TTLs."""
    var entries: List[HandleEntry]

    def __init__(out self):
        self.entries = List[HandleEntry]()

    def put(mut self, header: proto.LatentHeader, fd: Int32) -> String:
        var now_s = sys.sys_clock_monotonic_s()
        var addr = header.content_address()
        for i in range(len(self.entries)):
            if self.entries[i].address == addr:
                self.entries[i].refcount += 1
                self.entries[i].expires_at_s = now_s + Int64(header.ttl_s)
                return addr
        var entry = HandleEntry(addr, header, fd, now_s)
        self.entries.append(entry^)
        return addr

    def get(mut self, address: String) -> Tuple[proto.LatentHeader, Int32]:
        var now_s = sys.sys_clock_monotonic_s()
        for i in range(len(self.entries)):
            if self.entries[i].address == address:
                if not self.entries[i].is_expired(now_s):
                    # Renew lease on access
                    self.entries[i].expires_at_s = now_s + Int64(self.entries[i].header.ttl_s)
                    return (self.entries[i].header.copy(), self.entries[i].fd)
                else:
                    return (proto.LatentHeader(), -1)
        return (proto.LatentHeader(), -1)

    def evict_expired(mut self, now_s: Int64) -> Int:
        """Closes expired memfds and prunes them from the store. Returns count evicted."""
        var kept = List[HandleEntry]()
        var evicted = 0
        for i in range(len(self.entries)):
            if self.entries[i].is_expired(now_s) and self.entries[i].refcount <= 1:
                _ = sys.sys_close(self.entries[i].fd)
                evicted += 1
            else:
                kept.append(self.entries[i].copy())
        self.entries = kept^
        return evicted

    def count(self) -> Int:
        return len(self.entries)
