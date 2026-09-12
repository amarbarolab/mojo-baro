comptime H = 2048
comptime VOCAB = 248320
comptime FFN = 512
comptime HD = 256
comptime NQH = 16
comptime NKVH = 2
comptime N_LAYERS = 40
comptime N_SSM = 30
comptime N_ATT = 10
comptime QF = 8192
comptime KV = 512
comptime NROT = 64
comptime YARN_LOW = Float32(14.0)
comptime YARN_HIGH = Float32(22.0)
comptime FREQ_BASE = Float32(1e7)
# RegesCore's GGUF declares NO rope scaling: no rope.scaling.type, no
# rope.scaling.factor, only rope.freq_base 1e7 and rope.dimension_count 64.
# llama.cpp therefore runs plain rope for this model (freq_scale 1.0,
# ext_factor 0, mscale 1.0) and so must we. The 0.25 here was copied from
# the qwen35 dense profile, whose GGUF really does declare yarn factor 4.0;
# it made us interpolate where the reference extrapolates.
comptime FREQ_SCALE = Float32(1.0)
comptime MSCALE = Float32(1.0)
comptime MEGA_ALLOWED = False
