from max.gpu.host import DeviceContext, DeviceBuffer
from window import WindowBufs
from registry import f32


def realign_expected_embedding(
    ctx: DeviceContext,
    mut b: WindowBufs,
    e_dev: DeviceBuffer[f32],
) raises:
    raise Error("REALIGN not merged")
