using ILGPU;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.Models;

public record DeviceRandomHolder(
    int PageSize,
    MemoryBuffer2D<double, Stride2D.DenseX> DeviceRandomBuffer)
{
    public Guid Id { get; } = Guid.NewGuid();

    public DeviceRandom GetDeviceRandom()
    {
        var deviceRandom = new DeviceRandom(PageSize, DeviceRandomBuffer.View);
        return deviceRandom;
    }
}