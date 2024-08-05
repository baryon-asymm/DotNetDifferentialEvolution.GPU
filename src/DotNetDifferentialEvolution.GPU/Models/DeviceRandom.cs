using ILGPU;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.Models;

public readonly struct DeviceRandom(
    int pageSize,
    ArrayView2D<double, Stride2D.DenseX> numbers)
{
    public int PageSize { get; } = pageSize;
    public double NextDouble(int index, int page, int step) => numbers[index, page * pageSize + step];

    public int Next(int index, int page, int step, int maxNumber) =>
        (int)Math.Round(maxNumber * numbers[index, page * pageSize + step]);
}
