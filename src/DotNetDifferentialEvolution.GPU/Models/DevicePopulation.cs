using ILGPU;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.Models;

public readonly struct DevicePopulation
{
    public ArrayView1D<double, Stride1D.Dense> FitnessFunctionValues { get; }
    public ArrayView2D<double, Stride2D.DenseX> Individuals { get; }

    public DevicePopulation(
        ArrayView1D<double, Stride1D.Dense> fitnessFunctionValues,
        ArrayView2D<double, Stride2D.DenseX> individuals)
    {
        FitnessFunctionValues = fitnessFunctionValues;
        Individuals = individuals;
    }

    public int PopulationSize => (int)Individuals.Extent.X;
    public int VectorSize => (int)Individuals.Extent.Y;
}
