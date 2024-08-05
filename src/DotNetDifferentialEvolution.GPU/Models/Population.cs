using ILGPU;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.Models;

public readonly struct Population(
    ArrayView<double> fitnessFunctionValues,
    ArrayView2D<double, Stride2D.DenseX> individuals)
{
    public ArrayView<double> FitnessFunctionValues { get; } = fitnessFunctionValues;
    public ArrayView2D<double, Stride2D.DenseX> Individuals { get; } = individuals;

    public int IndividualsLength => (int)Individuals.Extent.X;
    public int IndividualVectorLength => (int)Individuals.Extent.Y;
}
