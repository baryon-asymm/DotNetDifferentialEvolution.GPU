using ILGPU;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.Models;

public record PopulationHolder(
    MemoryBuffer1D<double, Stride1D.Dense> FitnessFunctionValues,
    MemoryBuffer2D<double, Stride2D.DenseX> Individuals)
{
    public Population GetPopulation()
    {
        var population = new Population(FitnessFunctionValues.View, Individuals.View);
        return population;
    }
}
