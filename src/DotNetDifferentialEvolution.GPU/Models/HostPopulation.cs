using ILGPU;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.Models;

public class HostPopulation
{
    public MemoryBuffer1D<double, Stride1D.Dense> FitnessFunctionValues { get; }
    public MemoryBuffer2D<double, Stride2D.DenseX> Individuals { get; }
    
    public HostPopulation(
        MemoryBuffer1D<double, Stride1D.Dense> fitnessFunctionValues,
        MemoryBuffer2D<double, Stride2D.DenseX> individuals)
    {
        FitnessFunctionValues = fitnessFunctionValues;
        Individuals = individuals;
    }
    
    public DevicePopulation GetDevicePopulation()
    {
        var population = new DevicePopulation(
            FitnessFunctionValues.View,
            Individuals.View);
        
        return population;
    }
}
