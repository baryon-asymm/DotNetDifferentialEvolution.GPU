using DotNetDifferentialEvolution.GPU.Models;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.TerminationStrategies.Interfaces;

public interface ITerminationStrategy
{
    public bool IsMustTerminate(Accelerator device, int generation, HostPopulation population);
}
