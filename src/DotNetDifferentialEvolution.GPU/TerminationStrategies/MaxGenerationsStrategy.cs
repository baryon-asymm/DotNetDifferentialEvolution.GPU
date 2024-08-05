using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.TerminationStrategies.Interfaces;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.TerminationStrategies;

public class MaxGenerationsStrategy(int maxGenerations) : ITerminationStrategy
{
    public bool IsMustTerminate(Accelerator device, int generation, Population population)
    {
        return generation >= maxGenerations;
    }
}
