using System.Runtime.CompilerServices;
using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.TerminationStrategies.Interfaces;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.TerminationStrategies;

public class MaxGenerationStrategy(int maxGenerationCount) : ITerminationStrategy
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool IsMustTerminate(Accelerator device, int generation, HostPopulation population)
    {
        return generation >= maxGenerationCount;
    }
}
