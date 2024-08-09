using DotNetDifferentialEvolution.GPU.Models;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.TerminationStrategies.Interfaces;

/// <summary>
/// Defines the contract for a termination strategy in a differential evolution algorithm.
/// </summary>
/// <remarks>
/// Implementations of this interface determine the conditions under which the optimization process should be terminated. This allows for flexibility in defining when the algorithm should stop running.
/// </remarks>
public interface ITerminationStrategy
{
    /// <summary>
    /// Determines whether the optimization process should be terminated based on the current state of the algorithm.
    /// </summary>
    /// <param name="device">The GPU device that is running the algorithm.</param>
    /// <param name="generation">The current generation number of the optimization process.</param>
    /// <param name="population">The current population of individuals.</param>
    /// <returns>Returns <see langword="true"/> if the optimization process should terminate; otherwise, <see langword="false"/>.</returns>
    /// <remarks>
    /// Termination strategies can be based on various criteria, such as reaching a maximum number of generations, achieving a desired fitness level, or detecting convergence.
    /// </remarks>
    public bool IsMustTerminate(Accelerator device, int generation, HostPopulation population);
}
