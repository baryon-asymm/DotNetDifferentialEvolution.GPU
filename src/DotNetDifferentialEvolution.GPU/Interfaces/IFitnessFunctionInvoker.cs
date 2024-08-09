using DotNetDifferentialEvolution.GPU.Models;

namespace DotNetDifferentialEvolution.GPU.Interfaces;

/// <summary>
/// Defines the contract for invoking a fitness function on a population individual.
/// </summary>
public interface IFitnessFunctionInvoker
{
    /// <summary>
    /// Invokes the fitness function on a specified individual within a population on the GPU.
    /// </summary>
    /// <param name="individualIndex">The index of the individual in the population.</param>
    /// <param name="devicePopulation">The population data residing on the GPU.</param>
    public void Invoke(int individualIndex, DevicePopulation devicePopulation);
}
