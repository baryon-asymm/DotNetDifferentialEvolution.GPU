using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.RandomGenerators.Interfaces;

namespace DotNetDifferentialEvolution.GPU.MutationStrategies.Interfaces;

/// <summary>
/// Defines the contract for a mutation strategy in a differential evolution algorithm.
/// </summary>
/// <typeparam name="TRandomGenerator">The type of the random generator used for generating random values.</typeparam>
/// <remarks>
/// Implementations of this interface define how a mutation is applied to individuals in the population 
/// to generate new candidate solutions.
/// </remarks>
public interface IMutationStrategy<in TRandomGenerator>
    where TRandomGenerator : struct, IRandomGenerator
{
    /// <summary>
    /// Applies the mutation strategy to generate a trial individual from the current population.
    /// </summary>
    /// <param name="index">The index of the individual in the population to mutate.</param>
    /// <param name="currentPopulation">The current population residing in GPU memory.</param>
    /// <param name="trialPopulation">The population to store the mutated individuals.</param>
    /// <param name="random">The random generator used for mutation and crossover decisions.</param>
    /// <remarks>
    /// The method applies a differential mutation by selecting random individuals from the population,
    /// applying mutation and crossover strategies, and then storing the resulting trial individual.
    /// </remarks>
    public void Mutate(
        int index,
        DevicePopulation currentPopulation,
        DevicePopulation trialPopulation,
        TRandomGenerator random);
}
