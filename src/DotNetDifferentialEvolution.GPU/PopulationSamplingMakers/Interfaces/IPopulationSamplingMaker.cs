namespace DotNetDifferentialEvolution.GPU.PopulationSamplingMakers.Interfaces;

/// <summary>
/// Defines the contract for creating initial samples of a population for a differential evolution algorithm.
/// </summary>
/// <remarks>
/// Implementations of this interface generate a population of individuals, where each individual is represented by a vector of values within specified bounds.
/// </remarks>
public interface IPopulationSamplingMaker
{
    /// <summary>
    /// Gets the size of the population to be generated.
    /// </summary>
    /// <returns>The number of individuals in the population.</returns>
    public int GetPopulationSize();
    
    /// <summary>
    /// Generates a sample population with individuals whose values are randomly distributed within specified bounds.
    /// </summary>
    /// <returns>
    /// A 2D array where each row represents an individual, and each column represents a dimension of the individual's vector.
    /// </returns>
    /// <remarks>
    /// The population is generated in parallel to improve performance for larger populations.
    /// </remarks>
    public double[,] TakeSamples();
}
