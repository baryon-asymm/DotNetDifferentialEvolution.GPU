using DotNetDifferentialEvolution.GPU.Models;

namespace DotNetDifferentialEvolution.GPU.SelectionStrategies.Interfaces;

/// <summary>
/// Defines the contract for a selection strategy in a differential evolution algorithm.
/// </summary>
/// <remarks>
/// Implementations of this interface determine how the next generation of individuals is selected from the current population and the trial population based on fitness values.
/// </remarks>
public interface ISelectionStrategy
{
    /// <summary>
    /// Selects an individual for the next generation based on a comparison of fitness values between the current and trial populations.
    /// </summary>
    /// <param name="index">The index of the individual to evaluate.</param>
    /// <param name="currentPopulation">The current population residing in GPU memory.</param>
    /// <param name="nextPopulation">The population that will store the selected individuals for the next generation.</param>
    /// <param name="trialPopulation">The trial population containing mutated and potentially better individuals.</param>
    /// <remarks>
    /// The method compares the fitness of the individual in the trial population with the corresponding individual in the current population and selects the better one to be part of the next generation.
    /// </remarks>
    public void Select(
        int index,
        DevicePopulation currentPopulation,
        DevicePopulation nextPopulation,
        DevicePopulation trialPopulation);
}
