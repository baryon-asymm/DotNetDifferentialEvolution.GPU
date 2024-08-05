using DotNetDifferentialEvolution.GPU.Models;

namespace DotNetDifferentialEvolution.GPU.SelectionStrategies.Interfaces;

public interface ISelectionStrategy
{
    public void Select(
        int individualIndex,
        Population currentPopulation,
        Population nextPopulation,
        Population trialPopulation);
}
