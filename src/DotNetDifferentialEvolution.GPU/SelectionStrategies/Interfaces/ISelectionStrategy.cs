using DotNetDifferentialEvolution.GPU.Models;

namespace DotNetDifferentialEvolution.GPU.SelectionStrategies.Interfaces;

public interface ISelectionStrategy
{
    public void Select(
        int index,
        DevicePopulation currentPopulation,
        DevicePopulation nextPopulation,
        DevicePopulation trialPopulation);
}
