using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.RandomGenerators.Interfaces;

namespace DotNetDifferentialEvolution.GPU.MutationStrategies.Interfaces;

public interface IMutationStrategy<in TRandomGenerator>
    where TRandomGenerator : struct, IRandomGenerator
{
    public void Mutate(
        int index,
        DevicePopulation currentPopulation,
        DevicePopulation trialPopulation,
        TRandomGenerator random);
}
