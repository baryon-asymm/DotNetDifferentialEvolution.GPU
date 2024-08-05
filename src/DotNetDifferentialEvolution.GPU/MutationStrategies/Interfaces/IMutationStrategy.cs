using DotNetDifferentialEvolution.GPU.Models;

namespace DotNetDifferentialEvolution.GPU.MutationStrategies.Interfaces;

public interface IMutationStrategy
{
    public void Mutate(
        int indexOfCurrentIndividual,
        Population currentPopulation,
        Population trialPopulation,
        int pageOfRandom,
        DeviceRandom random);

    public int GetMaxRandomNumbersPerIndividual(int individualVectorSize);
}
