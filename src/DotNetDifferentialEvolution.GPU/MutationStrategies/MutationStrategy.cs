using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.MutationStrategies.Interfaces;
using ILGPU;

namespace DotNetDifferentialEvolution.GPU.MutationStrategies;

public readonly struct MutationStrategy(
    ArrayView<double> lowerBound,
    ArrayView<double> upperBound,
    double mutationForce = 0.3,
    double crossoverFactor = 0.8)
    : IMutationStrategy
{
    private const int NumberOfIndividualsToChoose = 3;

    public void Mutate(
        int indexOfCurrentIndividual,
        Population currentPopulation,
        Population trialPopulation,
        int pageOfRandom,
        DeviceRandom random)
    {
        var randomStep = 0;
        var indexes = new int[NumberOfIndividualsToChoose];

        for (var i = 0; i < NumberOfIndividualsToChoose; i++)
        {
            indexes[i] = random.Next(indexOfCurrentIndividual, pageOfRandom, randomStep++, currentPopulation.IndividualsLength - 2);
            if (indexes[i] >= indexOfCurrentIndividual) indexes[i]++;

            for (var j = 0; j < i; j++)
            {
                if (indexes[i] == indexes[j])
                {
                    indexes[i]++;
                    indexes[i] %= currentPopulation.IndividualsLength - 1;
                    if (indexes[i] == indexOfCurrentIndividual)
                        indexes[i]++;
                    j = 0;
                }
            }
        }

        var trialIndividualLength = trialPopulation.IndividualVectorLength;
        for (var i = 0; i < trialIndividualLength; i++)
            if (random.NextDouble(indexOfCurrentIndividual, pageOfRandom, randomStep++) <= crossoverFactor)
            {
                var trialValue = currentPopulation.Individuals[indexes[0], i]
                                 + mutationForce * (currentPopulation.Individuals[indexes[1], i]
                                                    - currentPopulation.Individuals[indexes[2], i]);
                if (trialValue >= lowerBound[i] && trialValue <= upperBound[i])
                    trialPopulation.Individuals[indexOfCurrentIndividual, i] = trialValue;
                else
                    trialPopulation.Individuals[indexOfCurrentIndividual, i] =
                        random.NextDouble(indexOfCurrentIndividual, pageOfRandom, randomStep++)
                        * (upperBound[i] - lowerBound[i]) + lowerBound[i];
            }
            else
            {
                trialPopulation.Individuals[indexOfCurrentIndividual, i] =
                    currentPopulation.Individuals[indexOfCurrentIndividual, i];
            }
    }

    public int GetMaxRandomNumbersPerIndividual(int individualVectorSize)
    {
        const int maxRandomNumbersForGene = 2;

        return NumberOfIndividualsToChoose + maxRandomNumbersForGene * individualVectorSize;
    }
}
