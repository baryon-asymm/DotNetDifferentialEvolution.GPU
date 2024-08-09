using System.Runtime.CompilerServices;
using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.MutationStrategies.Interfaces;
using DotNetDifferentialEvolution.GPU.RandomGenerators.Interfaces;
using ILGPU;

namespace DotNetDifferentialEvolution.GPU.MutationStrategies;

public readonly struct MutationStrategy<TRandomGenerator>(
    ArrayView<double> lowerBound,
    ArrayView<double> upperBound,
    double mutationForce = 0.3,
    double crossoverFactor = 0.8) : IMutationStrategy<TRandomGenerator>
    where TRandomGenerator : struct, IRandomGenerator
{
    private const int NumberOfIndividualsToChoose = 3;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Mutate(
        int index,
        DevicePopulation currentPopulation,
        DevicePopulation trialPopulation,
        TRandomGenerator random)
    {
        var indexes = new int[NumberOfIndividualsToChoose];

        for (var i = 0; i < NumberOfIndividualsToChoose; i++)
        {
            var sizeWithoutCurrent = currentPopulation.PopulationSize - 1;
            indexes[i] = random.Next(index) % sizeWithoutCurrent;
            if (indexes[i] >= index) indexes[i]++;

            for (var j = 0; j < i; j++)
                if (indexes[i] == indexes[j])
                {
                    i--;
                    break;
                }
        }

        var vectorSize = trialPopulation.VectorSize;
        for (var i = 0; i < vectorSize; i++)
            if (random.NextDouble(index) <= crossoverFactor)
            {
                var trialValue = currentPopulation.Individuals[indexes[0], i]
                                 + mutationForce * (currentPopulation.Individuals[indexes[1], i]
                                                    - currentPopulation.Individuals[indexes[2], i]);
                if (trialValue >= lowerBound[i] && trialValue <= upperBound[i])
                    trialPopulation.Individuals[index, i] = trialValue;
                else
                    trialPopulation.Individuals[index, i] =
                        random.NextDouble(index)
                        * (upperBound[i] - lowerBound[i]) + lowerBound[i];
            }
            else
            {
                trialPopulation.Individuals[index, i] =
                    currentPopulation.Individuals[index, i];
            }
    }
}
