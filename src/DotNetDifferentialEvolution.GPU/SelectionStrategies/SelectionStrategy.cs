using System.Runtime.CompilerServices;
using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.SelectionStrategies.Interfaces;

namespace DotNetDifferentialEvolution.GPU.SelectionStrategies;

public readonly struct SelectionStrategy : ISelectionStrategy
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Select(
        int index,
        DevicePopulation currentPopulation,
        DevicePopulation nextPopulation,
        DevicePopulation trialPopulation)
    {
        var vectorSize = nextPopulation.VectorSize;
        
        if (trialPopulation.FitnessFunctionValues[index] < currentPopulation.FitnessFunctionValues[index])
        {
            for (var i = 0; i < vectorSize; i++)
                nextPopulation.Individuals[index, i] = trialPopulation.Individuals[index, i];
            
            nextPopulation.FitnessFunctionValues[index] = trialPopulation.FitnessFunctionValues[index];
        }
        else
        {
            for (var i = 0; i < vectorSize; i++)
                nextPopulation.Individuals[index, i] = currentPopulation.Individuals[index, i];
            
            nextPopulation.FitnessFunctionValues[index] = currentPopulation.FitnessFunctionValues[index];
        }
    }
}