using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.SelectionStrategies.Interfaces;

namespace DotNetDifferentialEvolution.GPU.SelectionStrategies;

public readonly struct SelectionStrategy : ISelectionStrategy
{
    public void Select(
        int individualIndex,
        Population currentPopulation,
        Population nextPopulation,
        Population trialPopulation)
    {
        var vectorLength = nextPopulation.IndividualVectorLength;
        
        if (trialPopulation.FitnessFunctionValues[individualIndex]
            < currentPopulation.FitnessFunctionValues[individualIndex])
        {
            for (var i = 0; i < vectorLength; i++)
                nextPopulation.Individuals[individualIndex, i] =
                    trialPopulation.Individuals[individualIndex, i];
            
            nextPopulation.FitnessFunctionValues[individualIndex] =
                trialPopulation.FitnessFunctionValues[individualIndex];
        }
        else
        {
            for (var i = 0; i < vectorLength; i++)
                nextPopulation.Individuals[individualIndex, i] =
                    currentPopulation.Individuals[individualIndex, i];
            
            nextPopulation.FitnessFunctionValues[individualIndex] =
                currentPopulation.FitnessFunctionValues[individualIndex];
        }
    }
}