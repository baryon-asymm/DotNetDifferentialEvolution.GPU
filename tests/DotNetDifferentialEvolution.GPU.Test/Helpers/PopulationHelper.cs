using DotNetDifferentialEvolution.GPU.Models;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.Test.Helpers;

public static class PopulationHelper
{
    public static Population GetRandomPopulation(
        Accelerator device,
        int populationSize,
        int individualVectorSize,
        double maxFitnessFunctionValue = 100,
        double maxIndividualVectorValue = 1000)
    {
        var random = Random.Shared;

        var individuals = new double[populationSize, individualVectorSize];
        var fitnessFunctionValues = new double[populationSize];
        
        for (var i = 0; i < populationSize; i++)
        {
            for (var j = 0; j < individualVectorSize; j++)
            {
                individuals[i, j] = random.NextDouble() * maxIndividualVectorValue;
            }
            
            fitnessFunctionValues[i] = random.NextDouble() * maxFitnessFunctionValue;
        }

        var deviceIndividuals = device.Allocate2DDenseX(individuals);
        var deviceFitnessFunctionValues = device.Allocate1D(fitnessFunctionValues);
        var population = new Population(deviceFitnessFunctionValues.View, deviceIndividuals.View);

        return population;
    }

    public static Population GetPopulation(
        Accelerator device,
        int populationSize,
        int individualVectorSize,
        double fitnessFunctionValue = 0,
        double individualVectorValue = 0)
    {
        var individuals = new double[populationSize, individualVectorSize];
        var fitnessFunctionValues = new double[populationSize];
        
        for (var i = 0; i < populationSize; i++)
        {
            for (var j = 0; j < individualVectorSize; j++)
            {
                individuals[i, j] = individualVectorValue;
            }
            
            fitnessFunctionValues[i] = fitnessFunctionValue;
        }

        var deviceIndividuals = device.Allocate2DDenseX(individuals);
        var deviceFitnessFunctionValues = device.Allocate1D(fitnessFunctionValues);
        var population = new Population(deviceFitnessFunctionValues.View, deviceIndividuals.View);

        return population;
    }
}