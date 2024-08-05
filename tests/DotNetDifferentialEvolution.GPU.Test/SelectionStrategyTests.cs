using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.SelectionStrategies;
using DotNetDifferentialEvolution.GPU.Test.Helpers;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Xunit.Abstractions;

namespace DotNetDifferentialEvolution.GPU.Test;

public class SelectionStrategyTests
{
    private readonly ITestOutputHelper _output;

    private const int IndividualVectorSize = 300;
    private const int PopulationSize = 100_000;

    private readonly Context _context;
    private readonly Accelerator _device;

    private readonly Population _currentPopulation;
    private readonly Population _nextPopulation;
    private readonly Population _trialPopulation;
    private readonly Population _expectedResultPopulation;

    private readonly SelectionStrategy _selectionStrategy;

    public SelectionStrategyTests(ITestOutputHelper output)
    {
        _output = output;

        _context = Context.Create(builder => builder.EnableAlgorithms().Cuda());
        _device = _context.GetPreferredDevice(preferCPU: false).CreateAccelerator(_context);

        _currentPopulation = PopulationHelper.GetRandomPopulation(_device, PopulationSize, IndividualVectorSize);
        _nextPopulation = PopulationHelper.GetRandomPopulation(_device, PopulationSize, IndividualVectorSize);
        _trialPopulation = PopulationHelper.GetRandomPopulation(_device, PopulationSize, IndividualVectorSize);

        _expectedResultPopulation = GetExpectedResultPopulation();

        _selectionStrategy = new SelectionStrategy();
    }

    private Population GetExpectedResultPopulation()
    {
        var expectedResultPopulation =
            PopulationHelper.GetRandomPopulation(_device, PopulationSize, IndividualVectorSize);

        var hostExpectedIndividuals = expectedResultPopulation.Individuals.GetAsArray2D();
        var hostExpectedFFValues = expectedResultPopulation.FitnessFunctionValues.GetAsArray();

        var hostCurrentIndividuals = _currentPopulation.Individuals.GetAsArray2D();
        var hostCurrentFFValues = _currentPopulation.FitnessFunctionValues.GetAsArray();

        var hostTrialIndividuals = _trialPopulation.Individuals.GetAsArray2D();
        var hostTrialFFValues = _trialPopulation.FitnessFunctionValues.GetAsArray();

        for (var i = 0; i < PopulationSize; i++)
        {
            if (hostCurrentFFValues[i] < hostTrialFFValues[i])
            {
                CopyIndividual(i, hostCurrentFFValues, hostExpectedFFValues, hostCurrentIndividuals, hostExpectedIndividuals);
            }
            else
            {
                CopyIndividual(i, hostTrialFFValues, hostExpectedFFValues, hostTrialIndividuals, hostExpectedIndividuals);
            }
        }
        
        expectedResultPopulation.Individuals.CopyFromCPU(hostExpectedIndividuals);
        expectedResultPopulation.FitnessFunctionValues.CopyFromCPU(hostExpectedFFValues);

        return expectedResultPopulation;
    }

    private static void CopyIndividual(
        int individualIndex,
        double[] ffValuesFrom,
        double[] ffValuesTo,
        double[,] individualsFrom,
        double[,] individualsTo)
    {
        for (var j = 0; j < IndividualVectorSize; j++)
        {
            individualsTo[individualIndex, j] = individualsFrom[individualIndex, j];
        }

        ffValuesTo[individualIndex] = ffValuesFrom[individualIndex];
    }

    public static void Kernel(
        Index1D index,
        Population currentPopulation,
        Population nextPopulation,
        Population trialPopulation,
        SelectionStrategy selectionStrategy)
    {
        selectionStrategy.Select(
            index,
            currentPopulation,
            nextPopulation,
            trialPopulation);
    }

    [Fact]
    public void TestDefaultCase()
    {
        _output.WriteLine($"Device name: {_device.Name}");

        var kernel = _device
            .LoadAutoGroupedStreamKernel<
                Index1D,
                Population,
                Population,
                Population,
                SelectionStrategy>(Kernel);

        kernel(PopulationSize, _currentPopulation, _nextPopulation, _trialPopulation, _selectionStrategy);
        _device.Synchronize();

        var resultFitnessFunctionValues = _nextPopulation.FitnessFunctionValues.GetAsArray();
        var resultIndividuals = _nextPopulation.Individuals.GetAsArray2D();

        var expectedFitnessFunctionValues = _expectedResultPopulation.FitnessFunctionValues.GetAsArray();
        var expectedIndividuals = _expectedResultPopulation.Individuals.GetAsArray2D();

        Assert.Equal(expectedFitnessFunctionValues, resultFitnessFunctionValues);
        Assert.Equal(expectedIndividuals, resultIndividuals);
    }
}