using System.Text.Json;
using DotNetDifferentialEvolution.GPU.Controllers.Kernels;
using DotNetDifferentialEvolution.GPU.Interfaces;
using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.MutationStrategies;
using DotNetDifferentialEvolution.GPU.PopulationSamplingMakers;
using DotNetDifferentialEvolution.GPU.RandomGenerators;
using DotNetDifferentialEvolution.GPU.SelectionStrategies;
using DotNetDifferentialEvolution.GPU.TerminationStrategies;
using DotNetDifferentialEvolution.GPU.Test.FitnessFunctions;
using ILGPU;
using ILGPU.Algorithms.Random;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using Xunit.Abstractions;

namespace DotNetDifferentialEvolution.GPU.Test;

public class DifferentialEvolutionOptimizerTests
{
    private readonly ITestOutputHelper _output;

    private const int MaxGenerationCount = 1000;
    private const int PopulationSize = 10000;

    public DifferentialEvolutionOptimizerTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private DifferentialEvolutionOptimizer GetOptimizer<TFitnessFunctionInvoker>(
        double lowerValue,
        double upperValue,
        int individualSize,
        TFitnessFunctionInvoker fitnessFunction)
        where TFitnessFunctionInvoker : struct, IFitnessFunctionInvoker
    {
        var context = Context.Create(builder =>
        {
            builder.OpenCL();
            builder.EnableAlgorithms();
        });
        var device = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        var lowerBound = Enumerable.Repeat(lowerValue, individualSize).ToArray();
        var deviceLowerBound = device.Allocate1D(lowerBound);

        var upperBound = Enumerable.Repeat(upperValue, individualSize).ToArray();
        var deviceUpperBound = device.Allocate1D(upperBound);

        var populationSamplingMaker = new PopulationSamplingMaker(PopulationSize, upperBound, lowerBound);

        var random = Random.Shared;
        var xorShifts = new XorShift32[PopulationSize];
        for (var i = 0; i < xorShifts.Length; i++)
            xorShifts[i] = new XorShift32((uint)random.Next());
        var deviceXorShifts = device.Allocate1D(xorShifts);
        var randomGenerator = new RandomGenerator(deviceXorShifts.View);

        var mutationStrategy = new MutationStrategy<RandomGenerator>(deviceLowerBound.View, deviceUpperBound.View);
        var selectionStrategy = new SelectionStrategy();

        var terminationStrategy = new MaxGenerationStrategy(MaxGenerationCount);

        var workerKernel =
            new KernelController<TFitnessFunctionInvoker, RandomGenerator, MutationStrategy<RandomGenerator>,
                SelectionStrategy>(
                context,
                device,
                populationSamplingMaker,
                fitnessFunction,
                randomGenerator,
                mutationStrategy,
                selectionStrategy,
                terminationStrategy);

        var optimizer = new DifferentialEvolutionOptimizer(workerKernel);

        return optimizer;
    }

    [Fact]
    public async Task TestRosenbrockCase()
    {
        using var optimizer = GetOptimizer(
            lowerValue: -2000,
            upperValue: 2000,
            individualSize: RosenbrockFunction.IndividualSize,
            new RosenbrockFunction());
        var optimizationResult = await optimizer.RunAsync();

        _output.WriteLine($"ResultFFValue is {optimizationResult.FitnessFunctionValue}");
        _output.WriteLine($"ResultVector is {JsonSerializer.Serialize(optimizationResult.Individual)}");

        const double tolerance = 1e-6;
        Assert.Equal(RosenbrockFunction.GetFfValueResult(), optimizationResult.FitnessFunctionValue, tolerance);
        Assert.Equal(RosenbrockFunction.GetIndividualResult(), optimizationResult.Individual,
            (l, r) => Math.Abs(l - r) <= tolerance);
    }

    [Fact]
    public async Task TestPolynomialApproximationFunctionCase()
    {
        using var optimizer = GetOptimizer(
            lowerValue: -2000,
            upperValue: 2000,
            individualSize: PolynomialApproximationFunction.IndividualSize,
            new PolynomialApproximationFunction());
        var optimizationResult = await optimizer.RunAsync();

        _output.WriteLine($"ResultFFValue is {optimizationResult.FitnessFunctionValue}");
        _output.WriteLine($"ResultVector is {JsonSerializer.Serialize(optimizationResult.Individual)}");

        const double tolerance = 1e-8;
        Assert.Equal(
            PolynomialApproximationFunction.GetFfValueResult(),
            optimizationResult.FitnessFunctionValue,
            tolerance);
        Assert.Equal(
            PolynomialApproximationFunction.GetIndividualResult(),
            optimizationResult.Individual,
            (l, r) => Math.Abs(l - r) <= tolerance);
    }
}
