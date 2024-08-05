using DotNetDifferentialEvolution.GPU.Controllers;
using DotNetDifferentialEvolution.GPU.Interfaces;
using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.MutationStrategies;
using DotNetDifferentialEvolution.GPU.PopulationSamplingMakers;
using DotNetDifferentialEvolution.GPU.SelectionStrategies;
using DotNetDifferentialEvolution.GPU.TerminationStrategies;
using DotNetDifferentialEvolution.GPU.Test.FitnessFunctions;
using DotNetDifferentialEvolution.GPU.WorkerKernels;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Xunit.Abstractions;

namespace DotNetDifferentialEvolution.GPU.Test;

public class DEOptimizerTests
{
    private readonly ITestOutputHelper _output;

    private const int GenerationsCount = 10000;
    private const int IndividualVectorSize = 2;
    private const int PopulationSize = 1024;

    private const int DeviceRandomPoolSize = 10;
    private const int NumberOfPages = 20000;

    public DEOptimizerTests(ITestOutputHelper output)
    {
        _output = output;
    }

    private DEOptimizer GetOptimizer()
    {
        var context = Context.Create(builder => builder.EnableAlgorithms().Cuda());
        var device = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        const double lowerValue = 1;
        var lowerBound = Enumerable.Repeat(lowerValue, IndividualVectorSize);
        var deviceLowerBound = device.Allocate1D(lowerBound.ToArray());

        const double upperValue = 1000;
        var upperBound = Enumerable.Repeat(upperValue, IndividualVectorSize);
        var deviceUpperBound = device.Allocate1D(upperBound.ToArray());

        var populationSamplingMaker = new PopulationSamplingMaker(PopulationSize, upperBound, lowerBound);

        var rosenbrockFunction = new RosenbrockFunction();
        var mutationStrategy = new MutationStrategy(deviceLowerBound.View, deviceUpperBound.View);
        var selectionStrategy = new SelectionStrategy();

        var deviceRandomController = new DeviceRandomController(
            DeviceRandomPoolSize,
            PopulationSize,
            mutationStrategy.GetMaxRandomNumbersPerIndividual(IndividualVectorSize),
            NumberOfPages,
            device);

        var terminationStrategy = new MaxGenerationsStrategy(GenerationsCount);

        var updateHandler = new UpdateHandler(_output);

        var workerKernel = new WorkerKernel<RosenbrockFunction, MutationStrategy, SelectionStrategy>(
            context,
            device,
            populationSamplingMaker,
            deviceRandomController,
            rosenbrockFunction,
            mutationStrategy,
            selectionStrategy,
            terminationStrategy,
            updateHandler);

        var optimizer = new DEOptimizer(workerKernel);
        
        return optimizer;
    }

    [Fact]
    public async Task TestRosenbrockCase()
    {
        using var optimizer = GetOptimizer();
        var optimizationResult = await optimizer.RunAsync();

        var v = optimizationResult.IndividualVector;
        
        _output.WriteLine($"ResultFFValue is {optimizationResult.FitnessFunctionValue}");
        _output.WriteLine($"ResultVector is [{v[0]} {v[1]}]");

        const double tolerance = 1e-8;
        Assert.Equal(RosenbrockFunction.GetFFValueResult(), optimizationResult.FitnessFunctionValue, tolerance);
        Assert.Equal(RosenbrockFunction.GetVectorResult(), optimizationResult.IndividualVector,
            (l, r) => Math.Abs(l - r) <= tolerance);
    }
    
    private DEOptimizer GetOptimizerWithReuses()
    {
        var context = Context.Create(builder => builder.EnableAlgorithms().Cuda());
        var device = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        const double lowerValue = 1;
        var lowerBound = Enumerable.Repeat(lowerValue, IndividualVectorSize);
        var deviceLowerBound = device.Allocate1D(lowerBound.ToArray());

        const double upperValue = 1000;
        var upperBound = Enumerable.Repeat(upperValue, IndividualVectorSize);
        var deviceUpperBound = device.Allocate1D(upperBound.ToArray());

        var populationSamplingMaker = new PopulationSamplingMaker(PopulationSize, upperBound, lowerBound);

        var rosenbrockFunction = new RosenbrockFunction();
        var mutationStrategy = new MutationStrategy(deviceLowerBound.View, deviceUpperBound.View);
        var selectionStrategy = new SelectionStrategy();

        var deviceRandomController = new DeviceRandomController(
            DeviceRandomPoolSize,
            PopulationSize,
            mutationStrategy.GetMaxRandomNumbersPerIndividual(IndividualVectorSize),
            NumberOfPages,
            device);

        var terminationStrategy = new MaxGenerationsStrategy(GenerationsCount);

        var updateHandler = new UpdateHandler(_output);

        const int numberOfReuses = 1;
        var workerKernel = new ReusingDeviceRandomWorkerKernel<RosenbrockFunction, MutationStrategy, SelectionStrategy>(
            context,
            device,
            numberOfReuses,
            populationSamplingMaker,
            deviceRandomController,
            rosenbrockFunction,
            mutationStrategy,
            selectionStrategy,
            terminationStrategy,
            updateHandler);

        var optimizer = new DEOptimizer(workerKernel);
        
        return optimizer;
    }
    
    [Fact]
    public async Task TestRosenbrockCaseWithReuses()
    {
        using var optimizer = GetOptimizerWithReuses();
        var optimizationResult = await optimizer.RunAsync();

        var v = optimizationResult.IndividualVector;
        
        _output.WriteLine($"ResultFFValue is {optimizationResult.FitnessFunctionValue}");
        _output.WriteLine($"ResultVector is [{v[0]} {v[1]}]");

        const double tolerance = 1e-8;
        Assert.Equal(RosenbrockFunction.GetFFValueResult(), optimizationResult.FitnessFunctionValue, tolerance);
        Assert.Equal(RosenbrockFunction.GetVectorResult(), optimizationResult.IndividualVector,
            (l, r) => Math.Abs(l - r) <= tolerance);
    }
    
    private class UpdateHandler(ITestOutputHelper output) : IOptimizerUpdateHandler
    {
        private const int NumberOfSkippedGenerations = 25000;
        
        public void Handle(OptimizerState state, Accelerator device, int generation, Population population)
        {
            if (state == OptimizerState.Starting)
                output.WriteLine($"Device name: {device.Name}");

            if (state == OptimizerState.Running)
            {
                if (generation % NumberOfSkippedGenerations == 0)
                    output.WriteLine($"Current generation is {generation}");
            }
            
            if (state == OptimizerState.Terminating)
                output.WriteLine($"Terminating, current generation is {generation}");
        }
    }
}
