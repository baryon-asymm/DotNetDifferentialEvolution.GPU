using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.MutationStrategies;
using DotNetDifferentialEvolution.GPU.Test.Helpers;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using Xunit.Abstractions;

namespace DotNetDifferentialEvolution.GPU.Test;

public class MutationStrategyTests
{
    private readonly ITestOutputHelper _output;

    private const int IndividualVectorSize = 300;
    private const int PopulationSize = 100_000;

    private readonly Context _context;
    private readonly Accelerator _device;

    private readonly Population _currentPopulation;
    private readonly Population _trialPopulation;

    private readonly MutationStrategy _mutationStrategy;

    public MutationStrategyTests(ITestOutputHelper output)
    {
        _output = output;

        _context = Context.Create(builder => builder.EnableAlgorithms().Cuda());
        _device = _context.GetPreferredDevice(preferCPU: false).CreateAccelerator(_context);

        _currentPopulation = PopulationHelper.GetRandomPopulation(_device, PopulationSize, IndividualVectorSize);
        _trialPopulation = PopulationHelper.GetPopulation(_device, PopulationSize, IndividualVectorSize);

        const double lowerValue = 1;
        var lowerBound = Enumerable.Repeat(lowerValue, IndividualVectorSize);
        var deviceLowerBound = _device.Allocate1D(lowerBound.ToArray());

        const double upperValue = 1000;
        var upperBound = Enumerable.Repeat(upperValue, IndividualVectorSize);
        var deviceUpperBound = _device.Allocate1D(lowerBound.ToArray());

        _mutationStrategy = new MutationStrategy(deviceLowerBound.View, deviceUpperBound.View);
    }

    public static void Kernel(
        Index1D index,
        Population currentPopulation,
        Population trialPopulation,
        DeviceRandom random,
        MutationStrategy mutationStrategy)
    {
        const int pageOfRandom = 0;
        mutationStrategy.Mutate(index, currentPopulation, trialPopulation, pageOfRandom, random);
    }

    [Fact]
    public void TestDefaultCase()
    {
        _output.WriteLine($"Device name: {_device.Name}");

        var random = Random.Shared;
        var randomNumbers = new double[PopulationSize,
            _mutationStrategy.GetMaxRandomNumbersPerIndividual(IndividualVectorSize)];
        const int individualsDimension = 0;
        const int vectorsDimension = 1;
        for (var i = 0; i < randomNumbers.GetLength(individualsDimension); i++)
        {
            for (var j = 0; j < randomNumbers.GetLength(vectorsDimension); j++)
            {
                randomNumbers[i, j] = random.NextDouble();
            }
        }

        var deviceRandomNumbers = _device.Allocate2DDenseX(randomNumbers);
        var deviceRandom = new DeviceRandom(
            _mutationStrategy.GetMaxRandomNumbersPerIndividual(IndividualVectorSize),
            deviceRandomNumbers.View);

        var kernel = _device
            .LoadAutoGroupedStreamKernel<
                Index1D,
                Population,
                Population,
                DeviceRandom,
                MutationStrategy>(Kernel);

        kernel(PopulationSize, _currentPopulation, _trialPopulation, deviceRandom, _mutationStrategy);
        _device.Synchronize();

        var mutatedIndividuals = _trialPopulation.Individuals.GetAsArray2D();
        var notExpectedMutatedIndividuals =
            PopulationHelper.GetPopulation(_device, PopulationSize, IndividualVectorSize).Individuals.GetAsArray2D();

        Assert.NotEqual(notExpectedMutatedIndividuals, mutatedIndividuals);
    }
}
