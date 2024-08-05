using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.Test.Helpers;
using ILGPU;
using ILGPU.Runtime;
using Xunit.Abstractions;

namespace DotNetDifferentialEvolution.GPU.Test;

public class PopulationTests
{
    private readonly ITestOutputHelper _output;

    private const int IndividualVectorSize = 300;
    private const int PopulationSize = 100_000;

    private readonly Context _context;
    private readonly Accelerator _device;

    private readonly Population _population;

    private readonly double[] _expectedResult;

    public PopulationTests(ITestOutputHelper output)
    {
        _output = output;

        _context = Context.CreateDefault();
        _device = _context.GetPreferredDevice(preferCPU: false).CreateAccelerator(_context);

        _population = PopulationHelper.GetRandomPopulation(_device, PopulationSize, IndividualVectorSize);

        _expectedResult = GetExpectedResultArray();
    }

    private double[] GetExpectedResultArray()
    {
        var result = new double[PopulationSize];
        var hostIndividuals = _population.Individuals.GetAsArray2D();

        for (var i = 0; i < PopulationSize; i++)
        {
            var sum = 0.0;
            for (var j = 0; j < IndividualVectorSize; j++)
            {
                sum += hostIndividuals[i, j];
            }

            result[i] = sum;
        }

        return result;
    }

    public static void Kernel(
        Index1D index,
        Population population,
        ArrayView<double> result)
    {
        var extent = population.Individuals.Extent;

        var sum = 0.0;
        for (var i = 0; i < extent.Y; i++)
        {
            var index2D = new Index2D(index, i);
            sum += population.Individuals[index2D];
        }

        result[index] = sum;
    }

    [Fact]
    public void TestSumCase()
    {
        _output.WriteLine($"Device name: {_device.Name}");

        var deviceResult = _device.Allocate1D<double>(PopulationSize);

        var kernel = _device.LoadAutoGroupedStreamKernel<Index1D, Population, ArrayView<double>>(Kernel);

        kernel(PopulationSize, _population, deviceResult.View);
        _device.Synchronize();

        var hostResult = deviceResult.GetAsArray1D();

        Assert.Equal(_expectedResult, hostResult);
    }
}