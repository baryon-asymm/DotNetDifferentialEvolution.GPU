using System.Runtime.CompilerServices;
using DotNetDifferentialEvolution.GPU.Interfaces;
using DotNetDifferentialEvolution.GPU.Models;

namespace DotNetDifferentialEvolution.GPU.Test.FitnessFunctions;

public struct RosenbrockFunction : IFitnessFunctionInvoker
{
    private const double A = 1;
    private const double B = 100;
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Invoke(int individualIndex, DevicePopulation devicePopulation)
    {
        var x = devicePopulation.Individuals[individualIndex, 0];
        var y = devicePopulation.Individuals[individualIndex, 1];
        
        var result = Math.Pow(A - x, 2) + B * Math.Pow(y - x * x, 2);

        devicePopulation.FitnessFunctionValues[individualIndex] = result;
    }

    public static double GetFfValueResult() => 0;

    public static IEnumerable<double> GetIndividualResult() => [1, 1];

    public static int IndividualSize => 2;
}
