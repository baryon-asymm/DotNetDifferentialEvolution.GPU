using DotNetDifferentialEvolution.GPU.Interfaces;
using DotNetDifferentialEvolution.GPU.Models;

namespace DotNetDifferentialEvolution.GPU.Test.FitnessFunctions;

public struct RosenbrockFunction : IFitnessFunctionInvoker
{
    private const double A = 1;
    private const double B = 100;
    
    public void Invoke(int individualIndex, Population population)
    {
        var x = population.Individuals[individualIndex, 0];
        var y = population.Individuals[individualIndex, 1];
        
        var result = Math.Pow(A - x, 2) + B * Math.Pow(y - x * x, 2);

        population.FitnessFunctionValues[individualIndex] = result;
    }

    public static double GetFFValueResult() => 0;

    public static IEnumerable<double> GetVectorResult() => [1, 1];
}