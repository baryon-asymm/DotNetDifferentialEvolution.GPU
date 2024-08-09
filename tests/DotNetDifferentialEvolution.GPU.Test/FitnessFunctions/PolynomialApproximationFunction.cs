using System.Runtime.CompilerServices;
using DotNetDifferentialEvolution.GPU.Interfaces;
using DotNetDifferentialEvolution.GPU.Models;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.Test.FitnessFunctions;

public readonly struct PolynomialApproximationFunction : IFitnessFunctionInvoker
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Invoke(int individualIndex, DevicePopulation devicePopulation)
    {
        double[] functionValues = [0.264, 0.228, 0.194, 0.176, 0.162, 0.15, 0.14, 0.134, 0.13, 0.122, 0.12, 0.114];
        double[] argValues = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5];

        var result = 0.0;
        for (var i = 0; i < functionValues.Length; i++)
        {
            result += XMath.Pow(
                functionValues[i] - GetFunctionValue(individualIndex, devicePopulation.Individuals, argValues[i]), 2);
        }

        devicePopulation.FitnessFunctionValues[individualIndex] = result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double GetFunctionValue(int individualIndex, ArrayView2D<double, Stride2D.DenseX> individuals,
        double argValue)
    {
        var length = individuals.Extent.Y;
        var result = 0.0;
        for (var i = 0; i < length; i++)
        {
            result += individuals[individualIndex, i] * XMath.Pow(argValue, i);
        }

        return result;
    }

    public static double GetFfValueResult() => 2.3295763060466132E-05;

    public static IEnumerable<double> GetIndividualResult() =>
    [
        0.38718881229629304,
        -0.1599304697255068,
        0.043238654955395556,
        -0.0063164682846944525,
        0.0004719594740013259,
        -1.4479641152019519E-05
    ];

    public static int IndividualSize => 6;
}
