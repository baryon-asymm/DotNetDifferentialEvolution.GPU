using System.Collections.ObjectModel;

namespace DotNetDifferentialEvolution.GPU.Models;

public class OptimizationResult
{
    public double FitnessFunctionValue { get; }
    public ReadOnlyCollection<double> Individual { get; }

    public OptimizationResult(double fitnessFunctionValue, IEnumerable<double> individual)
    {
        FitnessFunctionValue = fitnessFunctionValue;
        Individual = individual.ToArray().AsReadOnly();
    }
}
