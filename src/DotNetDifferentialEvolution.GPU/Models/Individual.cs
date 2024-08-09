using System.Collections.ObjectModel;

namespace DotNetDifferentialEvolution.GPU.Models;

public class Individual
{
    public double FitnessFunctionValue { get; }
    public ReadOnlyCollection<double> Vector { get; }

    public Individual(
        double fitnessFunctionValue,
        IEnumerable<double> vector)
    {
        FitnessFunctionValue = fitnessFunctionValue;
        Vector = vector.ToArray().AsReadOnly();
    }
}
