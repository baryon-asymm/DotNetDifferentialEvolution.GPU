using System.Collections.ObjectModel;

namespace DotNetDifferentialEvolution.GPU.Models;

public record OptimizationResult(double FitnessFunctionValue, IEnumerable<double> individualVector)
{
    public ReadOnlyCollection<double> IndividualVector { get; } = individualVector.ToArray().AsReadOnly();
}
