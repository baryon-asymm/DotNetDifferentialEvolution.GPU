using System.Collections.ObjectModel;

namespace DotNetDifferentialEvolution.GPU.Models;

public record Individual(
    double FitnessFunctionValue,
    IEnumerable<double> vector)
{
    public ReadOnlyCollection<double> Vector { get; } = vector.ToArray().AsReadOnly();
}
