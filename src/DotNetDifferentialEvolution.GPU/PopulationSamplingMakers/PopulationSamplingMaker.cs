using System.Collections.ObjectModel;
using DotNetDifferentialEvolution.GPU.PopulationSamplingMakers.Interfaces;

namespace DotNetDifferentialEvolution.GPU.PopulationSamplingMakers;

public class PopulationSamplingMaker : IPopulationSamplingMaker
{
    private readonly Random _random = Random.Shared;

    private readonly int _populationSize;
    private readonly ReadOnlyCollection<double> _upperBound;
    private readonly ReadOnlyCollection<double> _lowerBound;

    public int GetPopulationSize() => _populationSize;

    public PopulationSamplingMaker(
        int populationSize,
        IEnumerable<double> upperBound,
        IEnumerable<double> lowerBound)
    {
        if (populationSize <= 0)
            throw new ArgumentException("The population size must be greater than zero.");
        
        _populationSize = populationSize;
        _upperBound = upperBound.ToArray().AsReadOnly();
        _lowerBound = lowerBound.ToArray().AsReadOnly();

        if (_upperBound.Count != _lowerBound.Count)
            throw new ArgumentException("The upper bound array must be the size of the lower bound array.");
    }
    
    public double[,] TakeSamples()
    {
        var vectorSize = _upperBound.Count;
        var individualsBuffer = new double[_populationSize, vectorSize];
        
        const int startIndex = 0;
        Parallel.For(startIndex, _populationSize, i =>
        {
            for (var j = 0; j < vectorSize; j++)
            {
                individualsBuffer[i, j] = _lowerBound[j] + _random.NextDouble() * (_upperBound[j] - _lowerBound[j]);
            }
        });

        return individualsBuffer;
    }
}
