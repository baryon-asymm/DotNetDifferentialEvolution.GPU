namespace DotNetDifferentialEvolution.GPU.PopulationSamplingMakers.Interfaces;

public interface IPopulationSamplingMaker
{
    public int GetPopulationSize();
    public double[,] TakeSamples();
}
