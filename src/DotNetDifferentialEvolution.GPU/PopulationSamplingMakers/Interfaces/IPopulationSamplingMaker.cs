namespace DotNetDifferentialEvolution.GPU.PopulationSamplingMakers.Interfaces;

public interface IPopulationSamplingMaker
{
    public double[,] TakeSamples();
    public int GetPopulationSize();
}
