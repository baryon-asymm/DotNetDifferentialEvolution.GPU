using DotNetDifferentialEvolution.GPU.Models;

namespace DotNetDifferentialEvolution.GPU.Interfaces;

public interface IFitnessFunctionInvoker
{
    public void Invoke(int individualIndex, Population population);
}
