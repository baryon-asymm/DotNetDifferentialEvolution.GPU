using DotNetDifferentialEvolution.GPU.Models;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.Interfaces;

public enum OptimizerState : byte
{
    Starting = 0,
    Running,
    Terminating
}

public interface IUpdateOptimizerHandler
{
    public void Handle(OptimizerState state, Accelerator device, int generation, HostPopulation population);
}
