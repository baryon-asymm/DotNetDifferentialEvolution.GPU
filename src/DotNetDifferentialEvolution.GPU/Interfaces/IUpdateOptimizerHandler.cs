using DotNetDifferentialEvolution.GPU.Models;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.Interfaces;

/// <summary>
/// Represents the different states of the optimizer during its lifecycle.
/// </summary>
public enum OptimizerState : byte
{
    /// <summary>
    /// Indicates that the optimizer is starting.
    /// </summary>
    Starting = 0,
    
    /// <summary>
    /// Indicates that the optimizer is currently running.
    /// </summary>
    Running,
    
    /// <summary>
    /// Indicates that the optimizer is terminating.
    /// </summary>
    Terminating
}

/// <summary>
/// Defines the contract for handling updates during the optimization process.
/// </summary>
public interface IUpdateOptimizerHandler
{
    /// <summary>
    /// Handles an update of the optimizer's state, including device and population information.
    /// </summary>
    /// <param name="state">The current state of the optimizer.</param>
    /// <param name="device">The GPU accelerator used for optimization.</param>
    /// <param name="generation">The current generation of the population in the optimization process.</param>
    /// <param name="population">The population data residing on the host.</param>
    public void Handle(OptimizerState state, Accelerator device, int generation, HostPopulation population);
}
