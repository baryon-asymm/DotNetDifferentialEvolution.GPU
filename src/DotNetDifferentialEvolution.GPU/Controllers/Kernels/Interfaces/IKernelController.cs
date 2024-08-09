using DotNetDifferentialEvolution.GPU.Models;

namespace DotNetDifferentialEvolution.GPU.Controllers.Kernels.Interfaces;

/// <summary>
/// Defines the contract for controlling the execution of GPU-based differential evolution algorithms.
/// </summary>
/// <remarks>
/// Implementations of this interface manage the compilation of GPU kernels, allocation of GPU memory,
/// and execution of optimization routines.
/// </remarks>
public interface IKernelController : IDisposable
{
    /// <summary>
    /// Compiles the necessary GPU kernels and allocates memory on the GPU for the population data.
    /// </summary>
    /// <remarks>
    /// This method must be called before any operations that involve GPU computation can be performed.
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown if the kernels have already been compiled or the GPU memory is already allocated.
    /// </exception>
    public void CompileAndGpuMemoryAlloc();

    /// <summary>
    /// Initializes the population data on the GPU by invoking the fitness function on each individual.
    /// </summary>
    /// <remarks>
    /// This method must be called after <see cref="CompileAndGpuMemoryAlloc"/> and before <see cref="Run"/>.
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown if the population data or the kernel initialization method is not properly set up.
    /// </exception>
    public void Init();
    
    /// <summary>
    /// Runs the differential evolution optimization process on the GPU.
    /// </summary>
    /// <remarks>
    /// This method executes the optimization without cancellation support.
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown if the necessary GPU memory has not been allocated or if the kernels have not been compiled.
    /// </exception>
    public void Run();
    
    /// <summary>
    /// Runs the differential evolution optimization process on the GPU with cancellation support.
    /// </summary>
    /// <param name="cancellationToken">A token that can be used to cancel the optimization process.</param>
    /// <remarks>
    /// This method allows the optimization process to be canceled by the provided token.
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown if the necessary GPU memory has not been allocated or if the kernels have not been compiled.
    /// </exception>
    public void Run(CancellationToken cancellationToken);

    /// <summary>
    /// Retrieves the current population from the GPU memory or returns null if the population is not initialized.
    /// </summary>
    /// <returns>The current population in host memory or null if the population is not initialized.</returns>
    public HostPopulation? GetCurrentPopulationOrNull();
}
