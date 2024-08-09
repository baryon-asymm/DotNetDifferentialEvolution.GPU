namespace DotNetDifferentialEvolution.GPU.Interfaces;

/// <summary>
/// Defines the contract for a differential evolution optimizer.
/// </summary>
/// <typeparam name="T">The type of the solution that the optimizer will return.</typeparam>
public interface IDifferentialEvolutionOptimizer<T>
{
    /// <summary>
    /// Asynchronously runs the differential evolution optimization process.
    /// </summary>
    /// <returns>A task that represents the asynchronous operation. 
    /// The task result contains the best solution found during the optimization process.</returns>
    public Task<T> RunAsync();
    
    /// <summary>
    /// Asynchronously runs the differential evolution optimization process with cancellation support.
    /// </summary>
    /// <param name="cancellationToken">A token that can be used to cancel the optimization process.</param>
    /// <returns>A task that represents the asynchronous operation. 
    /// The task result contains the best solution found during the optimization process.</returns>
    public Task<T> RunAsync(CancellationToken cancellationToken);
}