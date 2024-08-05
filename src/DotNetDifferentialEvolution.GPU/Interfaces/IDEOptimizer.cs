namespace DotNetDifferentialEvolution.GPU.Interfaces;

public interface IDEOptimizer<T>
{
    public Task<T> RunAsync();
    
    public Task<T> RunAsync(CancellationToken cancellationToken);
}