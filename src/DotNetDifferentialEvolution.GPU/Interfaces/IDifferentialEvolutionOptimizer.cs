namespace DotNetDifferentialEvolution.GPU.Interfaces;

public interface IDifferentialEvolutionOptimizer<T>
{
    public Task<T> RunAsync();
    
    public Task<T> RunAsync(CancellationToken cancellationToken);
}