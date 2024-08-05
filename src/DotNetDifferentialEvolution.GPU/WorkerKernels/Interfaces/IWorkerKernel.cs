using DotNetDifferentialEvolution.GPU.Models;

namespace DotNetDifferentialEvolution.GPU.WorkerKernels.Interfaces;

public interface IWorkerKernel : IDisposable
{
    public void CompileAndGPUMemoryAlloc();

    public void Init();
    
    public void Run();
    
    public void Run(CancellationToken cancellationToken);

    public Population GetCurrentPopulation();
}
