using DotNetDifferentialEvolution.GPU.Models;

namespace DotNetDifferentialEvolution.GPU.Controllers.Kernels.Interfaces;

public interface IKernelController : IDisposable
{
    public void CompileAndGpuMemoryAlloc();

    public void Init();
    
    public void Run();
    
    public void Run(CancellationToken cancellationToken);

    public HostPopulation? GetCurrentPopulationOrNull();
}
