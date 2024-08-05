using DotNetDifferentialEvolution.GPU.Interfaces;
using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.WorkerKernels.Interfaces;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU;

public class DEOptimizer : IDEOptimizer<OptimizationResult>, IDisposable
{
    private readonly IWorkerKernel _workerKernel;
    
    public DEOptimizer(IWorkerKernel workerKernel)
    {
        _workerKernel = workerKernel;
        
        _workerKernel.CompileAndGPUMemoryAlloc();
    }

    public Task<OptimizationResult> RunAsync() => RunAsync(CancellationToken.None);

    public Task<OptimizationResult> RunAsync(CancellationToken cancellationToken)
    {
        _workerKernel.Init();
        
        _workerKernel.Run(cancellationToken);

        var bestIndividual = GetBestIndividual(_workerKernel);
        var result = new OptimizationResult(bestIndividual.FitnessFunctionValue, bestIndividual.Vector);

        return Task.FromResult(result);
    }

    private static Individual GetBestIndividual(IWorkerKernel workerKernel)
    {
        var population = workerKernel.GetCurrentPopulation();
        var hostFFValues = population.FitnessFunctionValues.GetAsArray();
        var hostIndividuals = population.Individuals.GetAsArray2D();

        var bestIndividualIndex = GetBestIndividualIndex(hostFFValues.AsSpan());
        var bestIndividualVector = GetIndividualVector(bestIndividualIndex, hostIndividuals);
        var bestIndividualFFValue = hostFFValues[bestIndividualIndex];
        var bestIndividual = new Individual(bestIndividualFFValue, bestIndividualVector);

        return bestIndividual;
    }

    private static int GetBestIndividualIndex(Span<double> fitnessFunctionValues)
    {
        var bestIndividualIndex = 0;
        for (var i = 0; i < fitnessFunctionValues.Length; i++)
        {
            if (fitnessFunctionValues[i] < fitnessFunctionValues[bestIndividualIndex])
                bestIndividualIndex = i;
        }

        return bestIndividualIndex;
    }

    private static IEnumerable<double> GetIndividualVector(int index, double[,] individuals)
    {
        const int vectorDimension = 1;
        var vectorSize = individuals.GetLength(vectorDimension);
        var individualVector = new double[vectorSize];

        for (var i = 0; i < vectorSize; i++)
            individualVector[i] = individuals[index, i];

        return individualVector;
    }

    public void Dispose()
    {
        _workerKernel.Dispose();
        
        GC.Collect();
    }
}
