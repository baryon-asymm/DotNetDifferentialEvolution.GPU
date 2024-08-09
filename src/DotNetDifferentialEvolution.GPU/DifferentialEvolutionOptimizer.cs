using DotNetDifferentialEvolution.GPU.Controllers.Kernels.Interfaces;
using DotNetDifferentialEvolution.GPU.Interfaces;
using DotNetDifferentialEvolution.GPU.Models;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU;

public class DifferentialEvolutionOptimizer : IDifferentialEvolutionOptimizer<OptimizationResult>, IDisposable
{
    private readonly IKernelController _kernelController;
    
    public DifferentialEvolutionOptimizer(IKernelController kernelController)
    {
        _kernelController = kernelController;
        
        _kernelController.CompileAndGpuMemoryAlloc();
    }

    public Task<OptimizationResult> RunAsync() => RunAsync(CancellationToken.None);

    public Task<OptimizationResult> RunAsync(CancellationToken cancellationToken)
    {
        _kernelController.Init();
        
        _kernelController.Run(cancellationToken);

        var bestIndividual = GetBestIndividual(_kernelController);
        var result = new OptimizationResult(bestIndividual.FitnessFunctionValue, bestIndividual.Vector);

        return Task.FromResult(result);
    }

    private static Individual GetBestIndividual(IKernelController kernelController)
    {
        var population = kernelController.GetCurrentPopulationOrNull();
        if (population is null)
            throw new InvalidOperationException("The current population is null.");
        
        var hostFfValues = population.FitnessFunctionValues.GetAsArray1D();
        var hostIndividuals = population.Individuals.GetAsArray2D();

        var bestIndividualIndex = GetBestIndividualIndex(hostFfValues.AsSpan());
        var bestIndividualVector = GetIndividualVector(bestIndividualIndex, hostIndividuals);
        var bestIndividualFfValue = hostFfValues[bestIndividualIndex];
        var bestIndividual = new Individual(bestIndividualFfValue, bestIndividualVector);

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
        _kernelController.Dispose();
        
        GC.Collect();
    }
}
