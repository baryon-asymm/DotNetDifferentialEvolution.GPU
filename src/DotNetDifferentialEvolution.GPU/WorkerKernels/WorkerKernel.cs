using DotNetDifferentialEvolution.GPU.Controllers;
using DotNetDifferentialEvolution.GPU.Interfaces;
using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.MutationStrategies.Interfaces;
using DotNetDifferentialEvolution.GPU.PopulationSamplingMakers.Interfaces;
using DotNetDifferentialEvolution.GPU.SelectionStrategies.Interfaces;
using DotNetDifferentialEvolution.GPU.TerminationStrategies.Interfaces;
using DotNetDifferentialEvolution.GPU.WorkerKernels.Interfaces;
using ILGPU;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.WorkerKernels;

public class WorkerKernel<TFitnessFunctionInvoker, TMutationStrategy, TSelectionStrategy>(
    Context context,
    Accelerator device,
    IPopulationSamplingMaker populationSamplingMaker,
    DeviceRandomController deviceRandomController,
    TFitnessFunctionInvoker fitnessFunction,
    TMutationStrategy mutationStrategy,
    TSelectionStrategy selectionStrategy,
    ITerminationStrategy terminationStrategy,
    IOptimizerUpdateHandler? updatedPopulationHandler = null) : IWorkerKernel
    where TFitnessFunctionInvoker : struct, IFitnessFunctionInvoker
    where TMutationStrategy : struct, IMutationStrategy
    where TSelectionStrategy : struct, ISelectionStrategy
{
    private Action<Index1D, Population, TFitnessFunctionInvoker> _kernelInit;

    private Action<Index1D, Population, Population, Population, int, DeviceRandom, TMutationStrategy,
        TFitnessFunctionInvoker, TSelectionStrategy> _kernelRun;

    private PopulationHolder _currentPopulationHolder;
    private PopulationHolder _nextPopulationHolder;
    private PopulationHolder _trialPopulationHolder;

    public static void KernelInit(
        Index1D index,
        Population currentPopulation,
        TFitnessFunctionInvoker functionInvoker)
    {
        functionInvoker.Invoke(index, currentPopulation);
    }

    public static void KernelRun(
        Index1D index,
        Population currentPopulation,
        Population nextPopulation,
        Population trialPopulation,
        int pageOfRandom,
        DeviceRandom random,
        TMutationStrategy mutationStrategy,
        TFitnessFunctionInvoker functionInvoker,
        TSelectionStrategy selectionStrategy)
    {
        mutationStrategy.Mutate(index, currentPopulation, trialPopulation, pageOfRandom, random);

        functionInvoker.Invoke(index, trialPopulation);

        selectionStrategy.Select(index, currentPopulation, nextPopulation, trialPopulation);
    }

    public void CompileAndGPUMemoryAlloc()
    {
        if (_kernelInit is not null || _kernelRun is not null)
            throw new InvalidOperationException("The Kernel's invokers must have a null reference.");

        CompileKernels();
        GPUMemoryAlloc();
    }

    private void CompileKernels()
    {
        _kernelInit = device
            .LoadAutoGroupedStreamKernel<
                Index1D,
                Population,
                TFitnessFunctionInvoker>(KernelInit);
        _kernelRun = device
            .LoadAutoGroupedStreamKernel<
                Index1D,
                Population,
                Population,
                Population,
                int,
                DeviceRandom,
                TMutationStrategy,
                TFitnessFunctionInvoker,
                TSelectionStrategy>(KernelRun);
    }

    private void GPUMemoryAlloc()
    {
        StartAndMAllocForDeviceRandoms();
        MAllocForPopulations();
    }

    private void StartAndMAllocForDeviceRandoms() => deviceRandomController.Start();

    private void MAllocForPopulations()
    {
        var hostIndividualsBuffer = populationSamplingMaker.TakeSamples();
        var populationSize = populationSamplingMaker.GetPopulationSize();
        var hostFFValuesBuffer = new double[populationSize];

        _currentPopulationHolder = GetGPUAllocatedPopulationHolder(device, hostFFValuesBuffer, hostIndividualsBuffer);
        _nextPopulationHolder = GetGPUAllocatedPopulationHolder(device, hostFFValuesBuffer, hostIndividualsBuffer);
        _trialPopulationHolder = GetGPUAllocatedPopulationHolder(device, hostFFValuesBuffer, hostIndividualsBuffer);
    }

    private static PopulationHolder GetGPUAllocatedPopulationHolder(
        Accelerator device,
        double[] hostFFValuesBuffer,
        double[,] hostIndividualsBuffer)
    {
        var deviceFFValues = device.Allocate1D(hostFFValuesBuffer);
        var deviceIndividuals = device.Allocate2DDenseX(hostIndividualsBuffer);
        var population = new PopulationHolder(deviceFFValues, deviceIndividuals);

        return population;
    }

    public void Init()
    {
        var populationSize = populationSamplingMaker.GetPopulationSize();
        var devicePopulation = _currentPopulationHolder.GetPopulation();
        _kernelInit(populationSize, devicePopulation, fitnessFunction);
        device.Synchronize();
    }

    public void Run() => Run(CancellationToken.None);

    public void Run(CancellationToken cancellationToken)
    {
        var pageSize = deviceRandomController.PageSize;
        var numberOfPages = deviceRandomController.NumberOfPages;
        var populationSize = populationSamplingMaker.GetPopulationSize();
        var currentHolder = deviceRandomController.GetFirst();

        var deviceRandom = currentHolder.GetDeviceRandom();
        
        var currentPopulation = _currentPopulationHolder.GetPopulation();
        var nextPopulation = _nextPopulationHolder.GetPopulation();
        var trialPopulation = _trialPopulationHolder.GetPopulation();

        var page = 0;
        var generation = 0;

        updatedPopulationHandler?.Handle(OptimizerState.Starting, device, generation, currentPopulation);

        do
        {
            generation++;

            if (page >= numberOfPages)
            {
                currentHolder = deviceRandomController.ToRecycleAndGetNew(currentHolder);
                deviceRandom = new DeviceRandom(pageSize, currentHolder.DeviceRandomBuffer.View);

                page = 0;
            }

            _kernelRun(
                populationSize,
                currentPopulation,
                nextPopulation,
                trialPopulation,
                page,
                deviceRandom,
                mutationStrategy,
                fitnessFunction,
                selectionStrategy);
            device.Synchronize();

            (currentPopulation, nextPopulation) = (nextPopulation, currentPopulation);
            (_currentPopulationHolder, _nextPopulationHolder) = (_nextPopulationHolder, _currentPopulationHolder);

            updatedPopulationHandler?.Handle(OptimizerState.Running, device, generation, currentPopulation);

            page++;
        } while (terminationStrategy.IsMustTerminate(device, generation, currentPopulation) == false
                 && cancellationToken.IsCancellationRequested == false);

        updatedPopulationHandler?.Handle(OptimizerState.Terminating, device, generation, currentPopulation);
    }

    public Population GetCurrentPopulation() => _currentPopulationHolder.GetPopulation();

    public void Dispose()
    {
        deviceRandomController.Dispose();

        FreeGPUMemoryPopulation(_currentPopulationHolder);
        FreeGPUMemoryPopulation(_nextPopulationHolder);
        FreeGPUMemoryPopulation(_trialPopulationHolder);

        device.Dispose();
        context.Dispose();
    }

    private static void FreeGPUMemoryPopulation(PopulationHolder populationHolder)
    {
        populationHolder.FitnessFunctionValues.Dispose();
        populationHolder.Individuals.Dispose();
    }
}
