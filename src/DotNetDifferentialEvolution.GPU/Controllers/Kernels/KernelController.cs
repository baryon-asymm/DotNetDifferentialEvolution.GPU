using DotNetDifferentialEvolution.GPU.Controllers.Kernels.Interfaces;
using DotNetDifferentialEvolution.GPU.Interfaces;
using DotNetDifferentialEvolution.GPU.Models;
using DotNetDifferentialEvolution.GPU.MutationStrategies.Interfaces;
using DotNetDifferentialEvolution.GPU.PopulationSamplingMakers.Interfaces;
using DotNetDifferentialEvolution.GPU.RandomGenerators.Interfaces;
using DotNetDifferentialEvolution.GPU.SelectionStrategies.Interfaces;
using DotNetDifferentialEvolution.GPU.TerminationStrategies.Interfaces;
using ILGPU;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.Controllers.Kernels;

public class KernelController<TFitnessFunctionInvoker, TRandomGenerator, TMutationStrategy, TSelectionStrategy>(
    Context context,
    Accelerator device,
    IPopulationSamplingMaker populationSamplingMaker,
    TFitnessFunctionInvoker fitnessFunction,
    TRandomGenerator randomGenerator,
    TMutationStrategy mutationStrategy,
    TSelectionStrategy selectionStrategy,
    ITerminationStrategy terminationStrategy,
    IUpdateOptimizerHandler? updatedPopulationHandler = null) : IKernelController
    where TFitnessFunctionInvoker : struct, IFitnessFunctionInvoker
    where TRandomGenerator : struct, IRandomGenerator
    where TMutationStrategy : struct, IMutationStrategy<TRandomGenerator>
    where TSelectionStrategy : struct, ISelectionStrategy
{
    private Action<Index1D, DevicePopulation, TFitnessFunctionInvoker>? _kernelInit;

    private Action<Index1D, DevicePopulation, DevicePopulation, DevicePopulation, TFitnessFunctionInvoker,
        TRandomGenerator, TMutationStrategy, TSelectionStrategy>? _kernelRun;

    private HostPopulation? _currentPopulation;
    private HostPopulation? _nextPopulation;
    private HostPopulation? _trialPopulation;

    private static void KernelInit(
        Index1D index,
        DevicePopulation currentPopulation,
        TFitnessFunctionInvoker fitnessFunction)
    {
        fitnessFunction.Invoke(index, currentPopulation);
    }

    private static void KernelRun(
        Index1D index,
        DevicePopulation currentPopulation,
        DevicePopulation nextPopulation,
        DevicePopulation trialPopulation,
        TFitnessFunctionInvoker fitnessFunction,
        TRandomGenerator random,
        TMutationStrategy mutationStrategy,
        TSelectionStrategy selectionStrategy)
    {
        mutationStrategy.Mutate(index, currentPopulation, trialPopulation, random);

        fitnessFunction.Invoke(index, trialPopulation);

        selectionStrategy.Select(index, currentPopulation, nextPopulation, trialPopulation);
    }

    public void CompileAndGpuMemoryAlloc()
    {
        if (_kernelInit is not null || _kernelRun is not null)
            throw new InvalidOperationException("The Kernel's invokers must have a null reference.");

        CompileKernels();
        GpuMemoryAlloc();
    }

    private void CompileKernels()
    {
        _kernelInit = device
            .LoadAutoGroupedStreamKernel<
                Index1D,
                DevicePopulation,
                TFitnessFunctionInvoker>(KernelInit);
        _kernelRun = device
            .LoadAutoGroupedStreamKernel<
                Index1D,
                DevicePopulation,
                DevicePopulation,
                DevicePopulation,
                TFitnessFunctionInvoker,
                TRandomGenerator,
                TMutationStrategy,
                TSelectionStrategy>(KernelRun);
    }

    private void GpuMemoryAlloc()
    {
        MAllocForPopulations();
    }

    private void MAllocForPopulations()
    {
        var individualsBuffer = populationSamplingMaker.TakeSamples();
        var populationSize = populationSamplingMaker.GetPopulationSize();
        var ffValuesBuffer = new double[populationSize];

        _currentPopulation = GetGpuAllocatedPopulationHolder(device, ffValuesBuffer, individualsBuffer);
        _nextPopulation = GetGpuAllocatedPopulationHolder(device, ffValuesBuffer, individualsBuffer);
        _trialPopulation = GetGpuAllocatedPopulationHolder(device, ffValuesBuffer, individualsBuffer);
    }

    private static HostPopulation GetGpuAllocatedPopulationHolder(
        Accelerator device,
        double[] ffValuesBuffer,
        double[,] individualsBuffer)
    {
        var deviceFfValues = device.Allocate1D(ffValuesBuffer);
        var deviceIndividuals = device.Allocate2DDenseX(individualsBuffer);
        var population = new HostPopulation(deviceFfValues, deviceIndividuals);

        return population;
    }

    public void Init()
    {
        if (_currentPopulation is null)
            throw new InvalidOperationException("The current population is null.");

        if (_kernelInit is null)
            throw new InvalidOperationException("The KernelInit is not compiled.");

        var populationSize = populationSamplingMaker.GetPopulationSize();
        var devicePopulation = _currentPopulation.GetDevicePopulation();

        _kernelInit(populationSize, devicePopulation, fitnessFunction);
        device.Synchronize();
    }

    public void Run() => Run(CancellationToken.None);

    public void Run(CancellationToken cancellationToken)
    {
        if (_currentPopulation is null)
            throw new InvalidOperationException("The current population is null.");

        if (_nextPopulation is null)
            throw new InvalidOperationException("The next population is null.");

        if (_trialPopulation is null)
            throw new InvalidOperationException("The trial population is null.");

        if (_kernelRun is null)
            throw new InvalidOperationException("The KernelRun is not compiled.");

        var populationSize = populationSamplingMaker.GetPopulationSize();

        var generation = 0;

        updatedPopulationHandler?.Handle(OptimizerState.Starting, device, generation, _currentPopulation);

        do
        {
            generation++;

            var currentPopulation = _currentPopulation.GetDevicePopulation();
            var nextPopulation = _nextPopulation.GetDevicePopulation();
            var trialPopulation = _trialPopulation.GetDevicePopulation();

            _kernelRun(
                populationSize,
                currentPopulation,
                nextPopulation,
                trialPopulation,
                fitnessFunction,
                randomGenerator,
                mutationStrategy,
                selectionStrategy);
            device.Synchronize();

            (_currentPopulation, _nextPopulation) = (_nextPopulation, _currentPopulation);

            updatedPopulationHandler?.Handle(OptimizerState.Running, device, generation, _currentPopulation);
        } while (terminationStrategy.IsMustTerminate(device, generation, _currentPopulation) == false
                 && cancellationToken.IsCancellationRequested == false);

        updatedPopulationHandler?.Handle(OptimizerState.Terminating, device, generation, _currentPopulation);
    }

    public HostPopulation? GetCurrentPopulationOrNull() => _currentPopulation;

    public void Dispose()
    {
        if (_currentPopulation != null) FreeGpuMemoryPopulation(_currentPopulation);
        if (_nextPopulation != null) FreeGpuMemoryPopulation(_nextPopulation);
        if (_trialPopulation != null) FreeGpuMemoryPopulation(_trialPopulation);

        device.Dispose();
        context.Dispose();
    }

    private static void FreeGpuMemoryPopulation(HostPopulation hostPopulation)
    {
        hostPopulation.FitnessFunctionValues.Dispose();
        hostPopulation.Individuals.Dispose();
    }
}