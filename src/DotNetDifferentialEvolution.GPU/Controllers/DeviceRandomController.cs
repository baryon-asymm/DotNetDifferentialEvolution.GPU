using System.Collections.Concurrent;
using DotNetDifferentialEvolution.GPU.Models;
using ILGPU.Runtime;

namespace DotNetDifferentialEvolution.GPU.Controllers;

public class DeviceRandomController : IDisposable
{
    private readonly Dictionary<Guid, DeviceRandomHolder> _holders;

    private readonly BlockingCollection<DeviceRandomHolder> _goingOutHolders = new();
    private readonly BlockingCollection<DeviceRandomHolder> _incomingHolders = new();

    private readonly double[,] _hostRandomBuffer;

    private readonly Random _random = Random.Shared;

    private Task _taskRunningLoop;
    private readonly CancellationTokenSource _cancellationTokenSource = new();

    private readonly object _syncLock = new();

    public int PageSize { get; }

    public int NumberOfPages { get; }

    public DeviceRandomController(
        int poolSize,
        int populationSize,
        int numberOfRandomNumbersPerPage,
        int numberOfPages,
        Accelerator device)
    {
        PageSize = numberOfRandomNumbersPerPage;
        NumberOfPages = numberOfPages;

        _holders = new Dictionary<Guid, DeviceRandomHolder>(poolSize);

        var totalRandomNumbers = numberOfRandomNumbersPerPage * numberOfPages;
        _hostRandomBuffer = new double[populationSize, totalRandomNumbers];

        InitDeviceRandomHolders(poolSize, device);
    }

    private void InitDeviceRandomHolders(int poolSize, Accelerator device)
    {
        for (var i = 0; i < poolSize; i++)
        {
            var holder = GetDeviceRandomHolder(device);
            _holders[holder.Id] = holder;

            _incomingHolders.Add(_holders[holder.Id]);
        }
    }

    private DeviceRandomHolder GetDeviceRandomHolder(Accelerator device)
    {
        var deviceRandomBuffer = device.Allocate2DDenseX(_hostRandomBuffer);

        var holder = new DeviceRandomHolder(PageSize, deviceRandomBuffer);

        return holder;
    }

    public void Start()
    {
        if (_cancellationTokenSource.IsCancellationRequested)
            throw new InvalidOperationException("The worker process executing the loop has already stopped.");

        if (_taskRunningLoop is not null)
            throw new InvalidOperationException("The worker process executing the loop has already started.");

        var cancellationToken = _cancellationTokenSource.Token;
        _taskRunningLoop = Task.Factory.StartNew(RunningLoop, cancellationToken, TaskCreationOptions.LongRunning);
    }

    public void Stop()
    {
        if (_taskRunningLoop is null)
            throw new InvalidOperationException("The worker process executing the loop has not yet started.");

        _cancellationTokenSource.Cancel();
    }

    private void RunningLoop(object? arg)
    {
        try
        {
            lock (_syncLock)
                TryRunningLoop(arg);
        }
        catch (OperationCanceledException)
        {
        }
    }

    private void TryRunningLoop(object? arg)
    {
        if (arg is not CancellationToken)
            throw new ArgumentException($"{nameof(RunningLoop)}'s argument must have a CancellationToken type.");

        var cancellationToken = (CancellationToken)arg;

        while (cancellationToken.IsCancellationRequested == false)
        {
            var holder = _incomingHolders.Take(cancellationToken);
            UpdateDeviceRandom(holder);
            _goingOutHolders.Add(holder);
        }
    }

    private void UpdateDeviceRandom(DeviceRandomHolder holder)
    {
        const int startIndex = 0;
        const int populationsDimension = 0;
        const int individualsDimension = 1;
        Parallel.For(startIndex, _hostRandomBuffer.GetLength(populationsDimension), i =>
        {
            for (var j = 0; j < individualsDimension; j++)
            {
                _hostRandomBuffer[i, j] = _random.NextDouble();
            }
        });

        holder.DeviceRandomBuffer.CopyFromCPU(_hostRandomBuffer);
    }

    public DeviceRandomHolder ToRecycleAndGetNew(DeviceRandomHolder oldHolder)
    {
        _incomingHolders.Add(_holders[oldHolder.Id]);

        var holder = _goingOutHolders.Take();
        return holder;
    }

    public DeviceRandomHolder GetFirst()
    {
        var holder = _goingOutHolders.Take();
        return holder;
    }

    public void Dispose()
    {
        if (_cancellationTokenSource.IsCancellationRequested == false)
            _cancellationTokenSource.Cancel();

        lock (_syncLock)
        {
            FreeGPUMemory();

            _cancellationTokenSource.Dispose();
            _incomingHolders.Dispose();
            _goingOutHolders.Dispose();
        }
    }

    private void FreeGPUMemory()
    {
        foreach (var memoryBuffer in _holders.Select(holderPair => holderPair.Value)
                     .Select(holder => holder.DeviceRandomBuffer))
        {
            memoryBuffer.Dispose();
        }
    }
}
