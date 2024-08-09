# DotNetDifferentialEvolution.GPU

## Introduction

Differential Evolution (DE) is a stochastic optimization algorithm used for finding global minima or maxima of functions in multi-dimensional spaces. It was introduced by Kenneth Price and Rainer Storn in 1997. DE is known for its simplicity and effectiveness, especially for complex optimization problems. For more details on the algorithm, you can refer to the [Wikipedia page](https://en.wikipedia.org/wiki/Differential_evolution).

This library implements the Differential Evolution algorithm with GPU acceleration using [ILGPU](https://github.com/m4rs-mt/ILGPU/), which significantly speeds up the optimization process. The library is designed to be flexible and customizable, allowing users to define their own algorithm components through interfaces.

## Features

- **Support for various mutation, selection, and termination strategies**: Adaptable to specific tasks and problem domains.
- **GPU acceleration using ILGPU**: Improves performance by utilizing GPU computation.
- **Customizable algorithm components**: Implement your own strategies by defining interfaces.

## Installation

To use this library, you need:
- .NET SDK version 6.0 or higher.
- ILGPU package for GPU computation.

You can install the library via NuGet:

```bash
dotnet add package DotNetDifferentialEvolution.GPU
```

## Usage

Here's a complete example showing how to use the library for optimizing a polynomial approximation function:

```csharp
using System;
using System.Linq;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;

public class Program
{
    private const int MaxGenerationCount = 1000;
    private const int PopulationSize = 10000;

    public static async Task Main(string[] args)
    {
        // Set up GPU context and device
        using var context = Context.Create(builder =>
        {
            builder.OpenCL(); // Use OpenCL or Cuda depending on your GPU
            builder.EnableAlgorithms();
        });

        var device = context.GetPreferredDevice(preferCPU: false).CreateAccelerator(context);

        // Define bounds and create PopulationSamplingMaker
        double lowerValue = -2000;
        double upperValue = 2000;
        int individualSize = PolynomialApproximationFunction.IndividualSize;

        var lowerBound = Enumerable.Repeat(lowerValue, individualSize).ToArray();
        var upperBound = Enumerable.Repeat(upperValue, individualSize).ToArray();
        var populationSamplingMaker = new PopulationSamplingMaker(PopulationSize, upperBound, lowerBound);

        // Initialize random generator
        var random = new Random();
        var xorShifts = new XorShift32[PopulationSize];
        for (var i = 0; i < xorShifts.Length; i++)
            xorShifts[i] = new XorShift32((uint)random.Next());
        var deviceXorShifts = device.Allocate1D(xorShifts);
        var randomGenerator = new RandomGenerator(deviceXorShifts.View);

        // Create strategies
        var mutationStrategy = new MutationStrategy<RandomGenerator>(device.Allocate1D(lowerBound).View, device.Allocate1D(upperBound).View);
        var selectionStrategy = new SelectionStrategy();
        var terminationStrategy = new MaxGenerationStrategy(MaxGenerationCount);

        // Create and initialize optimizer
        var optimizer = new DifferentialEvolutionOptimizer(
            new KernelController<PolynomialApproximationFunction, RandomGenerator, MutationStrategy<RandomGenerator>, SelectionStrategy>(
                context,
                device,
                populationSamplingMaker,
                new PolynomialApproximationFunction(),
                randomGenerator,
                mutationStrategy,
                selectionStrategy,
                terminationStrategy
            ));

        // Run optimization
        var result = await optimizer.RunAsync();

        // Output results
        Console.WriteLine($"Result Fitness Function Value: {result.FitnessFunctionValue}");
        Console.WriteLine($"Result Vector: {string.Join(", ", result.Individual)}");
    }
}

public readonly struct PolynomialApproximationFunction : IFitnessFunctionInvoker
{
    public static int IndividualSize => 6;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Invoke(int individualIndex, DevicePopulation devicePopulation)
    {
        double[] functionValues = { 0.264, 0.228, 0.194, 0.176, 0.162, 0.15, 0.14, 0.134, 0.13, 0.122, 0.12, 0.114 };
        double[] argValues = { 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5 };

        var result = 0.0;
        for (var i = 0; i < functionValues.Length; i++)
        {
            result += XMath.Pow(functionValues[i] - GetFunctionValue(individualIndex, devicePopulation.Individuals, argValues[i]), 2);
        }

        devicePopulation.FitnessFunctionValues[individualIndex] = result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double GetFunctionValue(int individualIndex, ArrayView2D<double, Stride2D.DenseX> individuals, double argValue)
    {
        var length = individuals.Extent.Y;
        var result = 0.0;
        for (var i = 0; i < length; i++)
        {
            result += individuals[individualIndex, i] * XMath.Pow(argValue, i);
        }

        return result;
    }
}
```

### Explanation

1. **Context and Device Setup**: 
   - Initializes the GPU context and selects the appropriate device for computations.

2. **Population and Strategy Initialization**: 
   - Sets up bounds, sampling, random generators, mutation, selection, and termination strategies.

3. **Optimizer Creation and Execution**: 
   - Creates an instance of `DifferentialEvolutionOptimizer` with a `KernelController` and runs the optimization.

4. **Results Output**: 
   - Prints the results of the optimization, including the fitness function value and the optimized vector.

This example demonstrates how to configure and run the Differential Evolution algorithm using the provided library, allowing you to optimize a polynomial approximation function on a GPU.

### Flexibility

The library allows you to create custom algorithm components by implementing the following interfaces:

- `IDifferentialEvolutionOptimizer<T>` — Interface for implementing the Differential Evolution optimizer.
- `IFitnessFunctionInvoker` — Interface for invoking the fitness function.
- `IKernelController` — Interface for managing GPU kernels.
- `IMutationStrategy<TRandomGenerator>` — Interface for implementing mutation strategies.
- `ISelectionStrategy` — Interface for implementing selection strategies.
- `ITerminationStrategy` — Interface for implementing termination strategies.
- `IPopulationSamplingMaker` — Interface for creating initial populations.
- `IRandomGenerator` — Interface for generating random numbers.

This allows you to create your own versions of these components to fit different scenarios and needs.

### GPU Execution

The library utilizes [ILGPU](https://github.com/m4rs-mt/ILGPU/) for GPU execution. It automatically detects the suitable device (GPU or CPU) and uses it for computations. Example code for creating a context and selecting a device is shown above.

## Contributing and License

This library is open-source and distributed under the [MIT License](https://github.com/baryon-asymm/DotNetDifferentialEvolution.GPU/blob/master/LICENSE). You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software. For details on the terms and conditions, please refer to the full [LICENSE](https://github.com/baryon-asymm/DotNetDifferentialEvolution.GPU/blob/master/LICENSE) file.

### How to Contribute

Contributions are welcome! If you'd like to contribute to the project, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Make your changes and ensure they are well-documented.
4. Submit a pull request with a detailed description of your changes.

If you have any questions or suggestions, please open an issue in the [project repository](https://github.com/baryon-asymm/DotNetDifferentialEvolution.GPU).

### Third-Party Libraries

This project utilizes [ILGPU](https://github.com/m4rs-mt/ILGPU/), which is licensed under the [University of Illinois/NCSA Open Source License](https://github.com/m4rs-mt/ILGPU/blob/master/LICENSE.txt). ILGPU is used for GPU acceleration in this library.

A copy of the ILGPU license is provided in the file [ILGPU_LICENSE](https://github.com/baryon-asymm/DotNetDifferentialEvolution.GPU/blob/master/ILGPU_LICENSE).
