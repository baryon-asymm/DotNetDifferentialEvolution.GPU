namespace DotNetDifferentialEvolution.GPU.RandomGenerators.Interfaces;

/// <summary>
/// Defines the contract for generating random values in a parallelized environment, typically used in GPU-based algorithms.
/// </summary>
/// <remarks>
/// Implementations of this interface provide random number generation functionality, which can be indexed to support parallel operations across multiple threads or GPU cores.
/// </remarks>
public interface IRandomGenerator
{
    /// <summary>
    /// Generates a random double-precision floating-point number for a specific index.
    /// </summary>
    /// <param name="index">The index that determines the specific generator instance to use.</param>
    /// <returns>A random double-precision floating-point number.</returns>
    public double NextDouble(int index);

    /// <summary>
    /// Generates a random single-precision floating-point number for a specific index.
    /// </summary>
    /// <param name="index">The index that determines the specific generator instance to use.</param>
    /// <returns>A random single-precision floating-point number.</returns>
    public float NextFloat(int index);

    /// <summary>
    /// Generates a random unsigned integer for a specific index.
    /// </summary>
    /// <param name="index">The index that determines the specific generator instance to use.</param>
    /// <returns>A random unsigned integer.</returns>
    public uint NextUInt(int index);

    /// <summary>
    /// Generates a random integer for a specific index.
    /// </summary>
    /// <param name="index">The index that determines the specific generator instance to use.</param>
    /// <returns>A random integer.</returns>
    public int Next(int index);
}