using System.Runtime.CompilerServices;
using DotNetDifferentialEvolution.GPU.RandomGenerators.Interfaces;
using ILGPU;
using ILGPU.Algorithms.Random;

namespace DotNetDifferentialEvolution.GPU.RandomGenerators;

public struct RandomGenerator : IRandomGenerator
{
    private readonly ArrayView<XorShift32> _xorShifts;

    public RandomGenerator(ArrayView<XorShift32> xorShifts)
    {
        _xorShifts = xorShifts;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double NextDouble(int index)
    {
        var xorShift = _xorShifts[index];
        var result = xorShift.NextDouble();
        _xorShifts[index] = xorShift.NextProvider();
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float NextFloat(int index)
    {
        var xorShift = _xorShifts[index];
        var result = xorShift.NextFloat();
        _xorShifts[index] = xorShift.NextProvider();
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public uint NextUInt(int index)
    {
        var xorShift = _xorShifts[index];
        var result = xorShift.NextUInt();
        _xorShifts[index] = xorShift.NextProvider();
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int Next(int index)
    {
        var xorShift = _xorShifts[index];
        var result = xorShift.Next();
        _xorShifts[index] = xorShift.NextProvider();
        return result;
    }
}
