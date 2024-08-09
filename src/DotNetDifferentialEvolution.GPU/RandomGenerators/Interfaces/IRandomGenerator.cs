namespace DotNetDifferentialEvolution.GPU.RandomGenerators.Interfaces;

public interface IRandomGenerator
{
    public double NextDouble(int index);

    public float NextFloat(int index);

    public uint NextUInt(int index);

    public int Next(int index);
}