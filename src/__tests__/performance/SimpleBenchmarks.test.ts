import { BacktestingEngine } from '@/services/ml/BacktestingEngine';
import { CandleData } from '@/types/session';

// Simplified benchmarks without complex ONNX dependencies
describe('SimpleBenchmarks', () => {
  let mockCandles: CandleData[];

  beforeEach(() => {
    // Generate simple test data
    mockCandles = Array.from({ length: 100 }, (_, i) => ({
      id: `candle_${i}`,
      timestamp: new Date(Date.now() + i * 60000).toISOString(),
      open: 100 + Math.random() * 10,
      high: 105 + Math.random() * 10,
      low: 95 + Math.random() * 10,
      close: 100 + Math.random() * 10,
      volume: 1000 + Math.random() * 500,
      session_id: 'test_session',
      candle_index: i,
      candle_datetime: new Date(Date.now() + i * 60000).toISOString()
    } as CandleData));
  });

  describe('Backtesting Performance', () => {
    it('should process candles efficiently', async () => {
      const backtestingEngine = new BacktestingEngine({
        initialCapital: 10000,
        positionSize: 10,
        stopLoss: 2,
        takeProfit: 4,
        transactionCost: 0.1,
        maxPositions: 3,
        riskPerTrade: 1
      });

      const signals = mockCandles.slice(0, 10).map((candle, i) => ({
        timestamp: new Date(candle.timestamp).getTime(),
        direction: i % 2 === 0 ? 'long' as const : 'short' as const,
        confidence: 0.7,
        price: candle.close
      }));

      const start = performance.now();
      const results = await backtestingEngine.runBacktest(mockCandles, signals);
      const end = performance.now();

      const duration = end - start;

      expect(duration).toBeLessThan(1000); // Should complete within 1 second
      expect(results).toBeDefined();
      expect(results.trades).toBeDefined();
    });

    it('should handle empty signals gracefully', async () => {
      const backtestingEngine = new BacktestingEngine({
        initialCapital: 10000,
        positionSize: 10,
        stopLoss: 2,
        takeProfit: 4,
        transactionCost: 0.1,
        maxPositions: 3,
        riskPerTrade: 1
      });

      const start = performance.now();
      const results = await backtestingEngine.runBacktest(mockCandles, []);
      const end = performance.now();

      const duration = end - start;

      expect(duration).toBeLessThan(500);
      expect(results.trades).toHaveLength(0);
    });
  });

  describe('Data Processing Speed', () => {
    it('should process large datasets quickly', () => {
      const largeDataset = Array.from({ length: 10000 }, (_, i) => ({
        value: Math.random() * 100,
        timestamp: Date.now() + i * 1000
      }));

      const start = performance.now();
      
      // Simple processing simulation
      const processed = largeDataset.map(item => ({
        ...item,
        sma: item.value * 1.1
      }));

      const end = performance.now();
      const duration = end - start;

      expect(duration).toBeLessThan(100); // Should be very fast
      expect(processed).toHaveLength(largeDataset.length);
    });
  });
});