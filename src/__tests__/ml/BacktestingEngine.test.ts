import { BacktestingEngine, BacktestConfig, PredictionSignal } from '@/services/ml/BacktestingEngine';
import { CandleData } from '@/types/session';

describe('BacktestingEngine', () => {
  let engine: BacktestingEngine;
  let mockCandles: CandleData[];
  let mockSignals: PredictionSignal[];

  const defaultConfig: BacktestConfig = {
    initialCapital: 10000,
    positionSize: 10,
    stopLoss: 2,
    takeProfit: 4,
    transactionCost: 0.1,
    maxPositions: 3,
    riskPerTrade: 1
  };

  beforeEach(() => {
    engine = new BacktestingEngine(defaultConfig);
    
    // Create mock candle data
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

    // Create mock signals
    mockSignals = [
      {
        timestamp: new Date(mockCandles[10].timestamp).getTime(),
        direction: 'long',
        confidence: 0.8,
        price: mockCandles[10].close
      },
      {
        timestamp: new Date(mockCandles[50].timestamp).getTime(),
        direction: 'short',
        confidence: 0.7,
        price: mockCandles[50].close
      }
    ];
  });

  describe('Trade Execution', () => {
    it('should open trades based on signals', async () => {
      const results = await engine.runBacktest(mockCandles, mockSignals);
      
      expect(results.trades.length).toBeGreaterThan(0);
      expect(results.metrics.totalTrades).toBe(results.trades.filter(t => t.status === 'closed').length);
    });

    it('should respect position size limits', async () => {
      const manySignals = mockCandles.slice(0, 10).map(candle => ({
        timestamp: new Date(candle.timestamp).getTime(),
        direction: 'long' as const,
        confidence: 0.8,
        price: candle.close
      }));

      const results = await engine.runBacktest(mockCandles, manySignals);
      const openTradesAtOnce = results.trades.filter(t => t.status === 'open').length;
      
      expect(openTradesAtOnce).toBeLessThanOrEqual(defaultConfig.maxPositions);
    });

    it('should calculate stop loss and take profit correctly', async () => {
      const signal: PredictionSignal = {
        timestamp: new Date(mockCandles[10].timestamp).getTime(),
        direction: 'long',
        confidence: 0.8,
        price: mockCandles[10].close
      };

      const results = await engine.runBacktest(mockCandles, [signal]);
      const trade = results.trades[0];

      if (trade.stopLoss && trade.takeProfit) {
        const expectedStopLoss = trade.entryPrice * (1 - defaultConfig.stopLoss! / 100);
        const expectedTakeProfit = trade.entryPrice * (1 + defaultConfig.takeProfit! / 100);
        
        expect(trade.stopLoss).toBeCloseTo(expectedStopLoss, 2);
        expect(trade.takeProfit).toBeCloseTo(expectedTakeProfit, 2);
      }
    });
  });

  describe('Performance Metrics', () => {
    it('should calculate win rate correctly', async () => {
      const results = await engine.runBacktest(mockCandles, mockSignals);
      
      const winningTrades = results.trades.filter(t => t.status === 'closed' && t.pnl! > 0).length;
      const totalTrades = results.trades.filter(t => t.status === 'closed').length;
      const expectedWinRate = totalTrades > 0 ? winningTrades / totalTrades : 0;
      
      expect(results.metrics.winRate).toBeCloseTo(expectedWinRate, 3);
    });

    it('should calculate Sharpe ratio', async () => {
      const results = await engine.runBacktest(mockCandles, mockSignals);
      
      expect(results.metrics.sharpeRatio).toBeDefined();
      expect(typeof results.metrics.sharpeRatio).toBe('number');
    });

    it('should track maximum drawdown', async () => {
      const results = await engine.runBacktest(mockCandles, mockSignals);
      
      expect(results.metrics.maxDrawdown).toBeGreaterThanOrEqual(0);
      expect(results.metrics.maxDrawdownPercent).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Risk Management', () => {
    it('should filter low confidence signals', async () => {
      const lowConfidenceSignals = mockSignals.map(s => ({ ...s, confidence: 0.3 }));
      const results = await engine.runBacktest(mockCandles, lowConfidenceSignals);
      
      expect(results.trades.length).toBe(0);
    });

    it('should calculate transaction costs', async () => {
      const results = await engine.runBacktest(mockCandles, mockSignals);
      
      expect(results.metrics.transactionCosts).toBeGreaterThan(0);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty signals', async () => {
      const results = await engine.runBacktest(mockCandles, []);
      
      expect(results.trades.length).toBe(0);
      expect(results.metrics.totalTrades).toBe(0);
    });

    it('should handle single candle', async () => {
      const singleCandle = [mockCandles[0]];
      const results = await engine.runBacktest(singleCandle, []);
      
      expect(results.equityCurve.length).toBeGreaterThan(0);
    });
  });
});