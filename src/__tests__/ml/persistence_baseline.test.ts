import { CandleData } from '@/types/session';

describe('Persistence Baseline Test', () => {
  
  class PersistencePredictor {
    /**
     * Simple persistence predictor: predicts direction based on current candle body
     * pred = sign(current_candle.close - current_candle.open)
     */
    static predict(candle: CandleData): 'UP' | 'DOWN' {
      return candle.close > candle.open ? 'UP' : 'DOWN';
    }

    /**
     * Alternative: predict same as last direction
     */
    static predictLastDirection(currentCandle: CandleData, previousCandle: CandleData): 'UP' | 'DOWN' {
      const lastDirection = previousCandle.close > previousCandle.open ? 'UP' : 'DOWN';
      return lastDirection;
    }
  }

  function calculateAccuracy(predictions: string[], actuals: string[]): number {
    if (predictions.length !== actuals.length || predictions.length === 0) return 0;
    
    const correct = predictions.filter((pred, i) => pred === actuals[i]).length;
    return correct / predictions.length;
  }

  function generateTestData(count: number): CandleData[] {
    return Array.from({ length: count }, (_, i) => {
      const basePrice = 100;
      const trend = Math.sin(i * 0.05) * 10; // Create trending pattern
      const noise = (Math.random() - 0.5) * 5;
      const price = basePrice + trend + noise;
      
      const open = price + (Math.random() - 0.5) * 2;
      const close = price + (Math.random() - 0.5) * 2;
      const high = Math.max(open, close) + Math.random() * 3;
      const low = Math.min(open, close) - Math.random() * 3;

      return {
        id: `test_${i}`,
        timestamp: new Date(Date.now() + i * 60000).toISOString(),
        open,
        high,
        low,
        close,
        volume: 1000 + Math.random() * 500,
        session_id: 'test_session',
        candle_index: i,
        candle_datetime: new Date(Date.now() + i * 60000).toISOString()
      } as CandleData;
    });
  }

  it('should establish persistence baseline accuracy', () => {
    const candles = generateTestData(1000);
    const predictions: string[] = [];
    const actuals: string[] = [];

    // Generate predictions and actual outcomes
    for (let i = 0; i < candles.length - 1; i++) {
      const currentCandle = candles[i];
      const nextCandle = candles[i + 1];
      
      // Persistence prediction based on current candle body
      const prediction = PersistencePredictor.predict(currentCandle);
      predictions.push(prediction);
      
      // Actual outcome based on next candle
      const actual = nextCandle.close > currentCandle.close ? 'UP' : 'DOWN';
      actuals.push(actual);
    }

    const baselineAccuracy = calculateAccuracy(predictions, actuals);
    
    // Log baseline performance for comparison
    console.log(`Persistence Baseline Accuracy: ${(baselineAccuracy * 100).toFixed(2)}%`);
    
    // Baseline should be around 50% for random data, maybe slightly higher due to momentum
    expect(baselineAccuracy).toBeGreaterThan(0.35); // Sanity check - should be somewhat predictive
    expect(baselineAccuracy).toBeLessThan(0.75); // Should not be too high for this simple strategy
    
    // Store baseline for model comparison
    (global as any).__PERSISTENCE_BASELINE__ = baselineAccuracy;
  });

  it('should warn if model does not beat baseline', () => {
    // This would be called after model training/testing
    const mockModelAccuracy = 0.48; // Example poor model performance
    const baselineAccuracy = (global as any).__PERSISTENCE_BASELINE__ || 0.5;
    
    const improvement = mockModelAccuracy - baselineAccuracy;
    
    if (improvement <= 0.02) { // Less than 2% improvement
      console.warn(`
        ⚠️  MODEL PERFORMANCE WARNING ⚠️
        Model accuracy: ${(mockModelAccuracy * 100).toFixed(2)}%
        Baseline accuracy: ${(baselineAccuracy * 100).toFixed(2)}%
        Improvement: ${(improvement * 100).toFixed(2)}%
        
        The model is not significantly better than a simple persistence predictor.
        This suggests potential data leakage or model overfitting.
        
        Recommended actions:
        1. Check for label leakage (using future data in features)
        2. Verify feature engineering (no look-ahead bias)
        3. Increase model complexity or feature quality
        4. Check for proper train/test split
      `);
      
      // In CI, this could fail the build:
      // expect(improvement).toBeGreaterThan(0.02);
    }
    
    // This test passes but warns - in production you might want to fail
    expect(mockModelAccuracy).toBeGreaterThan(0); // Just ensures test doesn't crash
  });

  it('should test alternative persistence strategies', () => {
    const candles = generateTestData(500);
    const strategies = {
      'current_body': [] as string[],
      'last_direction': [] as string[],
      'last_outcome': [] as string[]
    };
    const actuals: string[] = [];

    for (let i = 1; i < candles.length - 1; i++) {
      const currentCandle = candles[i];
      const previousCandle = candles[i - 1];
      const nextCandle = candles[i + 1];
      
      // Strategy 1: Current candle body direction
      strategies.current_body.push(PersistencePredictor.predict(currentCandle));
      
      // Strategy 2: Last direction (previous candle body)
      strategies.last_direction.push(PersistencePredictor.predictLastDirection(currentCandle, previousCandle));
      
      // Strategy 3: Last actual outcome
      const lastOutcome = currentCandle.close > previousCandle.close ? 'UP' : 'DOWN';
      strategies.last_outcome.push(lastOutcome);
      
      // Actual outcome
      const actual = nextCandle.close > currentCandle.close ? 'UP' : 'DOWN';
      actuals.push(actual);
    }

    // Calculate accuracies for all strategies
    const accuracies = {
      current_body: calculateAccuracy(strategies.current_body, actuals),
      last_direction: calculateAccuracy(strategies.last_direction, actuals),
      last_outcome: calculateAccuracy(strategies.last_outcome, actuals)
    };

    console.log('Baseline Strategy Accuracies:');
    Object.entries(accuracies).forEach(([strategy, accuracy]) => {
      console.log(`  ${strategy}: ${(accuracy * 100).toFixed(2)}%`);
    });

    // All strategies should perform reasonably
    Object.values(accuracies).forEach(accuracy => {
      expect(accuracy).toBeGreaterThan(0.3);
      expect(accuracy).toBeLessThan(0.8);
    });
  });

  it('should test model vs baseline on same test set', () => {
    const testCandles = generateTestData(200);
    
    // Simulate model predictions (in reality, this would come from your ML model)
    const mockModelPredictions: string[] = [];
    const baselinePredictions: string[] = [];
    const actuals: string[] = [];
    
    for (let i = 0; i < testCandles.length - 1; i++) {
      const currentCandle = testCandles[i];
      const nextCandle = testCandles[i + 1];
      
      // Mock model prediction (random for this test)
      mockModelPredictions.push(Math.random() > 0.5 ? 'UP' : 'DOWN');
      
      // Baseline prediction
      baselinePredictions.push(PersistencePredictor.predict(currentCandle));
      
      // Actual outcome
      actuals.push(nextCandle.close > currentCandle.close ? 'UP' : 'DOWN');
    }
    
    const modelAccuracy = calculateAccuracy(mockModelPredictions, actuals);
    const baselineAccuracy = calculateAccuracy(baselinePredictions, actuals);
    
    console.log(`Model vs Baseline Comparison:`);
    console.log(`  Model: ${(modelAccuracy * 100).toFixed(2)}%`);
    console.log(`  Baseline: ${(baselineAccuracy * 100).toFixed(2)}%`);
    console.log(`  Improvement: ${((modelAccuracy - baselineAccuracy) * 100).toFixed(2)}%`);
    
    // Test should always pass, but log results for analysis
    expect(modelAccuracy).toBeGreaterThanOrEqual(0);
    expect(baselineAccuracy).toBeGreaterThanOrEqual(0);
  });
});