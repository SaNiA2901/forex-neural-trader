import { CandleData } from '@/types/session';
import { AdvancedMLTrainingService } from '@/services/ml/AdvancedMLTrainingService';

describe('Label Shift Validation', () => {
  let service: AdvancedMLTrainingService;
  
  beforeEach(() => {
    service = new AdvancedMLTrainingService();
  });

  it('should use future candle for label generation (UP case)', async () => {
    // Create artificial series with known values
    const candles: CandleData[] = [
      // Setup candles with predictable pattern
      ...Array.from({ length: 50 }, (_, i) => ({
        id: `candle_${i}`,
        timestamp: new Date(Date.now() + i * 60000).toISOString(),
        open: 100,
        high: 105,
        low: 95,
        close: 100,
        volume: 1000,
        session_id: 'test_session',
        candle_index: i,
        candle_datetime: new Date(Date.now() + i * 60000).toISOString()
      } as CandleData)),
      // Current candle
      {
        id: 'current_candle',
        timestamp: new Date(Date.now() + 50 * 60000).toISOString(),
        open: 100,
        high: 105,
        low: 95,
        close: 100, // Current close: 100
        volume: 1000,
        session_id: 'test_session',
        candle_index: 50,
        candle_datetime: new Date(Date.now() + 50 * 60000).toISOString()
      } as CandleData,
      // Next candle (should be used for label)
      {
        id: 'next_candle',
        timestamp: new Date(Date.now() + 51 * 60000).toISOString(),
        open: 102,
        high: 107,
        low: 97,
        close: 105, // Next close: 105 > 100 -> UP (label = 1)
        volume: 1200,
        session_id: 'test_session',
        candle_index: 51,
        candle_datetime: new Date(Date.now() + 51 * 60000).toISOString()
      } as CandleData
    ];

    const config = {
      modelType: 'regression' as const,
      lookbackPeriod: 20,
      features: ['price', 'volume'],
      trainingRatio: 0.7,
      validationRatio: 0.15
    };

    const features = await service.engineerFeatures(candles, config);
    
    // Find the feature vector for candle at index 50
    const targetFeature = features.find(f => f.timestamp === Number(candles[50].timestamp));
    
    expect(targetFeature).toBeDefined();
    expect(targetFeature!.label).toBe(1); // Should be 1 because next candle close (105) > current close (100)
  });

  it('should use future candle for label generation (DOWN case)', async () => {
    const candles: CandleData[] = [
      // Setup candles
      ...Array.from({ length: 50 }, (_, i) => ({
        id: `candle_${i}`,
        timestamp: new Date(Date.now() + i * 60000).toISOString(),
        open: 100,
        high: 105,
        low: 95,
        close: 100,
        volume: 1000,
        session_id: 'test_session',
        candle_index: i,
        candle_datetime: new Date(Date.now() + i * 60000).toISOString()
      } as CandleData)),
      // Current candle
      {
        id: 'current_candle',
        timestamp: new Date(Date.now() + 50 * 60000).toISOString(),
        open: 100,
        high: 105,    
        low: 95,
        close: 100, // Current close: 100
        volume: 1000,
        session_id: 'test_session',
        candle_index: 50,
        candle_datetime: new Date(Date.now() + 50 * 60000).toISOString()
      } as CandleData,
      // Next candle (should be used for label)
      {
        id: 'next_candle',
        timestamp: new Date(Date.now() + 51 * 60000).toISOString(),
        open: 98,
        high: 102,
        low: 92,
        close: 95, // Next close: 95 < 100 -> DOWN (label = 0)
        volume: 1200,
        session_id: 'test_session',
        candle_index: 51,
        candle_datetime: new Date(Date.now() + 51 * 60000).toISOString()
      } as CandleData
    ];

    const config = {
      modelType: 'regression' as const,
      lookbackPeriod: 20,
      features: ['price', 'volume'],
      trainingRatio: 0.7,
      validationRatio: 0.15
    };

    const features = await service.engineerFeatures(candles, config);
    
    // Find the feature vector for candle at index 50
    const targetFeature = features.find(f => f.timestamp === Number(candles[50].timestamp));
    
    expect(targetFeature).toBeDefined();
    expect(targetFeature!.label).toBe(0); // Should be 0 because next candle close (95) < current close (100)
  });

  it('should not have look-ahead bias in features', async () => {
    const candles: CandleData[] = Array.from({ length: 100 }, (_, i) => ({
      id: `candle_${i}`,
      timestamp: new Date(Date.now() + i * 60000).toISOString(),
      open: 100 + Math.sin(i * 0.1) * 5,
      high: 105 + Math.sin(i * 0.1) * 5,
      low: 95 + Math.sin(i * 0.1) * 5,
      close: 100 + Math.sin(i * 0.1) * 5,
      volume: 1000 + Math.random() * 500,
      session_id: 'test_session',
      candle_index: i,
      candle_datetime: new Date(Date.now() + i * 60000).toISOString()
    } as CandleData));

    const config = {
      modelType: 'regression' as const,
      lookbackPeriod: 20,
      features: ['price', 'volume'],
      trainingRatio: 0.7,
      validationRatio: 0.15
    };

    const features = await service.engineerFeatures(candles, config);
    
    // Features should only be generated for candles where we have next candle for label
    // So features.length should be candles.length - lookbackPeriod - 1
    const expectedLength = candles.length - config.lookbackPeriod - 1;
    expect(features.length).toBe(expectedLength);
    
    // Each feature should only use data up to its timestamp (no future data)
    for (let i = 0; i < features.length; i++) {
      const feature = features[i];
      const candleIndex = candles.findIndex(c => Number(c.timestamp) === feature.timestamp);
      
      // Feature should only use candles[0] to candles[candleIndex] (no future data)
      expect(candleIndex).toBeGreaterThanOrEqual(config.lookbackPeriod);
      expect(candleIndex).toBeLessThan(candles.length - 1); // Must have next candle for label
    }
  });

  it('should fail if current candle is used for label (data leakage test)', async () => {
    // This test ensures we're not accidentally using current candle for label
    const candles: CandleData[] = [
      // Setup with identical candles except the target one
      ...Array.from({ length: 50 }, (_, i) => ({
        id: `candle_${i}`,
        timestamp: new Date(Date.now() + i * 60000).toISOString(),
        open: 100,
        high: 105,
        low: 95,
        close: 100,
        volume: 1000,
        session_id: 'test_session',
        candle_index: i,
        candle_datetime: new Date(Date.now() + i * 60000).toISOString()
      } as CandleData)),
      // Current candle with specific pattern
      {
        id: 'current_candle',
        timestamp: new Date(Date.now() + 50 * 60000).toISOString(),
        open: 90,   // Open much lower
        high: 110,  // High much higher 
        low: 85,    // Low much lower
        close: 108, // Close much higher -> if used for label would be UP
        volume: 2000,
        session_id: 'test_session',
        candle_index: 50,
        candle_datetime: new Date(Date.now() + 50 * 60000).toISOString()
      } as CandleData,
      // Next candle should determine the label, not current
      {
        id: 'next_candle',
        timestamp: new Date(Date.now() + 51 * 60000).toISOString(),
        open: 108,
        high: 110,
        low: 95,
        close: 98, // Next close: 98 < 108 -> DOWN (label = 0)
        volume: 1200,
        session_id: 'test_session',
        candle_index: 51,
        candle_datetime: new Date(Date.now() + 51 * 60000).toISOString()
      } as CandleData
    ];

    const config = {
      modelType: 'regression' as const,
      lookbackPeriod: 20,
      features: ['price', 'volume'],
      trainingRatio: 0.7,
      validationRatio: 0.15
    };

    const features = await service.engineerFeatures(candles, config);
    const targetFeature = features.find(f => f.timestamp === Number(candles[50].timestamp));
    
    expect(targetFeature).toBeDefined();
    
    // Label should be 0 (DOWN) based on next candle (98 < 108)
    // NOT 1 (UP) which would be if we incorrectly used current candle pattern
    expect(targetFeature!.label).toBe(0);
    
    // Double check: if we were using current candle incorrectly, 
    // we would compare current close (108) vs some reference and get UP
    // But correct implementation uses next close (98) vs current close (108) = DOWN
    const correctLabel = candles[51].close > candles[50].close ? 1 : 0;
    expect(targetFeature!.label).toBe(correctLabel);
  });
});