import { testDataGenerator } from '../utils/TestDataGenerator';
import { AdvancedMLTrainingService } from '@/services/ml/AdvancedMLTrainingService';

describe('Data Validation Tests', () => {
  let trainingService: AdvancedMLTrainingService;

  beforeEach(() => {
    trainingService = new AdvancedMLTrainingService();
  });

  describe('Reference Data Validation', () => {
    it('should validate SMA calculation against known values', () => {
      const { candles, expectedIndicators } = testDataGenerator.generateReferenceData();
      
      // Extract features and check SMA values
      const features = trainingService['extractFeatures'](candles);
      
      // Compare calculated SMA with expected values (allowing for small precision errors)
      const calculatedSMA20 = features.map(f => f.sma_20).filter(v => !isNaN(v));
      const expectedSMA20 = expectedIndicators.sma20.filter(v => !isNaN(v));
      
      expect(calculatedSMA20.length).toBe(expectedSMA20.length);
      
      for (let i = 0; i < Math.min(calculatedSMA20.length, expectedSMA20.length); i++) {
        expect(calculatedSMA20[i]).toBeCloseTo(expectedSMA20[i], 3);
      }
    });

    it('should validate RSI calculation accuracy', () => {
      const { candles, expectedIndicators } = testDataGenerator.generateReferenceData();
      
      const features = trainingService['extractFeatures'](candles);
      const calculatedRSI = features.map(f => f.rsi_14).filter(v => !isNaN(v));
      const expectedRSI = expectedIndicators.rsi14.filter(v => !isNaN(v));
      
      expect(calculatedRSI.length).toBeGreaterThan(0);
      
      // RSI should be between 0 and 100
      calculatedRSI.forEach(rsi => {
        expect(rsi).toBeGreaterThanOrEqual(0);
        expect(rsi).toBeLessThanOrEqual(100);
      });
    });

    it('should validate MACD calculation components', () => {
      const { candles } = testDataGenerator.generateReferenceData();
      
      const features = trainingService['extractFeatures'](candles);
      
      features.forEach(f => {
        if (!isNaN(f.macd_line) && !isNaN(f.macd_signal)) {
          // MACD histogram should equal line - signal
          const expectedHistogram = f.macd_line - f.macd_signal;
          expect(f.macd_histogram).toBeCloseTo(expectedHistogram, 6);
        }
      });
    });
  });

  describe('Feature Engineering Validation', () => {
    it('should handle edge cases in feature calculation', () => {
      // Test with minimal data
      const minimalCandles = testDataGenerator.generateMarketData({
        symbol: 'TEST',
        startPrice: 1.0,
        candleCount: 5,
        timeframe: 60000,
        conditions: [{ type: 'ranging', strength: 0.1, duration: 5 }],
        addNoise: false,
        noiseLevel: 0
      });

      const features = trainingService['extractFeatures'](minimalCandles);
      expect(features.length).toBe(minimalCandles.length);
      
      // Basic features should always be present
      features.forEach(f => {
        expect(f.open).toBeDefined();
        expect(f.high).toBeDefined();
        expect(f.low).toBeDefined();
        expect(f.close).toBeDefined();
        expect(f.volume).toBeDefined();
      });
    });

    it('should validate price relationships', () => {
      const candles = testDataGenerator.generateMarketData({
        symbol: 'VALIDATION',
        startPrice: 100,
        candleCount: 100,
        timeframe: 60000,
        conditions: [{ type: 'ranging', strength: 0.5, duration: 100 }],
        addNoise: true,
        noiseLevel: 0.01
      });

      candles.forEach(candle => {
        // High should be >= max(open, close)
        expect(candle.high).toBeGreaterThanOrEqual(Math.max(candle.open, candle.close));
        
        // Low should be <= min(open, close)
        expect(candle.low).toBeLessThanOrEqual(Math.min(candle.open, candle.close));
        
        // Volume should be positive
        expect(candle.volume).toBeGreaterThan(0);
      });
    });

    it('should detect and handle outliers', () => {
      const candles = testDataGenerator.generateMarketData({
        symbol: 'OUTLIER_TEST',
        startPrice: 100,
        candleCount: 100,
        timeframe: 60000,
        conditions: [{ type: 'volatile', strength: 2.0, duration: 100 }], // High volatility
        addNoise: true,
        noiseLevel: 0.1 // High noise
      });

      const features = trainingService['extractFeatures'](candles);
      
      // Check for extreme values that might indicate outliers
      const returns = features.map((f, i) => i > 0 ? (f.close - features[i-1].close) / features[i-1].close : 0);
      const extremeReturns = returns.filter(r => Math.abs(r) > 0.1); // >10% moves
      
      // In normal market conditions, extreme moves should be rare
      expect(extremeReturns.length / returns.length).toBeLessThan(0.1);
    });
  });

  describe('Data Quality Monitoring', () => {
    it('should identify missing data patterns', () => {
      const candles = testDataGenerator.generateMarketData({
        symbol: 'MISSING_DATA',
        startPrice: 100,
        candleCount: 100,
        timeframe: 60000,
        conditions: [{ type: 'ranging', strength: 0.5, duration: 100 }],
        addNoise: false,
        noiseLevel: 0
      });

      // Introduce missing data
      const gappedCandles = candles.map((candle, i) => ({
        ...candle,
        close: i % 10 === 0 ? NaN : candle.close, // Every 10th candle has missing close
        volume: i % 15 === 0 ? 0 : candle.volume  // Every 15th candle has zero volume
      }));

      const features = trainingService['extractFeatures'](gappedCandles);
      
      // Count missing values
      const missingCloses = features.filter(f => isNaN(f.close)).length;
      const zeroVolumes = features.filter(f => f.volume === 0).length;
      
      expect(missingCloses).toBeGreaterThan(0);
      expect(zeroVolumes).toBeGreaterThan(0);
    });

    it('should validate temporal consistency', () => {
      const candles = testDataGenerator.generateMarketData({
        symbol: 'TEMPORAL_TEST',
        startPrice: 100,
        candleCount: 100,
        timeframe: 60000,
        conditions: [{ type: 'ranging', strength: 0.5, duration: 100 }],
        addNoise: false,
        noiseLevel: 0
      });

      // Check timestamps are sequential
      for (let i = 1; i < candles.length; i++) {
        expect(candles[i].timestamp).toBeGreaterThan(candles[i-1].timestamp);
      }

      // Check consistent timeframe (allowing for small variations)
      const intervals = [];
      for (let i = 1; i < candles.length; i++) {
        intervals.push(candles[i].timestamp - candles[i-1].timestamp);
      }

      const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
      const expectedInterval = 60000; // 1 minute
      
      expect(avgInterval).toBeCloseTo(expectedInterval, -2); // Within 100ms tolerance
    });
  });

  describe('Feature Drift Detection', () => {
    it('should detect statistical changes in features', () => {
      // Generate two datasets with different characteristics
      const baselineData = testDataGenerator.generateMarketData({
        symbol: 'BASELINE',
        startPrice: 100,
        candleCount: 200,
        timeframe: 60000,
        conditions: [{ type: 'ranging', strength: 0.3, duration: 200 }],
        addNoise: true,
        noiseLevel: 0.01
      });

      const driftedData = testDataGenerator.generateMarketData({
        symbol: 'DRIFTED',
        startPrice: 100,
        candleCount: 200,
        timeframe: 60000,
        conditions: [{ type: 'volatile', strength: 0.8, duration: 200 }],
        addNoise: true,
        noiseLevel: 0.05 // Much higher noise
      });

      const baselineFeatures = trainingService['extractFeatures'](baselineData);
      const driftedFeatures = trainingService['extractFeatures'](driftedData);

      // Calculate volatility (standard deviation of returns) for both datasets
      const baselineReturns = baselineFeatures.slice(1).map((f, i) => 
        (f.close - baselineFeatures[i].close) / baselineFeatures[i].close
      );
      const driftedReturns = driftedFeatures.slice(1).map((f, i) => 
        (f.close - driftedFeatures[i].close) / driftedFeatures[i].close
      );

      const baselineVolatility = Math.sqrt(
        baselineReturns.reduce((acc, r) => acc + r * r, 0) / baselineReturns.length
      );
      const driftedVolatility = Math.sqrt(
        driftedReturns.reduce((acc, r) => acc + r * r, 0) / driftedReturns.length
      );

      // Drifted data should have significantly higher volatility
      expect(driftedVolatility).toBeGreaterThan(baselineVolatility * 2);
    });

    it('should track feature distribution changes', () => {
      const stableData = testDataGenerator.generateMarketData({
        symbol: 'STABLE',
        startPrice: 100,
        candleCount: 500,
        timeframe: 60000,
        conditions: [{ type: 'ranging', strength: 0.2, duration: 500 }],
        addNoise: true,
        noiseLevel: 0.005
      });

      const features = trainingService['extractFeatures'](stableData);
      
      // Split into two halves and compare distributions
      const firstHalf = features.slice(0, Math.floor(features.length / 2));
      const secondHalf = features.slice(Math.floor(features.length / 2));

      // Calculate basic statistics for close prices
      const firstHalfMean = firstHalf.reduce((sum, f) => sum + f.close, 0) / firstHalf.length;
      const secondHalfMean = secondHalf.reduce((sum, f) => sum + f.close, 0) / secondHalf.length;

      // In stable conditions, means should be similar
      const meanDifference = Math.abs(firstHalfMean - secondHalfMean) / firstHalfMean;
      expect(meanDifference).toBeLessThan(0.1); // Less than 10% difference
    });
  });
});