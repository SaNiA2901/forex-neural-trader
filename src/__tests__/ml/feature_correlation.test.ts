import { CandleData } from '@/types/session';
import { AdvancedMLTrainingService } from '@/services/ml/AdvancedMLTrainingService';
import * as fs from 'fs';
import * as path from 'path';

describe('Feature Correlation Analysis', () => {
  let service: AdvancedMLTrainingService;
  
  beforeEach(() => {
    service = new AdvancedMLTrainingService();
  });

  function calculateCorrelation(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length === 0) return 0;
    
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
    const sumXX = x.reduce((acc, xi) => acc + xi * xi, 0);
    const sumYY = y.reduce((acc, yi) => acc + yi * yi, 0);
    
    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
    
    return denominator === 0 ? 0 : numerator / denominator;
  }

  function generateRealisticMarketData(count: number): CandleData[] {
    const data: CandleData[] = [];
    let price = 100;
    
    for (let i = 0; i < count; i++) {
      // Market microstructure: trending with mean reversion
      const trendComponent = Math.sin(i * 0.02) * 0.5;
      const momentumComponent = Math.sin(i * 0.1) * 0.3;
      const noiseComponent = (Math.random() - 0.5) * 0.8;
      
      const priceChange = trendComponent + momentumComponent + noiseComponent;
      price += priceChange;
      
      const open = price + (Math.random() - 0.5) * 0.5;
      const close = price + (Math.random() - 0.5) * 0.5;
      const high = Math.max(open, close) + Math.random() * 0.8;
      const low = Math.min(open, close) - Math.random() * 0.8;
      const volume = 1000 + Math.random() * 2000 + (Math.abs(priceChange) * 1000); // Volume increases with volatility
      
      data.push({
        id: `candle_${i}`,
        timestamp: new Date(Date.now() + i * 60000).toISOString(),
        open,
        high,
        low,
        close,
        volume,
        session_id: 'test_session',
        candle_index: i,
        candle_datetime: new Date(Date.now() + i * 60000).toISOString()
      } as CandleData);
    }
    
    return data;
  }

  function extractPreviousDirection(candles: CandleData[], index: number): number {
    if (index === 0) return 0;
    const prevCandle = candles[index - 1];
    return prevCandle.close > prevCandle.open ? 1 : 0;
  }

  it('should analyze feature correlation with target and previous direction', async () => {
    const candles = generateRealisticMarketData(500);
    
    const config = {
      modelType: 'regression' as const,
      lookbackPeriod: 20,
      features: ['price', 'volume', 'technical'],
      trainingRatio: 0.7,
      validationRatio: 0.15
    };

    const features = await service.engineerFeatures(candles, config);
    
    // Extract target and previous direction for each feature vector
    const targets: number[] = [];
    const previousDirections: number[] = [];
    const featureMatrix: number[][] = [];
    
    features.forEach((fv, index) => {
      if (fv.label !== undefined) {
        targets.push(fv.label);
        
        // Find the candle index for this feature vector
        const candleIndex = candles.findIndex(c => Number(c.timestamp) === fv.timestamp);
        if (candleIndex > 0) {
          previousDirections.push(extractPreviousDirection(candles, candleIndex));
          featureMatrix.push(fv.features);
        }
      }
    });

    expect(targets.length).toBeGreaterThan(50); // Ensure we have enough data
    expect(targets.length).toBe(previousDirections.length);
    expect(targets.length).toBe(featureMatrix.length);

    // Calculate correlations
    const correlationReport: any = {
      timestamp: new Date().toISOString(),
      totalSamples: targets.length,
      targetMean: targets.reduce((a, b) => a + b, 0) / targets.length,
      featureCorrelations: [] as any[],
      metaAnalysis: {} as any
    };

    // Analyze each feature
    const featureNames = [
      // OHLCV features (10)
      'avg_return', 'volatility', 'skewness', 'kurtosis', 'max_return', 'min_return',
      'avg_volume', 'volume_volatility', 'total_return', 'sharpe_ratio',
      // Technical features (variable count)
      'rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position', 'bb_bandwidth', 'bb_squeeze',
      'price_vs_sma5', 'price_vs_sma20', 'sma5_vs_sma20', 'ema12_vs_ema26', 'stoch_k', 'stoch_d', 'atr', 'price_vs_vwap',
      // Pattern features (8)
      'body_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio', 'bullish_candle', 'body_shadow_ratio',
      'position_in_range', 'distance_to_resistance', 'distance_to_support',
      // Volume features (3)
      'volume_trend', 'current_vs_avg_volume', 'price_volume_correlation',
      // Time features (4)
      'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
    ];

    for (let i = 0; i < Math.min(featureNames.length, featureMatrix[0].length); i++) {
      const featureValues = featureMatrix.map(row => row[i]);
      
      // Filter out NaN values
      const validIndices = featureValues
        .map((val, idx) => ({ val, idx }))
        .filter(({ val }) => !isNaN(val) && isFinite(val))
        .map(({ idx }) => idx);
      
      if (validIndices.length < 10) continue; // Skip features with too few valid values
      
      const validFeatures = validIndices.map(idx => featureValues[idx]);
      const validTargets = validIndices.map(idx => targets[idx]);
      const validPrevDirs = validIndices.map(idx => previousDirections[idx]);
      
      const targetCorrelation = calculateCorrelation(validFeatures, validTargets);
      const prevDirectionCorrelation = calculateCorrelation(validFeatures, validPrevDirs);
      
      correlationReport.featureCorrelations.push({
        featureName: featureNames[i] || `feature_${i}`,
        featureIndex: i,
        validSamples: validIndices.length,
        mean: validFeatures.reduce((a, b) => a + b, 0) / validFeatures.length,
        std: Math.sqrt(validFeatures.reduce((acc, val) => {
          const mean = validFeatures.reduce((a, b) => a + b, 0) / validFeatures.length;
          return acc + Math.pow(val - mean, 2);
        }, 0) / validFeatures.length),
        targetCorrelation,
        previousDirectionCorrelation: prevDirectionCorrelation,
        // Flag potential data leakage
        suspiciouslyHighCorrelation: Math.abs(targetCorrelation) > 0.8,
        strongPreviousDirectionCorrelation: Math.abs(prevDirectionCorrelation) > 0.5
      });
    }

    // Meta analysis
    const targetCorrelations = correlationReport.featureCorrelations.map((f: any) => f.targetCorrelation);
    const prevDirCorrelations = correlationReport.featureCorrelations.map((f: any) => f.previousDirectionCorrelation);
    
    correlationReport.metaAnalysis = {
      avgTargetCorrelation: targetCorrelations.reduce((a, b) => a + b, 0) / targetCorrelations.length,
      maxTargetCorrelation: Math.max(...targetCorrelations.map(Math.abs)),
      avgPrevDirectionCorrelation: prevDirCorrelations.reduce((a, b) => a + b, 0) / prevDirCorrelations.length,
      maxPrevDirectionCorrelation: Math.max(...prevDirCorrelations.map(Math.abs)),
      suspiciousFeatures: correlationReport.featureCorrelations.filter((f: any) => f.suspiciouslyHighCorrelation).length,
      strongPrevDirectionFeatures: correlationReport.featureCorrelations.filter((f: any) => f.strongPreviousDirectionCorrelation).length
    };

    // Sort by absolute target correlation (most predictive features first)
    correlationReport.featureCorrelations.sort((a: any, b: any) => 
      Math.abs(b.targetCorrelation) - Math.abs(a.targetCorrelation)
    );

    // Create reports directory if it doesn't exist
    const reportsDir = path.join(process.cwd(), 'reports');
    if (!fs.existsSync(reportsDir)) {
      fs.mkdirSync(reportsDir, { recursive: true });
    }
    
    // Save correlation report
    const reportPath = path.join(reportsDir, 'feature_correlation.json');
    fs.writeFileSync(reportPath, JSON.stringify(correlationReport, null, 2));
    
    console.log('Feature Correlation Analysis:');
    console.log(`Total samples: ${correlationReport.totalSamples}`);
    console.log(`Average target correlation: ${correlationReport.metaAnalysis.avgTargetCorrelation.toFixed(4)}`);
    console.log(`Max target correlation: ${correlationReport.metaAnalysis.maxTargetCorrelation.toFixed(4)}`);
    console.log(`Average previous direction correlation: ${correlationReport.metaAnalysis.avgPrevDirectionCorrelation.toFixed(4)}`);
    console.log(`Suspicious features (|corr| > 0.8): ${correlationReport.metaAnalysis.suspiciousFeatures}`);
    console.log(`Strong prev direction features (|corr| > 0.5): ${correlationReport.metaAnalysis.strongPrevDirectionFeatures}`);
    
    // Print top 5 most predictive features
    console.log('\nTop 5 predictive features:');
    correlationReport.featureCorrelations.slice(0, 5).forEach((f: any) => {
      console.log(`  ${f.featureName}: target_corr=${f.targetCorrelation.toFixed(4)}, prev_dir_corr=${f.previousDirectionCorrelation.toFixed(4)}`);
    });

    // Assertions
    expect(correlationReport.totalSamples).toBeGreaterThan(50);
    expect(correlationReport.featureCorrelations.length).toBeGreaterThan(10);
    
    // Check for data leakage warning
    if (correlationReport.metaAnalysis.suspiciousFeatures > 0) {
      console.warn(`⚠️  Found ${correlationReport.metaAnalysis.suspiciousFeatures} features with suspiciously high correlation (>0.8) - possible data leakage!`);
    }
    
    // Check for strong previous direction correlation (potential bug indicator)
    if (correlationReport.metaAnalysis.strongPrevDirectionFeatures > 3) {
      console.warn(`⚠️  Found ${correlationReport.metaAnalysis.strongPrevDirectionFeatures} features strongly correlated with previous direction - model might be copying last signal!`);
    }
  });

  it('should detect if features are overly correlated with previous direction', async () => {
    // Create specific test case where features accidentally leak previous direction
    const candles: CandleData[] = [];
    
    for (let i = 0; i < 200; i++) {
      const prevDirection = i > 0 ? (candles[i-1].close > candles[i-1].open ? 1 : -1) : 1;
      
      // Intentionally create features that correlate with previous direction
      const basePrice = 100 + i * 0.1;
      const leakyNoise = prevDirection * 2; // Features will be biased by previous direction
      
      const open = basePrice + (Math.random() - 0.5) + leakyNoise * 0.5;
      const close = basePrice + (Math.random() - 0.5) + leakyNoise * 0.3;
      const high = Math.max(open, close) + Math.random() * 2;
      const low = Math.min(open, close) - Math.random() * 2;
      const volume = 1000 + Math.random() * 500 + Math.abs(leakyNoise) * 200; // Volume biased too
      
      candles.push({
        id: `candle_${i}`,
        timestamp: new Date(Date.now() + i * 60000).toISOString(),
        open,
        high,
        low,
        close,
        volume,
        session_id: 'test_session',
        candle_index: i,
        candle_datetime: new Date(Date.now() + i * 60000).toISOString()
      } as CandleData);
    }

    const config = {
      modelType: 'regression' as const,
      lookbackPeriod: 20,
      features: ['price', 'volume'],
      trainingRatio: 0.7,
      validationRatio: 0.15
    };

    const features = await service.engineerFeatures(candles, config);
    
    let highPrevDirCorrelations = 0;
    
    for (let i = 0; i < Math.min(10, features[0].features.length); i++) {
      const featureValues = features.map(fv => fv.features[i]).filter(val => !isNaN(val) && isFinite(val));
      const prevDirections = features.map((fv, idx) => {
        const candleIndex = candles.findIndex(c => Number(c.timestamp) === fv.timestamp);
        return extractPreviousDirection(candles, candleIndex);
      });
      
      const correlation = calculateCorrelation(featureValues, prevDirections);
      
      if (Math.abs(correlation) > 0.5) {
        highPrevDirCorrelations++;
      }
    }
    
    // This leaky dataset should trigger the warning
    expect(highPrevDirCorrelations).toBeGreaterThan(2);
    console.log(`Detected ${highPrevDirCorrelations} features with high previous direction correlation - this is expected for this leaky test data`);
  });
});