import { AdvancedMLTrainingService, ModelConfig } from '@/services/ml/AdvancedMLTrainingService';
import { CandleData } from '@/types/session';

describe('AdvancedMLTrainingService', () => {
  let service: AdvancedMLTrainingService;
  let mockCandles: CandleData[];

  beforeEach(() => {
    service = new AdvancedMLTrainingService();
    
    mockCandles = Array.from({ length: 1000 }, (_, i) => ({
      id: `candle_${i}`,
      timestamp: Date.now() + i * 60000,
      open: 100 + Math.sin(i * 0.1) * 5 + Math.random() * 2,
      high: 105 + Math.sin(i * 0.1) * 5 + Math.random() * 2,
      low: 95 + Math.sin(i * 0.1) * 5 + Math.random() * 2,
      close: 100 + Math.sin(i * 0.1) * 5 + Math.random() * 2,
      volume: 1000 + Math.random() * 500,
      sessionId: 'test_session'
    } as CandleData));
  });

  describe('Feature Engineering', () => {
    it('should extract basic features', async () => {
      const features = await service['extractFeatures'](mockCandles);
      
      expect(features.length).toBeGreaterThan(0);
      expect(features[0]).toHaveProperty('open');
      expect(features[0]).toHaveProperty('high');
      expect(features[0]).toHaveProperty('low');
      expect(features[0]).toHaveProperty('close');
      expect(features[0]).toHaveProperty('volume');
    });

    it('should calculate technical indicators', async () => {
      const features = await service['extractFeatures'](mockCandles);
      
      // Check if technical indicators are included
      expect(features[0]).toHaveProperty('sma_20');
      expect(features[0]).toHaveProperty('rsi_14');
      expect(features[0]).toHaveProperty('bb_upper');
      expect(features[0]).toHaveProperty('macd_line');
    });

    it('should handle missing data gracefully', async () => {
      const incompleteCandles = mockCandles.slice(0, 5); // Too few for some indicators
      const features = await service['extractFeatures'](incompleteCandles);
      
      expect(features.length).toBeGreaterThan(0);
      // Should still return features, possibly with NaN values that get handled
    });
  });

  describe('Data Preparation', () => {
    it('should create time series splits correctly', async () => {
      const features = await service['extractFeatures'](mockCandles);
      const featureSet: FeatureSet = {
        features: features.map(f => Object.values(f)),
        labels: features.map(() => Math.random() > 0.5 ? 1 : 0),
        timestamps: mockCandles.map(c => c.timestamp)
      };

      const splits = await service['createTimeSeriesSplits'](featureSet, 5);
      
      expect(splits).toHaveLength(5);
      expect(splits[0]).toHaveProperty('trainFeatures');
      expect(splits[0]).toHaveProperty('trainLabels');
      expect(splits[0]).toHaveProperty('validationFeatures');
      expect(splits[0]).toHaveProperty('validationLabels');
    });

    it('should normalize features', async () => {
      const features = Array.from({ length: 100 }, () => 
        Array.from({ length: 10 }, () => Math.random() * 100)
      );

      const { normalizedFeatures, scaler } = await service['normalizeFeatures'](features);
      
      expect(normalizedFeatures).toHaveLength(features.length);
      expect(scaler).toHaveProperty('mean');
      expect(scaler).toHaveProperty('std');
      
      // Check if normalization is applied (values should be roughly centered around 0)
      const flatNormalized = normalizedFeatures.flat();
      const mean = flatNormalized.reduce((a, b) => a + b, 0) / flatNormalized.length;
      expect(Math.abs(mean)).toBeLessThan(0.1); // Should be close to 0
    });
  });

  describe('Model Training', () => {
    const mockConfig: ModelConfig = {
      modelType: 'xgboost',
      hyperparameters: {
        n_estimators: 100,
        max_depth: 6,
        learning_rate: 0.1
      },
      crossValidationFolds: 3,
      testSize: 0.2,
      randomState: 42
    };

    it('should validate model configuration', () => {
      const isValid = service['validateModelConfig'](mockConfig);
      expect(isValid).toBe(true);
    });

    it('should reject invalid configuration', () => {
      const invalidConfig = { ...mockConfig, hyperparameters: {} };
      const isValid = service['validateModelConfig'](invalidConfig);
      expect(isValid).toBe(false);
    });

    it('should start training experiment', async () => {
      const experimentId = await service.startExperiment(
        'test-experiment',
        mockCandles,
        mockConfig
      );
      
      expect(experimentId).toBeDefined();
      expect(typeof experimentId).toBe('string');
      
      const experiment = service.getExperiment(experimentId);
      expect(experiment).toBeDefined();
      expect(experiment?.name).toBe('test-experiment');
      expect(experiment?.status).toBe('running');
    });
  });

  describe('Model Evaluation', () => {
    it('should calculate performance metrics', () => {
      const yTrue = [1, 0, 1, 1, 0, 0, 1, 0];
      const yPred = [1, 0, 0, 1, 0, 1, 1, 0];
      
      const metrics = service['calculateMetrics'](yTrue, yPred);
      
      expect(metrics).toHaveProperty('accuracy');
      expect(metrics).toHaveProperty('precision');
      expect(metrics).toHaveProperty('recall');
      expect(metrics).toHaveProperty('f1Score');
      expect(metrics.accuracy).toBeGreaterThan(0);
      expect(metrics.accuracy).toBeLessThanOrEqual(1);
    });

    it('should calculate trading-specific metrics', () => {
      const returns = [0.02, -0.01, 0.03, -0.02, 0.01];
      const metrics = service['calculateTradingMetrics'](returns);
      
      expect(metrics).toHaveProperty('sharpeRatio');
      expect(metrics).toHaveProperty('maxDrawdown');
      expect(metrics).toHaveProperty('winRate');
      expect(metrics).toHaveProperty('profitFactor');
    });
  });

  describe('Experiment Management', () => {
    it('should track multiple experiments', async () => {
      const config1: ModelConfig = {
        modelType: 'random_forest',
        hyperparameters: { n_estimators: 50 },
        crossValidationFolds: 3,
        testSize: 0.2,
        randomState: 42
      };

      const config2: ModelConfig = {
        modelType: 'xgboost',
        hyperparameters: { n_estimators: 100 },
        crossValidationFolds: 3,
        testSize: 0.2,
        randomState: 42
      };

      const exp1 = await service.startExperiment('exp1', mockCandles, config1);
      const exp2 = await service.startExperiment('exp2', mockCandles, config2);
      
      const allExperiments = service.getAllExperiments();
      expect(allExperiments).toHaveLength(2);
      expect(allExperiments.map(e => e.id)).toContain(exp1);
      expect(allExperiments.map(e => e.id)).toContain(exp2);
    });

    it('should compare experiments', () => {
      // Mock completed experiments
      const exp1 = {
        id: 'exp1',
        name: 'Experiment 1',
        status: 'completed' as const,
        config: { modelType: 'random_forest' } as ModelConfig,
        metrics: { accuracy: 0.85, sharpeRatio: 1.2 },
        startTime: Date.now() - 10000,
        endTime: Date.now() - 5000
      };

      const exp2 = {
        id: 'exp2',
        name: 'Experiment 2', 
        status: 'completed' as const,
        config: { modelType: 'xgboost' } as ModelConfig,
        metrics: { accuracy: 0.87, sharpeRatio: 1.1 },
        startTime: Date.now() - 8000,
        endTime: Date.now() - 3000
      };

      // Add experiments to service (normally done through startExperiment)
      service['experiments'].set('exp1', exp1);
      service['experiments'].set('exp2', exp2);

      const comparison = service.compareExperiments(['exp1', 'exp2']);
      
      expect(comparison).toHaveProperty('experiments');
      expect(comparison.experiments).toHaveLength(2);
      expect(comparison).toHaveProperty('bestAccuracy');
      expect(comparison).toHaveProperty('bestSharpeRatio');
    });
  });

  describe('Model Persistence', () => {
    it('should generate model metadata', () => {
      const mockScaler = { mean: [1, 2, 3], std: [0.5, 1.0, 1.5] };
      const metadata = service['generateModelMetadata']('test-model', mockScaler);
      
      expect(metadata).toHaveProperty('name', 'test-model');
      expect(metadata).toHaveProperty('version');
      expect(metadata).toHaveProperty('scalerParams');
      expect(metadata.scalerParams).toEqual(mockScaler);
      expect(metadata).toHaveProperty('createdAt');
    });
  });
});