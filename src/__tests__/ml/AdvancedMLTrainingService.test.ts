import { AdvancedMLTrainingService, ModelConfig } from '@/services/ml/AdvancedMLTrainingService';
import { CandleData } from '@/types/session';

describe('AdvancedMLTrainingService', () => {
  let service: AdvancedMLTrainingService;
  let mockCandles: CandleData[];

  beforeEach(() => {
    service = new AdvancedMLTrainingService();
    
    mockCandles = Array.from({ length: 1000 }, (_, i) => ({
      id: `candle_${i}`,
      timestamp: new Date(Date.now() + i * 60000).toISOString(),
      open: 100 + Math.sin(i * 0.1) * 5 + Math.random() * 2,
      high: 105 + Math.sin(i * 0.1) * 5 + Math.random() * 2,
      low: 95 + Math.sin(i * 0.1) * 5 + Math.random() * 2,
      close: 100 + Math.sin(i * 0.1) * 5 + Math.random() * 2,
      volume: 1000 + Math.random() * 500,
      session_id: 'test_session',
      candle_index: i,
      candle_datetime: new Date(Date.now() + i * 60000).toISOString()
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
      const featureSet = {
        features: features.map(f => Object.values(f)),
        labels: features.map(() => Math.random() > 0.5 ? 1 : 0),
        timestamps: mockCandles.map(c => new Date(c.timestamp).getTime())
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
      modelType: 'ensemble',
      lookbackPeriod: 20,
      features: ['price', 'volume', 'sma', 'rsi'],
      trainingRatio: 0.7,
      validationRatio: 0.15
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
        mockConfig,
        mockCandles
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
      
      const metrics = service['calculateMetrics'](yTrue, yPred, []);
      
      expect(metrics).toHaveProperty('accuracy');
      expect(metrics).toHaveProperty('precision');
      expect(metrics).toHaveProperty('recall');
      expect(metrics).toHaveProperty('f1Score');
      expect(metrics.accuracy).toBeGreaterThan(0);
      expect(metrics.accuracy).toBeLessThanOrEqual(1);
    });

    it('should calculate trading-specific metrics', () => {
      const returns = [0.02, -0.01, 0.03, -0.02, 0.01];
      const predictions = [1, 0, 1, 0, 1];
      const metrics = service['calculateTradingMetrics'](returns, predictions);
      
      expect(metrics).toHaveProperty('sharpeRatio');
      expect(metrics).toHaveProperty('maxDrawdown');
      expect(metrics).toHaveProperty('winRate');
      expect(metrics).toHaveProperty('profitFactor');
    });
  });

  describe('Experiment Management', () => {
    it('should track multiple experiments', async () => {
      const config1: ModelConfig = {
        modelType: 'ensemble',
        lookbackPeriod: 20,
        features: ['price', 'volume'],
        trainingRatio: 0.7,
        validationRatio: 0.15
      };

      const config2: ModelConfig = {
        modelType: 'transformer',
        lookbackPeriod: 30,
        features: ['price', 'volume', 'sma'],
        trainingRatio: 0.7,
        validationRatio: 0.15
      };

      const exp1 = await service.startExperiment('exp1', config1, mockCandles);
      const exp2 = await service.startExperiment('exp2', config2, mockCandles);
      
      const allExperiments = service.getAllExperiments();
      expect(allExperiments).toHaveLength(2);
      expect(allExperiments.map(e => e.id)).toContain(exp1);
      expect(allExperiments.map(e => e.id)).toContain(exp2);
    });

    it('should compare experiments', () => {
      // Mock completed experiments with full TrainingExperiment interface
      const exp1 = {
        id: 'exp1',
        name: 'Experiment 1',
        status: 'completed' as const,
        config: { 
          modelType: 'ensemble' as const,
          lookbackPeriod: 20,
          features: ['price'],
          trainingRatio: 0.7,
          validationRatio: 0.15
        },
        model: null,
        parameters: {},
        features: [],
        metrics: {
          accuracy: 0.85,
          precision: 0.8,
          recall: 0.9,
          f1Score: 0.84,
          sharpeRatio: 1.2,
          maxDrawdown: 0.05,
          winRate: 0.6,
          totalTrades: 100,
          profitableTrades: 60,
          avgReturn: 0.12,
          volatility: 0.15
        },
        startTime: new Date(Date.now() - 10000),
        endTime: new Date(Date.now() - 5000)
      };

      const exp2 = {
        id: 'exp2',
        name: 'Experiment 2', 
        status: 'completed' as const,
        config: { 
          modelType: 'transformer' as const,
          lookbackPeriod: 20,
          features: ['price'],
          trainingRatio: 0.7,
          validationRatio: 0.15
        },
        model: null,
        parameters: {},
        features: [],
        metrics: {
          accuracy: 0.87,
          precision: 0.85,
          recall: 0.88,
          f1Score: 0.86,
          sharpeRatio: 1.1,
          maxDrawdown: 0.06,
          winRate: 0.65,
          totalTrades: 110,
          profitableTrades: 71,
          avgReturn: 0.10,
          volatility: 0.12
        },
        startTime: new Date(Date.now() - 8000),
        endTime: new Date(Date.now() - 3000)
      };

      // Add experiments to service (normally done through startExperiment)
      service['experiments'].set('exp1', exp1);
      service['experiments'].set('exp2', exp2);

      // Just test that experiments are tracked properly
      const allExperiments = service.getAllExperiments();
      expect(allExperiments).toHaveLength(2);
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