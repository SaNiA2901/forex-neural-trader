import { CandleData } from '@/types/session';

// Type for ML pipeline - using any to avoid complex type issues
type MLPipeline = any;

export interface TrainingExperiment {
  id: string;
  name: string;
  model: string;
  parameters: Record<string, any>;
  metrics: ModelMetrics;
  status: 'running' | 'completed' | 'failed';
  startTime: Date;
  endTime?: Date;
  features: string[];
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  totalTrades: number;
  profitableTrades: number;
  avgReturn: number;
  volatility: number;
}

export interface ModelConfig {
  modelType: 'transformer' | 'regression' | 'ensemble';
  lookbackPeriod: number;
  features: string[];
  trainingRatio: number;
  validationRatio: number;
  epochs?: number;
  batchSize?: number;
  learningRate?: number;
}

export interface FeatureVector {
  timestamp: number;
  features: number[];
  label?: number;
  metadata?: Record<string, any>;
}

export class AdvancedMLTrainingService {
  private experiments: Map<string, TrainingExperiment> = new Map();
  private models: Map<string, any> = new Map();
  private featurePipeline: MLPipeline | null = null;

  async initialize() {
    try {
      // Initialize feature extraction pipeline for embeddings
      const { pipeline } = await import('@huggingface/transformers');
      this.featurePipeline = await pipeline(
        'feature-extraction',
        'mixedbread-ai/mxbai-embed-xsmall-v1',
        { device: 'webgpu' }
      );
    } catch (error) {
      console.warn('ML pipeline initialization failed, using fallback:', error);
      this.featurePipeline = null;
    }
  }

  /**
   * Advanced feature engineering with technical indicators and price patterns
   */
  async engineerFeatures(candles: CandleData[], config: ModelConfig): Promise<FeatureVector[]> {
    if (candles.length < config.lookbackPeriod + 50) {
      throw new Error(`Insufficient data: need at least ${config.lookbackPeriod + 50} candles`);
    }

    const features: FeatureVector[] = [];
    
    for (let i = config.lookbackPeriod; i < candles.length - 1; i++) {
      const lookbackCandles = candles.slice(i - config.lookbackPeriod, i);
      const currentCandle = candles[i];
      const nextCandle = candles[i + 1];

      // Basic OHLCV features
      const ohlcvFeatures = this.extractOHLCVFeatures(lookbackCandles);
      
      // Technical indicator features
      const technicalFeatures = this.extractTechnicalFeatures(lookbackCandles, i);
      
      // Price pattern features
      const patternFeatures = this.extractPatternFeatures(lookbackCandles);
      
      // Volume profile features
      const volumeFeatures = this.extractVolumeFeatures(lookbackCandles);
      
      // Time-based features
      const timeFeatures = this.extractTimeFeatures(Number(currentCandle.timestamp));

      // Combine all features
      const allFeatures = [
        ...ohlcvFeatures,
        ...technicalFeatures,
        ...patternFeatures,
        ...volumeFeatures,
        ...timeFeatures
      ];

      // Create label (1 for up, 0 for down)
      const label = nextCandle.close > currentCandle.close ? 1 : 0;

      features.push({
        timestamp: Number(currentCandle.timestamp),
        features: allFeatures,
        label,
        metadata: {
          open: currentCandle.open,
          high: currentCandle.high,
          low: currentCandle.low,
          close: currentCandle.close,
          volume: currentCandle.volume
        }
      });
    }

    return features;
  }

  private extractOHLCVFeatures(candles: CandleData[]): number[] {
    const features: number[] = [];
    
    // Price statistics
    const closes = candles.map(c => c.close);
    const volumes = candles.map(c => c.volume);
    
    // Returns and volatility
    const returns = closes.slice(1).map((close, i) => (close - closes[i]) / closes[i]);
    
    features.push(
      this.mean(returns), // Average return
      this.standardDeviation(returns), // Volatility
      this.skewness(returns), // Skewness
      this.kurtosis(returns), // Kurtosis
      Math.max(...returns), // Max return
      Math.min(...returns), // Min return
      this.mean(volumes), // Average volume
      this.standardDeviation(volumes), // Volume volatility
      closes[closes.length - 1] / closes[0] - 1, // Total return
      this.sharpeRatio(returns) // Sharpe ratio
    );

    return features;
  }

  private extractTechnicalFeatures(candles: CandleData[], currentIndex: number): number[] {
    const features: number[] = [];
    const closes = candles.map(c => c.close);
    const highs = candles.map(c => c.high);
    const lows = candles.map(c => c.low);
    const volumes = candles.map(c => c.volume);

    // RSI
    const rsi = this.calculateRSI(closes, 14);
    features.push(rsi);

    // MACD
    const macd = this.calculateMACD(closes);
    features.push(macd.macd, macd.signal, macd.histogram);

    // Bollinger Bands
    const bb = this.calculateBollingerBands(closes, 20, 2);
    features.push(bb.position, bb.bandwidth, bb.squeeze);

    // Moving averages
    const sma5 = this.sma(closes, 5);
    const sma20 = this.sma(closes, 20);
    const ema12 = this.ema(closes, 12);
    const ema26 = this.ema(closes, 26);
    
    features.push(
      closes[closes.length - 1] / sma5 - 1, // Price vs SMA5
      closes[closes.length - 1] / sma20 - 1, // Price vs SMA20
      sma5 / sma20 - 1, // SMA5 vs SMA20
      ema12 / ema26 - 1 // EMA12 vs EMA26
    );

    // Stochastic
    const stoch = this.calculateStochastic(highs, lows, closes, 14);
    features.push(stoch.k, stoch.d);

    // ATR (Average True Range)
    const atr = this.calculateATR(highs, lows, closes, 14);
    features.push(atr);

    // VWAP
    const vwap = this.calculateVWAP(candles);
    features.push(closes[closes.length - 1] / vwap - 1);

    return features;
  }

  private extractPatternFeatures(candles: CandleData[]): number[] {
    const features: number[] = [];
    
    // Candlestick patterns (simplified)
    const lastCandle = candles[candles.length - 1];
    const bodySize = Math.abs(lastCandle.close - lastCandle.open);
    const upperShadow = lastCandle.high - Math.max(lastCandle.open, lastCandle.close);
    const lowerShadow = Math.min(lastCandle.open, lastCandle.close) - lastCandle.low;
    const range = lastCandle.high - lastCandle.low;

    features.push(
      bodySize / range, // Body ratio
      upperShadow / range, // Upper shadow ratio
      lowerShadow / range, // Lower shadow ratio
      lastCandle.close > lastCandle.open ? 1 : 0, // Bullish candle
      bodySize / ((upperShadow + lowerShadow) || 1) // Body to shadow ratio
    );

    // Support and resistance levels
    const highs = candles.map(c => c.high);
    const lows = candles.map(c => c.low);
    const resistance = Math.max(...highs);
    const support = Math.min(...lows);
    
    features.push(
      (lastCandle.close - support) / (resistance - support), // Position in range
      (resistance - lastCandle.close) / resistance, // Distance to resistance
      (lastCandle.close - support) / support // Distance to support
    );

    return features;
  }

  private extractVolumeFeatures(candles: CandleData[]): number[] {
    const features: number[] = [];
    const volumes = candles.map(c => c.volume);
    const prices = candles.map(c => c.close);
    
    // Volume trend
    const recentVolumes = volumes.slice(-5);
    const olderVolumes = volumes.slice(-10, -5);
    const volumeTrend = this.mean(recentVolumes) / this.mean(olderVolumes) - 1;
    
    features.push(
      volumeTrend,
      volumes[volumes.length - 1] / this.mean(volumes) - 1, // Current vs average volume
      this.correlation(prices, volumes) // Price-volume correlation
    );

    return features;
  }

  private extractTimeFeatures(timestamp: number): number[] {
    const date = new Date(timestamp);
    const hour = date.getHours();
    const dayOfWeek = date.getDay();
    
    // Convert to cyclical features
    return [
      Math.sin(2 * Math.PI * hour / 24), // Hour sine
      Math.cos(2 * Math.PI * hour / 24), // Hour cosine
      Math.sin(2 * Math.PI * dayOfWeek / 7), // Day sine
      Math.cos(2 * Math.PI * dayOfWeek / 7) // Day cosine
    ];
  }

  /**
   * Time-series aware train/validation split
   */
  splitTimeSeriesData(features: FeatureVector[], config: ModelConfig) {
    const totalSamples = features.length;
    const trainSize = Math.floor(totalSamples * config.trainingRatio);
    const validSize = Math.floor(totalSamples * config.validationRatio);
    
    return {
      train: features.slice(0, trainSize),
      validation: features.slice(trainSize, trainSize + validSize),
      test: features.slice(trainSize + validSize)
    };
  }

  /**
   * Walk-forward validation for time series
   */
  async walkForwardValidation(
    features: FeatureVector[],
    config: ModelConfig,
    windowSize: number = 1000
  ): Promise<ModelMetrics> {
    const results: number[] = [];
    const predictions: number[] = [];
    const actuals: number[] = [];

    for (let i = windowSize; i < features.length - windowSize; i += windowSize) {
      const trainData = features.slice(Math.max(0, i - windowSize * 2), i);
      const testData = features.slice(i, i + windowSize);

      // Simple logistic regression model
      const model = await this.trainSimpleModel(trainData);
      
      for (const sample of testData) {
        if (sample.label !== undefined) {
          const prediction = this.predictSimple(model, sample.features);
          predictions.push(prediction);
          actuals.push(sample.label);
        }
      }
    }

    return this.calculateMetrics(predictions, actuals, features);
  }

  private async trainSimpleModel(data: FeatureVector[]) {
    // Simple logistic regression implementation
    const features = data.map(d => d.features);
    const labels = data.map(d => d.label || 0);
    
    const weights = new Array(features[0].length).fill(0);
    const learningRate = 0.01;
    const epochs = 100;

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let i = 0; i < features.length; i++) {
        const prediction = this.sigmoid(this.dotProduct(weights, features[i]));
        const error = labels[i] - prediction;
        
        for (let j = 0; j < weights.length; j++) {
          weights[j] += learningRate * error * features[i][j];
        }
      }
    }

    return { weights };
  }

  private predictSimple(model: any, features: number[]): number {
    const logit = this.dotProduct(model.weights, features);
    return this.sigmoid(logit) > 0.5 ? 1 : 0;
  }

  private calculateMetrics(predictions: number[], actuals: number[], features: FeatureVector[]): ModelMetrics {
    const correct = predictions.filter((pred, i) => pred === actuals[i]).length;
    const accuracy = correct / predictions.length;

    // Confusion matrix
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (let i = 0; i < predictions.length; i++) {
      if (actuals[i] === 1 && predictions[i] === 1) tp++;
      else if (actuals[i] === 0 && predictions[i] === 1) fp++;
      else if (actuals[i] === 0 && predictions[i] === 0) tn++;
      else if (actuals[i] === 1 && predictions[i] === 0) fn++;
    }

    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1Score = 2 * (precision * recall) / (precision + recall) || 0;

    // Trading metrics
    const returns = this.calculateStrategyReturns(predictions, features);
    const sharpeRatio = this.sharpeRatio(returns);
    const maxDrawdown = this.calculateMaxDrawdown(returns);
    const winRate = tp / (tp + fn) || 0;

    return {
      accuracy,
      precision,
      recall,
      f1Score,
      sharpeRatio,
      maxDrawdown,
      winRate,
      totalTrades: predictions.filter(p => p === 1).length,
      profitableTrades: tp,
      avgReturn: this.mean(returns),
      volatility: this.standardDeviation(returns)
    };
  }

  private calculateStrategyReturns(predictions: number[], features: FeatureVector[]): number[] {
    const returns: number[] = [];
    
    for (let i = 0; i < predictions.length - 1; i++) {
      if (predictions[i] === 1 && features[i].metadata && features[i + 1].metadata) {
        const entry = features[i].metadata!.close;
        const exit = features[i + 1].metadata!.close;
        returns.push((exit - entry) / entry);
      }
    }

    return returns;
  }

  async startExperiment(name: string, config: ModelConfig, candles: CandleData[]): Promise<string> {
    const experimentId = `exp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const experiment: TrainingExperiment = {
      id: experimentId,
      name,
      model: config.modelType,
      parameters: config,
      metrics: {
        accuracy: 0, precision: 0, recall: 0, f1Score: 0,
        sharpeRatio: 0, maxDrawdown: 0, winRate: 0, totalTrades: 0,
        profitableTrades: 0, avgReturn: 0, volatility: 0
      },
      status: 'running',
      startTime: new Date(),
      features: config.features
    };

    this.experiments.set(experimentId, experiment);

    // Run training in background
    this.runExperiment(experimentId, config, candles);

    return experimentId;
  }

  private async runExperiment(experimentId: string, config: ModelConfig, candles: CandleData[]) {
    try {
      const experiment = this.experiments.get(experimentId)!;
      
      // Engineer features
      const features = await this.engineerFeatures(candles, config);
      
      // Perform walk-forward validation
      const metrics = await this.walkForwardValidation(features, config);
      
      experiment.metrics = metrics;
      experiment.status = 'completed';
      experiment.endTime = new Date();
      
      console.log(`Experiment ${experimentId} completed:`, metrics);
      
    } catch (error) {
      const experiment = this.experiments.get(experimentId)!;
      experiment.status = 'failed';
      experiment.endTime = new Date();
      console.error(`Experiment ${experimentId} failed:`, error);
    }
  }

  getExperiment(id: string): TrainingExperiment | undefined {
    return this.experiments.get(id);
  }

  getAllExperiments(): TrainingExperiment[] {
    return Array.from(this.experiments.values());
  }

  getExperimentsByStatus(status: TrainingExperiment['status']): TrainingExperiment[] {
    return this.getAllExperiments().filter(exp => exp.status === status);
  }

  // Utility functions
  private mean(arr: number[]): number {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  private standardDeviation(arr: number[]): number {
    const mean = this.mean(arr);
    const variance = arr.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / arr.length;
    return Math.sqrt(variance);
  }

  private skewness(arr: number[]): number {
    const mean = this.mean(arr);
    const std = this.standardDeviation(arr);
    return arr.reduce((acc, val) => acc + Math.pow((val - mean) / std, 3), 0) / arr.length;
  }

  private kurtosis(arr: number[]): number {
    const mean = this.mean(arr);
    const std = this.standardDeviation(arr);
    return arr.reduce((acc, val) => acc + Math.pow((val - mean) / std, 4), 0) / arr.length - 3;
  }

  private sharpeRatio(returns: number[]): number {
    if (returns.length === 0) return 0;
    const avgReturn = this.mean(returns);
    const std = this.standardDeviation(returns);
    return std === 0 ? 0 : avgReturn / std * Math.sqrt(252); // Annualized
  }

  private calculateMaxDrawdown(returns: number[]): number {
    let maxDrawdown = 0;
    let peak = 0;
    let cumReturn = 0;

    for (const ret of returns) {
      cumReturn = (1 + cumReturn) * (1 + ret) - 1;
      peak = Math.max(peak, cumReturn);
      const drawdown = (peak - cumReturn) / (1 + peak);
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }

    return maxDrawdown;
  }

  private correlation(x: number[], y: number[]): number {
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((acc, val, i) => acc + val * y[i], 0);
    const sumXX = x.reduce((acc, val) => acc + val * val, 0);
    const sumYY = y.reduce((acc, val) => acc + val * val, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));

    return denominator === 0 ? 0 : numerator / denominator;
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  private dotProduct(a: number[], b: number[]): number {
    return a.reduce((sum, val, i) => sum + val * b[i], 0);
  }

  // Technical indicator calculations (simplified versions)
  private calculateRSI(prices: number[], period: number): number {
    if (prices.length < period + 1) return 50;

    let gains = 0, losses = 0;
    for (let i = prices.length - period; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      if (change > 0) gains += change;
      else losses -= change;
    }

    const avgGain = gains / period;
    const avgLoss = losses / period;
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  private calculateMACD(prices: number[]): { macd: number; signal: number; histogram: number } {
    const ema12 = this.ema(prices, 12);
    const ema26 = this.ema(prices, 26);
    const macd = ema12 - ema26;
    const signal = macd; // Simplified
    const histogram = macd - signal;

    return { macd, signal, histogram };
  }

  private calculateBollingerBands(prices: number[], period: number, stdDev: number) {
    const sma = this.sma(prices, period);
    const std = this.standardDeviation(prices.slice(-period));
    const upper = sma + stdDev * std;
    const lower = sma - stdDev * std;
    const current = prices[prices.length - 1];
    
    return {
      position: (current - lower) / (upper - lower),
      bandwidth: (upper - lower) / sma,
      squeeze: std < this.mean(prices.slice(-period * 2, -period)) * 0.1 ? 1 : 0
    };
  }

  private calculateStochastic(highs: number[], lows: number[], closes: number[], period: number) {
    const recentHighs = highs.slice(-period);
    const recentLows = lows.slice(-period);
    const currentClose = closes[closes.length - 1];
    
    const highestHigh = Math.max(...recentHighs);
    const lowestLow = Math.min(...recentLows);
    
    const k = ((currentClose - lowestLow) / (highestHigh - lowestLow)) * 100;
    const d = k; // Simplified
    
    return { k, d };
  }

  private calculateATR(highs: number[], lows: number[], closes: number[], period: number): number {
    const trs: number[] = [];
    
    for (let i = 1; i < Math.min(highs.length, period + 1); i++) {
      const tr = Math.max(
        highs[i] - lows[i],
        Math.abs(highs[i] - closes[i - 1]),
        Math.abs(lows[i] - closes[i - 1])
      );
      trs.push(tr);
    }
    
    return this.mean(trs);
  }

  private calculateVWAP(candles: CandleData[]): number {
    let totalVolume = 0;
    let totalVolumePrice = 0;
    
    for (const candle of candles) {
      const typical = (candle.high + candle.low + candle.close) / 3;
      totalVolumePrice += typical * candle.volume;
      totalVolume += candle.volume;
    }
    
    return totalVolume === 0 ? 0 : totalVolumePrice / totalVolume;
  }

  private sma(prices: number[], period: number): number {
    return this.mean(prices.slice(-period));
  }

  private ema(prices: number[], period: number): number {
    if (prices.length === 0) return 0;
    
    const multiplier = 2 / (period + 1);
    let ema = prices[0];
    
    for (let i = 1; i < prices.length; i++) {
      ema = (prices[i] - ema) * multiplier + ema;
    }
    
    return ema;
  }
}

export const advancedMLTrainingService = new AdvancedMLTrainingService();