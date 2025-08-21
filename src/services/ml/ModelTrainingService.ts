import { CandleData } from '@/types/session';

export interface TrainingData {
  features: number[][];
  labels: number[];
  weights?: number[];
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: number[][];
}

export interface TrainingConfig {
  epochs: number;
  learningRate: number;
  batchSize: number;
  validationSplit: number;
  earlyStoppingPatience: number;
}

export class ModelTrainingService {
  private defaultConfig: TrainingConfig = {
    epochs: 100,
    learningRate: 0.001,
    batchSize: 32,
    validationSplit: 0.2,
    earlyStoppingPatience: 10
  };

  /**
   * Подготавливает данные для обучения модели
   */
  prepareTrainingData(candles: CandleData[], lookbackPeriod: number = 10): TrainingData {
    if (candles.length < lookbackPeriod + 1) {
      throw new Error('Недостаточно данных для подготовки обучающей выборки');
    }

    const features: number[][] = [];
    const labels: number[] = [];

    // Нормализуем данные свечей
    const normalizedCandles = this.normalizeCandles(candles);

    for (let i = lookbackPeriod; i < normalizedCandles.length; i++) {
      // Создаем набор признаков из последних свечей
      const candleFeatures = [];
      
      for (let j = i - lookbackPeriod; j < i; j++) {
        const candle = normalizedCandles[j];
        candleFeatures.push(
          candle.open,
          candle.high,
          candle.low,
          candle.close,
          candle.volume,
          // Дополнительные технические индикаторы
          this.calculateRSI(normalizedCandles, j, 14),
          this.calculateMACD(normalizedCandles, j),
          this.calculateBollingerBands(normalizedCandles, j)
        );
      }

      features.push(candleFeatures);

      // Определяем целевую переменную (направление движения)
      const currentCandle = normalizedCandles[i];
      const previousCandle = normalizedCandles[i - 1];
      const label = currentCandle.close > previousCandle.close ? 1 : 0;
      labels.push(label);
    }

    return { features, labels };
  }

  /**
   * Нормализует данные свечей
   */
  private normalizeCandles(candles: CandleData[]): CandleData[] {
    const prices = candles.flatMap(c => [c.open, c.high, c.low, c.close]);
    const volumes = candles.map(c => c.volume);
    
    const priceMin = Math.min(...prices);
    const priceMax = Math.max(...prices);
    const volumeMin = Math.min(...volumes);
    const volumeMax = Math.max(...volumes);

    return candles.map(candle => ({
      ...candle,
      open: (candle.open - priceMin) / (priceMax - priceMin),
      high: (candle.high - priceMin) / (priceMax - priceMin),
      low: (candle.low - priceMin) / (priceMax - priceMin),
      close: (candle.close - priceMin) / (priceMax - priceMin),
      volume: volumeMax > volumeMin ? (candle.volume - volumeMin) / (volumeMax - volumeMin) : 0
    }));
  }

  /**
   * Вычисляет RSI
   */
  private calculateRSI(candles: CandleData[], index: number, period: number = 14): number {
    if (index < period) return 0.5; // Нейтральное значение

    let gains = 0;
    let losses = 0;

    for (let i = index - period + 1; i <= index; i++) {
      const change = candles[i].close - candles[i - 1].close;
      if (change > 0) {
        gains += change;
      } else {
        losses -= change;
      }
    }

    const avgGain = gains / period;
    const avgLoss = losses / period;
    
    if (avgLoss === 0) return 1;
    
    const rs = avgGain / avgLoss;
    return 1 - (1 / (1 + rs));
  }

  /**
   * Вычисляет MACD
   */
  private calculateMACD(candles: CandleData[], index: number): number {
    if (index < 26) return 0; // Нейтральное значение

    const ema12 = this.calculateEMA(candles, index, 12);
    const ema26 = this.calculateEMA(candles, index, 26);
    
    return ema12 - ema26;
  }

  /**
   * Вычисляет EMA
   */
  private calculateEMA(candles: CandleData[], index: number, period: number): number {
    if (index < period - 1) return candles[index].close;

    const multiplier = 2 / (period + 1);
    let ema = candles[index - period + 1].close;

    for (let i = index - period + 2; i <= index; i++) {
      ema = (candles[i].close - ema) * multiplier + ema;
    }

    return ema;
  }

  /**
   * Вычисляет полосы Боллинджера
   */
  private calculateBollingerBands(candles: CandleData[], index: number, period: number = 20): number {
    if (index < period - 1) return 0.5; // Нейтральное значение

    const closes = candles.slice(index - period + 1, index + 1).map(c => c.close);
    const sma = closes.reduce((sum, close) => sum + close, 0) / period;
    
    const variance = closes.reduce((sum, close) => sum + Math.pow(close - sma, 2), 0) / period;
    const stdDev = Math.sqrt(variance);
    
    const currentPrice = candles[index].close;
    const upperBand = sma + (2 * stdDev);
    const lowerBand = sma - (2 * stdDev);
    
    // Возвращаем позицию цены относительно полос (0-1)
    if (upperBand === lowerBand) return 0.5;
    return (currentPrice - lowerBand) / (upperBand - lowerBand);
  }

  /**
   * Обучает модель (заглушка для будущей реализации)
   */
  async trainModel(trainingData: TrainingData, config?: Partial<TrainingConfig>): Promise<ModelMetrics> {
    const finalConfig = { ...this.defaultConfig, ...config };
    
    console.log('Training model with config:', finalConfig);
    console.log('Training data size:', trainingData.features.length);
    
    // Заглушка для обучения модели
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Возвращаем мок-метрики
    return {
      accuracy: 0.65 + Math.random() * 0.2,
      precision: 0.6 + Math.random() * 0.25,
      recall: 0.55 + Math.random() * 0.3,
      f1Score: 0.6 + Math.random() * 0.2,
      confusionMatrix: [
        [45, 15],
        [20, 40]
      ]
    };
  }

  /**
   * Валидирует модель на тестовых данных
   */
  validateModel(testData: TrainingData): ModelMetrics {
    // Заглушка для валидации
    return {
      accuracy: 0.7 + Math.random() * 0.15,
      precision: 0.65 + Math.random() * 0.2,
      recall: 0.6 + Math.random() * 0.25,
      f1Score: 0.65 + Math.random() * 0.15,
      confusionMatrix: [
        [38, 12],
        [18, 32]
      ]
    };
  }
}

export const modelTrainingService = new ModelTrainingService();