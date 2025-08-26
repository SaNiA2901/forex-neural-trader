import { CandleData } from '@/types/session';
import { PredictionResult, PredictionConfig } from '@/types/trading';
import { TechnicalIndicatorService } from '../indicators/TechnicalIndicators';
import { PatternAnalysisService } from '../patterns/PatternAnalysis';

interface NetworkWeights {
  input_hidden: number[][];
  hidden_output: number[];
  hidden_bias: number[];
  output_bias: number;
}

interface TrainingExample {
  features: number[];
  target: number; // 1 for UP, 0 for DOWN
  timestamp: number;
  actualOutcome?: number;
}

interface FeatureSet {
  technical: number[];
  pattern: number[];
  volume: number[];
  price: number[];
  momentum: number[];
}

export class RealMLService {
  private static instance: RealMLService;
  private weights: NetworkWeights;
  private trainingData: TrainingExample[] = [];
  private readonly inputSize = 30;
  private readonly hiddenSize = 64;
  private readonly learningRate = 0.001;
  private readonly momentum = 0.9;
  private previousGradients: any = null;
  private modelAccuracy = 0.55;
  private lastTrainingTime = 0;
  private readonly minTrainingInterval = 60000; // 1 минута

  private constructor() {
    this.initializeNetwork();
    this.loadTrainingHistory();
  }

  static getInstance(): RealMLService {
    if (!RealMLService.instance) {
      RealMLService.instance = new RealMLService();
    }
    return RealMLService.instance;
  }

  private initializeNetwork(): void {
    // Xavier/Glorot инициализация
    const initWeight = (fanIn: number, fanOut: number) => {
      const limit = Math.sqrt(6 / (fanIn + fanOut));
      return (Math.random() * 2 - 1) * limit;
    };

    this.weights = {
      input_hidden: Array(this.inputSize).fill(0).map(() =>
        Array(this.hiddenSize).fill(0).map(() => initWeight(this.inputSize, this.hiddenSize))
      ),
      hidden_output: Array(this.hiddenSize).fill(0).map(() => initWeight(this.hiddenSize, 1)),
      hidden_bias: Array(this.hiddenSize).fill(0).map(() => initWeight(1, this.hiddenSize)),
      output_bias: initWeight(1, 1)
    };
  }

  private async extractFeatures(candles: CandleData[], currentIndex: number): Promise<FeatureSet> {
    const lookback = Math.min(20, currentIndex);
    // CRITICAL FIX: Only use candles UP TO currentIndex (no look-ahead)
    const recentCandles = candles.slice(Math.max(0, currentIndex - lookback), currentIndex);
    const current = candles[currentIndex];

    // Технические индикаторы
    const technical = await TechnicalIndicatorService.calculateAll(candles, currentIndex);
    
    // Паттерны
    const patterns = PatternAnalysisService.analyzePatterns(candles, currentIndex);
    
    // Объемы
    const volume = PatternAnalysisService.analyzeVolume(candles, currentIndex);

    return {
      technical: [
        this.normalize(technical.rsi, 0, 100),
        this.normalize(technical.macd.line, -1, 1),
        this.normalize(technical.macd.signal, -1, 1),
        this.normalize(technical.macd.histogram, -0.5, 0.5),
        this.normalize(technical.stochastic.k, 0, 100),
        this.normalize(technical.stochastic.d, 0, 100),
        this.normalize(technical.atr, 0, current.close * 0.1),
        this.normalize(technical.adx, 0, 100),
        this.normalize(technical.bollingerBands.upper - technical.bollingerBands.lower, 0, current.close * 0.2),
        this.calculateBBPosition(current.close, technical.bollingerBands)
      ],
      
      pattern: [
        patterns.strength,
        patterns.isReversal ? 1 : 0,
        patterns.isContinuation ? 1 : 0,
        this.encodePattern(patterns.candlestickPattern)
      ],
      
      volume: [
        this.normalize(volume.volumeOscillator, -100, 100),
        this.normalize(volume.onBalanceVolume, -1000000, 1000000),
        volume.volumeTrend === 'increasing' ? 1 : volume.volumeTrend === 'decreasing' ? -1 : 0,
        this.normalize(current.volume, 0, this.getMaxVolume(recentCandles))
      ],
      
      price: [
        this.normalize(current.open / current.close, 0.95, 1.05),
        this.normalize(current.high / current.close, 1, 1.1),
        this.normalize(current.low / current.close, 0.9, 1),
        this.calculatePriceVelocity(recentCandles),
        this.calculateTrendStrength(recentCandles),
        this.calculateVolatility(recentCandles)
      ],
      
      momentum: [
        this.calculateMomentum(recentCandles, 1),
        this.calculateMomentum(recentCandles, 3),
        this.calculateMomentum(recentCandles, 5),
        this.calculateMomentum(recentCandles, 10),
        this.calculateAcceleration(recentCandles),
        this.calculateMeanReversion(recentCandles)
      ]
    };
  }

  private flattenFeatures(features: FeatureSet): number[] {
    return [
      ...features.technical,
      ...features.pattern,
      ...features.volume,
      ...features.price,
      ...features.momentum
    ].slice(0, this.inputSize); // Обрезаем до нужного размера
  }

  async generatePrediction(
    candles: CandleData[], 
    currentIndex: number, 
    config: PredictionConfig
  ): Promise<PredictionResult | null> {
    try {
      if (candles.length < 20 || currentIndex < 19) {
        return null;
      }

      const features = await this.extractFeatures(candles, currentIndex);
      const inputVector = this.flattenFeatures(features);
      
      // Нормализуем вектор
      const normalizedInput = this.normalizeVector(inputVector);
      
      // Прогон через сеть
      const { hiddenOutputs, output } = this.feedForward(normalizedInput);
      
      // Определяем направление и вероятность
      const probability = this.sigmoid(output) * 100;
      const direction = probability > 50 ? 'UP' : 'DOWN';
      const adjustedProbability = direction === 'UP' ? probability : 100 - probability;
      
      // Расчет уверенности на основе силы сигнала
      const confidence = this.calculateConfidence(normalizedInput, output, features);
      
      // CRITICAL FIX: Don't create training example with prediction as target!
      // Training examples should only be created when we have actual outcomes
      // this.addTrainingExample(normalizedInput, probability > 50 ? 1 : 0);
      
      // Периодическое обучение
      if (this.shouldTrain()) {
        await this.trainNetwork();
      }

      return {
        direction,
        probability: Math.round(adjustedProbability * 10) / 10,
        confidence: Math.round(confidence * 10) / 10,
        interval: config.predictionInterval,
        factors: this.generateFactors(features, output),
        recommendation: this.generateRecommendation(direction, adjustedProbability, confidence, features)
      };

    } catch (error) {
      console.error('Error in ML prediction:', error);
      return null;
    }
  }

  private feedForward(input: number[]): { hiddenOutputs: number[]; output: number } {
    // Скрытый слой
    const hiddenOutputs = this.weights.hidden_bias.map((bias, i) => {
      let sum = bias;
      for (let j = 0; j < input.length; j++) {
        sum += input[j] * this.weights.input_hidden[j][i];
      }
      return this.relu(sum);
    });

    // Выходной слой
    let output = this.weights.output_bias;
    for (let i = 0; i < hiddenOutputs.length; i++) {
      output += hiddenOutputs[i] * this.weights.hidden_output[i];
    }

    return { hiddenOutputs, output };
  }

  private async trainNetwork(): Promise<void> {
    if (this.trainingData.length < 10) return;

    const batchSize = Math.min(32, this.trainingData.length);
    const epochs = 5;
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      // Случайное перемешивание данных
      const shuffled = [...this.trainingData].sort(() => Math.random() - 0.5);
      
      for (let i = 0; i < shuffled.length; i += batchSize) {
        const batch = shuffled.slice(i, i + batchSize);
        await this.trainBatch(batch);
      }
    }

    this.lastTrainingTime = Date.now();
    this.calculateAccuracy();
    console.log(`🧠 Сеть обучена. Точность: ${(this.modelAccuracy * 100).toFixed(1)}%`);
  }

  private async trainBatch(batch: TrainingExample[]): Promise<void> {
    const totalGradients = this.initializeGradients();
    
    for (const example of batch) {
      const { hiddenOutputs, output } = this.feedForward(example.features);
      const error = example.target - this.sigmoid(output);
      
      // Обратное распространение
      const gradients = this.backpropagate(example.features, hiddenOutputs, output, error);
      this.accumulateGradients(totalGradients, gradients);
    }
    
    // Применяем градиенты с momentum
    this.applyGradients(totalGradients, batch.length);
  }

  private backpropagate(
    input: number[], 
    hiddenOutputs: number[], 
    output: number, 
    error: number
  ): any {
    const sigmoidOutput = this.sigmoid(output);
    const outputGradient = error * sigmoidOutput * (1 - sigmoidOutput);
    
    // Градиенты для скрытого слоя
    const hiddenGradients = this.weights.hidden_output.map((weight, i) => {
      const hiddenError = outputGradient * weight;
      return hiddenError * this.reluDerivative(hiddenOutputs[i]);
    });
    
    return {
      input_hidden: input.map(inp => 
        hiddenGradients.map(hiddenGrad => inp * hiddenGrad)
      ),
      hidden_output: hiddenOutputs.map(hidden => hidden * outputGradient),
      hidden_bias: hiddenGradients,
      output_bias: outputGradient
    };
  }

  // Активационные функции
  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
  }

  private relu(x: number): number {
    return Math.max(0, x);
  }

  private reluDerivative(x: number): number {
    return x > 0 ? 1 : 0;
  }

  // Вспомогательные функции для признаков
  private normalize(value: number, min: number, max: number): number {
    if (max === min) return 0;
    return Math.max(-1, Math.min(1, (value - min) / (max - min) * 2 - 1));
  }

  private normalizeVector(vector: number[]): number[] {
    const mean = vector.reduce((sum, val) => sum + val, 0) / vector.length;
    const std = Math.sqrt(vector.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / vector.length);
    
    if (std === 0) return vector;
    return vector.map(val => (val - mean) / std);
  }

  private calculateBBPosition(price: number, bb: any): number {
    if (bb.upper === bb.lower) return 0.5;
    return (price - bb.lower) / (bb.upper - bb.lower);
  }

  private encodePattern(pattern: string | null): number {
    const patterns: { [key: string]: number } = {
      'Doji': 0.1, 'Hammer': 0.2, 'Shooting Star': 0.3,
      'Bullish Engulfing': 0.4, 'Bearish Engulfing': 0.5,
      'Three White Soldiers': 0.6, 'Three Black Crows': 0.7
    };
    return pattern ? (patterns[pattern] || 0) : 0;
  }

  private getMaxVolume(candles: CandleData[]): number {
    return Math.max(...candles.map(c => c.volume));
  }

  private calculatePriceVelocity(candles: CandleData[]): number {
    if (candles.length < 3) return 0;
    const prices = candles.map(c => c.close);
    const velocity = (prices[prices.length - 1] - prices[prices.length - 2]) - 
                     (prices[prices.length - 2] - prices[prices.length - 3]);
    return Math.tanh(velocity / prices[prices.length - 1]);
  }

  private calculateTrendStrength(candles: CandleData[]): number {
    if (candles.length < 5) return 0;
    const prices = candles.map(c => c.close);
    const firstPrice = prices[0];
    const lastPrice = prices[prices.length - 1];
    const totalChange = (lastPrice - firstPrice) / firstPrice;
    
    let trendCount = 0;
    for (let i = 1; i < prices.length; i++) {
      if (totalChange > 0 && prices[i] > prices[i-1]) trendCount++;
      if (totalChange < 0 && prices[i] < prices[i-1]) trendCount++;
    }
    
    const trendStrength = trendCount / (prices.length - 1);
    return Math.tanh(totalChange * trendStrength);
  }

  private calculateVolatility(candles: CandleData[]): number {
    if (candles.length < 2) return 0;
    const returns = [];
    for (let i = 1; i < candles.length; i++) {
      returns.push((candles[i].close - candles[i-1].close) / candles[i-1].close);
    }
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    return Math.tanh(Math.sqrt(variance) * 100);
  }

  private calculateMomentum(candles: CandleData[], period: number): number {
    if (candles.length <= period) return 0;
    const current = candles[candles.length - 1].close;
    const past = candles[candles.length - 1 - period].close;
    return Math.tanh((current - past) / past);
  }

  private calculateAcceleration(candles: CandleData[]): number {
    if (candles.length < 4) return 0;
    const prices = candles.map(c => c.close);
    const len = prices.length;
    const velocity1 = prices[len - 1] - prices[len - 2];
    const velocity2 = prices[len - 2] - prices[len - 3];
    const acceleration = velocity1 - velocity2;
    return Math.tanh(acceleration / prices[len - 1]);
  }

  private calculateMeanReversion(candles: CandleData[]): number {
    if (candles.length < 10) return 0;
    const prices = candles.map(c => c.close);
    const sma = prices.reduce((sum, p) => sum + p, 0) / prices.length;
    const currentPrice = prices[prices.length - 1];
    return Math.tanh((currentPrice - sma) / sma);
  }

  private calculateConfidence(input: number[], output: number, features: FeatureSet): number {
    // Базовая уверенность на основе силы сигнала
    const signalStrength = Math.abs(output);
    let confidence = 60 + (signalStrength * 15);
    
    // Модификаторы уверенности
    if (features.pattern.length > 0 && features.pattern[0] > 0.7) confidence += 5;
    if (features.volume[2] !== 0) confidence += 3; // Объемный тренд
    if (Math.abs(features.momentum[0]) > 0.5) confidence += 2; // Сильный моментум
    
    // Учитываем согласованность индикаторов
    const technicalAgreement = this.calculateTechnicalAgreement(features.technical);
    confidence += technicalAgreement * 10;
    
    return Math.min(90, Math.max(55, confidence));
  }

  private calculateTechnicalAgreement(technical: number[]): number {
    const signals = [
      technical[0] > 0.4 ? 1 : technical[0] < -0.4 ? -1 : 0, // RSI
      technical[1] > technical[2] ? 1 : -1, // MACD
      technical[4] > 0.6 ? 1 : technical[4] < -0.6 ? -1 : 0  // Stochastic
    ];
    
    const agreement = Math.abs(signals.reduce((sum, signal) => sum + signal, 0)) / signals.length;
    return agreement;
  }

  private generateFactors(features: FeatureSet, output: number): any {
    return {
      technical_strength: features.technical[0], // RSI нормализованный
      momentum_factor: features.momentum[0],
      pattern_confidence: features.pattern[0],
      volume_factor: features.volume[0],
      trend_factor: features.price[4],
      model_output: Math.tanh(output)
    };
  }

  private generateRecommendation(
    direction: 'UP' | 'DOWN', 
    probability: number, 
    confidence: number, 
    features: FeatureSet
  ): string {
    const strength = probability > 70 ? 'strong' : probability > 60 ? 'moderate' : 'weak';
    const patternInfo = features.pattern[0] > 0.5 ? ' Pattern detected.' : '';
    const volumeInfo = features.volume[2] !== 0 ? ` Volume ${features.volume[2] > 0 ? 'increasing' : 'decreasing'}.` : '';
    
    return `${strength.charAt(0).toUpperCase() + strength.slice(1)} ${direction.toLowerCase()} signal (${probability.toFixed(1)}%).${patternInfo}${volumeInfo}`;
  }

  // Управление обучающими данными
  private addTrainingExample(features: number[], target: number): void {
    this.trainingData.push({
      features,
      target,
      timestamp: Date.now()
    });
    
    // Ограничиваем размер обучающей выборки
    if (this.trainingData.length > 1000) {
      this.trainingData = this.trainingData.slice(-800);
    }
  }

  // CRITICAL FIX: Proper training with actual outcomes
  addTrainingExampleWithActualOutcome(
    features: number[], 
    actualDirection: 'UP' | 'DOWN',
    predictionTimestamp: number
  ): void {
    const target = actualDirection === 'UP' ? 1 : 0;
    this.addTrainingExample(features, target);
    
    // Trigger training if we have enough examples
    if (this.shouldTrain()) {
      this.trainNetwork();
    }
  }

  updateWithActualResult(predictionIndex: number, actualDirection: 'UP' | 'DOWN'): void {
    if (predictionIndex < this.trainingData.length) {
      this.trainingData[predictionIndex].actualOutcome = actualDirection === 'UP' ? 1 : 0;
      // Update target with actual outcome
      this.trainingData[predictionIndex].target = this.trainingData[predictionIndex].actualOutcome!;
      // Перетренируем с новыми данными
      this.trainNetwork();
    }
  }

  private shouldTrain(): boolean {
    return Date.now() - this.lastTrainingTime > this.minTrainingInterval && 
           this.trainingData.length >= 20;
  }

  private calculateAccuracy(): void {
    const completed = this.trainingData.filter(t => t.actualOutcome !== undefined);
    if (completed.length === 0) return;
    
    const correct = completed.filter(t => 
      (t.target === 1 && t.actualOutcome === 1) || 
      (t.target === 0 && t.actualOutcome === 0)
    ).length;
    
    this.modelAccuracy = correct / completed.length;
  }

  private initializeGradients(): any {
    return {
      input_hidden: Array(this.inputSize).fill(0).map(() => Array(this.hiddenSize).fill(0)),
      hidden_output: Array(this.hiddenSize).fill(0),
      hidden_bias: Array(this.hiddenSize).fill(0),
      output_bias: 0
    };
  }

  private accumulateGradients(total: any, gradients: any): void {
    for (let i = 0; i < this.inputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        total.input_hidden[i][j] += gradients.input_hidden[i][j];
      }
    }
    
    for (let i = 0; i < this.hiddenSize; i++) {
      total.hidden_output[i] += gradients.hidden_output[i];
      total.hidden_bias[i] += gradients.hidden_bias[i];
    }
    
    total.output_bias += gradients.output_bias;
  }

  private applyGradients(gradients: any, batchSize: number): void {
    const lr = this.learningRate / batchSize;
    
    // Применяем градиенты с momentum
    for (let i = 0; i < this.inputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        const gradient = gradients.input_hidden[i][j] * lr;
        this.weights.input_hidden[i][j] += gradient;
      }
    }
    
    for (let i = 0; i < this.hiddenSize; i++) {
      this.weights.hidden_output[i] += gradients.hidden_output[i] * lr;
      this.weights.hidden_bias[i] += gradients.hidden_bias[i] * lr;
    }
    
    this.weights.output_bias += gradients.output_bias * lr;
  }

  private loadTrainingHistory(): void {
    // В реальной реализации здесь была бы загрузка из постоянного хранилища
    this.trainingData = [];
  }

  getModelStats() {
    return {
      accuracy: this.modelAccuracy,
      trainingExamples: this.trainingData.length,
      lastTraining: this.lastTrainingTime,
      inputSize: this.inputSize,
      hiddenSize: this.hiddenSize
    };
  }
}

export const realMLService = RealMLService.getInstance();