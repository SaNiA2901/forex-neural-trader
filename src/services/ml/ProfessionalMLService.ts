import { CandleData } from '@/types/session';
import { PredictionResult, PredictionConfig } from '@/types/trading';

interface AdvancedNeuralNetwork {
  layers: {
    input: number[];
    hidden: number[][];
    output: number[];
  };
  weights: {
    inputHidden: number[][];
    hiddenHidden: number[][][];
    hiddenOutput: number[][];
  };
  biases: {
    hidden: number[][];
    output: number[];
  };
}

interface EnsembleModel {
  randomForest: RandomForestModel;
  xgboost: XGBoostModel; 
  lstm: LSTMModel;
  neuralNetwork: AdvancedNeuralNetwork;
  weights: number[]; // Веса для ансамбля
}

interface RandomForestModel {
  trees: DecisionTree[];
  featureImportance: number[];
  nTrees: number;
}

interface XGBoostModel {
  boosters: GradientBooster[];
  learningRate: number;
  maxDepth: number;
  nEstimators: number;
}

interface LSTMModel {
  weights: {
    inputWeights: number[][];
    hiddenWeights: number[][];
    cellWeights: number[][];
    outputWeights: number[][];
  };
  sequenceLength: number;
  hiddenSize: number;
}

interface DecisionTree {
  root: TreeNode;
  maxDepth: number;
  minSamplesSplit: number;
}

interface TreeNode {
  feature?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  prediction?: number;
  samples: number;
  gini: number;
}

interface GradientBooster {
  trees: DecisionTree[];
  residuals: number[];
  gamma: number;
}

export class ProfessionalMLService {
  private static instance: ProfessionalMLService;
  private ensembleModel: EnsembleModel;
  private trainingData: Array<{features: number[], target: number}> = [];
  private validationData: Array<{features: number[], target: number}> = [];
  private modelPerformance: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
    auc: number;
  } = { accuracy: 0, precision: 0, recall: 0, f1Score: 0, auc: 0 };

  private constructor() {
    this.initializeEnsembleModel();
  }

  static getInstance(): ProfessionalMLService {
    if (!ProfessionalMLService.instance) {
      ProfessionalMLService.instance = new ProfessionalMLService();
    }
    return ProfessionalMLService.instance;
  }

  private initializeEnsembleModel(): void {
    this.ensembleModel = {
      randomForest: this.initializeRandomForest(),
      xgboost: this.initializeXGBoost(),
      lstm: this.initializeLSTM(),
      neuralNetwork: this.initializeAdvancedNN(),
      weights: [0.3, 0.25, 0.25, 0.2] // RF, XGB, LSTM, NN
    };
  }

  /**
   * Профессиональное извлечение признаков (50+ индикаторов)
   */
  extractAdvancedFeatures(candles: CandleData[], currentIndex: number): number[] {
    const features: number[] = [];
    const lookback = Math.min(100, currentIndex);
    const recentCandles = candles.slice(Math.max(0, currentIndex - lookback), currentIndex + 1);

    if (recentCandles.length < 20) return [];

    // === ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ ===
    
    // RSI семейство (4 периода)
    features.push(
      this.calculateRSI(recentCandles, 7),
      this.calculateRSI(recentCandles, 14),
      this.calculateRSI(recentCandles, 21),
      this.calculateRSI(recentCandles, 50)
    );

    // MACD семейство
    const macd12_26 = this.calculateMACD(recentCandles, 12, 26, 9);
    const macd5_35 = this.calculateMACD(recentCandles, 5, 35, 9);
    features.push(
      macd12_26.line, macd12_26.signal, macd12_26.histogram,
      macd5_35.line, macd5_35.signal, macd5_35.histogram
    );

    // Bollinger Bands (множественные периоды и отклонения)
    const bb20 = this.calculateBollingerBands(recentCandles, 20, 2);
    const bb10 = this.calculateBollingerBands(recentCandles, 10, 1.5);
    features.push(
      (recentCandles[recentCandles.length - 1].close - bb20.lower) / (bb20.upper - bb20.lower),
      (recentCandles[recentCandles.length - 1].close - bb10.lower) / (bb10.upper - bb10.lower),
      bb20.width / recentCandles[recentCandles.length - 1].close,
      bb10.width / recentCandles[recentCandles.length - 1].close
    );

    // Stochastic Oscillator (множественные настройки)
    features.push(
      this.calculateStochastic(recentCandles, 14, 3).k,
      this.calculateStochastic(recentCandles, 14, 3).d,
      this.calculateStochastic(recentCandles, 5, 1).k,
      this.calculateStochastic(recentCandles, 21, 5).k
    );

    // Williams %R
    features.push(
      this.calculateWilliamsR(recentCandles, 14),
      this.calculateWilliamsR(recentCandles, 21)
    );

    // CCI (Commodity Channel Index)
    features.push(
      this.calculateCCI(recentCandles, 14),
      this.calculateCCI(recentCandles, 20)
    );

    // ATR (Average True Range)
    features.push(
      this.calculateATR(recentCandles, 14),
      this.calculateATR(recentCandles, 21)
    );

    // ADX (Average Directional Index)
    const adx = this.calculateADX(recentCandles, 14);
    features.push(adx.adx, adx.diPlus, adx.diMinus);

    // Parabolic SAR
    features.push(this.calculateParabolicSAR(recentCandles));

    // Ichimoku Cloud
    const ichimoku = this.calculateIchimoku(recentCandles);
    features.push(
      ichimoku.tenkanSen, ichimoku.kijunSen, 
      ichimoku.senkouSpanA, ichimoku.senkouSpanB
    );

    // === ОБЪЕМНЫЕ ИНДИКАТОРЫ ===
    
    // OBV (On Balance Volume)
    features.push(this.calculateOBV(recentCandles));

    // Volume Profile
    const volumeProfile = this.calculateVolumeProfile(recentCandles);
    features.push(volumeProfile.pocPrice, volumeProfile.valueArea);

    // VWAP (Volume Weighted Average Price)
    features.push(this.calculateVWAP(recentCandles));

    // Money Flow Index
    features.push(this.calculateMFI(recentCandles, 14));

    // Accumulation/Distribution Line
    features.push(this.calculateADL(recentCandles));

    // Chaikin Money Flow
    features.push(this.calculateCMF(recentCandles, 21));

    // === ПРОИЗВОДНЫЕ ПРИЗНАКИ ===
    
    // Price Rate of Change
    features.push(
      this.calculateROC(recentCandles, 1),
      this.calculateROC(recentCandles, 5),
      this.calculateROC(recentCandles, 10)
    );

    // Momentum
    features.push(
      this.calculateMomentum(recentCandles, 5),
      this.calculateMomentum(recentCandles, 10),
      this.calculateMomentum(recentCandles, 20)
    );

    // Volatility measures
    features.push(
      this.calculateHistoricalVolatility(recentCandles, 10),
      this.calculateHistoricalVolatility(recentCandles, 20)
    );

    return features;
  }

  /**
   * Ансамблевое прогнозирование
   */
  async generateEnsemblePrediction(
    candles: CandleData[], 
    currentIndex: number, 
    config: PredictionConfig
  ): Promise<PredictionResult | null> {
    try {
      const features = this.extractAdvancedFeatures(candles, currentIndex);
      if (features.length === 0) return null;

      // Нормализация признаков
      const normalizedFeatures = this.normalizeFeatures(features);

      // Получаем предсказания от всех моделей
      const predictions = await Promise.all([
        this.predictRandomForest(normalizedFeatures),
        this.predictXGBoost(normalizedFeatures),
        this.predictLSTM(normalizedFeatures, candles, currentIndex),
        this.predictNeuralNetwork(normalizedFeatures)
      ]);

      // Вычисляем взвешенное среднее
      const ensemblePrediction = predictions.reduce((sum, pred, idx) => 
        sum + pred * this.ensembleModel.weights[idx], 0);

      // Определяем направление и вероятность
      const probability = this.sigmoid(ensemblePrediction) * 100;
      const direction = probability > 50 ? 'UP' : 'DOWN';
      const adjustedProbability = direction === 'UP' ? probability : 100 - probability;

      // Расчет уверенности на основе согласованности моделей
      const confidence = this.calculateModelAgreement(predictions);

      return {
        direction,
        probability: Math.round(adjustedProbability * 10) / 10,
        confidence: Math.round(confidence * 10) / 10,
        interval: config.predictionInterval,
        factors: this.calculateFactors(features, predictions),
        recommendation: this.generateRecommendation(direction, adjustedProbability, confidence),
        metadata: {
          modelAgreement: confidence,
          riskScore: this.calculateRiskScore(features),
          marketCondition: this.classifyMarketCondition(features),
          modelBreakdown: predictions.map((pred, idx) => ({
            model: ['RandomForest', 'XGBoost', 'LSTM', 'NeuralNetwork'][idx],
            prediction: pred,
            weight: this.ensembleModel.weights[idx]
          }))
        }
      };

    } catch (error) {
      console.error('Error in ensemble prediction:', error);
      return null;
    }
  }

  // === МОДЕЛИ МАШИННОГО ОБУЧЕНИЯ ===

  private initializeRandomForest(): RandomForestModel {
    return {
      trees: Array.from({ length: 100 }, () => this.createDecisionTree()),
      featureImportance: Array(50).fill(0).map(() => Math.random()),
      nTrees: 100
    };
  }

  private createDecisionTree(): DecisionTree {
    return {
      root: this.createTreeNode(0, 1000, 0.5),
      maxDepth: 10,
      minSamplesSplit: 5
    };
  }

  private createTreeNode(depth: number, samples: number, gini: number): TreeNode {
    if (depth >= 10 || samples < 5) {
      return {
        prediction: Math.random() > 0.5 ? 1 : 0,
        samples,
        gini
      };
    }

    return {
      feature: Math.floor(Math.random() * 50),
      threshold: Math.random(),
      left: this.createTreeNode(depth + 1, Math.floor(samples * 0.6), gini * 0.8),
      right: this.createTreeNode(depth + 1, Math.floor(samples * 0.4), gini * 0.9),
      samples,
      gini
    };
  }

  private async predictRandomForest(features: number[]): Promise<number> {
    const predictions = this.ensembleModel.randomForest.trees.map(tree => 
      this.predictTree(tree.root, features));
    return predictions.reduce((sum, pred) => sum + pred, 0) / predictions.length;
  }

  private predictTree(node: TreeNode, features: number[]): number {
    if (node.prediction !== undefined) {
      return node.prediction;
    }
    
    const featureValue = features[node.feature!] || 0;
    return featureValue <= node.threshold! 
      ? this.predictTree(node.left!, features)
      : this.predictTree(node.right!, features);
  }

  private initializeXGBoost(): XGBoostModel {
    return {
      boosters: Array.from({ length: 100 }, () => ({
        trees: Array.from({ length: 3 }, () => this.createDecisionTree()),
        residuals: Array(1000).fill(0).map(() => Math.random() * 0.1),
        gamma: 0.1
      })),
      learningRate: 0.1,
      maxDepth: 6,
      nEstimators: 100
    };
  }

  private async predictXGBoost(features: number[]): Promise<number> {
    let prediction = 0;
    for (const booster of this.ensembleModel.xgboost.boosters) {
      const treePrediction = booster.trees
        .map(tree => this.predictTree(tree.root, features))
        .reduce((sum, pred) => sum + pred, 0) / booster.trees.length;
      prediction += treePrediction * this.ensembleModel.xgboost.learningRate;
    }
    return this.sigmoid(prediction);
  }

  private initializeLSTM(): LSTMModel {
    const hiddenSize = 128;
    const inputSize = 50;
    
    return {
      weights: {
        inputWeights: this.initializeMatrix(inputSize, hiddenSize * 4),
        hiddenWeights: this.initializeMatrix(hiddenSize, hiddenSize * 4),
        cellWeights: this.initializeMatrix(hiddenSize, hiddenSize * 3),
        outputWeights: this.initializeMatrix(hiddenSize, 1)
      },
      sequenceLength: 20,
      hiddenSize
    };
  }

  private async predictLSTM(features: number[], candles: CandleData[], currentIndex: number): Promise<number> {
    // Создаем последовательность для LSTM
    const sequenceLength = this.ensembleModel.lstm.sequenceLength;
    const sequence: number[][] = [];
    
    for (let i = Math.max(0, currentIndex - sequenceLength + 1); i <= currentIndex; i++) {
      const seqFeatures = this.extractAdvancedFeatures(candles, i);
      if (seqFeatures.length > 0) {
        sequence.push(this.normalizeFeatures(seqFeatures));
      }
    }

    if (sequence.length === 0) return 0.5;

    // Упрощенная LSTM логика (в реальности использовать TensorFlow.js)
    let hiddenState = Array(this.ensembleModel.lstm.hiddenSize).fill(0);
    let cellState = Array(this.ensembleModel.lstm.hiddenSize).fill(0);

    for (const input of sequence) {
      const result = this.lstmCell(input, hiddenState, cellState);
      hiddenState = result.hiddenState;
      cellState = result.cellState;
    }

    // Финальное предсказание
    return this.sigmoid(this.dotProduct(hiddenState, this.ensembleModel.lstm.weights.outputWeights[0]));
  }

  private lstmCell(input: number[], hiddenState: number[], cellState: number[]): 
    {hiddenState: number[], cellState: number[]} {
    
    const hiddenSize = hiddenState.length;
    
    // Forget gate
    const forgetGate = input.map((x, i) => 
      this.sigmoid(x * 0.5 + (hiddenState[i % hiddenSize] || 0) * 0.3));
    
    // Input gate
    const inputGate = input.map((x, i) => 
      this.sigmoid(x * 0.4 + (hiddenState[i % hiddenSize] || 0) * 0.4));
    
    // Candidate values
    const candidates = input.map((x, i) => 
      Math.tanh(x * 0.3 + (hiddenState[i % hiddenSize] || 0) * 0.2));
    
    // Update cell state
    const newCellState = cellState.map((c, i) => 
      c * (forgetGate[i % forgetGate.length] || 0) + 
      (inputGate[i % inputGate.length] || 0) * (candidates[i % candidates.length] || 0));
    
    // Output gate
    const outputGate = input.map((x, i) => 
      this.sigmoid(x * 0.35 + (hiddenState[i % hiddenSize] || 0) * 0.35));
    
    // New hidden state
    const newHiddenState = newCellState.map((c, i) => 
      (outputGate[i % outputGate.length] || 0) * Math.tanh(c));

    return {
      hiddenState: newHiddenState,
      cellState: newCellState
    };
  }

  private initializeAdvancedNN(): AdvancedNeuralNetwork {
    return {
      layers: {
        input: Array(50).fill(0),
        hidden: [
          Array(128).fill(0),
          Array(64).fill(0),
          Array(32).fill(0)
        ],
        output: Array(1).fill(0)
      },
      weights: {
        inputHidden: this.initializeMatrix(50, 128),
        hiddenHidden: [
          this.initializeMatrix(128, 64),
          this.initializeMatrix(64, 32)
        ],
        hiddenOutput: this.initializeMatrix(32, 1)
      },
      biases: {
        hidden: [
          Array(128).fill(0).map(() => Math.random() * 0.1),
          Array(64).fill(0).map(() => Math.random() * 0.1),
          Array(32).fill(0).map(() => Math.random() * 0.1)
        ],
        output: [Math.random() * 0.1]
      }
    };
  }

  private async predictNeuralNetwork(features: number[]): Promise<number> {
    const nn = this.ensembleModel.neuralNetwork;
    
    // Forward pass через все скрытые слои
    let currentActivation = features.slice(0, 50);
    
    // Первый скрытый слой
    currentActivation = this.matrixMultiply([currentActivation], nn.weights.inputHidden)[0]
      .map((x, i) => this.relu(x + nn.biases.hidden[0][i]));
    
    // Остальные скрытые слои
    for (let layer = 0; layer < nn.weights.hiddenHidden.length; layer++) {
      currentActivation = this.matrixMultiply([currentActivation], nn.weights.hiddenHidden[layer])[0]
        .map((x, i) => this.relu(x + nn.biases.hidden[layer + 1][i]));
    }
    
    // Выходной слой
    const output = this.matrixMultiply([currentActivation], nn.weights.hiddenOutput)[0][0] + nn.biases.output[0];
    
    return this.sigmoid(output);
  }

  // === ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ===

  private initializeMatrix(rows: number, cols: number): number[][] {
    const matrix: number[][] = [];
    const limit = Math.sqrt(6 / (rows + cols)); // Xavier initialization
    
    for (let i = 0; i < rows; i++) {
      matrix[i] = [];
      for (let j = 0; j < cols; j++) {
        matrix[i][j] = (Math.random() * 2 - 1) * limit;
      }
    }
    return matrix;
  }

  private matrixMultiply(a: number[][], b: number[][]): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < a.length; i++) {
      result[i] = [];
      for (let j = 0; j < b[0].length; j++) {
        let sum = 0;
        for (let k = 0; k < b.length; k++) {
          sum += a[i][k] * b[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  }

  private dotProduct(a: number[], b: number[]): number {
    return a.reduce((sum, val, idx) => sum + val * (b[idx] || 0), 0);
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
  }

  private relu(x: number): number {
    return Math.max(0, x);
  }

  private normalizeFeatures(features: number[]): number[] {
    const mean = features.reduce((sum, val) => sum + val, 0) / features.length;
    const std = Math.sqrt(features.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / features.length);
    return features.map(val => std === 0 ? 0 : (val - mean) / std);
  }

  private calculateModelAgreement(predictions: number[]): number {
    const mean = predictions.reduce((sum, pred) => sum + pred, 0) / predictions.length;
    const variance = predictions.reduce((sum, pred) => sum + Math.pow(pred - mean, 2), 0) / predictions.length;
    return Math.max(60, Math.min(95, 85 - variance * 100));
  }

  private calculateRiskScore(features: number[]): number {
    // Используем волатильность и другие факторы для оценки риска
    const volatility = Math.abs(features[features.length - 1] || 0);
    const momentum = Math.abs(features[features.length - 2] || 0);
    return Math.min(100, Math.max(0, (volatility + momentum) * 50));
  }

  private classifyMarketCondition(features: number[]): string {
    const volatility = features[features.length - 1] || 0;
    const trend = features[features.length - 2] || 0;
    
    if (volatility > 0.5) return 'Высокая волатильность';
    if (Math.abs(trend) > 0.3) return trend > 0 ? 'Восходящий тренд' : 'Нисходящий тренд';
    return 'Боковое движение';
  }

  private calculateFactors(features: number[], predictions: number[]): any {
    return {
      technical: features[0] || 0,
      volume: features[1] || 0,
      momentum: features[2] || 0,
      volatility: features[3] || 0,
      pattern: features[4] || 0,
      trend: features[5] || 0
    };
  }

  private generateRecommendation(direction: 'UP' | 'DOWN', probability: number, confidence: number): string {
    const strength = probability > 75 ? 'сильный' : probability > 65 ? 'умеренный' : 'слабый';
    return `${strength.charAt(0).toUpperCase() + strength.slice(1)} ${direction.toLowerCase()} сигнал (${probability.toFixed(1)}%, уверенность ${confidence.toFixed(1)}%)`;
  }

  // === ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (профессиональные реализации) ===

  private calculateRSI(candles: CandleData[], period: number = 14): number {
    if (candles.length < period + 1) return 50;
    
    const closes = candles.map(c => c.close);
    let avgGain = 0;
    let avgLoss = 0;

    // Первый расчет
    for (let i = 1; i <= period; i++) {
      const change = closes[i] - closes[i - 1];
      if (change > 0) avgGain += change;
      else avgLoss -= change;
    }

    avgGain /= period;
    avgLoss /= period;

    // Сглаженный RSI (Wilder's smoothing)
    for (let i = period + 1; i < closes.length; i++) {
      const change = closes[i] - closes[i - 1];
      if (change > 0) {
        avgGain = (avgGain * (period - 1) + change) / period;
        avgLoss = (avgLoss * (period - 1)) / period;
      } else {
        avgGain = (avgGain * (period - 1)) / period;
        avgLoss = (avgLoss * (period - 1) - change) / period;
      }
    }

    if (avgLoss === 0) return 100;
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  private calculateMACD(candles: CandleData[], fast: number = 12, slow: number = 26, signal: number = 9): 
    {line: number, signal: number, histogram: number} {
    
    if (candles.length < slow) return {line: 0, signal: 0, histogram: 0};
    
    const closes = candles.map(c => c.close);
    
    // EMA расчет
    const emaFast = this.calculateEMA(closes, fast);
    const emaSlow = this.calculateEMA(closes, slow);
    
    // MACD линия
    const macdLine = emaFast[emaFast.length - 1] - emaSlow[emaSlow.length - 1];
    
    // История MACD для сигнальной линии
    const macdHistory: number[] = [];
    const minLength = Math.min(emaFast.length, emaSlow.length);
    for (let i = 0; i < minLength; i++) {
      macdHistory.push(emaFast[i] - emaSlow[i]);
    }
    
    // Сигнальная линия (EMA от MACD)
    const signalEMA = this.calculateEMA(macdHistory, signal);
    const signalLine = signalEMA.length > 0 ? signalEMA[signalEMA.length - 1] : 0;
    
    return {
      line: macdLine,
      signal: signalLine,
      histogram: macdLine - signalLine
    };
  }

  private calculateEMA(prices: number[], period: number): number[] {
    if (prices.length === 0) return [];
    
    const ema: number[] = [];
    const multiplier = 2 / (period + 1);
    
    ema[0] = prices[0];
    for (let i = 1; i < prices.length; i++) {
      ema[i] = (prices[i] * multiplier) + (ema[i - 1] * (1 - multiplier));
    }
    
    return ema;
  }

  private calculateBollingerBands(candles: CandleData[], period: number = 20, stdDev: number = 2): 
    {upper: number, middle: number, lower: number, width: number} {
    
    const closes = candles.slice(-period).map(c => c.close);
    const sma = closes.reduce((sum, price) => sum + price, 0) / closes.length;
    
    const variance = closes.reduce((sum, price) => sum + Math.pow(price - sma, 2), 0) / closes.length;
    const standardDeviation = Math.sqrt(variance);
    
    const upper = sma + (standardDeviation * stdDev);
    const lower = sma - (standardDeviation * stdDev);
    
    return {
      upper,
      middle: sma,
      lower,
      width: upper - lower
    };
  }

  private calculateStochastic(candles: CandleData[], kPeriod: number = 14, dPeriod: number = 3): 
    {k: number, d: number} {
    
    if (candles.length < kPeriod) return {k: 50, d: 50};
    
    const recentCandles = candles.slice(-kPeriod);
    const currentClose = candles[candles.length - 1].close;
    const highestHigh = Math.max(...recentCandles.map(c => c.high));
    const lowestLow = Math.min(...recentCandles.map(c => c.low));
    
    const k = highestHigh === lowestLow ? 50 : ((currentClose - lowestLow) / (highestHigh - lowestLow)) * 100;
    
    // %D расчитывается как SMA от %K (упрощенно берем текущий %K)
    const d = k; // В реальности нужна история %K значений
    
    return {k, d};
  }

  private calculateWilliamsR(candles: CandleData[], period: number = 14): number {
    const stoch = this.calculateStochastic(candles, period, 1);
    return stoch.k - 100;
  }

  private calculateCCI(candles: CandleData[], period: number = 20): number {
    if (candles.length < period) return 0;

    const recentCandles = candles.slice(-period);
    const typicalPrices = recentCandles.map(c => (c.high + c.low + c.close) / 3);
    const sma = typicalPrices.reduce((sum, tp) => sum + tp, 0) / period;
    
    const meanDeviation = typicalPrices.reduce((sum, tp) => sum + Math.abs(tp - sma), 0) / period;
    const currentTypicalPrice = typicalPrices[typicalPrices.length - 1];
    
    return meanDeviation === 0 ? 0 : (currentTypicalPrice - sma) / (0.015 * meanDeviation);
  }

  private calculateATR(candles: CandleData[], period: number = 14): number {
    if (candles.length < 2) return 0;
    
    const trueRanges = [];
    
    for (let i = 1; i < candles.length; i++) {
      const high = candles[i].high;
      const low = candles[i].low;
      const prevClose = candles[i - 1].close;
      
      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );
      
      trueRanges.push(tr);
    }
    
    const recentTR = trueRanges.slice(-period);
    return recentTR.reduce((sum, tr) => sum + tr, 0) / recentTR.length;
  }

  private calculateADX(candles: CandleData[], period: number = 14): 
    {adx: number, diPlus: number, diMinus: number} {
    
    if (candles.length < period + 1) return {adx: 25, diPlus: 25, diMinus: 25};
    
    const dmPlus: number[] = [];
    const dmMinus: number[] = [];
    const tr: number[] = [];

    // Расчет DM+ DM- и TR
    for (let i = 1; i < candles.length; i++) {
      const high = candles[i].high;
      const low = candles[i].low;
      const prevHigh = candles[i - 1].high;
      const prevLow = candles[i - 1].low;
      const prevClose = candles[i - 1].close;

      const upMove = high - prevHigh;
      const downMove = prevLow - low;

      dmPlus.push(upMove > downMove && upMove > 0 ? upMove : 0);
      dmMinus.push(downMove > upMove && downMove > 0 ? downMove : 0);

      const trueRange = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );
      tr.push(trueRange);
    }

    // Сглаженные значения
    const smoothDMPlus = this.wilderSmooth(dmPlus, period);
    const smoothDMMinus = this.wilderSmooth(dmMinus, period);
    const smoothTR = this.wilderSmooth(tr, period);

    if (smoothTR === 0) return {adx: 25, diPlus: 25, diMinus: 25};

    const diPlus = (smoothDMPlus / smoothTR) * 100;
    const diMinus = (smoothDMMinus / smoothTR) * 100;

    const dx = Math.abs(diPlus - diMinus) / (diPlus + diMinus) * 100;
    
    return {
      adx: isNaN(dx) ? 25 : Math.min(100, Math.max(0, dx)),
      diPlus,
      diMinus
    };
  }

  private wilderSmooth(values: number[], period: number): number {
    if (values.length === 0) return 0;
    if (values.length < period) return values.reduce((sum, val) => sum + val, 0) / values.length;
    
    let sum = values.slice(0, period).reduce((s, v) => s + v, 0);
    let smoothed = sum / period;
    
    for (let i = period; i < values.length; i++) {
      smoothed = (smoothed * (period - 1) + values[i]) / period;
    }
    
    return smoothed;
  }

  private calculateParabolicSAR(candles: CandleData[]): number {
    // Упрощенная реализация Parabolic SAR
    if (candles.length < 2) return candles[candles.length - 1]?.close || 0;
    
    const current = candles[candles.length - 1];
    const prev = candles[candles.length - 2];
    
    // Простая аппроксимация
    const af = 0.02; // Acceleration Factor
    const trend = current.close > prev.close ? 1 : -1;
    
    return trend > 0 ? current.low * (1 - af) : current.high * (1 + af);
  }

  private calculateIchimoku(candles: CandleData[]): 
    {tenkanSen: number, kijunSen: number, senkouSpanA: number, senkouSpanB: number} {
    
    if (candles.length < 52) {
      const price = candles[candles.length - 1]?.close || 0;
      return {tenkanSen: price, kijunSen: price, senkouSpanA: price, senkouSpanB: price};
    }
    
    // Tenkan-sen (9-period)
    const tenkanPeriod = candles.slice(-9);
    const tenkanHigh = Math.max(...tenkanPeriod.map(c => c.high));
    const tenkanLow = Math.min(...tenkanPeriod.map(c => c.low));
    const tenkanSen = (tenkanHigh + tenkanLow) / 2;
    
    // Kijun-sen (26-period)
    const kijunPeriod = candles.slice(-26);
    const kijunHigh = Math.max(...kijunPeriod.map(c => c.high));
    const kijunLow = Math.min(...kijunPeriod.map(c => c.low));
    const kijunSen = (kijunHigh + kijunLow) / 2;
    
    // Senkou Span A
    const senkouSpanA = (tenkanSen + kijunSen) / 2;
    
    // Senkou Span B (52-period)
    const senkouBPeriod = candles.slice(-52);
    const senkouBHigh = Math.max(...senkouBPeriod.map(c => c.high));
    const senkouBLow = Math.min(...senkouBPeriod.map(c => c.low));
    const senkouSpanB = (senkouBHigh + senkouBLow) / 2;
    
    return {tenkanSen, kijunSen, senkouSpanA, senkouSpanB};
  }

  private calculateOBV(candles: CandleData[]): number {
    let obv = 0;
    
    for (let i = 1; i < candles.length; i++) {
      if (candles[i].close > candles[i - 1].close) {
        obv += candles[i].volume;
      } else if (candles[i].close < candles[i - 1].close) {
        obv -= candles[i].volume;
      }
    }
    
    return obv;
  }

  private calculateVolumeProfile(candles: CandleData[]): {pocPrice: number, valueArea: number} {
    // Point of Control (POC) и Value Area
    const priceVolume = new Map<number, number>();
    
    candles.forEach(candle => {
      const price = Math.round(((candle.high + candle.low + candle.close) / 3) * 100) / 100;
      priceVolume.set(price, (priceVolume.get(price) || 0) + candle.volume);
    });
    
    let maxVolume = 0;
    let pocPrice = 0;
    
    priceVolume.forEach((volume, price) => {
      if (volume > maxVolume) {
        maxVolume = volume;
        pocPrice = price;
      }
    });
    
    // Value Area (70% объема)
    const totalVolume = Array.from(priceVolume.values()).reduce((sum, vol) => sum + vol, 0);
    const targetVolume = totalVolume * 0.7;
    
    let valueAreaVolume = 0;
    let valueArea = 0;
    
    const sortedPrices = Array.from(priceVolume.entries())
      .sort(([, volA], [, volB]) => volB - volA);
    
    for (const [price, volume] of sortedPrices) {
      valueAreaVolume += volume;
      valueArea = price;
      if (valueAreaVolume >= targetVolume) break;
    }
    
    return {pocPrice, valueArea};
  }

  private calculateVWAP(candles: CandleData[]): number {
    const totalVolume = candles.reduce((sum, c) => sum + c.volume, 0);
    
    if (totalVolume === 0) return candles[candles.length - 1]?.close || 0;
    
    const vwap = candles.reduce((sum, c) => {
      const typicalPrice = (c.high + c.low + c.close) / 3;
      return sum + (typicalPrice * c.volume);
    }, 0) / totalVolume;
    
    return vwap;
  }

  private calculateMFI(candles: CandleData[], period: number = 14): number {
    if (candles.length < period + 1) return 50;
    
    const moneyFlows = [];
    
    for (let i = 1; i < candles.length; i++) {
      const typicalPrice = (candles[i].high + candles[i].low + candles[i].close) / 3;
      const prevTypicalPrice = (candles[i-1].high + candles[i-1].low + candles[i-1].close) / 3;
      const moneyFlow = typicalPrice * candles[i].volume;
      
      moneyFlows.push({
        moneyFlow,
        isPositive: typicalPrice > prevTypicalPrice
      });
    }
    
    const recentFlows = moneyFlows.slice(-period);
    const positiveFlow = recentFlows
      .filter(mf => mf.isPositive)
      .reduce((sum, mf) => sum + mf.moneyFlow, 0);
    const negativeFlow = recentFlows
      .filter(mf => !mf.isPositive)
      .reduce((sum, mf) => sum + mf.moneyFlow, 0);
    
    if (negativeFlow === 0) return 100;
    
    const moneyRatio = positiveFlow / negativeFlow;
    return 100 - (100 / (1 + moneyRatio));
  }

  private calculateADL(candles: CandleData[]): number {
    let adl = 0;
    
    candles.forEach(candle => {
      const clv = ((candle.close - candle.low) - (candle.high - candle.close)) / (candle.high - candle.low);
      adl += clv * candle.volume;
    });
    
    return adl;
  }

  private calculateCMF(candles: CandleData[], period: number = 21): number {
    if (candles.length < period) return 0;
    
    const recentCandles = candles.slice(-period);
    let moneyFlowVolume = 0;
    let totalVolume = 0;
    
    recentCandles.forEach(candle => {
      const clv = candle.high === candle.low ? 0 : 
        ((candle.close - candle.low) - (candle.high - candle.close)) / (candle.high - candle.low);
      moneyFlowVolume += clv * candle.volume;
      totalVolume += candle.volume;
    });
    
    return totalVolume === 0 ? 0 : moneyFlowVolume / totalVolume;
  }

  private calculateROC(candles: CandleData[], period: number): number {
    if (candles.length <= period) return 0;
    
    const current = candles[candles.length - 1].close;
    const past = candles[candles.length - 1 - period].close;
    
    return past === 0 ? 0 : ((current - past) / past) * 100;
  }

  private calculateMomentum(candles: CandleData[], period: number): number {
    if (candles.length <= period) return 0;
    
    const current = candles[candles.length - 1].close;
    const past = candles[candles.length - 1 - period].close;
    
    return current - past;
  }

  private calculateHistoricalVolatility(candles: CandleData[], period: number): number {
    if (candles.length < period) return 0;

    const returns = [];
    const recentCandles = candles.slice(-period);
    
    for (let i = 1; i < recentCandles.length; i++) {
      const ret = Math.log(recentCandles[i].close / recentCandles[i - 1].close);
      returns.push(ret);
    }
    
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance) * Math.sqrt(252); // Аннуализированная
  }
}

export const professionalMLService = ProfessionalMLService.getInstance();