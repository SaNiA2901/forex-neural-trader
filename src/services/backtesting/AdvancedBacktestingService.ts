import { CandleData } from '@/types/session';

export interface AdvancedBacktestConfig {
  startDate: Date;
  endDate: Date;
  initialCapital: number;
  commission: number;
  slippage: number;
  maxPositions: number;
  riskPerTrade: number;
  leverage: number;
  marginRequirement: number;
  
  // Продвинутые настройки
  reinvestProfits: boolean;
  compoundReturns: boolean;
  dynamicPositionSizing: boolean;
  stopLossPercent: number;
  takeProfitPercent: number;
  trailingStop: boolean;
  
  // Бенчмарки
  benchmarkSymbol?: string;
  riskFreeRate: number;
}

export interface AdvancedBacktestResult {
  // Основные метрики
  totalReturn: number;
  annualizedReturn: number;
  maxDrawdown: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  
  // Торговые метрики
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  avgTradeReturn: number;
  avgWinningTrade: number;
  avgLosingTrade: number;
  largestWin: number;
  largestLoss: number;
  
  // Risk-adjusted метрики
  informationRatio: number;
  treynorRatio: number;
  jensenAlpha: number;
  beta: number;
  trackingError: number;
  
  // Продвинутая аналитика
  valueAtRisk95: number;
  expectedShortfall: number;
  ulcerIndex: number;
  recoverFactor: number;
  
  // Временной анализ
  bestMonth: number;
  worstMonth: number;
  avgMonthlyReturn: number;
  monthlyWinRate: number;
  maxConsecutiveWins: number;
  maxConsecutiveLosses: number;
  
  // Детальная информация
  trades: AdvancedTrade[];
  equityCurve: AdvancedEquityPoint[];
  monthlyReturns: MonthlyPerformance[];
  drawdownPeriods: DrawdownPeriod[];
  performanceAttribution: PerformanceAttribution;
  riskDecomposition: RiskDecomposition;
}

export interface AdvancedTrade {
  id: string;
  entryDate: Date;
  exitDate: Date;
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  side: 'long' | 'short';
  
  // P&L
  grossPnL: number;
  netPnL: number;
  pnlPercent: number;
  commission: number;
  slippage: number;
  
  // Анализ
  holdingPeriod: number;
  maxFavorableExcursion: number;
  maxAdverseExcursion: number;
  entryReason: string;
  exitReason: string;
  
  // Risk metrics
  riskRewardRatio: number;
  volatilityDuringTrade: number;
}

export interface AdvancedEquityPoint {
  date: Date;
  equity: number;
  drawdown: number;
  drawdownPercent: number;
  
  // Дополнительные метрики
  rollingReturn: number;
  rollingVolatility: number;
  rollingSharp: number;
  
  // Позиции
  longPositions: number;
  shortPositions: number;
  cash: number;
  leverage: number;
}

export interface MonthlyPerformance {
  year: number;
  month: number;
  return: number;
  trades: number;
  winRate: number;
  volatility: number;
  maxDrawdown: number;
}

export interface DrawdownPeriod {
  startDate: Date;
  endDate: Date;
  peak: number;
  trough: number;
  drawdown: number;
  duration: number;
  recoveryTime: number;
}

export interface PerformanceAttribution {
  alpha: number;
  beta: number;
  stockSelection: number;
  marketTiming: number;
  interactionEffect: number;
}

export interface RiskDecomposition {
  systematicRisk: number;
  specificRisk: number;
  concentrationRisk: number;
  liquidityRisk: number;
}

export interface AdvancedStrategy {
  name: string;
  description: string;
  parameters: { [key: string]: any };
  
  // Основные методы
  initialize: (config: AdvancedBacktestConfig) => void;
  generateSignal: (candles: CandleData[], index: number, portfolio: Portfolio) => AdvancedSignal | null;
  onTrade: (trade: AdvancedTrade) => void;
  onMarketClose: (date: Date, portfolio: Portfolio) => void;
  
  // Управление рисками
  calculatePositionSize: (signal: AdvancedSignal, portfolio: Portfolio) => number;
  shouldExit: (position: Position, currentPrice: number, portfolio: Portfolio) => boolean;
}

export interface AdvancedSignal {
  type: 'buy' | 'sell' | 'hold';
  strength: number; // 0-1
  confidence: number; // 0-1
  
  // Управление рисками
  stopLoss?: number;
  takeProfit?: number;
  positionSize?: number;
  
  // Метаданные
  reason: string;
  indicators: { [key: string]: number };
  riskLevel: 'low' | 'medium' | 'high';
}

export interface Portfolio {
  cash: number;
  equity: number;
  positions: Position[];
  
  // Метрики портфеля
  totalValue: number;
  leverage: number;
  marginUsed: number;
  freeMargin: number;
  
  // Performance
  returns: number[];
  drawdowns: number[];
  trades: AdvancedTrade[];
}

export interface Position {
  id: string;
  symbol: string;
  quantity: number;
  entryPrice: number;
  entryDate: Date;
  side: 'long' | 'short';
  
  // Risk management
  stopLoss?: number;
  takeProfit?: number;
  trailingStop?: number;
  
  // P&L tracking
  unrealizedPnL: number;
  maxFavorableExcursion: number;
  maxAdverseExcursion: number;
}

export class AdvancedBacktestingService {
  private static instance: AdvancedBacktestingService;
  
  private constructor() {}

  static getInstance(): AdvancedBacktestingService {
    if (!AdvancedBacktestingService.instance) {
      AdvancedBacktestingService.instance = new AdvancedBacktestingService();
    }
    return AdvancedBacktestingService.instance;
  }

  /**
   * Запуск продвинутого бэктестинга
   */
  async runAdvancedBacktest(
    strategy: AdvancedStrategy,
    historicalData: CandleData[],
    config: AdvancedBacktestConfig,
    benchmarkData?: CandleData[]
  ): Promise<AdvancedBacktestResult> {
    
    console.log(`🚀 Starting advanced backtest: ${strategy.name}`);
    console.log(`📊 Data range: ${config.startDate.toISOString().split('T')[0]} to ${config.endDate.toISOString().split('T')[0]}`);
    console.log(`💰 Initial capital: $${config.initialCapital.toLocaleString()}`);

    // Инициализация
    strategy.initialize(config);
    const portfolio = this.initializePortfolio(config);
    const filteredData = this.filterDataByDateRange(historicalData, config.startDate, config.endDate);
    
    if (filteredData.length === 0) {
      throw new Error('No historical data available for the specified date range');
    }

    const equityCurve: AdvancedEquityPoint[] = [];
    const trades: AdvancedTrade[] = [];
    let peak = config.initialCapital;
    let maxDrawdown = 0;

    // Основной цикл бэктестинга
    for (let i = 1; i < filteredData.length; i++) {
      const currentCandle = filteredData[i];
      const currentPrice = currentCandle.close;
      const currentDate = new Date(currentCandle.candle_datetime);

      // Обновляем P&L открытых позиций
      this.updatePositions(portfolio, currentPrice);

      // Проверяем stop-loss и take-profit
      this.checkExitConditions(portfolio, currentPrice, config, trades);

      // Генерируем торговый сигнал
      const signal = strategy.generateSignal(filteredData.slice(0, i + 1), i, portfolio);
      
      if (signal && signal.type !== 'hold') {
        const trade = this.executeSignal(signal, currentCandle, portfolio, config, strategy);
        if (trade) {
          trades.push(trade);
          strategy.onTrade(trade);
        }
      }

      // Обновляем метрики портфеля
      portfolio.totalValue = portfolio.cash + this.calculatePositionsValue(portfolio, currentPrice);
      portfolio.leverage = this.calculateLeverage(portfolio);

      // Записываем точку equity curve
      peak = Math.max(peak, portfolio.totalValue);
      const currentDrawdown = (peak - portfolio.totalValue) / peak;
      maxDrawdown = Math.max(maxDrawdown, currentDrawdown);

      const equityPoint: AdvancedEquityPoint = {
        date: currentDate,
        equity: portfolio.totalValue,
        drawdown: peak - portfolio.totalValue,
        drawdownPercent: currentDrawdown,
        rollingReturn: this.calculateRollingReturn(equityCurve, 20),
        rollingVolatility: this.calculateRollingVolatility(equityCurve, 20),
        rollingSharp: this.calculateRollingSharpe(equityCurve, 20),
        longPositions: portfolio.positions.filter(p => p.side === 'long').length,
        shortPositions: portfolio.positions.filter(p => p.side === 'short').length,
        cash: portfolio.cash,
        leverage: portfolio.leverage
      };

      equityCurve.push(equityPoint);

      // Вызываем hook конца дня
      strategy.onMarketClose(currentDate, portfolio);
    }

    // Закрываем все открытые позиции
    this.closeAllPositions(portfolio, filteredData[filteredData.length - 1], config, trades);

    // Вычисляем результаты
    return this.calculateAdvancedResults(
      trades, 
      equityCurve, 
      config, 
      benchmarkData ? this.filterDataByDateRange(benchmarkData, config.startDate, config.endDate) : undefined
    );
  }

  /**
   * Создает продвинутую стратегию Mean Reversion
   */
  createMeanReversionStrategy(params: {
    lookbackPeriod: number;
    entryThreshold: number;
    exitThreshold: number;
    rsiPeriod: number;
    volatilityFilter: boolean;
  }): AdvancedStrategy {
    
    return {
      name: 'Advanced Mean Reversion',
      description: 'Mean reversion strategy with RSI and volatility filtering',
      parameters: params,
      
      initialize: (config: AdvancedBacktestConfig) => {
        console.log(`📋 Initializing Mean Reversion Strategy with parameters:`, params);
      },
      
      generateSignal: (candles: CandleData[], index: number, portfolio: Portfolio): AdvancedSignal | null => {
        if (index < params.lookbackPeriod + params.rsiPeriod) return null;

        const current = candles[index];
        const closes = candles.slice(index - params.lookbackPeriod, index + 1).map(c => c.close);
        
        // Расчет индикаторов
        const sma = closes.reduce((sum, price) => sum + price, 0) / closes.length;
        const deviation = (current.close - sma) / sma;
        const rsi = this.calculateRSI(candles, index, params.rsiPeriod);
        const volatility = this.calculateVolatility(candles, index, 20);
        
        // Фильтр волатильности
        if (params.volatilityFilter && volatility > 0.03) { // Высокая волатильность
          return { type: 'hold', strength: 0, confidence: 0, reason: 'High volatility filter', indicators: {}, riskLevel: 'high' };
        }

        // Сигналы
        if (deviation < -params.entryThreshold && rsi < 30) {
          return {
            type: 'buy',
            strength: Math.abs(deviation) / params.entryThreshold,
            confidence: (30 - rsi) / 30,
            reason: 'Mean reversion buy signal',
            indicators: { deviation, rsi, volatility },
            riskLevel: volatility > 0.02 ? 'medium' : 'low',
            stopLoss: current.close * 0.95,
            takeProfit: current.close * 1.05
          };
        }
        
        if (deviation > params.entryThreshold && rsi > 70) {
          return {
            type: 'sell',
            strength: deviation / params.entryThreshold,
            confidence: (rsi - 70) / 30,
            reason: 'Mean reversion sell signal',
            indicators: { deviation, rsi, volatility },
            riskLevel: volatility > 0.02 ? 'medium' : 'low',
            stopLoss: current.close * 1.05,
            takeProfit: current.close * 0.95
          };
        }

        return null;
      },
      
      calculatePositionSize: (signal: AdvancedSignal, portfolio: Portfolio): number => {
        const riskAdjuster = signal.riskLevel === 'high' ? 0.5 : signal.riskLevel === 'medium' ? 0.75 : 1.0;
        const baseSize = portfolio.totalValue * 0.02; // 2% risk per trade
        return baseSize * signal.confidence * riskAdjuster;
      },
      
      shouldExit: (position: Position, currentPrice: number, portfolio: Portfolio): boolean => {
        // Exit на противоположном сигнале или stop/target
        const priceChange = (currentPrice - position.entryPrice) / position.entryPrice;
        
        if (position.side === 'long') {
          return priceChange > params.exitThreshold || priceChange < -0.03; // 3% stop loss
        } else {
          return priceChange < -params.exitThreshold || priceChange > 0.03;
        }
      },
      
      onTrade: (trade: AdvancedTrade) => {
        // Logging или адаптация параметров
      },
      
      onMarketClose: (date: Date, portfolio: Portfolio) => {
        // Ночная обработка, перебалансировка и т.д.
      }
    };
  }

  /**
   * Создает momentum стратегию
   */
  createMomentumStrategy(params: {
    fastMA: number;
    slowMA: number;
    momentumPeriod: number;
    volumeFilter: boolean;
    minVolume: number;
  }): AdvancedStrategy {
    
    return {
      name: 'Advanced Momentum',
      description: 'Momentum strategy with moving averages and volume confirmation',
      parameters: params,
      
      initialize: (config: AdvancedBacktestConfig) => {
        console.log(`📋 Initializing Momentum Strategy:`, params);
      },
      
      generateSignal: (candles: CandleData[], index: number, portfolio: Portfolio): AdvancedSignal | null => {
        if (index < Math.max(params.slowMA, params.momentumPeriod)) return null;

        const current = candles[index];
        
        // Расчет индикаторов
        const fastMA = this.calculateSMA(candles, index, params.fastMA);
        const slowMA = this.calculateSMA(candles, index, params.slowMA);
        const prevFastMA = this.calculateSMA(candles, index - 1, params.fastMA);
        const prevSlowMA = this.calculateSMA(candles, index - 1, params.slowMA);
        const momentum = this.calculateMomentum(candles, index, params.momentumPeriod);
        const avgVolume = this.calculateAverageVolume(candles, index, 20);
        
        // Фильтр объема
        if (params.volumeFilter && current.volume < avgVolume * params.minVolume) {
          return { type: 'hold', strength: 0, confidence: 0, reason: 'Low volume filter', indicators: {}, riskLevel: 'medium' };
        }

        // Bullish crossover
        if (prevFastMA <= prevSlowMA && fastMA > slowMA && momentum > 0) {
          const strength = (fastMA - slowMA) / slowMA;
          return {
            type: 'buy',
            strength: Math.min(1, strength * 100),
            confidence: Math.min(1, current.volume / avgVolume),
            reason: 'Bullish MA crossover with positive momentum',
            indicators: { fastMA, slowMA, momentum, volumeRatio: current.volume / avgVolume },
            riskLevel: 'medium',
            stopLoss: current.close * 0.96,
            takeProfit: current.close * 1.08
          };
        }
        
        // Bearish crossover
        if (prevFastMA >= prevSlowMA && fastMA < slowMA && momentum < 0) {
          const strength = (slowMA - fastMA) / fastMA;
          return {
            type: 'sell',
            strength: Math.min(1, strength * 100),
            confidence: Math.min(1, current.volume / avgVolume),
            reason: 'Bearish MA crossover with negative momentum',
            indicators: { fastMA, slowMA, momentum, volumeRatio: current.volume / avgVolume },
            riskLevel: 'medium',
            stopLoss: current.close * 1.04,
            takeProfit: current.close * 0.92
          };
        }

        return null;
      },
      
      calculatePositionSize: (signal: AdvancedSignal, portfolio: Portfolio): number => {
        return portfolio.totalValue * 0.05 * signal.strength * signal.confidence; // 5% base risk
      },
      
      shouldExit: (position: Position, currentPrice: number, portfolio: Portfolio): boolean => {
        const pnlPercent = (currentPrice - position.entryPrice) / position.entryPrice;
        return Math.abs(pnlPercent) > 0.05; // 5% stop/target
      },
      
      onTrade: (trade: AdvancedTrade) => {},
      onMarketClose: (date: Date, portfolio: Portfolio) => {}
    };
  }

  // === ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ===

  private initializePortfolio(config: AdvancedBacktestConfig): Portfolio {
    return {
      cash: config.initialCapital,
      equity: config.initialCapital,
      positions: [],
      totalValue: config.initialCapital,
      leverage: 1,
      marginUsed: 0,
      freeMargin: config.initialCapital,
      returns: [],
      drawdowns: [],
      trades: []
    };
  }

  private filterDataByDateRange(data: CandleData[], startDate: Date, endDate: Date): CandleData[] {
    return data.filter(candle => {
      const candleDate = new Date(candle.candle_datetime);
      return candleDate >= startDate && candleDate <= endDate;
    });
  }

  private updatePositions(portfolio: Portfolio, currentPrice: number): void {
    portfolio.positions.forEach(position => {
      const priceDiff = currentPrice - position.entryPrice;
      position.unrealizedPnL = position.side === 'long' ? 
        priceDiff * position.quantity : 
        -priceDiff * position.quantity;
      
      // Update MAE и MFE
      if (position.side === 'long') {
        position.maxFavorableExcursion = Math.max(position.maxFavorableExcursion, priceDiff);
        position.maxAdverseExcursion = Math.min(position.maxAdverseExcursion, priceDiff);
      } else {
        position.maxFavorableExcursion = Math.max(position.maxFavorableExcursion, -priceDiff);
        position.maxAdverseExcursion = Math.min(position.maxAdverseExcursion, -priceDiff);
      }
    });
  }

  private checkExitConditions(
    portfolio: Portfolio, 
    currentPrice: number, 
    config: AdvancedBacktestConfig, 
    trades: AdvancedTrade[]
  ): void {
    const positionsToClose = portfolio.positions.filter(position => {
      // Stop Loss
      if (position.stopLoss) {
        if ((position.side === 'long' && currentPrice <= position.stopLoss) ||
            (position.side === 'short' && currentPrice >= position.stopLoss)) {
          return true;
        }
      }
      
      // Take Profit
      if (position.takeProfit) {
        if ((position.side === 'long' && currentPrice >= position.takeProfit) ||
            (position.side === 'short' && currentPrice <= position.takeProfit)) {
          return true;
        }
      }
      
      return false;
    });

    positionsToClose.forEach(position => {
      const trade = this.closePosition(position, currentPrice, config, 'Stop/Target hit');
      trades.push(trade);
      portfolio.positions = portfolio.positions.filter(p => p.id !== position.id);
      portfolio.cash += trade.netPnL;
    });
  }

  private executeSignal(
    signal: AdvancedSignal,
    candle: CandleData,
    portfolio: Portfolio,
    config: AdvancedBacktestConfig,
    strategy: AdvancedStrategy
  ): AdvancedTrade | null {
    
    const positionSize = strategy.calculatePositionSize(signal, portfolio);
    const price = candle.close * (1 + (signal.type === 'buy' ? config.slippage : -config.slippage));
    const quantity = positionSize / price;
    const commission = positionSize * config.commission;
    
    // Проверяем достаточность средств
    if (portfolio.cash < positionSize + commission) {
      return null;
    }

    // Создаем позицию
    const position: Position = {
      id: `pos-${Date.now()}-${Math.random()}`,
      symbol: 'SYMBOL',
      quantity,
      entryPrice: price,
      entryDate: new Date(candle.candle_datetime),
      side: signal.type === 'buy' ? 'long' : 'short',
      stopLoss: signal.stopLoss,
      takeProfit: signal.takeProfit,
      unrealizedPnL: 0,
      maxFavorableExcursion: 0,
      maxAdverseExcursion: 0
    };

    portfolio.positions.push(position);
    portfolio.cash -= positionSize + commission;

    // Если это краткосрочная стратегия, сразу планируем выход
    // (для демонстрации, в реальности определяется стратегией)
    return null; // Trade создается при закрытии позиции
  }

  private closePosition(
    position: Position, 
    currentPrice: number, 
    config: AdvancedBacktestConfig, 
    exitReason: string
  ): AdvancedTrade {
    
    const exitPrice = currentPrice * (1 + (position.side === 'long' ? -config.slippage : config.slippage));
    const commission = position.quantity * exitPrice * config.commission;
    
    let grossPnL: number;
    if (position.side === 'long') {
      grossPnL = (exitPrice - position.entryPrice) * position.quantity;
    } else {
      grossPnL = (position.entryPrice - exitPrice) * position.quantity;
    }
    
    const netPnL = grossPnL - commission;
    const pnlPercent = netPnL / (position.entryPrice * position.quantity);
    
    return {
      id: `trade-${Date.now()}`,
      entryDate: position.entryDate,
      exitDate: new Date(),
      entryPrice: position.entryPrice,
      exitPrice,
      quantity: position.quantity,
      side: position.side,
      grossPnL,
      netPnL,
      pnlPercent,
      commission,
      slippage: Math.abs(exitPrice - currentPrice),
      holdingPeriod: (new Date().getTime() - position.entryDate.getTime()) / (1000 * 60 * 60 * 24),
      maxFavorableExcursion: position.maxFavorableExcursion,
      maxAdverseExcursion: position.maxAdverseExcursion,
      entryReason: 'Strategy signal',
      exitReason,
      riskRewardRatio: position.maxFavorableExcursion / Math.abs(position.maxAdverseExcursion || 1),
      volatilityDuringTrade: 0.02 // Placeholder
    };
  }

  private closeAllPositions(
    portfolio: Portfolio, 
    lastCandle: CandleData, 
    config: AdvancedBacktestConfig, 
    trades: AdvancedTrade[]
  ): void {
    portfolio.positions.forEach(position => {
      const trade = this.closePosition(position, lastCandle.close, config, 'End of backtest');
      trades.push(trade);
      portfolio.cash += trade.netPnL;
    });
    portfolio.positions = [];
  }

  private calculatePositionsValue(portfolio: Portfolio, currentPrice: number): number {
    return portfolio.positions.reduce((total, position) => {
      return total + (position.quantity * currentPrice);
    }, 0);
  }

  private calculateLeverage(portfolio: Portfolio): number {
    const positionValue = portfolio.positions.reduce((total, position) => {
      return total + Math.abs(position.quantity * position.entryPrice);
    }, 0);
    
    return portfolio.totalValue > 0 ? positionValue / portfolio.totalValue : 1;
  }

  private calculateAdvancedResults(
    trades: AdvancedTrade[], 
    equityCurve: AdvancedEquityPoint[], 
    config: AdvancedBacktestConfig,
    benchmarkData?: CandleData[]
  ): AdvancedBacktestResult {
    
    if (trades.length === 0 || equityCurve.length === 0) {
      return this.getEmptyAdvancedResults();
    }

    // Основные расчеты
    const totalReturn = (equityCurve[equityCurve.length - 1].equity - config.initialCapital) / config.initialCapital;
    const daysInPeriod = equityCurve.length;
    const annualizedReturn = Math.pow(1 + totalReturn, 365 / daysInPeriod) - 1;
    
    const returns = equityCurve.map((point, i) => 
      i === 0 ? 0 : (point.equity - equityCurve[i - 1].equity) / equityCurve[i - 1].equity
    ).slice(1);
    
    // Risk metrics
    const maxDrawdown = Math.max(...equityCurve.map(point => point.drawdownPercent));
    const sharpeRatio = this.calculateSharpeRatio(returns, config.riskFreeRate);
    const sortinoRatio = this.calculateSortinoRatio(returns, config.riskFreeRate);
    const calmarRatio = maxDrawdown === 0 ? 0 : annualizedReturn / maxDrawdown;
    
    // Trade analysis
    const winningTrades = trades.filter(t => t.netPnL > 0);
    const losingTrades = trades.filter(t => t.netPnL < 0);
    const winRate = winningTrades.length / trades.length;
    
    const grossProfit = winningTrades.reduce((sum, t) => sum + t.netPnL, 0);
    const grossLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.netPnL, 0));
    const profitFactor = grossLoss === 0 ? (grossProfit > 0 ? Infinity : 0) : grossProfit / grossLoss;
    
    // Advanced metrics
    const valueAtRisk95 = this.calculateVaR(returns, 0.05);
    const expectedShortfall = this.calculateExpectedShortfall(returns, 0.05);
    const ulcerIndex = this.calculateUlcerIndex(equityCurve);
    
    // Monthly analysis
    const monthlyReturns = this.calculateMonthlyPerformance(equityCurve);
    const drawdownPeriods = this.identifyDrawdownPeriods(equityCurve);
    
    // Benchmark comparison
    let beta = 1;
    let informationRatio = 0;
    let jensenAlpha = 0;
    let trackingError = 0;
    
    if (benchmarkData) {
      const benchmarkReturns = this.calculateBenchmarkReturns(benchmarkData);
      beta = this.calculateBeta(returns, benchmarkReturns);
      jensenAlpha = this.calculateAlpha(returns, benchmarkReturns, config.riskFreeRate, beta);
      informationRatio = this.calculateInformationRatio(returns, benchmarkReturns);
      trackingError = this.calculateTrackingError(returns, benchmarkReturns);
    }

    return {
      // Basic metrics
      totalReturn,
      annualizedReturn,
      maxDrawdown,
      sharpeRatio,
      sortinoRatio,
      calmarRatio,
      
      // Trading metrics
      winRate,
      profitFactor,
      totalTrades: trades.length,
      avgTradeReturn: trades.reduce((sum, t) => sum + t.netPnL, 0) / trades.length,
      avgWinningTrade: winningTrades.length > 0 ? grossProfit / winningTrades.length : 0,
      avgLosingTrade: losingTrades.length > 0 ? -grossLoss / losingTrades.length : 0,
      largestWin: Math.max(...trades.map(t => t.netPnL)),
      largestLoss: Math.min(...trades.map(t => t.netPnL)),
      
      // Risk-adjusted metrics
      informationRatio,
      treynorRatio: beta !== 0 ? annualizedReturn / beta : 0,
      jensenAlpha,
      beta,
      trackingError,
      
      // Advanced analytics
      valueAtRisk95,
      expectedShortfall,
      ulcerIndex,
      recoverFactor: maxDrawdown !== 0 ? totalReturn / maxDrawdown : 0,
      
      // Time analysis
      bestMonth: monthlyReturns.length > 0 ? Math.max(...monthlyReturns.map(m => m.return)) : 0,
      worstMonth: monthlyReturns.length > 0 ? Math.min(...monthlyReturns.map(m => m.return)) : 0,
      avgMonthlyReturn: monthlyReturns.reduce((sum, m) => sum + m.return, 0) / monthlyReturns.length,
      monthlyWinRate: monthlyReturns.filter(m => m.return > 0).length / monthlyReturns.length,
      maxConsecutiveWins: this.calculateMaxConsecutive(trades, true),
      maxConsecutiveLosses: this.calculateMaxConsecutive(trades, false),
      
      // Details
      trades,
      equityCurve,
      monthlyReturns,
      drawdownPeriods,
      performanceAttribution: {
        alpha: jensenAlpha,
        beta,
        stockSelection: 0, // Placeholder
        marketTiming: 0,   // Placeholder
        interactionEffect: 0 // Placeholder
      },
      riskDecomposition: {
        systematicRisk: 0.6, // Placeholder
        specificRisk: 0.3,   // Placeholder
        concentrationRisk: 0.1, // Placeholder
        liquidityRisk: 0.05     // Placeholder
      }
    };
  }

  // === РАСЧЕТНЫЕ МЕТОДЫ ===

  private calculateSharpeRatio(returns: number[], riskFreeRate: number): number {
    if (returns.length === 0) return 0;
    
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const annualizedReturn = avgReturn * 252;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance * 252);
    
    return volatility === 0 ? 0 : (annualizedReturn - riskFreeRate) / volatility;
  }

  private calculateSortinoRatio(returns: number[], riskFreeRate: number): number {
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const annualizedReturn = avgReturn * 252;
    
    const negativeReturns = returns.filter(r => r < 0);
    if (negativeReturns.length === 0) return Infinity;
    
    const downwardDeviation = Math.sqrt(
      negativeReturns.reduce((sum, r) => sum + Math.pow(r, 2), 0) / negativeReturns.length * 252
    );
    
    return downwardDeviation === 0 ? 0 : (annualizedReturn - riskFreeRate) / downwardDeviation;
  }

  private calculateVaR(returns: number[], confidenceLevel: number): number {
    const sortedReturns = [...returns].sort((a, b) => a - b);
    const index = Math.floor(confidenceLevel * sortedReturns.length);
    return sortedReturns[index] || 0;
  }

  private calculateExpectedShortfall(returns: number[], confidenceLevel: number): number {
    const var95 = this.calculateVaR(returns, confidenceLevel);
    const tailReturns = returns.filter(r => r <= var95);
    return tailReturns.length > 0 
      ? tailReturns.reduce((sum, r) => sum + r, 0) / tailReturns.length 
      : 0;
  }

  private calculateUlcerIndex(equityCurve: AdvancedEquityPoint[]): number {
    const drawdowns = equityCurve.map(point => Math.pow(point.drawdownPercent, 2));
    const avgSquaredDrawdown = drawdowns.reduce((sum, dd) => sum + dd, 0) / drawdowns.length;
    return Math.sqrt(avgSquaredDrawdown);
  }

  private calculateBeta(returns: number[], marketReturns: number[]): number {
    const n = Math.min(returns.length, marketReturns.length);
    const returnsSlice = returns.slice(0, n);
    const marketSlice = marketReturns.slice(0, n);
    
    const meanReturn = returnsSlice.reduce((sum, r) => sum + r, 0) / n;
    const meanMarket = marketSlice.reduce((sum, r) => sum + r, 0) / n;
    
    let covariance = 0;
    let marketVariance = 0;
    
    for (let i = 0; i < n; i++) {
      covariance += (returnsSlice[i] - meanReturn) * (marketSlice[i] - meanMarket);
      marketVariance += Math.pow(marketSlice[i] - meanMarket, 2);
    }
    
    return marketVariance === 0 ? 1 : covariance / marketVariance;
  }

  private calculateAlpha(
    returns: number[], 
    marketReturns: number[], 
    riskFreeRate: number, 
    beta: number
  ): number {
    const portfolioReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length * 252;
    const marketReturn = marketReturns.reduce((sum, r) => sum + r, 0) / marketReturns.length * 252;
    
    return portfolioReturn - (riskFreeRate + beta * (marketReturn - riskFreeRate));
  }

  private calculateInformationRatio(returns: number[], benchmarkReturns: number[]): number {
    const n = Math.min(returns.length, benchmarkReturns.length);
    const excessReturns = returns.slice(0, n).map((r, i) => r - benchmarkReturns[i]);
    
    const meanExcessReturn = excessReturns.reduce((sum, r) => sum + r, 0) / n;
    const trackingError = Math.sqrt(
      excessReturns.reduce((sum, r) => sum + Math.pow(r - meanExcessReturn, 2), 0) / (n - 1)
    );
    
    return trackingError === 0 ? 0 : meanExcessReturn / trackingError;
  }

  private calculateTrackingError(returns: number[], benchmarkReturns: number[]): number {
    const n = Math.min(returns.length, benchmarkReturns.length);
    const excessReturns = returns.slice(0, n).map((r, i) => r - benchmarkReturns[i]);
    
    const meanExcessReturn = excessReturns.reduce((sum, r) => sum + r, 0) / n;
    const variance = excessReturns.reduce((sum, r) => sum + Math.pow(r - meanExcessReturn, 2), 0) / (n - 1);
    
    return Math.sqrt(variance * 252);
  }

  // Вспомогательные методы для расчета индикаторов
  private calculateRSI(candles: CandleData[], index: number, period: number): number {
    // Реализация аналогична предыдущему сервису
    return 50; // Placeholder
  }

  private calculateSMA(candles: CandleData[], index: number, period: number): number {
    if (index < period - 1) return 0;
    
    const sum = candles.slice(index - period + 1, index + 1)
      .reduce((acc, candle) => acc + candle.close, 0);
    return sum / period;
  }

  private calculateVolatility(candles: CandleData[], index: number, period: number): number {
    if (index < period) return 0;
    
    const returns = [];
    for (let i = index - period + 1; i <= index; i++) {
      if (i > 0) {
        returns.push(Math.log(candles[i].close / candles[i - 1].close));
      }
    }
    
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance);
  }

  private calculateMomentum(candles: CandleData[], index: number, period: number): number {
    if (index < period) return 0;
    return candles[index].close - candles[index - period].close;
  }

  private calculateAverageVolume(candles: CandleData[], index: number, period: number): number {
    if (index < period - 1) return 0;
    
    const sum = candles.slice(index - period + 1, index + 1)
      .reduce((acc, candle) => acc + candle.volume, 0);
    return sum / period;
  }

  private calculateRollingReturn(equityCurve: AdvancedEquityPoint[], period: number): number {
    if (equityCurve.length < period) return 0;
    const current = equityCurve[equityCurve.length - 1].equity;
    const past = equityCurve[equityCurve.length - period].equity;
    return (current - past) / past;
  }

  private calculateRollingVolatility(equityCurve: AdvancedEquityPoint[], period: number): number {
    if (equityCurve.length < period) return 0;
    
    const returns = [];
    const recentPoints = equityCurve.slice(-period);
    for (let i = 1; i < recentPoints.length; i++) {
      const ret = (recentPoints[i].equity - recentPoints[i-1].equity) / recentPoints[i-1].equity;
      returns.push(ret);
    }
    
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance);
  }

  private calculateRollingSharpe(equityCurve: AdvancedEquityPoint[], period: number): number {
    const rollingReturn = this.calculateRollingReturn(equityCurve, period);
    const rollingVol = this.calculateRollingVolatility(equityCurve, period);
    
    return rollingVol === 0 ? 0 : rollingReturn / rollingVol;
  }

  private calculateMonthlyPerformance(equityCurve: AdvancedEquityPoint[]): MonthlyPerformance[] {
    const monthlyData = new Map<string, AdvancedEquityPoint[]>();
    
    equityCurve.forEach(point => {
      const year = point.date.getFullYear();
      const month = point.date.getMonth();
      const key = `${year}-${month}`;
      
      if (!monthlyData.has(key)) {
        monthlyData.set(key, []);
      }
      monthlyData.get(key)!.push(point);
    });
    
    return Array.from(monthlyData.entries()).map(([key, points]) => {
      const [year, month] = key.split('-').map(Number);
      const startEquity = points[0].equity;
      const endEquity = points[points.length - 1].equity;
      const monthReturn = (endEquity - startEquity) / startEquity;
      
      // Дополнительная статистика за месяц
      const returns = points.map((point, i) => 
        i === 0 ? 0 : (point.equity - points[i-1].equity) / points[i-1].equity
      ).slice(1);
      
      const positiveReturns = returns.filter(r => r > 0).length;
      const winRate = returns.length > 0 ? positiveReturns / returns.length : 0;
      
      const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
      const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
      const volatility = Math.sqrt(variance);
      
      const maxDrawdown = Math.max(...points.map(p => p.drawdownPercent));
      
      return {
        year,
        month: month + 1,
        return: monthReturn,
        trades: 0, // Placeholder - нужна привязка к сделкам
        winRate,
        volatility,
        maxDrawdown
      };
    });
  }

  private identifyDrawdownPeriods(equityCurve: AdvancedEquityPoint[]): DrawdownPeriod[] {
    const periods: DrawdownPeriod[] = [];
    let inDrawdown = false;
    let peak = 0;
    let peakDate = equityCurve[0]?.date;
    let startDate = peakDate;
    
    equityCurve.forEach(point => {
      if (point.equity > peak) {
        // Новый пик
        if (inDrawdown) {
          // Конец периода просадки
          periods.push({
            startDate,
            endDate: point.date,
            peak,
            trough: Math.min(...equityCurve.filter(p => p.date >= startDate && p.date <= point.date).map(p => p.equity)),
            drawdown: (peak - Math.min(...equityCurve.filter(p => p.date >= startDate && p.date <= point.date).map(p => p.equity))) / peak,
            duration: (point.date.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24),
            recoveryTime: (point.date.getTime() - peakDate.getTime()) / (1000 * 60 * 60 * 24)
          });
          inDrawdown = false;
        }
        peak = point.equity;
        peakDate = point.date;
      } else if (!inDrawdown && point.equity < peak) {
        // Начало просадки
        inDrawdown = true;
        startDate = point.date;
      }
    });
    
    return periods;
  }

  private calculateMaxConsecutive(trades: AdvancedTrade[], wins: boolean): number {
    let maxConsecutive = 0;
    let currentConsecutive = 0;
    
    trades.forEach(trade => {
      const isWin = trade.netPnL > 0;
      if (isWin === wins) {
        currentConsecutive++;
        maxConsecutive = Math.max(maxConsecutive, currentConsecutive);
      } else {
        currentConsecutive = 0;
      }
    });
    
    return maxConsecutive;
  }

  private calculateBenchmarkReturns(benchmarkData: CandleData[]): number[] {
    const returns: number[] = [];
    for (let i = 1; i < benchmarkData.length; i++) {
      const ret = (benchmarkData[i].close - benchmarkData[i-1].close) / benchmarkData[i-1].close;
      returns.push(ret);
    }
    return returns;
  }

  private getEmptyAdvancedResults(): AdvancedBacktestResult {
    return {
      totalReturn: 0,
      annualizedReturn: 0,
      maxDrawdown: 0,
      sharpeRatio: 0,
      sortinoRatio: 0,
      calmarRatio: 0,
      winRate: 0,
      profitFactor: 0,
      totalTrades: 0,
      avgTradeReturn: 0,
      avgWinningTrade: 0,
      avgLosingTrade: 0,
      largestWin: 0,
      largestLoss: 0,
      informationRatio: 0,
      treynorRatio: 0,
      jensenAlpha: 0,
      beta: 1,
      trackingError: 0,
      valueAtRisk95: 0,
      expectedShortfall: 0,
      ulcerIndex: 0,
      recoverFactor: 0,
      bestMonth: 0,
      worstMonth: 0,
      avgMonthlyReturn: 0,
      monthlyWinRate: 0,
      maxConsecutiveWins: 0,
      maxConsecutiveLosses: 0,
      trades: [],
      equityCurve: [],
      monthlyReturns: [],
      drawdownPeriods: [],
      performanceAttribution: {
        alpha: 0,
        beta: 1,
        stockSelection: 0,
        marketTiming: 0,
        interactionEffect: 0
      },
      riskDecomposition: {
        systematicRisk: 0,
        specificRisk: 0,
        concentrationRisk: 0,
        liquidityRisk: 0
      }
    };
  }
}

export const advancedBacktestingService = AdvancedBacktestingService.getInstance();