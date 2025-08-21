import { CandleData } from '@/types/session';

export interface BacktestConfig {
  startDate: Date;
  endDate: Date;
  initialCapital: number;
  commission: number;
  slippage: number;
  maxPositions: number;
  riskPerTrade: number;
}

export interface BacktestResult {
  totalReturn: number;
  annualizedReturn: number;
  maxDrawdown: number;
  sharpeRatio: number;
  winRate: number;
  profitFactor: number;
  totalTrades: number;
  avgTradeReturn: number;
  bestTrade: number;
  worstTrade: number;
  calmarRatio: number;
  recoveryFactor: number;
  trades: Trade[];
  equityCurve: EquityPoint[];
  monthlyReturns: MonthlyReturn[];
}

export interface Trade {
  entryDate: Date;
  exitDate: Date;
  entryPrice: number;
  exitPrice: number;
  quantity: number;
  side: 'long' | 'short';
  pnl: number;
  pnlPercent: number;
  holdingPeriod: number;
  commission: number;
  slippage: number;
}

export interface EquityPoint {
  date: Date;
  equity: number;
  drawdown: number;
}

export interface MonthlyReturn {
  year: number;
  month: number;
  return: number;
}

export interface Strategy {
  name: string;
  generate: (candles: CandleData[], index: number) => Signal | null;
}

export interface Signal {
  type: 'buy' | 'sell';
  strength: number; // 0-1
  stopLoss?: number;
  takeProfit?: number;
  confidence: number;
}

export class BacktestingService {
  private static instance: BacktestingService;
  
  private constructor() {}

  static getInstance(): BacktestingService {
    if (!BacktestingService.instance) {
      BacktestingService.instance = new BacktestingService();
    }
    return BacktestingService.instance;
  }

  /**
   * Запускает бэктест стратегии
   */
  async runBacktest(
    strategy: Strategy,
    historicalData: CandleData[],
    config: BacktestConfig
  ): Promise<BacktestResult> {
    const trades: Trade[] = [];
    const equityCurve: EquityPoint[] = [];
    let currentEquity = config.initialCapital;
    let peak = config.initialCapital;
    let maxDrawdown = 0;
    let openPosition: any = null;

    // Фильтруем данные по датам
    const filteredData = this.filterDataByDateRange(historicalData, config.startDate, config.endDate);
    
    if (filteredData.length === 0) {
      throw new Error('Недостаточно исторических данных для бэктеста');
    }

    console.log(`Запуск бэктеста ${strategy.name}: ${filteredData.length} свечей`);

    for (let i = 1; i < filteredData.length; i++) {
      const currentCandle = filteredData[i];
      const signal = strategy.generate(filteredData.slice(0, i + 1), i);

      // Закрытие позиции
      if (openPosition) {
        const shouldClose = this.shouldClosePosition(openPosition, currentCandle, signal);
        
        if (shouldClose) {
          const trade = this.closePosition(openPosition, currentCandle, config);
          trades.push(trade);
          currentEquity += trade.pnl;
          openPosition = null;
        }
      }

      // Открытие новой позиции
      if (!openPosition && signal && trades.length < config.maxPositions * 100) {
        openPosition = this.openPosition(signal, currentCandle, currentEquity, config);
      }

      // Обновление кривой доходности
      peak = Math.max(peak, currentEquity);
      const drawdown = (peak - currentEquity) / peak;
      maxDrawdown = Math.max(maxDrawdown, drawdown);

      equityCurve.push({
        date: new Date(currentCandle.created_at),
        equity: currentEquity,
        drawdown: drawdown
      });
    }

    // Закрываем открытую позицию в конце
    if (openPosition) {
      const lastCandle = filteredData[filteredData.length - 1];
      const trade = this.closePosition(openPosition, lastCandle, config);
      trades.push(trade);
      currentEquity += trade.pnl;
    }

    return this.calculateResults(trades, equityCurve, config);
  }

  /**
   * Создает простую стратегию на основе скользящих средних
   */
  createMovingAverageStrategy(fastPeriod: number = 20, slowPeriod: number = 50): Strategy {
    return {
      name: `MA ${fastPeriod}/${slowPeriod}`,
      generate: (candles: CandleData[], index: number) => {
        if (index < slowPeriod) return null;

        const fastMA = this.calculateSMA(candles, index, fastPeriod);
        const slowMA = this.calculateSMA(candles, index, slowPeriod);
        const prevFastMA = this.calculateSMA(candles, index - 1, fastPeriod);
        const prevSlowMA = this.calculateSMA(candles, index - 1, slowPeriod);

        // Пересечение скользящих средних
        if (prevFastMA <= prevSlowMA && fastMA > slowMA) {
          return {
            type: 'buy',
            strength: 0.7,
            confidence: 0.6,
            stopLoss: candles[index].close * 0.98,
            takeProfit: candles[index].close * 1.04
          };
        }

        if (prevFastMA >= prevSlowMA && fastMA < slowMA) {
          return {
            type: 'sell',
            strength: 0.7,
            confidence: 0.6,
            stopLoss: candles[index].close * 1.02,
            takeProfit: candles[index].close * 0.96
          };
        }

        return null;
      }
    };
  }

  /**
   * Создает стратегию на основе RSI
   */
  createRSIStrategy(period: number = 14, overbought: number = 70, oversold: number = 30): Strategy {
    return {
      name: `RSI ${period}`,
      generate: (candles: CandleData[], index: number) => {
        if (index < period + 1) return null;

        const rsi = this.calculateRSI(candles, index, period);
        const prevRSI = this.calculateRSI(candles, index - 1, period);

        // Пересечение уровней перепроданности/перекупленности
        if (prevRSI < oversold && rsi >= oversold) {
          return {
            type: 'buy',
            strength: 0.8,
            confidence: 0.7,
            stopLoss: candles[index].close * 0.97,
            takeProfit: candles[index].close * 1.05
          };
        }

        if (prevRSI > overbought && rsi <= overbought) {
          return {
            type: 'sell',
            strength: 0.8,
            confidence: 0.7,
            stopLoss: candles[index].close * 1.03,
            takeProfit: candles[index].close * 0.95
          };
        }

        return null;
      }
    };
  }

  /**
   * Сравнивает несколько стратегий
   */
  async compareStrategies(
    strategies: Strategy[],
    historicalData: CandleData[],
    config: BacktestConfig
  ): Promise<{ strategy: string; results: BacktestResult }[]> {
    const results = [];

    for (const strategy of strategies) {
      try {
        const result = await this.runBacktest(strategy, historicalData, config);
        results.push({ strategy: strategy.name, results: result });
      } catch (error) {
        console.error(`Ошибка при тестировании стратегии ${strategy.name}:`, error);
      }
    }

    // Сортируем по Sharpe Ratio
    return results.sort((a, b) => b.results.sharpeRatio - a.results.sharpeRatio);
  }

  // Приватные методы

  private filterDataByDateRange(data: CandleData[], startDate: Date, endDate: Date): CandleData[] {
    return data.filter(candle => {
      const candleDate = new Date(candle.created_at);
      return candleDate >= startDate && candleDate <= endDate;
    });
  }

  private shouldClosePosition(position: any, candle: CandleData, signal: Signal | null): boolean {
    // Стоп-лосс или тейк-профит
    if (position.side === 'long') {
      if (candle.low <= position.stopLoss || candle.high >= position.takeProfit) {
        return true;
      }
    } else {
      if (candle.high >= position.stopLoss || candle.low <= position.takeProfit) {
        return true;
      }
    }

    // Противоположный сигнал
    if (signal && signal.type !== position.side.replace('long', 'buy').replace('short', 'sell')) {
      return true;
    }

    return false;
  }

  private openPosition(signal: Signal, candle: CandleData, equity: number, config: BacktestConfig): any {
    const riskAmount = equity * config.riskPerTrade;
    const entryPrice = candle.close * (1 + (signal.type === 'buy' ? config.slippage : -config.slippage));
    
    let stopLoss = signal.stopLoss || entryPrice * (signal.type === 'buy' ? 0.98 : 1.02);
    let takeProfit = signal.takeProfit || entryPrice * (signal.type === 'buy' ? 1.04 : 0.96);
    
    const riskPerShare = Math.abs(entryPrice - stopLoss);
    const quantity = riskAmount / riskPerShare;

    return {
      entryDate: new Date(candle.created_at),
      entryPrice,
      quantity,
      side: signal.type === 'buy' ? 'long' : 'short',
      stopLoss,
      takeProfit,
      signal
    };
  }

  private closePosition(position: any, candle: CandleData, config: BacktestConfig): Trade {
    const exitPrice = candle.close * (1 + (position.side === 'long' ? -config.slippage : config.slippage));
    const commission = position.quantity * exitPrice * config.commission;
    
    let pnl: number;
    if (position.side === 'long') {
      pnl = (exitPrice - position.entryPrice) * position.quantity - commission;
    } else {
      pnl = (position.entryPrice - exitPrice) * position.quantity - commission;
    }

    const pnlPercent = pnl / (position.entryPrice * position.quantity);
    const holdingPeriod = (new Date(candle.created_at).getTime() - position.entryDate.getTime()) / (1000 * 60 * 60 * 24);

    return {
      entryDate: position.entryDate,
      exitDate: new Date(candle.created_at),
      entryPrice: position.entryPrice,
      exitPrice,
      quantity: position.quantity,
      side: position.side,
      pnl,
      pnlPercent,
      holdingPeriod,
      commission,
      slippage: Math.abs(exitPrice - candle.close)
    };
  }

  private calculateResults(trades: Trade[], equityCurve: EquityPoint[], config: BacktestConfig): BacktestResult {
    if (trades.length === 0) {
      return this.getEmptyResults();
    }

    const totalPnL = trades.reduce((sum, trade) => sum + trade.pnl, 0);
    const totalReturn = totalPnL / config.initialCapital;
    
    const daysInPeriod = equityCurve.length;
    const annualizedReturn = Math.pow(1 + totalReturn, 365 / daysInPeriod) - 1;
    
    const maxDrawdown = Math.max(...equityCurve.map(point => point.drawdown));
    
    const returns = equityCurve.map((point, i) => 
      i === 0 ? 0 : (point.equity - equityCurve[i-1].equity) / equityCurve[i-1].equity
    ).slice(1);
    
    const sharpeRatio = this.calculateSharpeRatio(returns);
    const winningTrades = trades.filter(t => t.pnl > 0);
    const winRate = winningTrades.length / trades.length;
    
    const grossProfit = winningTrades.reduce((sum, t) => sum + t.pnl, 0);
    const grossLoss = Math.abs(trades.filter(t => t.pnl < 0).reduce((sum, t) => sum + t.pnl, 0));
    const profitFactor = grossLoss === 0 ? (grossProfit > 0 ? Infinity : 0) : grossProfit / grossLoss;
    
    const avgTradeReturn = totalPnL / trades.length;
    const bestTrade = Math.max(...trades.map(t => t.pnl));
    const worstTrade = Math.min(...trades.map(t => t.pnl));
    
    const calmarRatio = maxDrawdown === 0 ? 0 : annualizedReturn / maxDrawdown;
    const recoveryFactor = maxDrawdown === 0 ? 0 : totalReturn / maxDrawdown;
    
    const monthlyReturns = this.calculateMonthlyReturns(equityCurve);

    return {
      totalReturn,
      annualizedReturn,
      maxDrawdown,
      sharpeRatio,
      winRate,
      profitFactor,
      totalTrades: trades.length,
      avgTradeReturn,
      bestTrade,
      worstTrade,
      calmarRatio,
      recoveryFactor,
      trades,
      equityCurve,
      monthlyReturns
    };
  }

  private getEmptyResults(): BacktestResult {
    return {
      totalReturn: 0,
      annualizedReturn: 0,
      maxDrawdown: 0,
      sharpeRatio: 0,
      winRate: 0,
      profitFactor: 0,
      totalTrades: 0,
      avgTradeReturn: 0,
      bestTrade: 0,
      worstTrade: 0,
      calmarRatio: 0,
      recoveryFactor: 0,
      trades: [],
      equityCurve: [],
      monthlyReturns: []
    };
  }

  private calculateSMA(candles: CandleData[], index: number, period: number): number {
    if (index < period - 1) return 0;
    
    const sum = candles.slice(index - period + 1, index + 1)
      .reduce((acc, candle) => acc + candle.close, 0);
    return sum / period;
  }

  private calculateRSI(candles: CandleData[], index: number, period: number): number {
    if (index < period) return 50;

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

    if (avgLoss === 0) return 100;

    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  private calculateSharpeRatio(returns: number[], riskFreeRate: number = 0.02): number {
    if (returns.length === 0) return 0;

    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const annualizedReturn = avgReturn * 252; // Торговые дни в году
    
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance * 252);

    return volatility === 0 ? 0 : (annualizedReturn - riskFreeRate) / volatility;
  }

  private calculateMonthlyReturns(equityCurve: EquityPoint[]): MonthlyReturn[] {
    const monthlyData = new Map<string, { start: number; end: number }>();

    for (const point of equityCurve) {
      const year = point.date.getFullYear();
      const month = point.date.getMonth();
      const key = `${year}-${month}`;

      if (!monthlyData.has(key)) {
        monthlyData.set(key, { start: point.equity, end: point.equity });
      } else {
        monthlyData.get(key)!.end = point.equity;
      }
    }

    return Array.from(monthlyData.entries()).map(([key, data]) => {
      const [year, month] = key.split('-').map(Number);
      const returnValue = data.start === 0 ? 0 : (data.end - data.start) / data.start;
      
      return {
        year,
        month: month + 1, // JavaScript months are 0-indexed
        return: returnValue
      };
    });
  }
}

export const backtestingService = BacktestingService.getInstance();