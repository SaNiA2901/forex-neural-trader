import { CandleData } from '@/types/session';
import { ModelMetrics } from './AdvancedMLTrainingService';

export interface BacktestConfig {
  initialCapital: number;
  positionSize: number; // Percentage of capital per trade
  stopLoss?: number; // Percentage
  takeProfit?: number; // Percentage
  transactionCost: number; // Percentage per trade
  maxPositions: number;
  riskPerTrade: number; // Percentage of capital at risk
}

export interface Trade {
  id: string;
  entryTime: number;
  exitTime?: number;
  entryPrice: number;
  exitPrice?: number;
  direction: 'long' | 'short';
  size: number;
  pnl?: number;
  pnlPercent?: number;
  status: 'open' | 'closed' | 'stopped';
  stopLoss?: number;
  takeProfit?: number;
  reason?: 'signal' | 'stop_loss' | 'take_profit' | 'timeout';
}

export interface BacktestResults {
  trades: Trade[];
  metrics: {
    totalTrades: number;
    winningTrades: number;
    losingTrades: number;
    winRate: number;
    totalReturn: number;
    totalReturnPercent: number;
    sharpeRatio: number;
    maxDrawdown: number;
    maxDrawdownPercent: number;
    averageWin: number;
    averageLoss: number;
    profitFactor: number;
    expectancy: number;
    calmarRatio: number;
    sortinoRatio: number;
    valueAtRisk: number;
    transactionCosts: number;
  };
  equityCurve: { timestamp: number; equity: number; drawdown: number }[];
  monthlyReturns: { month: string; return: number }[];
}

export interface PredictionSignal {
  timestamp: number;
  direction: 'long' | 'short';
  confidence: number;
  price: number;
}

export class BacktestingEngine {
  private config: BacktestConfig;
  private trades: Trade[] = [];
  private currentEquity: number;
  private peakEquity: number;
  private equityCurve: { timestamp: number; equity: number; drawdown: number }[] = [];

  constructor(config: BacktestConfig) {
    this.config = config;
    this.currentEquity = config.initialCapital;
    this.peakEquity = config.initialCapital;
  }

  /**
   * Run backtest with prediction signals
   */
  async runBacktest(
    candles: CandleData[],
    signals: PredictionSignal[]
  ): Promise<BacktestResults> {
    this.reset();
    
    // Sort signals by timestamp
    const sortedSignals = signals.sort((a, b) => a.timestamp - b.timestamp);
    const candleMap = new Map(candles.map(c => [c.timestamp, c]));
    
    let signalIndex = 0;
    let openTrades: Trade[] = [];

    for (let i = 0; i < candles.length; i++) {
      const candle = candles[i];
      const currentTime = candle.timestamp;

      // Process any new signals at this time
      while (signalIndex < sortedSignals.length && 
             sortedSignals[signalIndex].timestamp === Number(currentTime)) {
        const signal = sortedSignals[signalIndex];
        
        if (openTrades.length < this.config.maxPositions) {
          const trade = this.openTrade(signal, candle);
          if (trade) {
            openTrades.push(trade);
          }
        }
        
        signalIndex++;
      }

      // Update open trades
      openTrades = this.updateOpenTrades(openTrades, candle);

      // Update equity curve
      this.updateEquityCurve(Number(currentTime), openTrades);
    }

    // Close any remaining open trades
    const lastCandle = candles[candles.length - 1];
    openTrades.forEach(trade => {
      this.closeTrade(trade, lastCandle, 'timeout');
    });

    return this.calculateResults();
  }

  private reset(): void {
    this.trades = [];
    this.currentEquity = this.config.initialCapital;
    this.peakEquity = this.config.initialCapital;
    this.equityCurve = [];
  }

  private openTrade(signal: PredictionSignal, candle: CandleData): Trade | null {
    // Calculate position size based on risk management
    const riskAmount = this.currentEquity * (this.config.riskPerTrade / 100);
    const positionValue = this.currentEquity * (this.config.positionSize / 100);
    
    // Don't trade if signal confidence is too low
    if (signal.confidence < 0.6) {
      return null;
    }

    const trade: Trade = {
      id: `trade_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
      entryTime: Number(candle.timestamp),
      entryPrice: candle.close,
      direction: signal.direction,
      size: positionValue / candle.close,
      status: 'open'
    };

    // Set stop loss and take profit
    if (this.config.stopLoss) {
      const stopDistance = candle.close * (this.config.stopLoss / 100);
      trade.stopLoss = signal.direction === 'long' 
        ? candle.close - stopDistance
        : candle.close + stopDistance;
    }

    if (this.config.takeProfit) {
      const profitDistance = candle.close * (this.config.takeProfit / 100);
      trade.takeProfit = signal.direction === 'long'
        ? candle.close + profitDistance
        : candle.close - profitDistance;
    }

    this.trades.push(trade);
    return trade;
  }

  private updateOpenTrades(openTrades: Trade[], candle: CandleData): Trade[] {
    const activeTrades: Trade[] = [];

    for (const trade of openTrades) {
      let shouldClose = false;
      let closeReason: Trade['reason'] = 'signal';

      // Check stop loss
      if (trade.stopLoss) {
        if ((trade.direction === 'long' && candle.low <= trade.stopLoss) ||
            (trade.direction === 'short' && candle.high >= trade.stopLoss)) {
          shouldClose = true;
          closeReason = 'stop_loss';
        }
      }

      // Check take profit
      if (trade.takeProfit && !shouldClose) {
        if ((trade.direction === 'long' && candle.high >= trade.takeProfit) ||
            (trade.direction === 'short' && candle.low <= trade.takeProfit)) {
          shouldClose = true;
          closeReason = 'take_profit';
        }
      }

      // Check for exit signal (simplified - could be enhanced)
      if (!shouldClose) {
        // Hold for now - in real implementation, this would check exit signals
        activeTrades.push(trade);
      } else {
        this.closeTrade(trade, candle, closeReason);
      }
    }

    return activeTrades;
  }

  private closeTrade(trade: Trade, candle: CandleData, reason: Trade['reason']): void {
    const exitPrice = reason === 'stop_loss' ? trade.stopLoss! :
                     reason === 'take_profit' ? trade.takeProfit! :
                     candle.close;

    trade.exitTime = Number(candle.timestamp);
    trade.exitPrice = exitPrice;
    trade.reason = reason;
    trade.status = 'closed';

    // Calculate P&L
    const priceChange = trade.direction === 'long' 
      ? exitPrice - trade.entryPrice
      : trade.entryPrice - exitPrice;
    
    const grossPnl = (priceChange / trade.entryPrice) * trade.size * trade.entryPrice;
    const transactionCost = (trade.size * trade.entryPrice * this.config.transactionCost / 100) * 2; // Entry + exit
    
    trade.pnl = grossPnl - transactionCost;
    trade.pnlPercent = (trade.pnl / (trade.size * trade.entryPrice)) * 100;

    this.currentEquity += trade.pnl;
  }

  private updateEquityCurve(timestamp: number, openTrades: Trade[]): void {
    // Calculate unrealized P&L from open trades
    let unrealizedPnl = 0;
    // This would need current market price to calculate properly
    // For now, we'll just use realized equity

    this.peakEquity = Math.max(this.peakEquity, this.currentEquity);
    const drawdown = (this.peakEquity - this.currentEquity) / this.peakEquity;

    this.equityCurve.push({
      timestamp,
      equity: this.currentEquity,
      drawdown
    });
  }

  private calculateResults(): BacktestResults {
    const completedTrades = this.trades.filter(t => t.status === 'closed');
    const winningTrades = completedTrades.filter(t => t.pnl! > 0);
    const losingTrades = completedTrades.filter(t => t.pnl! <= 0);

    const totalReturn = this.currentEquity - this.config.initialCapital;
    const totalReturnPercent = (totalReturn / this.config.initialCapital) * 100;

    // Calculate returns for Sharpe ratio
    const dailyReturns = this.calculateDailyReturns();
    const sharpeRatio = this.calculateSharpeRatio(dailyReturns);
    const sortinoRatio = this.calculateSortinoRatio(dailyReturns);

    // Calculate maximum drawdown
    const maxDrawdown = Math.max(...this.equityCurve.map(e => e.drawdown));
    const maxDrawdownPercent = maxDrawdown * 100;

    // Calculate other metrics
    const averageWin = winningTrades.length > 0 
      ? winningTrades.reduce((sum, t) => sum + t.pnl!, 0) / winningTrades.length
      : 0;
    
    const averageLoss = losingTrades.length > 0
      ? losingTrades.reduce((sum, t) => sum + Math.abs(t.pnl!), 0) / losingTrades.length
      : 0;

    const grossProfit = winningTrades.reduce((sum, t) => sum + t.pnl!, 0);
    const grossLoss = Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl!, 0));
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : 0;

    const winRate = completedTrades.length > 0 
      ? winningTrades.length / completedTrades.length
      : 0;

    const expectancy = (winRate * averageWin) - ((1 - winRate) * averageLoss);
    
    const calmarRatio = maxDrawdown > 0 ? totalReturnPercent / (maxDrawdownPercent / 100) : 0;
    
    const valueAtRisk = this.calculateVaR(dailyReturns);
    
    const transactionCosts = completedTrades.reduce((sum, t) => 
      sum + (t.size * t.entryPrice * this.config.transactionCost / 100) * 2, 0
    );

    const monthlyReturns = this.calculateMonthlyReturns();

    return {
      trades: this.trades,
      metrics: {
        totalTrades: completedTrades.length,
        winningTrades: winningTrades.length,
        losingTrades: losingTrades.length,
        winRate,
        totalReturn,
        totalReturnPercent,
        sharpeRatio,
        maxDrawdown,
        maxDrawdownPercent,
        averageWin,
        averageLoss,
        profitFactor,
        expectancy,
        calmarRatio,
        sortinoRatio,
        valueAtRisk,
        transactionCosts
      },
      equityCurve: this.equityCurve,
      monthlyReturns
    };
  }

  private calculateDailyReturns(): number[] {
    const returns: number[] = [];
    
    for (let i = 1; i < this.equityCurve.length; i++) {
      const prevEquity = this.equityCurve[i - 1].equity;
      const currentEquity = this.equityCurve[i].equity;
      const dailyReturn = (currentEquity - prevEquity) / prevEquity;
      returns.push(dailyReturn);
    }
    
    return returns;
  }

  private calculateSharpeRatio(returns: number[]): number {
    if (returns.length === 0) return 0;
    
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((acc, ret) => acc + Math.pow(ret - avgReturn, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance);
    
    return volatility > 0 ? (avgReturn / volatility) * Math.sqrt(252) : 0; // Annualized
  }

  private calculateSortinoRatio(returns: number[]): number {
    if (returns.length === 0) return 0;
    
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const negativeReturns = returns.filter(r => r < 0);
    
    if (negativeReturns.length === 0) return 0;
    
    const downwardVariance = negativeReturns.reduce((acc, ret) => acc + Math.pow(ret, 2), 0) / negativeReturns.length;
    const downwardVolatility = Math.sqrt(downwardVariance);
    
    return downwardVolatility > 0 ? (avgReturn / downwardVolatility) * Math.sqrt(252) : 0;
  }

  private calculateVaR(returns: number[], confidence: number = 0.05): number {
    if (returns.length === 0) return 0;
    
    const sortedReturns = [...returns].sort((a, b) => a - b);
    const index = Math.floor(confidence * sortedReturns.length);
    
    return Math.abs(sortedReturns[index] || 0);
  }

  private calculateMonthlyReturns(): { month: string; return: number }[] {
    const monthlyData: Map<string, { start: number; end: number }> = new Map();
    
    for (const point of this.equityCurve) {
      const date = new Date(point.timestamp);
      const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
      
      if (!monthlyData.has(monthKey)) {
        monthlyData.set(monthKey, { start: point.equity, end: point.equity });
      } else {
        monthlyData.get(monthKey)!.end = point.equity;
      }
    }
    
    return Array.from(monthlyData.entries()).map(([month, data]) => ({
      month,
      return: ((data.end - data.start) / data.start) * 100
    }));
  }
}

export const backtestingEngine = new BacktestingEngine({
  initialCapital: 10000,
  positionSize: 10,
  stopLoss: 2,
  takeProfit: 4,
  transactionCost: 0.1,
  maxPositions: 3,
  riskPerTrade: 1
});