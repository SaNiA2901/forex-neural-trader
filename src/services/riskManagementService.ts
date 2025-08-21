import { CandleData } from '@/types/session';
import { PredictionResult } from '@/types/trading';

interface PortfolioPosition {
  symbol: string;
  direction: 'CALL' | 'PUT';
  amount: number;
  entryPrice: number;
  entryTime: Date;
  expiryTime: Date;
  probability: number;
  confidence: number;
  status: 'open' | 'won' | 'lost' | 'expired';
  pnl?: number;
}

interface RiskMetrics {
  var95: number; // Value at Risk 95%
  var99: number; // Value at Risk 99%
  expectedShortfall: number; // Conditional VaR
  maxDrawdown: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  beta: number;
  alpha: number;
  informationRatio: number;
  trackingError: number;
}

interface PositionSizing {
  kellyOptimal: number;
  fixedFractional: number;
  riskParity: number;
  volatilityAdjusted: number;
  recommended: number;
  reasoning: string;
}

interface PortfolioAllocation {
  totalCapital: number;
  availableCapital: number;
  allocatedCapital: number;
  maxPositionSize: number;
  diversificationScore: number;
  correlationMatrix: number[][];
  sectorExposure: { [sector: string]: number };
  riskBudget: { [position: string]: number };
}

export class RiskManagementService {
  private static instance: RiskManagementService;
  private portfolio: PortfolioPosition[] = [];
  private capitalHistory: number[] = [];
  private maxCapital = 10000; // Максимальный капитал
  private currentCapital = 10000; // Текущий капитал
  private maxDrawdownLimit = 0.15; // 15% максимальная просадка
  private maxPositionSize = 0.05; // 5% максимальный размер позиции
  private maxDailyRisk = 0.02; // 2% максимальный дневной риск

  static getInstance(): RiskManagementService {
    if (!RiskManagementService.instance) {
      RiskManagementService.instance = new RiskManagementService();
    }
    return RiskManagementService.instance;
  }

  // Расчет оптимального размера позиции
  calculatePositionSize(
    prediction: PredictionResult,
    currentPrice: number,
    accountBalance: number
  ): PositionSizing {
    
    const winProbability = prediction.probability / 100;
    const winRate = this.getHistoricalWinRate();
    const avgWin = this.getAverageWin();
    const avgLoss = this.getAverageLoss();
    
    // Kelly Criterion
    const kellyFraction = ((winProbability * avgWin) - ((1 - winProbability) * avgLoss)) / avgWin;
    const kellyOptimal = Math.max(0, Math.min(0.25, kellyFraction)); // Ограничиваем 25%
    
    // Fixed Fractional (основан на максимальной просадке)
    const fixedFractional = this.maxPositionSize;
    
    // Risk Parity (основан на волатильности)
    const volatility = this.calculateVolatility();
    const riskParity = (this.maxDailyRisk / volatility) * accountBalance;
    
    // Volatility Adjusted
    const confidence = prediction.confidence / 100;
    const volatilityAdjusted = fixedFractional * confidence * (1 / volatility);
    
    // Рекомендуемый размер (консервативный подход)
    const sizes = [kellyOptimal * accountBalance, fixedFractional * accountBalance, riskParity, volatilityAdjusted * accountBalance];
    const recommended = Math.min(...sizes);
    
    let reasoning = 'Консервативный подход основан на ';
    if (recommended === kellyOptimal * accountBalance) reasoning += 'Kelly Criterion';
    else if (recommended === fixedFractional * accountBalance) reasoning += 'фиксированном проценте';
    else if (recommended === riskParity) reasoning += 'риск-паритете';
    else reasoning += 'волатильности';
    
    return {
      kellyOptimal: kellyOptimal * accountBalance,
      fixedFractional: fixedFractional * accountBalance,
      riskParity,
      volatilityAdjusted: volatilityAdjusted * accountBalance,
      recommended,
      reasoning
    };
  }

  // Анализ риска портфеля
  calculatePortfolioRisk(): RiskMetrics {
    const returns = this.calculateReturns();
    const sortedReturns = [...returns].sort((a, b) => a - b);
    
    // Value at Risk
    const var95Index = Math.floor(returns.length * 0.05);
    const var99Index = Math.floor(returns.length * 0.01);
    const var95 = sortedReturns[var95Index] || 0;
    const var99 = sortedReturns[var99Index] || 0;
    
    // Expected Shortfall (Conditional VaR)
    const tailLosses = sortedReturns.slice(0, var95Index);
    const expectedShortfall = tailLosses.length > 0 
      ? tailLosses.reduce((sum, ret) => sum + ret, 0) / tailLosses.length 
      : 0;
    
    // Maximum Drawdown
    const maxDrawdown = this.calculateMaxDrawdown();
    
    // Sharpe Ratio
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const volatility = this.calculateVolatility();
    const sharpeRatio = volatility > 0 ? avgReturn / volatility : 0;
    
    // Sortino Ratio (учитывает только негативную волатильность)
    const negativeReturns = returns.filter(ret => ret < 0);
    const downsideDeviation = negativeReturns.length > 0 
      ? Math.sqrt(negativeReturns.reduce((sum, ret) => sum + ret * ret, 0) / negativeReturns.length)
      : 0;
    const sortinoRatio = downsideDeviation > 0 ? avgReturn / downsideDeviation : 0;
    
    // Calmar Ratio
    const calmarRatio = maxDrawdown > 0 ? avgReturn / Math.abs(maxDrawdown) : 0;
    
    return {
      var95: Math.abs(var95),
      var99: Math.abs(var99),
      expectedShortfall: Math.abs(expectedShortfall),
      maxDrawdown,
      sharpeRatio,
      sortinoRatio,
      calmarRatio,
      beta: this.calculateBeta(),
      alpha: this.calculateAlpha(),
      informationRatio: this.calculateInformationRatio(),
      trackingError: this.calculateTrackingError()
    };
  }

  // Оценка корреляции позиций
  calculatePortfolioCorrelation(): number[][] {
    const symbols = [...new Set(this.portfolio.map(p => p.symbol))];
    const correlationMatrix: number[][] = [];
    
    for (let i = 0; i < symbols.length; i++) {
      correlationMatrix[i] = [];
      for (let j = 0; j < symbols.length; j++) {
        if (i === j) {
          correlationMatrix[i][j] = 1.0;
        } else {
          // Расчет корреляции между активами
          correlationMatrix[i][j] = this.calculatePairCorrelation(symbols[i], symbols[j]);
        }
      }
    }
    
    return correlationMatrix;
  }

  // Проверка лимитов риска
  validateRiskLimits(
    newPosition: Omit<PortfolioPosition, 'status' | 'entryTime'>
  ): { allowed: boolean; reason?: string; maxAllowed?: number } {
    
    // Проверка максимального размера позиции
    const positionPercentage = newPosition.amount / this.currentCapital;
    if (positionPercentage > this.maxPositionSize) {
      return {
        allowed: false,
        reason: `Размер позиции превышает лимит ${(this.maxPositionSize * 100).toFixed(1)}%`,
        maxAllowed: this.currentCapital * this.maxPositionSize
      };
    }
    
    // Проверка максимальной просадки
    const currentDrawdown = this.calculateCurrentDrawdown();
    if (currentDrawdown >= this.maxDrawdownLimit) {
      return {
        allowed: false,
        reason: `Достигнут лимит просадки ${(this.maxDrawdownLimit * 100).toFixed(1)}%`
      };
    }
    
    // Проверка концентрации по символу
    const symbolExposure = this.calculateSymbolExposure(newPosition.symbol);
    const newExposure = (symbolExposure + newPosition.amount) / this.currentCapital;
    if (newExposure > 0.3) { // Максимум 30% на один актив
      return {
        allowed: false,
        reason: `Превышена концентрация по активу ${newPosition.symbol}`
      };
    }
    
    // Проверка дневного лимита риска
    const dailyRisk = this.calculateDailyRisk();
    if (dailyRisk >= this.maxDailyRisk) {
      return {
        allowed: false,
        reason: `Достигнут дневной лимит риска ${(this.maxDailyRisk * 100).toFixed(1)}%`
      };
    }
    
    return { allowed: true };
  }

  // Оптимизация портфеля
  optimizePortfolio(): PortfolioAllocation {
    const totalCapital = this.currentCapital;
    const allocatedCapital = this.portfolio
      .filter(p => p.status === 'open')
      .reduce((sum, p) => sum + p.amount, 0);
    
    const availableCapital = totalCapital - allocatedCapital;
    const correlationMatrix = this.calculatePortfolioCorrelation();
    
    // Расчет диверсификации
    const diversificationScore = this.calculateDiversificationScore(correlationMatrix);
    
    // Секторное распределение
    const sectorExposure = this.calculateSectorExposure();
    
    // Risk budgeting
    const riskBudget = this.calculateRiskBudget();
    
    return {
      totalCapital,
      availableCapital,
      allocatedCapital,
      maxPositionSize: totalCapital * this.maxPositionSize,
      diversificationScore,
      correlationMatrix,
      sectorExposure,
      riskBudget
    };
  }

  // Stress testing
  stressTesting(scenarios: Array<{ name: string; marketShock: number }>): Array<{ scenario: string; pnl: number; newCapital: number }> {
    return scenarios.map(scenario => {
      let totalPnL = 0;
      
      this.portfolio.forEach(position => {
        if (position.status === 'open') {
          // Применяем шок к позиции
          const shockImpact = scenario.marketShock * position.amount;
          if (position.direction === 'CALL') {
            totalPnL += shockImpact;
          } else {
            totalPnL -= shockImpact;
          }
        }
      });
      
      return {
        scenario: scenario.name,
        pnl: totalPnL,
        newCapital: this.currentCapital + totalPnL
      };
    });
  }

  // Мониторинг позиций в реальном времени
  monitorPositions(currentPrices: { [symbol: string]: number }): Array<{
    position: PortfolioPosition;
    currentPnL: number;
    unrealizedPnL: number;
    riskScore: number;
    recommendation: 'hold' | 'close' | 'hedge';
  }> {
    return this.portfolio
      .filter(p => p.status === 'open')
      .map(position => {
        const currentPrice = currentPrices[position.symbol] || position.entryPrice;
        const priceChange = (currentPrice - position.entryPrice) / position.entryPrice;
        
        let unrealizedPnL = 0;
        if (position.direction === 'CALL' && priceChange > 0) {
          unrealizedPnL = position.amount * 0.8; // 80% прибыль
        } else if (position.direction === 'PUT' && priceChange < 0) {
          unrealizedPnL = position.amount * 0.8;
        } else {
          unrealizedPnL = -position.amount; // Потеря ставки
        }
        
        const riskScore = this.calculatePositionRisk(position, currentPrice);
        const recommendation = this.getPositionRecommendation(position, riskScore, unrealizedPnL);
        
        return {
          position,
          currentPnL: position.pnl || 0,
          unrealizedPnL,
          riskScore,
          recommendation
        };
      });
  }

  // Добавление новой позиции
  addPosition(position: Omit<PortfolioPosition, 'status' | 'entryTime'>): boolean {
    const validation = this.validateRiskLimits(position);
    
    if (!validation.allowed) {
      console.warn('Position rejected:', validation.reason);
      return false;
    }
    
    const newPosition: PortfolioPosition = {
      ...position,
      status: 'open',
      entryTime: new Date()
    };
    
    this.portfolio.push(newPosition);
    this.updateCapitalHistory();
    
    return true;
  }

  // Закрытие позиции
  closePosition(positionId: string, outcome: 'won' | 'lost' | 'expired', pnl: number): void {
    const position = this.portfolio.find(p => p.symbol === positionId && p.status === 'open');
    
    if (position) {
      position.status = outcome;
      position.pnl = pnl;
      this.currentCapital += pnl;
      this.updateCapitalHistory();
    }
  }

  // Вспомогательные методы
  private calculateReturns(): number[] {
    const returns: number[] = [];
    for (let i = 1; i < this.capitalHistory.length; i++) {
      const ret = (this.capitalHistory[i] - this.capitalHistory[i - 1]) / this.capitalHistory[i - 1];
      returns.push(ret);
    }
    return returns;
  }

  private calculateVolatility(): number {
    const returns = this.calculateReturns();
    if (returns.length < 2) return 0.02; // Значение по умолчанию
    
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length;
    
    return Math.sqrt(variance * 252); // Аннуализированная волатильность
  }

  private calculateMaxDrawdown(): number {
    let maxDrawdown = 0;
    let peak = this.capitalHistory[0] || this.maxCapital;
    
    for (const capital of this.capitalHistory) {
      if (capital > peak) {
        peak = capital;
      }
      const drawdown = (peak - capital) / peak;
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }
    
    return maxDrawdown;
  }

  private getHistoricalWinRate(): number {
    const closedPositions = this.portfolio.filter(p => p.status !== 'open');
    if (closedPositions.length === 0) return 0.6; // Значение по умолчанию
    
    const wins = closedPositions.filter(p => p.status === 'won').length;
    return wins / closedPositions.length;
  }

  private getAverageWin(): number {
    const wins = this.portfolio.filter(p => p.status === 'won');
    if (wins.length === 0) return 0.8; // 80% прибыль по умолчанию
    
    return wins.reduce((sum, p) => sum + (p.pnl || 0), 0) / wins.length;
  }

  private getAverageLoss(): number {
    const losses = this.portfolio.filter(p => p.status === 'lost');
    if (losses.length === 0) return 1.0; // 100% потеря ставки по умолчанию
    
    return Math.abs(losses.reduce((sum, p) => sum + (p.pnl || 0), 0) / losses.length);
  }

  private calculateCurrentDrawdown(): number {
    if (this.capitalHistory.length === 0) return 0;
    
    const peak = Math.max(...this.capitalHistory);
    return (peak - this.currentCapital) / peak;
  }

  private calculateSymbolExposure(symbol: string): number {
    return this.portfolio
      .filter(p => p.symbol === symbol && p.status === 'open')
      .reduce((sum, p) => sum + p.amount, 0);
  }

  private calculateDailyRisk(): number {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const todayPositions = this.portfolio.filter(p => 
      p.entryTime >= today && p.status !== 'open'
    );
    
    const dailyPnL = todayPositions.reduce((sum, p) => sum + (p.pnl || 0), 0);
    return Math.abs(dailyPnL) / this.currentCapital;
  }

  private updateCapitalHistory(): void {
    this.capitalHistory.push(this.currentCapital);
    
    // Ограничиваем историю последними 1000 записями
    if (this.capitalHistory.length > 1000) {
      this.capitalHistory.splice(0, 100);
    }
  }

  // Заглушки для сложных расчетов
  private calculateBeta(): number { return 1.0; }
  private calculateAlpha(): number { return 0.05; }
  private calculateInformationRatio(): number { return 0.5; }
  private calculateTrackingError(): number { return 0.02; }
  private calculatePairCorrelation(symbol1: string, symbol2: string): number { return 0.3; }
  private calculateDiversificationScore(matrix: number[][]): number { return 0.8; }
  private calculateSectorExposure(): { [sector: string]: number } { return { 'Crypto': 0.6, 'Forex': 0.4 }; }
  private calculateRiskBudget(): { [position: string]: number } { return {}; }
  private calculatePositionRisk(position: PortfolioPosition, currentPrice: number): number { return 0.3; }
  private getPositionRecommendation(position: PortfolioPosition, risk: number, pnl: number): 'hold' | 'close' | 'hedge' {
    if (risk > 0.7) return 'close';
    if (pnl < -position.amount * 0.5) return 'close';
    return 'hold';
  }

  // Геттеры для текущего состояния
  getCurrentCapital(): number { return this.currentCapital; }
  getPortfolio(): PortfolioPosition[] { return [...this.portfolio]; }
  getOpenPositions(): PortfolioPosition[] { return this.portfolio.filter(p => p.status === 'open'); }
}