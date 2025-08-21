import { CandleData } from '@/types/session';

export interface RiskMetrics {
  maxDrawdown: number;
  sharpeRatio: number;
  winRate: number;
  profitFactor: number;
  valueAtRisk: number;
  expectedShortfall: number;
  beta: number;
  alpha: number;
  informationRatio: number;
  calmarRatio: number;
}

export interface PositionSizing {
  kellyPercentage: number;
  fixedPercentage: number;
  volatilityAdjusted: number;
  maximumPosition: number;
  recommendedSize: number;
}

export interface RiskLimits {
  maxPositionSize: number;
  maxDailyLoss: number;
  maxDrawdown: number;
  maxCorrelatedPositions: number;
  maxSectorExposure: number;
}

export interface StressTestResult {
  scenario: string;
  priceChange: number;
  portfolioImpact: number;
  probability: number;
  timeHorizon: string;
}

export class RiskManagementService {
  private static instance: RiskManagementService;
  
  private constructor() {}

  static getInstance(): RiskManagementService {
    if (!RiskManagementService.instance) {
      RiskManagementService.instance = new RiskManagementService();
    }
    return RiskManagementService.instance;
  }

  /**
   * Вычисляет комплексные метрики риска
   */
  calculateRiskMetrics(
    returns: number[], 
    benchmarkReturns?: number[], 
    riskFreeRate: number = 0.02
  ): RiskMetrics {
    if (returns.length === 0) {
      return this.getDefaultRiskMetrics();
    }

    const annualizedReturns = this.annualizeReturns(returns);
    const volatility = this.calculateVolatility(returns);
    const maxDrawdown = this.calculateMaxDrawdown(returns);
    const winRate = this.calculateWinRate(returns);
    const profitFactor = this.calculateProfitFactor(returns);
    const sharpeRatio = this.calculateSharpeRatio(annualizedReturns, volatility, riskFreeRate);
    const var95 = this.calculateVaR(returns, 0.05);
    const expectedShortfall = this.calculateExpectedShortfall(returns, 0.05);
    
    let beta = 1;
    let alpha = 0;
    let informationRatio = 0;
    
    if (benchmarkReturns && benchmarkReturns.length === returns.length) {
      beta = this.calculateBeta(returns, benchmarkReturns);
      alpha = this.calculateAlpha(returns, benchmarkReturns, riskFreeRate, beta);
      informationRatio = this.calculateInformationRatio(returns, benchmarkReturns);
    }

    const calmarRatio = maxDrawdown !== 0 ? annualizedReturns / Math.abs(maxDrawdown) : 0;

    return {
      maxDrawdown,
      sharpeRatio,
      winRate,
      profitFactor,
      valueAtRisk: var95,
      expectedShortfall,
      beta,
      alpha,
      informationRatio,
      calmarRatio
    };
  }

  /**
   * Рассчитывает оптимальный размер позиции
   */
  calculatePositionSizing(
    accountBalance: number,
    winRate: number,
    avgWin: number,
    avgLoss: number,
    volatility: number,
    riskTolerance: number = 0.02
  ): PositionSizing {
    // Критерий Келли
    const kellyPercentage = this.calculateKellyRatio(winRate, avgWin, avgLoss);
    
    // Фиксированный процент риска
    const fixedPercentage = riskTolerance;
    
    // Размер, скорректированный на волатильность
    const baseVolatility = 0.15; // 15% базовая волатильность
    const volatilityAdjusted = (baseVolatility / volatility) * riskTolerance;
    
    // Максимальный размер позиции (не более 10% от счета)
    const maximumPosition = 0.10;
    
    // Рекомендуемый размер (консервативный подход)
    const recommendedSize = Math.min(
      kellyPercentage * 0.25, // Четверть от Келли
      fixedPercentage,
      volatilityAdjusted,
      maximumPosition
    );

    return {
      kellyPercentage,
      fixedPercentage,
      volatilityAdjusted,
      maximumPosition,
      recommendedSize: Math.max(0.001, recommendedSize) // Минимум 0.1%
    };
  }

  /**
   * Проводит стресс-тестирование портфеля
   */
  performStressTest(
    currentPositions: any[],
    correlationMatrix: number[][],
    historicalData: CandleData[]
  ): StressTestResult[] {
    const scenarios = [
      { name: 'Кризис 2008', priceChange: -0.45, probability: 0.02 },
      { name: 'Пандемия 2020', priceChange: -0.35, probability: 0.05 },
      { name: 'Флэш-крэш', priceChange: -0.20, probability: 0.10 },
      { name: 'Коррекция рынка', priceChange: -0.15, probability: 0.20 },
      { name: 'Умеренное снижение', priceChange: -0.10, probability: 0.30 },
      { name: 'Волатильность', priceChange: -0.05, probability: 0.40 }
    ];

    return scenarios.map(scenario => {
      const portfolioImpact = this.calculatePortfolioImpact(
        currentPositions,
        scenario.priceChange,
        correlationMatrix
      );

      return {
        scenario: scenario.name,
        priceChange: scenario.priceChange,
        portfolioImpact,
        probability: scenario.probability,
        timeHorizon: '1 день'
      };
    });
  }

  /**
   * Устанавливает лимиты риска
   */
  setRiskLimits(
    accountBalance: number,
    riskTolerance: 'conservative' | 'moderate' | 'aggressive'
  ): RiskLimits {
    const profiles = {
      conservative: {
        maxPositionSize: 0.05,    // 5%
        maxDailyLoss: 0.02,       // 2%
        maxDrawdown: 0.10,        // 10%
        maxCorrelatedPositions: 3,
        maxSectorExposure: 0.20   // 20%
      },
      moderate: {
        maxPositionSize: 0.10,    // 10%
        maxDailyLoss: 0.03,       // 3%
        maxDrawdown: 0.15,        // 15%
        maxCorrelatedPositions: 5,
        maxSectorExposure: 0.30   // 30%
      },
      aggressive: {
        maxPositionSize: 0.20,    // 20%
        maxDailyLoss: 0.05,       // 5%
        maxDrawdown: 0.25,        // 25%
        maxCorrelatedPositions: 7,
        maxSectorExposure: 0.50   // 50%
      }
    };

    return profiles[riskTolerance];
  }

  // Приватные методы для расчетов

  private getDefaultRiskMetrics(): RiskMetrics {
    return {
      maxDrawdown: 0,
      sharpeRatio: 0,
      winRate: 0,
      profitFactor: 0,
      valueAtRisk: 0,
      expectedShortfall: 0,
      beta: 1,
      alpha: 0,
      informationRatio: 0,
      calmarRatio: 0
    };
  }

  private annualizeReturns(returns: number[]): number {
    const totalReturn = returns.reduce((acc, r) => acc * (1 + r), 1) - 1;
    const periodsPerYear = 252; // Торговые дни
    const annualizedReturn = Math.pow(1 + totalReturn, periodsPerYear / returns.length) - 1;
    return annualizedReturn;
  }

  private calculateVolatility(returns: number[]): number {
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / (returns.length - 1);
    return Math.sqrt(variance * 252); // Аннуализированная волатильность
  }

  private calculateMaxDrawdown(returns: number[]): number {
    let peak = 0;
    let maxDrawdown = 0;
    let cumReturn = 0;

    for (const ret of returns) {
      cumReturn += ret;
      peak = Math.max(peak, cumReturn);
      const drawdown = (peak - cumReturn) / (1 + peak);
      maxDrawdown = Math.max(maxDrawdown, drawdown);
    }

    return maxDrawdown;
  }

  private calculateWinRate(returns: number[]): number {
    const winningTrades = returns.filter(r => r > 0).length;
    return winningTrades / returns.length;
  }

  private calculateProfitFactor(returns: number[]): number {
    const grossProfit = returns.filter(r => r > 0).reduce((sum, r) => sum + r, 0);
    const grossLoss = Math.abs(returns.filter(r => r < 0).reduce((sum, r) => sum + r, 0));
    
    return grossLoss === 0 ? (grossProfit > 0 ? Infinity : 0) : grossProfit / grossLoss;
  }

  private calculateSharpeRatio(
    annualizedReturn: number, 
    volatility: number, 
    riskFreeRate: number
  ): number {
    return volatility === 0 ? 0 : (annualizedReturn - riskFreeRate) / volatility;
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
    const portfolioReturn = this.annualizeReturns(returns);
    const marketReturn = this.annualizeReturns(marketReturns);
    
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

  private calculateKellyRatio(winRate: number, avgWin: number, avgLoss: number): number {
    if (avgLoss === 0) return 0;
    return winRate - ((1 - winRate) / (avgWin / Math.abs(avgLoss)));
  }

  private calculatePortfolioImpact(
    positions: any[],
    priceChange: number,
    correlationMatrix: number[][]
  ): number {
    // Упрощенный расчет воздействия на портфель
    // В реальной системе здесь была бы более сложная логика учета корреляций
    return positions.reduce((impact, position, index) => {
      const positionImpact = position.size * priceChange;
      const correlationFactor = correlationMatrix[index] ? 
        correlationMatrix[index].reduce((sum, corr) => sum + Math.abs(corr), 0) / correlationMatrix[index].length : 1;
      
      return impact + (positionImpact * correlationFactor);
    }, 0);
  }
}

export const riskManagementService = RiskManagementService.getInstance();