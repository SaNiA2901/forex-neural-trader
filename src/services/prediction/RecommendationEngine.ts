import { TechnicalIndicators } from '../indicators/TechnicalIndicators';

export class RecommendationEngine {
  static generateRecommendation(
    direction: 'UP' | 'DOWN',
    probability: number,
    interval: number,
    pattern: string | null,
    technical: TechnicalIndicators
  ): string {
    let recommendation = `Рекомендуем ${direction === 'UP' ? 'CALL' : 'PUT'} опцион на ${interval} мин`;
    
    if (pattern) {
      recommendation += `. Обнаружен паттерн "${pattern}"`;
    }
    
    if (technical.rsi > 80) {
      recommendation += '. Актив в зоне перекупленности';
    } else if (technical.rsi < 20) {
      recommendation += '. Актив в зоне перепроданности';
    }
    
    if (probability > 85) {
      recommendation += '. Высокая уверенность в прогнозе';
    } else if (probability < 65) {
      recommendation += '. Рекомендуется дополнительное подтверждение';
    }
    
    if (technical.macd.histogram > 0 && direction === 'UP') {
      recommendation += '. MACD поддерживает бычий тренд';
    } else if (technical.macd.histogram < 0 && direction === 'DOWN') {
      recommendation += '. MACD поддерживает медвежий тренд';
    }
    
    return recommendation;
  }

  static generateRiskWarning(
    probability: number,
    volatility: number,
    technical: TechnicalIndicators
  ): string[] {
    const warnings: string[] = [];
    
    if (probability < 65) {
      warnings.push('Низкая вероятность успеха - рекомендуется осторожность');
    }
    
    if (volatility > 85) {
      warnings.push('Высокая волатильность может увеличить риск');
    }
    
    if (technical.rsi > 70 && technical.rsi < 80) {
      warnings.push('RSI близок к зоне перекупленности');
    } else if (technical.rsi > 20 && technical.rsi < 30) {
      warnings.push('RSI близок к зоне перепроданности');
    }
    
    if (technical.adx < 25) {
      warnings.push('Слабый тренд может привести к неопределенности');
    }
    
    return warnings;
  }

  static calculateRiskScore(
    probability: number,
    confidence: number,
    volatility: number,
    technical: TechnicalIndicators
  ): number {
    let riskScore = 50; // Базовый риск
    
    // Снижаем риск при высокой вероятности
    riskScore -= (probability - 50) * 0.5;
    
    // Снижаем риск при высокой уверенности
    riskScore -= (confidence - 50) * 0.3;
    
    // Увеличиваем риск при высокой волатильности
    riskScore += (volatility - 50) * 0.4;
    
    // Корректируем на основе RSI
    if (technical.rsi > 80 || technical.rsi < 20) {
      riskScore += 10; // Экстремальные зоны увеличивают риск
    }
    
    // Корректируем на основе силы тренда
    if (technical.adx < 25) {
      riskScore += 15; // Слабый тренд увеличивает риск
    }
    
    return Math.max(0, Math.min(100, riskScore));
  }
}