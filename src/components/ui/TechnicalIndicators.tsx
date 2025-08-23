
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Database, Activity, TrendingUp, TrendingDown } from "lucide-react";
import { useTradingStore } from '@/store/TradingStore';
import { useState, useEffect } from 'react';
import { TechnicalIndicatorService } from '@/services/indicators/TechnicalIndicators';
import RSIIndicator from "@/components/ui/RSIIndicator";
import MACDIndicator from "@/components/ui/MACDIndicator";
import MovingAveragesIndicator from "@/components/ui/MovingAveragesIndicator";
import StochasticIndicator from "@/components/ui/StochasticIndicator";

interface TechnicalIndicatorsProps {
  pair: string;
  timeframe: string;
}

const TechnicalIndicators = ({ pair, timeframe }: TechnicalIndicatorsProps) => {
  const { state } = useTradingStore();

  // Вычисляем технические индикаторы на основе реальных данных
  const [indicators, setIndicators] = useState<any>(null);

  useEffect(() => {
    const calculateIndicators = async () => {
      if (!state.candles || state.candles.length < 20) {
        setIndicators(null);
        return;
      }

      const sortedCandles = [...state.candles].sort((a, b) => a.candle_index - b.candle_index);
      const currentIndex = sortedCandles.length - 1;
      
      try {
        const result = await TechnicalIndicatorService.calculateAll(sortedCandles, currentIndex);
        setIndicators(result);
      } catch (error) {
        console.error('Error calculating indicators:', error);
        setIndicators(null);
      }
    };

    calculateIndicators();
  }, [state.candles]);

  // Если нет активной сессии
  if (!state.currentSession) {
    return (
      <Card className="p-8 bg-card/30 border-border/50 text-center backdrop-blur-sm">
        <Database className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
        <div className="text-muted-foreground mb-2">
          Нет активной сессии для анализа
        </div>
        <div className="text-sm text-muted-foreground/70">
          Создайте или загрузите сессию в разделе "Бинарные опционы"
        </div>
      </Card>
    );
  }

  // Если недостаточно данных для расчета индикаторов
  if (!indicators || state.candles.length < 20) {
    return (
      <Card className="p-8 bg-card/30 border-border/50 text-center backdrop-blur-sm">
        <Activity className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
        <div className="text-muted-foreground mb-2">
          Недостаточно данных для расчета технических индикаторов
        </div>
        <div className="text-sm text-muted-foreground/70">
          Добавьте минимум 20 свечей для корректного расчета индикаторов
        </div>
        <Badge variant="outline" className="mt-3">
          Текущее количество: {state.candles.length} / 20 свечей
        </Badge>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <Activity className="h-6 w-6 text-primary" />
            <h3 className="text-lg font-semibold text-foreground">Технические индикаторы для {pair}</h3>
          </div>
          <div className="flex items-center space-x-2">
            <Badge variant="outline">Сессия: {state.currentSession.session_name}</Badge>
            <Badge variant="secondary">{state.candles.length} свечей</Badge>
          </div>
        </div>

        <div className="text-muted-foreground text-sm mb-4">
          Индикаторы рассчитаны на основе {state.candles.length} свечей из текущей сессии.
          Последнее обновление: {new Date().toLocaleTimeString()}
        </div>
      </Card>

      <RSIIndicator rsi={indicators.rsi} />
      <Card className="p-6 bg-card/50 border-border/50">
        <h3 className="text-lg font-semibold mb-4">MACD: {indicators.macd.line.toFixed(4)}</h3>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-sm text-muted-foreground">Линия</div>
            <div className="text-foreground font-medium">{indicators.macd.line.toFixed(4)}</div>
          </div>
          <div className="text-center">
            <div className="text-sm text-muted-foreground">Сигнал</div>
            <div className="text-foreground font-medium">{indicators.macd.signal.toFixed(4)}</div>
          </div>
          <div className="text-center">
            <div className="text-sm text-muted-foreground">Гистограмма</div>
            <div className="text-foreground font-medium">{indicators.macd.histogram.toFixed(4)}</div>
          </div>
        </div>
      </Card>
      <StochasticIndicator stochastic={indicators.stochastic} />
    </div>
  );
};

export default TechnicalIndicators;
