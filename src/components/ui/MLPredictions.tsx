
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { TrendingUp, TrendingDown, Brain, Target, Activity, Database } from "lucide-react";
import { useTradingStore } from '@/store/TradingStore';
import { usePredictionGeneration } from '@/hooks/usePredictionGeneration';
import { useMemo } from 'react';
import PredictionDisplay from './PredictionDisplay';

interface MLPredictionsProps {
  pair: string;
  timeframe: string;
}

const MLPredictions = ({ pair, timeframe }: MLPredictionsProps) => {
  const { state } = useTradingStore();
  const { getModelStats } = usePredictionGeneration();

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

  // Показываем реальные данные прогнозов из текущей сессии
  return (
    <div className="space-y-6">
      <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <Brain className="h-6 w-6 text-primary" />
            <h3 className="text-lg font-semibold text-foreground">AI Прогнозы для {pair}</h3>
          </div>
          <Badge variant="outline" className="bg-primary/10">
            Сессия: {state.currentSession.session_name}
          </Badge>
        </div>

        <div className="text-muted-foreground text-sm mb-4">
          Анализ основан на реальных данных свечей из текущей сессии. 
          Прогнозы генерируются автоматически при добавлении новых свечей.
        </div>
      </Card>

      {/* Используем реальный компонент прогнозов */}
      <PredictionDisplay 
        candles={state.candles} 
        currentSession={state.currentSession} 
      />
    </div>
  );
};

export default MLPredictions;
