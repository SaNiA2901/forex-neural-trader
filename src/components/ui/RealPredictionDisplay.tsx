import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TrendingUp, TrendingDown, Target, Brain, Database, Activity } from "lucide-react";
import { useTradingStore } from '@/store/TradingStore';
import { usePredictionGeneration } from '@/hooks/usePredictionGeneration';
import { useMemo, useState } from 'react';

interface RealPredictionDisplayProps {
  pair: string;
  timeframe: string;
}

const RealPredictionDisplay = ({ pair, timeframe }: RealPredictionDisplayProps) => {
  const { state } = useTradingStore();
  const { predictionResult, generatePrediction, isGenerating } = usePredictionGeneration();
  const [lastPredictionTime, setLastPredictionTime] = useState<Date | null>(null);

  // Получаем последнюю свечу для прогноза
  const lastCandle = useMemo(() => {
    if (!state.candles || state.candles.length === 0) return null;
    return [...state.candles].sort((a, b) => a.candle_index - b.candle_index).pop();
  }, [state.candles]);

  // Генерация нового прогноза
  const handleGeneratePrediction = async () => {
    if (!lastCandle || !state.currentSession) return;

    const predictionConfig = {
      predictionInterval: parseInt(timeframe.replace(/[^0-9]/g, '')) || 5,
      analysisMode: 'session' as const
    };

    const allCandles = [...state.candles].sort((a, b) => a.candle_index - b.candle_index);
    await generatePrediction(lastCandle, predictionConfig, allCandles);
    setLastPredictionTime(new Date());
  };

  // Получаем свечи с прогнозами
  const candlesWithPredictions = useMemo(() => {
    return state.candles.filter(candle => 
      candle.prediction_direction && candle.prediction_probability
    );
  }, [state.candles]);

  // Если нет активной сессии
  if (!state.currentSession) {
    return (
      <Card className="p-8 bg-card/30 border-border/50 text-center backdrop-blur-sm">
        <Database className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
        <div className="text-muted-foreground mb-2">
          Нет активной сессии для прогнозирования
        </div>
        <div className="text-sm text-muted-foreground/70">
          Создайте или загрузите сессию в разделе "Бинарные опционы"
        </div>
      </Card>
    );
  }

  // Если нет данных свечей
  if (!state.candles || state.candles.length === 0) {
    return (
      <Card className="p-8 bg-card/30 border-border/50 text-center backdrop-blur-sm">
        <Activity className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
        <div className="text-muted-foreground mb-2">
          Нет данных для генерации прогнозов
        </div>
        <div className="text-sm text-muted-foreground/70">
          Добавьте свечи в текущую сессию для получения прогнозов
        </div>
      </Card>
    );
  }

  const formatPrice = (price: number) => {
    return pair.includes("JPY") ? price.toFixed(2) : price.toFixed(4);
  };

  return (
    <div className="space-y-6">
      {/* Основной прогноз */}
      <Card className="p-6 bg-gradient-to-br from-primary/5 to-primary/10 border-primary/20 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <Brain className="h-6 w-6 text-primary" />
            <div>
              <h3 className="text-xl font-semibold text-foreground">
                Прогноз следующей свечи для {pair}
              </h3>
              <p className="text-sm text-muted-foreground">
                Сессия: {state.currentSession.session_name} • {timeframe}
              </p>
            </div>
          </div>
          
          <Button 
            onClick={handleGeneratePrediction}
            disabled={isGenerating || !lastCandle}
            className="bg-primary hover:bg-primary/90"
          >
            {isGenerating ? (
              <>
                <Activity className="h-4 w-4 mr-2 animate-spin" />
                Анализ...
              </>
            ) : (
              <>
                <Target className="h-4 w-4 mr-2" />
                Генерировать прогноз
              </>
            )}
          </Button>
        </div>

        {predictionResult ? (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-background/50 rounded-lg border border-border/50">
              <div className={`text-3xl font-bold mb-2 flex items-center justify-center space-x-2 ${
                predictionResult.direction === 'UP' ? 'text-green-500' : 'text-red-500'
              }`}>
                {predictionResult.direction === 'UP' ? (
                  <TrendingUp className="h-8 w-8" />
                ) : (
                  <TrendingDown className="h-8 w-8" />
                )}
                <span>{predictionResult.direction}</span>
              </div>
              <p className="text-muted-foreground text-sm">Направление</p>
            </div>
            
            <div className="text-center p-4 bg-background/50 rounded-lg border border-border/50">
              <div className="text-3xl font-bold text-primary mb-2">
                {predictionResult.probability}%
              </div>
              <p className="text-muted-foreground text-sm">Вероятность успеха</p>
            </div>
            
            <div className="text-center p-4 bg-background/50 rounded-lg border border-border/50">
              <div className="text-3xl font-bold text-foreground mb-2">
                {predictionResult.confidence}%
              </div>
              <p className="text-muted-foreground text-sm">Уверенность модели</p>
            </div>
            
            <div className="text-center p-4 bg-background/50 rounded-lg border border-border/50">
              <div className="text-3xl font-bold text-blue-500 mb-2">
                {predictionResult.interval}м
              </div>
              <p className="text-muted-foreground text-sm">Временной горизонт</p>
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <Target className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
            <div className="text-muted-foreground mb-2">
              Прогноз не сгенерирован
            </div>
            <div className="text-sm text-muted-foreground/70">
              Нажмите кнопку "Генерировать прогноз" для получения прогноза следующей свечи
            </div>
          </div>
        )}

        {predictionResult?.recommendation && (
          <div className="mt-6 p-4 bg-background/30 rounded-lg border border-border/50">
            <h4 className="font-medium text-foreground mb-2">Рекомендация:</h4>
            <p className="text-sm text-muted-foreground">
              {predictionResult.recommendation}
            </p>
          </div>
        )}

        {lastPredictionTime && (
          <div className="mt-4 text-xs text-muted-foreground text-center">
            Последний прогноз: {lastPredictionTime.toLocaleTimeString()}
          </div>
        )}
      </Card>

      {/* История прогнозов из свечей */}
      {candlesWithPredictions.length > 0 && (
        <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
          <h4 className="text-lg font-semibold text-foreground mb-4">
            История прогнозов сессии
          </h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {candlesWithPredictions.slice(-6).map((candle) => (
              <Card key={candle.id} className="p-4 bg-background/50 border-border/50">
                <div className="flex justify-between items-center mb-3">
                  <span className="text-sm text-muted-foreground">
                    Свеча #{candle.candle_index + 1}
                  </span>
                  <Badge 
                    variant="outline"
                    className={
                      candle.prediction_direction === 'UP' 
                        ? 'border-green-500/50 text-green-500' 
                        : 'border-red-500/50 text-red-500'
                    }
                  >
                    {candle.prediction_direction}
                  </Badge>
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Цена:</span>
                    <span className="text-foreground font-medium">
                      {formatPrice(candle.close)}
                    </span>
                  </div>
                  
                  {candle.prediction_probability && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Вероятность:</span>
                      <span className="text-primary font-medium">
                        {candle.prediction_probability.toFixed(1)}%
                      </span>
                    </div>
                  )}

                  {candle.prediction_confidence && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Уверенность:</span>
                      <span className="text-green-500 font-medium">
                        {candle.prediction_confidence.toFixed(1)}%
                      </span>
                    </div>
                  )}
                </div>
              </Card>
            ))}
          </div>

          {candlesWithPredictions.length > 6 && (
            <div className="text-center mt-4">
              <Badge variant="outline">
                Показано последние 6 из {candlesWithPredictions.length} прогнозов
              </Badge>
            </div>
          )}
        </Card>
      )}

      {/* Статистика */}
      {candlesWithPredictions.length > 0 && (
        <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
          <h4 className="text-lg font-semibold text-foreground mb-4">
            Статистика прогнозов
          </h4>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">
                {candlesWithPredictions.length}
              </div>
              <div className="text-sm text-muted-foreground">Всего прогнозов</div>
            </div>

            <div className="text-center">
              <div className="text-2xl font-bold text-green-500">
                {candlesWithPredictions.filter(c => c.prediction_direction === 'UP').length}
              </div>
              <div className="text-sm text-muted-foreground">CALL сигналов</div>
            </div>

            <div className="text-center">
              <div className="text-2xl font-bold text-red-500">
                {candlesWithPredictions.filter(c => c.prediction_direction === 'DOWN').length}
              </div>
              <div className="text-sm text-muted-foreground">PUT сигналов</div>
            </div>

            <div className="text-center">
              <div className="text-2xl font-bold text-blue-500">
                {(candlesWithPredictions.reduce((sum, c) => sum + (c.prediction_probability || 0), 0) / candlesWithPredictions.length).toFixed(1)}%
              </div>
              <div className="text-sm text-muted-foreground">Средняя вероятность</div>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};

export default RealPredictionDisplay;