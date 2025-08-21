import { memo, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { TrendingUp, TrendingDown, BarChart3, Brain, Activity, History, Eye, Target } from 'lucide-react';
import { CandleData, TradingSession } from '@/types/session';
import { usePredictionGeneration } from '@/hooks/usePredictionGeneration';
import PredictionHistory from '../PredictionHistory';

interface OptimizedPredictionDisplayProps {
  candles: CandleData[];
  currentSession: TradingSession | null;
}

/**
 * Optimized Prediction Display with memoization and performance improvements
 */
const OptimizedPredictionDisplay = memo(({ candles, currentSession }: OptimizedPredictionDisplayProps) => {
  const { getModelStats } = usePredictionGeneration();

  // Memoized calculations
  const candlesWithPredictions = useMemo(() => 
    candles.filter(candle => 
      candle.prediction_direction && candle.prediction_probability
    ), [candles]
  );

  const modelStats = useMemo(() => {
    if (candlesWithPredictions.length > 0) {
      return getModelStats();
    }
    return null;
  }, [candlesWithPredictions.length, getModelStats]);

  const predictionStats = useMemo(() => {
    if (candlesWithPredictions.length === 0) return null;

    const upPredictions = candlesWithPredictions.filter(c => c.prediction_direction === 'UP');
    const downPredictions = candlesWithPredictions.filter(c => c.prediction_direction === 'DOWN');
    
    const avgProbability = candlesWithPredictions.reduce((sum, c) => 
      sum + (c.prediction_probability || 0), 0
    ) / candlesWithPredictions.length;

    const avgConfidence = candlesWithPredictions.reduce((sum, c) => 
      sum + (c.prediction_confidence || 0), 0
    ) / candlesWithPredictions.length;

    return {
      total: candlesWithPredictions.length,
      upCount: upPredictions.length,
      downCount: downPredictions.length,
      upPercentage: (upPredictions.length / candlesWithPredictions.length) * 100,
      downPercentage: (downPredictions.length / candlesWithPredictions.length) * 100,
      avgProbability,
      avgConfidence
    };
  }, [candlesWithPredictions]);

  const recentPredictions = useMemo(() => 
    candlesWithPredictions
      .sort((a, b) => b.candle_index - a.candle_index)
      .slice(0, 5), 
    [candlesWithPredictions]
  );

  if (!currentSession) {
    return (
      <Card className="p-8 bg-card/30 border-border/50 text-center backdrop-blur-sm">
        <div className="text-muted-foreground">
          Создайте или загрузите сессию для просмотра прогнозов
        </div>
      </Card>
    );
  }

  if (candlesWithPredictions.length === 0) {
    return (
      <Card className="p-8 bg-card/30 border-border/50 text-center backdrop-blur-sm">
        <BarChart3 className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
        <div className="text-muted-foreground mb-2">
          Нет доступных прогнозов для текущей сессии
        </div>
        <div className="text-sm text-muted-foreground/70">
          Добавьте свечи, чтобы получить AI прогнозы
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">AI Прогнозы</h2>
          <p className="text-sm text-muted-foreground">
            {currentSession.pair} • {currentSession.timeframe} • {predictionStats?.total} прогнозов
          </p>
        </div>
        <Badge variant="outline" className="bg-primary/10">
          <Brain className="h-4 w-4 mr-1" />
          Активно
        </Badge>
      </div>

      {/* Prediction Statistics */}
      {predictionStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="p-4 bg-card/50 border-border/50 backdrop-blur-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Всего прогнозов</p>
                <p className="text-2xl font-bold text-foreground">{predictionStats.total}</p>
              </div>
              <Target className="h-8 w-8 text-chart-1" />
            </div>
          </Card>

          <Card className="p-4 bg-card/50 border-border/50 backdrop-blur-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">UP прогнозов</p>
                <p className="text-2xl font-bold text-green-500">
                  {predictionStats.upCount} ({predictionStats.upPercentage.toFixed(0)}%)
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-500" />
            </div>
          </Card>

          <Card className="p-4 bg-card/50 border-border/50 backdrop-blur-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">DOWN прогнозов</p>
                <p className="text-2xl font-bold text-red-500">
                  {predictionStats.downCount} ({predictionStats.downPercentage.toFixed(0)}%)
                </p>
              </div>
              <TrendingDown className="h-8 w-8 text-red-500" />
            </div>
          </Card>

          <Card className="p-4 bg-card/50 border-border/50 backdrop-blur-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Ср. уверенность</p>
                <p className="text-2xl font-bold text-foreground">
                  {predictionStats.avgConfidence.toFixed(1)}%
                </p>
              </div>
              <Activity className="h-8 w-8 text-chart-4" />
            </div>
          </Card>
        </div>
      )}

      <Tabs defaultValue="recent" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="recent">Последние</TabsTrigger>
          <TabsTrigger value="history">История</TabsTrigger>
          <TabsTrigger value="model">Модель</TabsTrigger>
        </TabsList>

        <TabsContent value="recent">
          <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-6">
              <Eye className="h-5 w-5 text-primary" />
              <h3 className="text-lg font-semibold">Последние прогнозы</h3>
            </div>
            <div className="space-y-4">
              {recentPredictions.map((candle) => (
                <div key={candle.id} className="flex items-center justify-between p-4 bg-background/50 rounded-lg border border-border/50">
                  <div className="flex items-center gap-4">
                    <div className="text-center">
                      <div className="text-sm text-muted-foreground">Свеча</div>
                      <div className="font-mono font-semibold">#{candle.candle_index}</div>
                    </div>
                    <div className="h-8 w-px bg-border" />
                    <div className="text-center">
                      <div className="text-sm text-muted-foreground">Направление</div>
                      <div className={`font-semibold flex items-center gap-1 ${
                        candle.prediction_direction === 'UP' ? 'text-green-500' : 'text-red-500'
                      }`}>
                        {candle.prediction_direction === 'UP' ? (
                          <TrendingUp className="h-4 w-4" />
                        ) : (
                          <TrendingDown className="h-4 w-4" />
                        )}
                        {candle.prediction_direction}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-6">
                    <div className="text-center">
                      <div className="text-sm text-muted-foreground">Вероятность</div>
                      <div className="font-semibold">{candle.prediction_probability?.toFixed(1)}%</div>
                    </div>
                    <div className="text-center">
                      <div className="text-sm text-muted-foreground">Уверенность</div>
                      <div className="font-semibold">{candle.prediction_confidence?.toFixed(1)}%</div>
                      <Progress 
                        value={candle.prediction_confidence || 0} 
                        className="h-1 w-16 mt-1"
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="history">
          <PredictionHistory candles={candlesWithPredictions} />
        </TabsContent>

        <TabsContent value="model">
          {modelStats ? (
            <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
              <div className="flex items-center gap-2 mb-6">
                <Brain className="h-5 w-5 text-primary" />
                <h3 className="text-lg font-semibold">Статистика модели</h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Общая точность</div>
                    <div className="text-2xl font-bold">{modelStats.overallAccuracy.toFixed(1)}%</div>
                    <Progress value={modelStats.overallAccuracy} className="h-2 mt-2" />
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Точность CALL</div>
                    <div className="text-xl font-semibold text-green-500">{modelStats.callAccuracy.toFixed(1)}%</div>
                    <Progress value={modelStats.callAccuracy} className="h-2 mt-2" />
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Точность PUT</div>
                    <div className="text-xl font-semibold text-red-500">{modelStats.putAccuracy.toFixed(1)}%</div>
                    <Progress value={modelStats.putAccuracy} className="h-2 mt-2" />
                  </div>
                </div>
                <div className="space-y-4">
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Всего прогнозов</div>
                    <div className="text-xl font-semibold">{modelStats.totalPredictions}</div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground mb-1">Точных прогнозов</div>
                    <div className="text-xl font-semibold text-green-500">{modelStats.accurateCount}</div>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Веса модели обновляются автоматически на основе производительности
                  </div>
                </div>
              </div>
            </Card>
          ) : (
            <Card className="p-8 bg-card/30 border-border/50 text-center backdrop-blur-sm">
              <div className="text-muted-foreground">
                Недостаточно данных для отображения статистики модели
              </div>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
});

OptimizedPredictionDisplay.displayName = 'OptimizedPredictionDisplay';

export default OptimizedPredictionDisplay;