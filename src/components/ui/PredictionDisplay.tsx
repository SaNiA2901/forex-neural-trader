
import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { TrendingUp, TrendingDown, BarChart3, Brain, Activity, History, Eye } from 'lucide-react';
import { CandleData, TradingSession } from '@/types/session';
import { usePredictionGeneration } from '@/hooks/usePredictionGeneration';
import PredictionHistory from './PredictionHistory';

interface PredictionDisplayProps {
  candles: CandleData[];
  currentSession: TradingSession | null;
}

const PredictionDisplay = ({ candles, currentSession }: PredictionDisplayProps) => {
  const { getModelStats } = usePredictionGeneration();
  const [modelStats, setModelStats] = useState<any>(null);

  const candlesWithPredictions = candles.filter(candle => 
    candle.prediction_direction && candle.prediction_probability
  );

  useEffect(() => {
    if (candlesWithPredictions.length > 0) {
      setModelStats(getModelStats());
    }
  }, [candlesWithPredictions.length, getModelStats]);

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
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-2 bg-card/50 backdrop-blur-sm">
          <TabsTrigger 
            value="overview" 
            className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
          >
            <Eye className="h-4 w-4 mr-2" />
            Обзор прогнозов
          </TabsTrigger>
          <TabsTrigger 
            value="history" 
            className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
          >
            <History className="h-4 w-4 mr-2" />
            История
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <Card className="p-6 bg-gradient-to-br from-card/80 to-card/50 border-border/50 backdrop-blur-sm">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-foreground">
                AI Прогнозы для {currentSession.session_name}
              </h3>
              <Badge variant="outline" className="bg-primary/10 border-primary/30">
                Последние {Math.min(6, candlesWithPredictions.length)} прогнозов
              </Badge>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {candlesWithPredictions.slice(-6).map((candle) => (
                <Card key={candle.id} className="p-4 bg-background/50 border-border/50 hover:bg-background/70 transition-all duration-200 hover:scale-105">
                  <div className="flex justify-between items-center mb-3">
                    <span className="text-sm text-muted-foreground">
                      Свеча #{candle.candle_index + 1}
                    </span>
                    <Badge 
                      className={
                        candle.prediction_direction === 'UP' 
                          ? 'bg-green-500/20 text-green-400 border-green-500/30' 
                          : 'bg-red-500/20 text-red-400 border-red-500/30'
                      }
                    >
                      <div className="flex items-center space-x-1">
                        {candle.prediction_direction === 'UP' ? (
                          <TrendingUp className="h-3 w-3" />
                        ) : (
                          <TrendingDown className="h-3 w-3" />
                        )}
                        <span>{candle.prediction_direction}</span>
                      </div>
                    </Badge>
                  </div>

                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Цена закрытия:</span>
                      <span className="text-foreground font-medium">{candle.close}</span>
                    </div>
                    
                    {candle.prediction_probability && (
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Вероятность:</span>
                        <span className="text-blue-400 font-medium">
                          {candle.prediction_probability.toFixed(1)}%
                        </span>
                      </div>
                    )}

                    {candle.prediction_confidence && (
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Уверенность:</span>
                        <span className="text-green-400 font-medium">
                          {candle.prediction_confidence.toFixed(1)}%
                        </span>
                      </div>
                    )}

                    <div className="mt-3 p-2 bg-muted/50 rounded-md text-xs space-y-1">
                      <div className="text-muted-foreground">
                        O: {candle.open} | H: {candle.high}
                      </div>
                      <div className="text-muted-foreground">
                        L: {candle.low} | C: {candle.close}
                      </div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </Card>

          {candlesWithPredictions.length > 0 && (
            <>
              <Card className="p-6 bg-gradient-to-br from-card/80 to-card/50 border-border/50 backdrop-blur-sm">
                <h4 className="text-lg font-medium text-foreground mb-4">Быстрая статистика</h4>
                
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

              {modelStats && (
                <Card className="p-6 bg-gradient-to-br from-card/80 to-card/50 border-border/50 backdrop-blur-sm">
                  <div className="flex items-center space-x-3 mb-4">
                    <Brain className="h-5 w-5 text-purple-400" />
                    <h4 className="text-lg font-medium text-foreground">Статистика модели ML</h4>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Общая точность */}
                    <div className="space-y-3">
                      <div className="text-center">
                        <div className="text-3xl font-bold text-foreground mb-2">
                          {modelStats.overallAccuracy.toFixed(1)}%
                        </div>
                        <Progress value={modelStats.overallAccuracy} className="mb-2 h-2" />
                        <p className="text-sm text-muted-foreground">Общая точность</p>
                      </div>
                    </div>

                    {/* CALL точность */}
                    <div className="space-y-3">
                      <div className="text-center">
                        <div className="text-3xl font-bold text-green-500 mb-2">
                          {modelStats.callAccuracy.toFixed(1)}%
                        </div>
                        <Progress value={modelStats.callAccuracy} className="mb-2 h-2" />
                        <p className="text-sm text-muted-foreground">Точность CALL</p>
                      </div>
                    </div>

                    {/* PUT точность */}
                    <div className="space-y-3">
                      <div className="text-center">
                        <div className="text-3xl font-bold text-red-500 mb-2">
                          {modelStats.putAccuracy.toFixed(1)}%
                        </div>
                        <Progress value={modelStats.putAccuracy} className="mb-2 h-2" />
                        <p className="text-sm text-muted-foreground">Точность PUT</p>
                      </div>
                    </div>
                  </div>

                  {/* Веса модели */}
                  <div className="mt-6">
                    <h5 className="text-foreground font-medium mb-3 flex items-center space-x-2">
                      <Activity className="h-4 w-4 text-blue-400" />
                      <span>Адаптивные веса факторов</span>
                    </h5>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {Object.entries(modelStats.currentWeights).map(([factor, weight]) => (
                        <div key={factor} className="bg-background/50 rounded-lg p-3 border border-border/50">
                          <div className="flex justify-between items-center mb-1">
                            <span className="text-sm text-muted-foreground capitalize">{factor}</span>
                            <span className="text-xs text-foreground font-medium">{((weight as number) * 100).toFixed(1)}%</span>
                          </div>
                          <Progress value={(weight as number) * 100} className="h-1" />
                        </div>
                      ))}
                    </div>
                  </div>
                </Card>
              )}
            </>
          )}
        </TabsContent>

        <TabsContent value="history">
          <PredictionHistory candles={candles} />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default PredictionDisplay;
