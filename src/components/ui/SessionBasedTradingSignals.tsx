import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { TrendingUp, TrendingDown, AlertTriangle, Clock, Activity, BarChart3 } from "lucide-react";
import { useNewApplicationState } from "@/hooks/useNewApplicationState";
import { useSignalGeneration } from "@/hooks/useSignalGeneration";

interface SessionBasedTradingSignalsProps {
  pair: string;
  timeframe: string;
}

const SessionBasedTradingSignals = ({ pair, timeframe }: SessionBasedTradingSignalsProps) => {
  const { currentSession, candles } = useNewApplicationState();
  const { signals, overallSentiment, confidence } = useSignalGeneration(pair, timeframe);

  if (!currentSession) {
    return (
      <Alert className="border-orange-600 bg-orange-600/20">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription className="text-orange-200">
          Для получения торговых сигналов необходимо создать активную сессию и добавить данные свечей.
        </AlertDescription>
      </Alert>
    );
  }

  if (candles.length < 10) {
    return (
      <Alert className="border-blue-600 bg-blue-600/20">
        <Activity className="h-4 w-4" />
        <AlertDescription className="text-blue-200">
          Недостаточно данных для анализа. Добавьте минимум 10 свечей для генерации сигналов.
          Текущее количество: {candles.length}
        </AlertDescription>
      </Alert>
    );
  }

  const formatPrice = (price: number) => {
    return pair.includes("JPY") ? price.toFixed(2) : price.toFixed(4);
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'BULLISH': return 'bg-green-600';
      case 'BEARISH': return 'bg-red-600';
      default: return 'bg-blue-600';
    }
  };

  const getSentimentText = (sentiment: string) => {
    switch (sentiment) {
      case 'BULLISH': return 'Бычий';
      case 'BEARISH': return 'Медвежий';
      default: return 'Нейтральный';
    }
  };

  return (
    <div className="space-y-6">
      {/* Market Analysis Overview */}
      <Card className="p-6 bg-gradient-to-br from-slate-800/80 to-slate-900/80 border-slate-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Анализ рынка на основе сессии</h3>
          <Badge className="bg-blue-600">
            Сессия: {currentSession.session_name || 'Активная сессия'}
          </Badge>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <Badge className={`${getSentimentColor(overallSentiment)} text-white mb-2`}>
              {getSentimentText(overallSentiment)}
            </Badge>
            <p className="text-slate-400 text-sm">Настроение рынка</p>
          </div>
          
          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <div className="text-2xl font-bold text-white mb-1">{confidence}%</div>
            <Progress value={confidence} className="mb-2 h-2" />
            <p className="text-slate-400 text-sm">Уверенность анализа</p>
          </div>
          
          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <div className="text-2xl font-bold text-blue-400 mb-1">{signals.length}</div>
            <p className="text-slate-400 text-sm">Активных сигналов</p>
          </div>
          
          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <div className="text-2xl font-bold text-purple-400 mb-1">{candles.length}</div>
            <p className="text-slate-400 text-sm">Проанализировано свечей</p>
          </div>
        </div>

        <div className="flex items-center space-x-2 text-sm text-slate-400">
          <BarChart3 className="h-4 w-4" />
          <span>Анализ основан на реальных данных сессии: {pair} | {timeframe}</span>
        </div>
      </Card>

      {/* Trading Signals */}
      <Card className="p-6 bg-slate-800/50 border-slate-700">
        <h3 className="text-lg font-semibold text-white mb-4">
          Торговые сигналы для бинарных опционов
        </h3>
        
        {signals.length === 0 ? (
          <div className="text-center py-8">
            <AlertTriangle className="h-12 w-12 text-slate-500 mx-auto mb-4" />
            <p className="text-slate-400">
              В данный момент нет сильных торговых сигналов.
              Продолжайте добавлять данные свечей для улучшения анализа.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {signals.map((signal) => (
              <div 
                key={signal.id} 
                className="border border-slate-600 rounded-lg p-4 bg-slate-700/30 hover:bg-slate-700/50 transition-colors"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <Badge 
                      className={`${signal.type === 'CALL' ? 'bg-green-600' : 'bg-red-600'} text-white flex items-center space-x-1`}
                    >
                      {signal.type === 'CALL' ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
                      <span>{signal.type}</span>
                    </Badge>
                    
                    <div className="text-white font-medium">
                      {pair} @ {formatPrice(signal.entry)}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2 text-slate-400">
                    <Clock className="h-4 w-4" />
                    <span className="text-sm">{signal.timeLeft} мин</span>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
                  <div>
                    <p className="text-slate-400 text-sm">Сила сигнала</p>
                    <div className="flex items-center space-x-2">
                      <Progress value={signal.strength} className="flex-1 h-2" />
                      <span className="text-white text-sm">{signal.strength.toFixed(0)}%</span>
                    </div>
                  </div>
                  
                  <div>
                    <p className="text-slate-400 text-sm">Вероятность успеха</p>
                    <div className="flex items-center space-x-2">
                      <Progress value={signal.probability} className="flex-1 h-2" />
                      <span className="text-white text-sm">{signal.probability.toFixed(0)}%</span>
                    </div>
                  </div>
                  
                  <div>
                    <p className="text-slate-400 text-sm">Рекомендация</p>
                    <Button 
                      size="sm" 
                      className={`w-full ${signal.type === 'CALL' ? 'bg-green-600 hover:bg-green-700' : 'bg-red-600 hover:bg-red-700'}`}
                    >
                      {signal.type === 'CALL' ? 'ПОКУПАТЬ' : 'ПРОДАВАТЬ'}
                    </Button>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-start space-x-2">
                    <AlertTriangle className="h-4 w-4 text-blue-400 mt-0.5 flex-shrink-0" />
                    <div>
                      <p className="text-slate-300 text-sm font-medium">{signal.reason}</p>
                      {signal.technicalBasis && (
                        <p className="text-slate-500 text-xs mt-1">
                          Техническая основа: {signal.technicalBasis}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Session-Based Trading Tips */}
      <Card className="p-6 bg-slate-800/50 border-slate-700">
        <h3 className="text-lg font-semibold text-white mb-4">
          Рекомендации на основе анализа сессии
        </h3>
        
        <div className="space-y-3">
          <div className="flex items-start space-x-3">
            <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
            <p className="text-slate-300 text-sm">
              Сигналы формируются на основе {candles.length} реальных свечей из вашей сессии
            </p>
          </div>
          <div className="flex items-start space-x-3">
            <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
            <p className="text-slate-300 text-sm">
              Используются множественные технические индикаторы и нейросеть для анализа
            </p>
          </div>
          <div className="flex items-start space-x-3">
            <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
            <p className="text-slate-300 text-sm">
              Чем больше данных в сессии, тем точнее прогнозы и сигналы
            </p>
          </div>
          <div className="flex items-start space-x-3">
            <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
            <p className="text-slate-300 text-sm">
              Сигналы с вероятностью выше 75% имеют наивысший приоритет
            </p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default SessionBasedTradingSignals;