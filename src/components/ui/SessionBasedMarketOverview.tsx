import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useNewApplicationState } from "@/hooks/useNewApplicationState";
import { sessionBasedDataService } from "@/services/data/SessionBasedDataService";
import { TrendingUp, TrendingDown, Activity, BarChart3, AlertTriangle } from "lucide-react";
import { useEffect, useState } from "react";

interface SessionBasedMarketOverviewProps {
  pair: string;
  timeframe: string;
}

interface MarketMetrics {
  priceChangePercent: number;
  volatilityIndex: number;
  momentumScore: number;
  supportLevel: number;
  resistanceLevel: number;
  trendStrength: number;
  trendDirection: 'UP' | 'DOWN' | 'SIDEWAYS';
  volumeAnalysis: {
    currentVolume: number;
    averageVolume: number;
    volumeRatio: number;
    volumeTrend: 'INCREASING' | 'DECREASING' | 'STABLE';
  };
}

const SessionBasedMarketOverview = ({ pair, timeframe }: SessionBasedMarketOverviewProps) => {
  const { currentSession, candles } = useNewApplicationState();
  const [metrics, setMetrics] = useState<MarketMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const updateMetrics = () => {
      if (!currentSession || candles.length < 5) {
        setMetrics(null);
        setIsLoading(false);
        return;
      }

      try {
        // Обновляем данные в сервисе
        sessionBasedDataService.updateSessionData(currentSession.id, candles);

        // Получаем метрики
        const realTimeMetrics = sessionBasedDataService.getRealTimeMetrics(currentSession.id);
        const trendDirection = sessionBasedDataService.getTrendDirection(currentSession.id);
        const volumeAnalysis = sessionBasedDataService.getVolumeAnalysis(currentSession.id);

        setMetrics({
          ...realTimeMetrics,
          trendDirection,
          volumeAnalysis
        });
      } catch (error) {
        console.error('Error calculating market metrics:', error);
        setMetrics(null);
      } finally {
        setIsLoading(false);
      }
    };

    updateMetrics();
    
    // Обновляем метрики каждые 30 секунд
    const interval = setInterval(updateMetrics, 30000);
    
    return () => clearInterval(interval);
  }, [currentSession, candles]);

  if (!currentSession) {
    return (
      <Alert className="border-orange-600 bg-orange-600/20">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription className="text-orange-200">
          Для отображения обзора рынка необходимо создать активную сессию.
        </AlertDescription>
      </Alert>
    );
  }

  if (candles.length < 5) {
    return (
      <Alert className="border-blue-600 bg-blue-600/20">
        <Activity className="h-4 w-4" />
        <AlertDescription className="text-blue-200">
          Недостаточно данных для анализа рынка. Добавьте минимум 5 свечей.
          Текущее количество: {candles.length}
        </AlertDescription>
      </Alert>
    );
  }

  if (isLoading || !metrics) {
    return (
      <Card className="p-6 bg-slate-800/50 border-slate-700">
        <div className="flex items-center justify-center h-48">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
          <span className="ml-3 text-slate-400">Анализ рыночных данных...</span>
        </div>
      </Card>
    );
  }

  const getTrendColor = (direction: string) => {
    switch (direction) {
      case 'UP': return 'text-green-400';
      case 'DOWN': return 'text-red-400';
      default: return 'text-blue-400';
    }
  };

  const getTrendIcon = (direction: string) => {
    switch (direction) {
      case 'UP': return <TrendingUp className="h-5 w-5" />;
      case 'DOWN': return <TrendingDown className="h-5 w-5" />;
      default: return <Activity className="h-5 w-5" />;
    }
  };

  const getVolumeColor = (trend: string) => {
    switch (trend) {
      case 'INCREASING': return 'text-green-400';
      case 'DECREASING': return 'text-red-400';
      default: return 'text-blue-400';
    }
  };

  const formatNumber = (value: number, decimals: number = 2) => {
    return value.toLocaleString('ru-RU', { 
      minimumFractionDigits: decimals, 
      maximumFractionDigits: decimals 
    });
  };

  return (
    <div className="space-y-6">
      {/* Основной обзор */}
      <Card className="p-6 bg-gradient-to-br from-slate-800/80 to-slate-900/80 border-slate-700">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <BarChart3 className="h-6 w-6 text-blue-400" />
            <h3 className="text-xl font-semibold text-white">Обзор рынка на основе сессии</h3>
          </div>
          <div className="flex items-center space-x-2">
            <Badge className="bg-blue-600">
              {pair} • {timeframe}
            </Badge>
            <Badge variant="outline">
              Свечей: {candles.length}
            </Badge>
          </div>
        </div>

        {/* Ключевые метрики */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <div className="flex items-center justify-center space-x-2 mb-2">
              {getTrendIcon(metrics.trendDirection)}
              <div className={`text-2xl font-bold ${getTrendColor(metrics.trendDirection)}`}>
                {metrics.priceChangePercent > 0 ? '+' : ''}{formatNumber(metrics.priceChangePercent)}%
              </div>
            </div>
            <p className="text-slate-400 text-sm">Изменение цены</p>
            <Badge className={metrics.trendDirection === 'UP' ? 'bg-green-600' : 
                            metrics.trendDirection === 'DOWN' ? 'bg-red-600' : 'bg-blue-600'}>
              {metrics.trendDirection === 'UP' ? 'Восходящий тренд' : 
               metrics.trendDirection === 'DOWN' ? 'Нисходящий тренд' : 'Боковое движение'}
            </Badge>
          </div>

          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <div className="text-2xl font-bold text-purple-400 mb-2">
              {formatNumber(metrics.volatilityIndex)}%
            </div>
            <Progress value={Math.min(100, metrics.volatilityIndex)} className="mb-2 h-2" />
            <p className="text-slate-400 text-sm">Индекс волатильности</p>
          </div>

          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <div className="text-2xl font-bold text-yellow-400 mb-2">
              {formatNumber(metrics.momentumScore)}
            </div>
            <Progress 
              value={Math.min(100, Math.abs(metrics.momentumScore))} 
              className="mb-2 h-2" 
            />
            <p className="text-slate-400 text-sm">Моментум</p>
          </div>

          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <div className="text-2xl font-bold text-orange-400 mb-2">
              {formatNumber(metrics.trendStrength)}
            </div>
            <Progress value={Math.min(100, metrics.trendStrength)} className="mb-2 h-2" />
            <p className="text-slate-400 text-sm">Сила тренда</p>
          </div>
        </div>

        {/* Дополнительная информация */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Анализ уровней */}
          <div className="bg-slate-700/30 rounded-lg p-4">
            <h4 className="text-white font-medium mb-3 flex items-center space-x-2">
              <BarChart3 className="h-4 w-4 text-blue-400" />
              <span>Ключевые уровни</span>
            </h4>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Сопротивление:</span>
                <span className="text-red-400 font-medium">
                  {formatNumber(metrics.resistanceLevel, 4)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Поддержка:</span>
                <span className="text-green-400 font-medium">
                  {formatNumber(metrics.supportLevel, 4)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Диапазон:</span>
                <span className="text-blue-400 font-medium">
                  {formatNumber(metrics.resistanceLevel - metrics.supportLevel, 4)}
                </span>
              </div>
            </div>
          </div>

          {/* Анализ объема */}
          <div className="bg-slate-700/30 rounded-lg p-4">
            <h4 className="text-white font-medium mb-3 flex items-center space-x-2">
              <Activity className="h-4 w-4 text-purple-400" />
              <span>Анализ объема</span>
            </h4>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Текущий объем:</span>
                <span className="text-white font-medium">
                  {formatNumber(metrics.volumeAnalysis.currentVolume, 0)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Средний объем:</span>
                <span className="text-slate-400 font-medium">
                  {formatNumber(metrics.volumeAnalysis.averageVolume, 0)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Отношение к среднему:</span>
                <span className="text-yellow-400 font-medium">
                  {formatNumber(metrics.volumeAnalysis.volumeRatio)}x
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-slate-400">Тренд объема:</span>
                <span className={`font-medium ${getVolumeColor(metrics.volumeAnalysis.volumeTrend)}`}>
                  {metrics.volumeAnalysis.volumeTrend === 'INCREASING' ? 'Растет' :
                   metrics.volumeAnalysis.volumeTrend === 'DECREASING' ? 'Падает' : 'Стабилен'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Рыночные условия */}
      <Card className="p-6 bg-slate-800/50 border-slate-700">
        <h4 className="text-lg font-semibold text-white mb-4">Текущие рыночные условия</h4>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-3 bg-slate-700/30 rounded-lg">
            <div className="text-sm text-slate-400 mb-1">Характер рынка</div>
            <div className="text-white font-medium">
              {metrics.volatilityIndex > 150 ? 'Высокая волатильность' :
               metrics.volatilityIndex > 100 ? 'Умеренная волатильность' : 'Низкая волатильность'}
            </div>
          </div>
          
          <div className="text-center p-3 bg-slate-700/30 rounded-lg">
            <div className="text-sm text-slate-400 mb-1">Торговая активность</div>
            <div className="text-white font-medium">
              {metrics.volumeAnalysis.volumeRatio > 1.5 ? 'Высокая активность' :
               metrics.volumeAnalysis.volumeRatio > 0.8 ? 'Нормальная активность' : 'Низкая активность'}
            </div>
          </div>
          
          <div className="text-center p-3 bg-slate-700/30 rounded-lg">
            <div className="text-sm text-slate-400 mb-1">Рекомендация</div>
            <div className="text-white font-medium">
              {metrics.trendStrength > 70 ? 'Следуй тренду' :
               metrics.volatilityIndex > 150 ? 'Будь осторожен' : 'Ожидай прорыва'}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default SessionBasedMarketOverview;