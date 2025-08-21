import React from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  Activity,
  Clock,
  Target
} from 'lucide-react';
import { useTradingStore } from '@/store/TradingStore';

export const CandleInputStats: React.FC = () => {
  const { state } = useTradingStore();
  
  if (!state.currentSession || state.candles.length === 0) {
    return null;
  }

  const candles = state.candles;
  const lastCandle = candles[candles.length - 1];
  const sessionStats = state.sessionStats;

  // Анализ трендов
  const bullishCandles = candles.filter(c => c.close > c.open).length;
  const bearishCandles = candles.filter(c => c.close < c.open).length;
  const dojiCandles = candles.filter(c => Math.abs(c.close - c.open) < 0.0001).length;

  // Волатильность
  const avgRange = candles.reduce((sum, c) => sum + (c.high - c.low), 0) / candles.length;
  const avgVolume = candles.reduce((sum, c) => sum + c.volume, 0) / candles.length;

  // Текущий тренд (последние 5 свечей)
  const recentCandles = candles.slice(-5);
  const recentTrend = recentCandles.length >= 2 ? 
    (recentCandles[recentCandles.length - 1].close > recentCandles[0].open ? 'up' : 'down') : 'neutral';

  const stats = [
    {
      label: 'Всего свечей',
      value: candles.length,
      icon: BarChart3,
      variant: 'default' as const
    },
    {
      label: 'Бычьи свечи',
      value: bullishCandles,
      percentage: Math.round((bullishCandles / candles.length) * 100),
      icon: TrendingUp,
      variant: 'default' as const
    },
    {
      label: 'Медвежьи свечи',
      value: bearishCandles,
      percentage: Math.round((bearishCandles / candles.length) * 100),
      icon: TrendingDown,
      variant: 'destructive' as const
    },
    {
      label: 'Доджи',
      value: dojiCandles,
      percentage: candles.length > 0 ? Math.round((dojiCandles / candles.length) * 100) : 0,
      icon: Activity,
      variant: 'secondary' as const
    }
  ];

  return (
    <div className="space-y-4">
      {/* Основная статистика */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {stats.map((stat, index) => (
          <Card key={index} className="p-4 bg-gradient-to-br from-background/95 to-background/80">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">{stat.label}</p>
                <div className="flex items-center gap-2">
                  <span className="text-2xl font-bold">{stat.value}</span>
                  {stat.percentage !== undefined && (
                    <Badge variant={stat.variant} className="text-xs">
                      {stat.percentage}%
                    </Badge>
                  )}
                </div>
              </div>
              <stat.icon className="h-8 w-8 text-muted-foreground/50" />
            </div>
          </Card>
        ))}
      </div>

      {/* Дополнительная аналитика */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4 bg-gradient-to-br from-background/95 to-background/80">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Activity className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Средний диапазон</p>
              <p className="text-lg font-semibold">{avgRange.toFixed(5)}</p>
            </div>
          </div>
        </Card>

        <Card className="p-4 bg-gradient-to-br from-background/95 to-background/80">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <BarChart3 className="h-5 w-5 text-primary" />
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Средний объем</p>
              <p className="text-lg font-semibold">{Math.round(avgVolume).toLocaleString()}</p>
            </div>
          </div>
        </Card>

        <Card className="p-4 bg-gradient-to-br from-background/95 to-background/80">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              {recentTrend === 'up' ? (
                <TrendingUp className="h-5 w-5 text-green-500" />
              ) : recentTrend === 'down' ? (
                <TrendingDown className="h-5 w-5 text-red-500" />
              ) : (
                <Target className="h-5 w-5 text-muted-foreground" />
              )}
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Текущий тренд</p>
              <Badge 
                variant={recentTrend === 'up' ? 'default' : recentTrend === 'down' ? 'destructive' : 'secondary'}
              >
                {recentTrend === 'up' ? 'Восходящий' : recentTrend === 'down' ? 'Нисходящий' : 'Боковой'}
              </Badge>
            </div>
          </div>
        </Card>
      </div>

      {/* Информация о последней свече */}
      {lastCandle && (
        <Card className="p-4 bg-gradient-to-br from-background/95 to-background/80">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-primary/10 rounded-lg">
                <Clock className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Последняя свеча #{lastCandle.candle_index}</p>
                <div className="flex items-center gap-4 mt-1">
                  <span className="text-sm">O: {lastCandle.open}</span>
                  <span className="text-sm">H: {lastCandle.high}</span>
                  <span className="text-sm">L: {lastCandle.low}</span>
                  <span className="text-sm">C: {lastCandle.close}</span>
                  <span className="text-sm">V: {lastCandle.volume}</span>
                </div>
              </div>
            </div>
            
            <Badge 
              variant={lastCandle.close > lastCandle.open ? 'default' : 
                      lastCandle.close < lastCandle.open ? 'destructive' : 'secondary'}
              className="flex items-center gap-1"
            >
              {lastCandle.close > lastCandle.open ? (
                <TrendingUp className="h-3 w-3" />
              ) : lastCandle.close < lastCandle.open ? (
                <TrendingDown className="h-3 w-3" />
              ) : (
                <Activity className="h-3 w-3" />
              )}
              {((lastCandle.close - lastCandle.open) / lastCandle.open * 100).toFixed(2)}%
            </Badge>
          </div>
        </Card>
      )}
    </div>
  );
};