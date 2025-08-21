
import { Card } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts";
import { Badge } from "@/components/ui/badge";
import { Database, AlertCircle, TrendingUp, TrendingDown } from "lucide-react";
import { useTradingStore } from '@/store/TradingStore';
import { useMemo } from 'react';

interface PriceChartProps {
  pair: string;
  timeframe: string;
}

const PriceChart = ({ pair, timeframe }: PriceChartProps) => {
  const { state } = useTradingStore();
  
  // Подготавливаем данные для графика
  const chartData = useMemo(() => {
    if (!state.candles || state.candles.length === 0) return [];
    
    return state.candles
      .sort((a, b) => a.candle_index - b.candle_index)
      .map((candle, index) => ({
        index: index + 1,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: candle.volume,
        name: `Свеча ${index + 1}`
      }));
  }, [state.candles]);

  // Вычисляем статистику
  const stats = useMemo(() => {
    if (chartData.length === 0) return { max: 0, min: 0, totalVolume: 0, trend: 'NEUTRAL' };
    
    const prices = chartData.flatMap(d => [d.high, d.low]);
    const max = Math.max(...prices);
    const min = Math.min(...prices);
    const totalVolume = chartData.reduce((sum, d) => sum + d.volume, 0);
    
    const firstClose = chartData[0]?.close || 0;
    const lastClose = chartData[chartData.length - 1]?.close || 0;
    const trend = lastClose > firstClose ? 'UP' : lastClose < firstClose ? 'DOWN' : 'NEUTRAL';
    
    return { max, min, totalVolume, trend };
  }, [chartData]);

  const formatPrice = (price: number) => {
    return pair.includes("JPY") ? price.toFixed(2) : price.toFixed(4);
  };

  // Если нет активной сессии
  if (!state.currentSession) {
    return (
      <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
        <div className="h-96 flex items-center justify-center">
          <div className="text-center space-y-4">
            <Database className="h-16 w-16 text-muted-foreground mx-auto opacity-50" />
            <div>
              <h4 className="text-foreground font-medium mb-2">Нет активной сессии</h4>
              <p className="text-muted-foreground text-sm max-w-md">
                Создайте или загрузите сессию в разделе "Бинарные опционы" для просмотра графика
              </p>
            </div>
          </div>
        </div>
      </Card>
    );
  }

  // Если нет данных свечей
  if (chartData.length === 0) {
    return (
      <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-6">
          <div>
            <div className="flex items-center space-x-3 mb-2">
              <h3 className="text-xl font-semibold text-foreground">{pair}</h3>
              <Badge variant="secondary">Сессия: {state.currentSession.session_name}</Badge>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-muted-foreground">
                Данные отображаются после ввода свечей
              </span>
            </div>
          </div>
          
          <div className="text-right">
            <p className="text-muted-foreground text-sm">Временной интервал</p>
            <p className="text-foreground font-medium">{timeframe}</p>
          </div>
        </div>

        <div className="h-96 flex items-center justify-center bg-muted/20 rounded-lg">
          <div className="text-center space-y-4">
            <Database className="h-16 w-16 text-muted-foreground mx-auto opacity-50" />
            <div>
              <h4 className="text-foreground font-medium mb-2">Нет данных для отображения</h4>
              <p className="text-muted-foreground text-sm max-w-md">
                Добавьте OHLC данные в текущую сессию для просмотра графика
              </p>
            </div>
            <div className="bg-primary/10 border border-primary/30 rounded-lg p-3 max-w-sm mx-auto">
              <div className="flex items-center space-x-2 mb-1">
                <AlertCircle className="h-4 w-4 text-primary" />
                <span className="text-primary text-sm font-medium">Информация</span>
              </div>
              <p className="text-primary/80 text-xs">
                График обновляется автоматически при добавлении свечей
              </p>
            </div>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center space-x-3 mb-2">
            <h3 className="text-xl font-semibold text-foreground">{pair}</h3>
            <Badge variant="default">Сессия: {state.currentSession.session_name}</Badge>
            <Badge 
              variant="outline" 
              className={`
                ${stats.trend === 'UP' ? 'border-green-500/50 text-green-500' : 
                  stats.trend === 'DOWN' ? 'border-red-500/50 text-red-500' : 
                  'border-muted-foreground/50 text-muted-foreground'}
              `}
            >
              {stats.trend === 'UP' && <TrendingUp className="h-3 w-3 mr-1" />}
              {stats.trend === 'DOWN' && <TrendingDown className="h-3 w-3 mr-1" />}
              {stats.trend}
            </Badge>
          </div>
          <div className="flex items-center space-x-4">
            <span className="text-muted-foreground text-sm">
              {chartData.length} свечей • Последняя цена: {formatPrice(chartData[chartData.length - 1]?.close || 0)}
            </span>
          </div>
        </div>
        
        <div className="text-right">
          <p className="text-muted-foreground text-sm">Временной интервал</p>
          <p className="text-foreground font-medium">{timeframe}</p>
        </div>
      </div>

      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.8} />
                <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis 
              dataKey="index" 
              stroke="hsl(var(--muted-foreground))" 
              fontSize={12}
              label={{ value: 'Номер свечи', position: 'insideBottom', offset: -5 }}
            />
            <YAxis 
              stroke="hsl(var(--muted-foreground))" 
              fontSize={12}
              domain={['dataMin - 0.001', 'dataMax + 0.001']}
              tickFormatter={formatPrice}
              label={{ value: 'Цена', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: 'hsl(var(--popover))',
                border: '1px solid hsl(var(--border))',
                borderRadius: '6px',
                color: 'hsl(var(--popover-foreground))'
              }}
              formatter={(value: any, name: string) => [
                typeof value === 'number' ? formatPrice(value) : value, 
                name === 'high' ? 'Максимум' :
                name === 'low' ? 'Минимум' :
                name === 'open' ? 'Открытие' :
                name === 'close' ? 'Закрытие' : name
              ]}
              labelFormatter={(label) => `Свеча ${label}`}
            />
            <Area 
              type="monotone" 
              dataKey="close" 
              stroke="hsl(var(--primary))" 
              strokeWidth={2}
              fillOpacity={1} 
              fill="url(#priceGradient)"
              name="close"
            />
            <Line 
              type="monotone" 
              dataKey="high" 
              stroke="hsl(var(--chart-1))" 
              strokeWidth={1}
              dot={false}
              name="high"
            />
            <Line 
              type="monotone" 
              dataKey="low" 
              stroke="hsl(var(--chart-2))" 
              strokeWidth={1}
              dot={false}
              name="low"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4 pt-4 border-t border-border">
        <div className="text-center">
          <p className="text-muted-foreground text-sm">Максимум</p>
          <p className="text-foreground font-medium">{formatPrice(stats.max)}</p>
        </div>
        <div className="text-center">
          <p className="text-muted-foreground text-sm">Минимум</p>
          <p className="text-foreground font-medium">{formatPrice(stats.min)}</p>
        </div>
        <div className="text-center">
          <p className="text-muted-foreground text-sm">Общий объем</p>
          <p className="text-foreground font-medium">{stats.totalVolume.toLocaleString()}</p>
        </div>
      </div>
    </Card>
  );
};

export default PriceChart;
