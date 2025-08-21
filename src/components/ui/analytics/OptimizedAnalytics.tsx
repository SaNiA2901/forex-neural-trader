import { memo, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { TrendingUp, Activity, BarChart3 } from 'lucide-react';
import { useAnalyticsData } from '@/hooks/analytics/useAnalyticsData';

interface OptimizedAnalyticsProps {
  pair: string;
  timeframe: string;
}

/**
 * Optimized Analytics component with memoization and performance improvements
 */
const OptimizedAnalytics = memo(({ pair, timeframe }: OptimizedAnalyticsProps) => {
  const { analyticsData, isDataReady } = useAnalyticsData(pair, timeframe);

  // Memoized chart configurations
  const chartConfig = useMemo(() => ({
    height: 300,
    margin: { top: 5, right: 30, left: 20, bottom: 5 }
  }), []);

  if (!isDataReady) {
    return (
      <Card className="p-8 bg-card/30 border-border/50 text-center backdrop-blur-sm">
        <Activity className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
        <div className="text-muted-foreground">
          Загрузка аналитических данных...
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">Расширенная Аналитика</h2>
          <p className="text-sm text-muted-foreground">
            {pair} • {timeframe} • Обновлено {new Date().toLocaleTimeString()}
          </p>
        </div>
        <Badge variant="outline" className="bg-primary/10">
          <TrendingUp className="h-4 w-4 mr-1" />
          Активно
        </Badge>
      </div>

      <Tabs defaultValue="market-depth" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="market-depth">Глубина рынка</TabsTrigger>
          <TabsTrigger value="order-flow">Поток заявок</TabsTrigger>
          <TabsTrigger value="sentiment">Настроения</TabsTrigger>
          <TabsTrigger value="volatility">Волатильность</TabsTrigger>
        </TabsList>

        <TabsContent value="market-depth">
          <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 className="h-5 w-5 text-primary" />
              <h3 className="text-lg font-semibold">Глубина рынка</h3>
            </div>
            <ResponsiveContainer width="100%" height={chartConfig.height}>
              <BarChart data={analyticsData.marketDepth} margin={chartConfig.margin}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis 
                  dataKey="price" 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                />
                <YAxis 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'hsl(var(--popover))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '6px'
                  }}
                />
                <Bar dataKey="bids" fill="hsl(var(--chart-1))" name="Покупки" />
                <Bar dataKey="asks" fill="hsl(var(--chart-2))" name="Продажи" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </TabsContent>

        <TabsContent value="order-flow">
          <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-4">
              <Activity className="h-5 w-5 text-primary" />
              <h3 className="text-lg font-semibold">Поток заявок (24ч)</h3>
            </div>
            <ResponsiveContainer width="100%" height={chartConfig.height}>
              <LineChart data={analyticsData.orderFlow} margin={chartConfig.margin}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis 
                  dataKey="hour" 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                />
                <YAxis 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'hsl(var(--popover))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '6px'
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="buyVolume" 
                  stroke="hsl(var(--chart-1))" 
                  strokeWidth={2}
                  name="Объем покупок"
                />
                <Line 
                  type="monotone" 
                  dataKey="sellVolume" 
                  stroke="hsl(var(--chart-2))" 
                  strokeWidth={2}
                  name="Объем продаж"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </TabsContent>

        <TabsContent value="sentiment">
          <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="h-5 w-5 text-primary" />
              <h3 className="text-lg font-semibold">Настроения рынка</h3>
            </div>
            <ResponsiveContainer width="100%" height={chartConfig.height}>
              <BarChart data={analyticsData.sentimentData} margin={chartConfig.margin}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis 
                  dataKey="date" 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                />
                <YAxis 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'hsl(var(--popover))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '6px'
                  }}
                />
                <Bar dataKey="bullish" stackId="a" fill="hsl(var(--chart-1))" name="Бычьи" />
                <Bar dataKey="bearish" stackId="a" fill="hsl(var(--chart-2))" name="Медвежьи" />
                <Bar dataKey="neutral" stackId="a" fill="hsl(var(--chart-3))" name="Нейтральные" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </TabsContent>

        <TabsContent value="volatility">
          <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-4">
              <Activity className="h-5 w-5 text-primary" />
              <h3 className="text-lg font-semibold">Анализ волатильности</h3>
            </div>
            <ResponsiveContainer width="100%" height={chartConfig.height}>
              <LineChart data={analyticsData.volatilityData} margin={chartConfig.margin}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis 
                  dataKey="period" 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                />
                <YAxis 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'hsl(var(--popover))',
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '6px'
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="impliedVol" 
                  stroke="hsl(var(--chart-1))" 
                  strokeWidth={2}
                  name="Подразумеваемая волатильность"
                />
                <Line 
                  type="monotone" 
                  dataKey="realizedVol" 
                  stroke="hsl(var(--chart-2))" 
                  strokeWidth={2}
                  name="Реализованная волатильность"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
});

OptimizedAnalytics.displayName = 'OptimizedAnalytics';

export default OptimizedAnalytics;