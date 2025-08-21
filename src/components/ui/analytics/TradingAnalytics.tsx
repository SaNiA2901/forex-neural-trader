import { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, TrendingDown, Target, Brain, Zap, Activity } from 'lucide-react';
import { CandleData } from '@/types/session';

interface TradingAnalyticsProps {
  candles: CandleData[];
}

interface AnalyticsData {
  totalPredictions: number;
  accuratePredictions: number;
  accuracy: number;
  profitability: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
}

interface PredictionMetrics {
  direction: 'UP' | 'DOWN';
  count: number;
  accuracy: number;
  avgConfidence: number;
}

const TradingAnalytics = ({ candles }: TradingAnalyticsProps) => {
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1h' | '4h' | '1d'>('1h');

  const analytics = useMemo((): AnalyticsData => {
    if (candles.length < 2) {
      return {
        totalPredictions: 0,
        accuratePredictions: 0,
        accuracy: 0,
        profitability: 0,
        sharpeRatio: 0,
        maxDrawdown: 0,
        winRate: 0,
        avgWin: 0,
        avgLoss: 0,
        profitFactor: 0
      };
    }

    const predictions = candles.filter(c => c.prediction_direction);
    let accurate = 0;
    let wins = 0;
    let losses = 0;
    let totalWin = 0;
    let totalLoss = 0;
    let equity = 1000; // Начальный капитал
    const equityCurve = [equity];
    let maxEquity = equity;
    let maxDrawdown = 0;

    for (let i = 0; i < predictions.length - 1; i++) {
      const current = predictions[i];
      const next = candles.find(c => c.candle_index === current.candle_index + 1);
      
      if (next) {
        const actualDirection = next.close > current.close ? 'UP' : 'DOWN';
        const correct = current.prediction_direction === actualDirection;
        
        if (correct) {
          accurate++;
          wins++;
          const profit = equity * 0.8; // 80% прибыль при правильном прогнозе
          totalWin += profit;
          equity += profit;
        } else {
          losses++;
          const loss = equity * 0.1; // 10% потеря при неправильном прогнозе
          totalLoss += loss;
          equity -= loss;
        }
        
        equityCurve.push(equity);
        maxEquity = Math.max(maxEquity, equity);
        const currentDrawdown = (maxEquity - equity) / maxEquity;
        maxDrawdown = Math.max(maxDrawdown, currentDrawdown);
      }
    }

    const accuracy = predictions.length > 0 ? (accurate / predictions.length) * 100 : 0;
    const profitability = ((equity - 1000) / 1000) * 100;
    const returns = equityCurve.slice(1).map((val, i) => Math.log(val / equityCurve[i]));
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const volatility = Math.sqrt(returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length);
    const sharpeRatio = volatility > 0 ? (avgReturn / volatility) * Math.sqrt(252) : 0;

    return {
      totalPredictions: predictions.length,
      accuratePredictions: accurate,
      accuracy,
      profitability,
      sharpeRatio,
      maxDrawdown: maxDrawdown * 100,
      winRate: wins + losses > 0 ? (wins / (wins + losses)) * 100 : 0,
      avgWin: wins > 0 ? totalWin / wins : 0,
      avgLoss: losses > 0 ? totalLoss / losses : 0,
      profitFactor: totalLoss > 0 ? totalWin / totalLoss : 0
    };
  }, [candles]);

  const predictionMetrics = useMemo((): PredictionMetrics[] => {
    const upPredictions = candles.filter(c => c.prediction_direction === 'UP');
    const downPredictions = candles.filter(c => c.prediction_direction === 'DOWN');

    const calculateAccuracy = (predictions: CandleData[]): number => {
      let accurate = 0;
      for (const pred of predictions) {
        const next = candles.find(c => c.candle_index === pred.candle_index + 1);
        if (next) {
          const actualDirection = next.close > pred.close ? 'UP' : 'DOWN';
          if (pred.prediction_direction === actualDirection) accurate++;
        }
      }
      return predictions.length > 0 ? (accurate / predictions.length) * 100 : 0;
    };

    return [
      {
        direction: 'UP',
        count: upPredictions.length,
        accuracy: calculateAccuracy(upPredictions),
        avgConfidence: upPredictions.length > 0 
          ? upPredictions.reduce((sum, p) => sum + (p.prediction_confidence || 0), 0) / upPredictions.length 
          : 0
      },
      {
        direction: 'DOWN',
        count: downPredictions.length,
        accuracy: calculateAccuracy(downPredictions),
        avgConfidence: downPredictions.length > 0 
          ? downPredictions.reduce((sum, p) => sum + (p.prediction_confidence || 0), 0) / downPredictions.length 
          : 0
      }
    ];
  }, [candles]);

  const chartData = useMemo(() => {
    return candles.slice(-50).map((candle, index) => ({
      index: index + 1,
      price: candle.close,
      prediction: candle.prediction_probability || 50,
      confidence: candle.prediction_confidence || 0,
      volume: candle.volume
    }));
  }, [candles]);

  const pieData = [
    { name: 'Точные прогнозы', value: analytics.accuratePredictions, color: '#22c55e' },
    { name: 'Неточные прогнозы', value: analytics.totalPredictions - analytics.accuratePredictions, color: '#ef4444' }
  ];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Основные метрики */}
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Точность модели</p>
                <p className="text-2xl font-bold">{analytics.accuracy.toFixed(1)}%</p>
              </div>
              <Target className="h-8 w-8 text-primary" />
            </div>
            <Progress value={analytics.accuracy} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Прибыльность</p>
                <p className={`text-2xl font-bold ${analytics.profitability >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  {analytics.profitability >= 0 ? '+' : ''}{analytics.profitability.toFixed(1)}%
                </p>
              </div>
              {analytics.profitability >= 0 ? 
                <TrendingUp className="h-8 w-8 text-green-500" /> : 
                <TrendingDown className="h-8 w-8 text-red-500" />
              }
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Коэф. Шарпа</p>
                <p className="text-2xl font-bold">{analytics.sharpeRatio.toFixed(2)}</p>
              </div>
              <Brain className="h-8 w-8 text-blue-500" />
            </div>
            <Badge variant={analytics.sharpeRatio > 1 ? 'default' : 'secondary'} className="mt-2">
              {analytics.sharpeRatio > 1 ? 'Отличный' : analytics.sharpeRatio > 0.5 ? 'Хороший' : 'Слабый'}
            </Badge>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Макс. просадка</p>
                <p className="text-2xl font-bold text-red-500">{analytics.maxDrawdown.toFixed(1)}%</p>
              </div>
              <Activity className="h-8 w-8 text-red-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="performance" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="performance">Производительность</TabsTrigger>
          <TabsTrigger value="predictions">Прогнозы</TabsTrigger>
          <TabsTrigger value="patterns">Паттерны</TabsTrigger>
          <TabsTrigger value="risk">Риск-менеджмент</TabsTrigger>
        </TabsList>

        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  График цены и прогнозов
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="index" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="price" stroke="#8884d8" strokeWidth={2} />
                    <Line type="monotone" dataKey="prediction" stroke="#82ca9d" strokeDasharray="5 5" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Распределение точности</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      dataKey="value"
                    >
                      {pieData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="predictions" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {predictionMetrics.map((metric) => (
              <Card key={metric.direction}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    {metric.direction === 'UP' ? 
                      <TrendingUp className="h-5 w-5 text-green-500" /> : 
                      <TrendingDown className="h-5 w-5 text-red-500" />
                    }
                    Прогнозы {metric.direction === 'UP' ? 'CALL' : 'PUT'}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Количество:</span>
                      <span className="font-medium">{metric.count}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Точность:</span>
                      <Badge variant={metric.accuracy > 60 ? 'default' : 'secondary'}>
                        {metric.accuracy.toFixed(1)}%
                      </Badge>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm text-muted-foreground">Ср. уверенность:</span>
                      <span className="font-medium">{metric.avgConfidence.toFixed(1)}%</span>
                    </div>
                  </div>
                  <Progress value={metric.accuracy} />
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="patterns" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Объемы торгов</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData.slice(-20)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="volume" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="risk" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Win Rate</p>
                  <p className="text-2xl font-bold text-green-500">{analytics.winRate.toFixed(1)}%</p>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Profit Factor</p>
                  <p className="text-2xl font-bold">{analytics.profitFactor.toFixed(2)}</p>
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="p-4">
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Ср. прибыль/убыток</p>
                  <p className="text-sm">
                    <span className="text-green-500">+{analytics.avgWin.toFixed(0)}</span>
                    {' / '}
                    <span className="text-red-500">-{analytics.avgLoss.toFixed(0)}</span>
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default TradingAnalytics;