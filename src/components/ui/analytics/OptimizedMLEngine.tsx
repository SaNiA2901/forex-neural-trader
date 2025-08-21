import { memo, useCallback, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Brain, Zap, Target, TrendingUp, Activity, Play, Pause } from 'lucide-react';
import { CandleData } from '@/types/session';
import { useAnalyticsData } from '@/hooks/analytics/useAnalyticsData';

interface OptimizedMLEngineProps {
  pair: string;
  timeframe: string;
  candles: CandleData[];
  onPredictionUpdate?: (prediction: any) => void;
}

/**
 * Optimized ML Engine component with performance improvements
 */
const OptimizedMLEngine = memo(({ 
  pair, 
  timeframe, 
  candles, 
  onPredictionUpdate 
}: OptimizedMLEngineProps) => {
  const { mlEngineData, updateMLEngineState, generateMLData } = useAnalyticsData(pair, timeframe, candles);

  // Memoized handlers to prevent unnecessary re-renders
  const handleStartTraining = useCallback(async () => {
    updateMLEngineState({ isTraining: true });
    
    // Simulate training process
    const trainingSteps = 10;
    for (let i = 0; i < trainingSteps; i++) {
      await new Promise(resolve => setTimeout(resolve, 500));
      // Update progress could be added here
    }
    
    generateMLData();
    updateMLEngineState({ isTraining: false });
  }, [updateMLEngineState, generateMLData]);

  const handleGeneratePrediction = useCallback(async () => {
    if (candles.length < 10) return;
    
    updateMLEngineState({ isGeneratingPrediction: true });
    
    // Simulate prediction generation
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    const prediction = {
      direction: Math.random() > 0.5 ? 'UP' : 'DOWN',
      confidence: Math.random() * 0.3 + 0.7,
      probability: Math.random() * 20 + 70,
      timeframe: parseInt(timeframe),
      generatedAt: new Date().toISOString()
    };
    
    updateMLEngineState({ 
      isGeneratingPrediction: false,
      ensembleResult: { ...mlEngineData.ensembleResult, ...prediction }
    });
    
    onPredictionUpdate?.(prediction);
  }, [candles.length, timeframe, updateMLEngineState, mlEngineData.ensembleResult, onPredictionUpdate]);

  // Memoized chart configurations
  const chartConfig = useMemo(() => ({
    height: 250,
    margin: { top: 5, right: 30, left: 20, bottom: 5 }
  }), []);

  // Memoized performance data
  const performanceData = useMemo(() => 
    mlEngineData.modelPerformance.slice(-20), // Show last 20 epochs
    [mlEngineData.modelPerformance]
  );

  if (candles.length < 10) {
    return (
      <Card className="p-8 bg-card/30 border-border/50 text-center backdrop-blur-sm">
        <Brain className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
        <div className="text-muted-foreground mb-2">
          Недостаточно данных для ML анализа
        </div>
        <div className="text-sm text-muted-foreground/70">
          Необходимо минимум 10 свечей для работы алгоритмов машинного обучения
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">AI Движок</h2>
          <p className="text-sm text-muted-foreground">
            Машинное обучение для {pair} • {timeframe}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={mlEngineData.isTraining ? "default" : "outline"}>
            {mlEngineData.isTraining ? 'Обучение...' : 'Готов'}
          </Badge>
          <Button
            onClick={handleStartTraining}
            disabled={mlEngineData.isTraining}
            size="sm"
            variant="outline"
          >
            {mlEngineData.isTraining ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {mlEngineData.isTraining ? 'Остановить' : 'Обучить'}
          </Button>
        </div>
      </div>

      {/* Model Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4 bg-card/50 border-border/50 backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Точность</p>
              <p className="text-2xl font-bold text-foreground">
                {(mlEngineData.modelStats.accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <Target className="h-8 w-8 text-chart-1" />
          </div>
        </Card>

        <Card className="p-4 bg-card/50 border-border/50 backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Точность</p>
              <p className="text-2xl font-bold text-foreground">
                {(mlEngineData.modelStats.precision * 100).toFixed(1)}%
              </p>
            </div>
            <Zap className="h-8 w-8 text-chart-2" />
          </div>
        </Card>

        <Card className="p-4 bg-card/50 border-border/50 backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Полнота</p>
              <p className="text-2xl font-bold text-foreground">
                {(mlEngineData.modelStats.recall * 100).toFixed(1)}%
              </p>
            </div>
            <Activity className="h-8 w-8 text-chart-3" />
          </div>
        </Card>

        <Card className="p-4 bg-card/50 border-border/50 backdrop-blur-sm">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">F1 Score</p>
              <p className="text-2xl font-bold text-foreground">
                {(mlEngineData.modelStats.f1Score * 100).toFixed(1)}%
              </p>
            </div>
            <TrendingUp className="h-8 w-8 text-chart-4" />
          </div>
        </Card>
      </div>

      {/* Ensemble Prediction */}
      {mlEngineData.ensembleResult && (
        <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" />
              <h3 className="text-lg font-semibold">Ансамблевый прогноз</h3>
            </div>
            <Button
              onClick={handleGeneratePrediction}
              disabled={mlEngineData.isGeneratingPrediction}
              size="sm"
            >
              {mlEngineData.isGeneratingPrediction ? 'Генерация...' : 'Новый прогноз'}
            </Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <div className="flex items-center gap-4 mb-4">
                <div className={`text-3xl font-bold ${
                  mlEngineData.ensembleResult.prediction === 'UP' ? 'text-green-500' : 'text-red-500'
                }`}>
                  {mlEngineData.ensembleResult.prediction}
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Уверенность</p>
                  <p className="text-xl font-semibold">
                    {(mlEngineData.ensembleResult.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
              <Progress 
                value={mlEngineData.ensembleResult.confidence * 100} 
                className="h-2"
              />
            </div>

            <div className="space-y-2">
              <p className="text-sm font-medium text-muted-foreground">Модели в ансамбле:</p>
              {mlEngineData.ensembleResult.models?.map((model: any, index: number) => (
                <div key={index} className="flex items-center justify-between text-sm">
                  <span>{model.name}</span>
                  <div className="flex items-center gap-2">
                    <span className={
                      model.prediction === 'UP' ? 'text-green-500' : 'text-red-500'
                    }>
                      {model.prediction}
                    </span>
                    <span className="text-muted-foreground">
                      {(model.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Card>
      )}

      <Tabs defaultValue="performance" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="performance">Производительность</TabsTrigger>
          <TabsTrigger value="features">Важность признаков</TabsTrigger>
        </TabsList>

        <TabsContent value="performance">
          <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-4">
              <Activity className="h-5 w-5 text-primary" />
              <h3 className="text-lg font-semibold">Обучение модели</h3>
            </div>
            <ResponsiveContainer width="100%" height={chartConfig.height}>
              <LineChart data={performanceData} margin={chartConfig.margin}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis 
                  dataKey="epoch" 
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
                  dataKey="accuracy" 
                  stroke="hsl(var(--chart-1))" 
                  strokeWidth={2}
                  name="Точность"
                />
                <Line 
                  type="monotone" 
                  dataKey="valAccuracy" 
                  stroke="hsl(var(--chart-2))" 
                  strokeWidth={2}
                  name="Валидационная точность"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </TabsContent>

        <TabsContent value="features">
          <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
            <div className="flex items-center gap-2 mb-4">
              <Target className="h-5 w-5 text-primary" />
              <h3 className="text-lg font-semibold">Важность признаков</h3>
            </div>
            <ResponsiveContainer width="100%" height={chartConfig.height}>
              <BarChart data={mlEngineData.featureImportance} margin={chartConfig.margin}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis 
                  dataKey="feature" 
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                  angle={-45}
                  textAnchor="end"
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
                <Bar 
                  dataKey="importance" 
                  fill="hsl(var(--chart-1))" 
                  name="Важность"
                />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
});

OptimizedMLEngine.displayName = 'OptimizedMLEngine';

export default OptimizedMLEngine;