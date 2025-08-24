import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Play, Square, Download, Upload, TrendingUp, Target, Award } from 'lucide-react';
import { advancedMLTrainingService, TrainingExperiment, ModelConfig } from '@/services/ml/AdvancedMLTrainingService';
import { experimentTracker, ExperimentComparison } from '@/services/ml/ExperimentTracker';
import { useTradingStore } from '@/store/TradingStore';
import { useToast } from '@/hooks/use-toast';

export const MLTrainingDashboard: React.FC = () => {
  const { toast } = useToast();
  const { state } = useTradingStore();
  const currentSession = state.currentSession;
  const candles = state.candles;
  
  const [experiments, setExperiments] = useState<TrainingExperiment[]>([]);
  const [selectedExperiments, setSelectedExperiments] = useState<string[]>([]);
  const [comparison, setComparison] = useState<ExperimentComparison | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  
  // Training configuration
  const [config, setConfig] = useState<ModelConfig>({
    modelType: 'transformer',
    lookbackPeriod: 50,
    features: ['ohlcv', 'technical', 'patterns', 'volume', 'time'],
    trainingRatio: 0.7,
    validationRatio: 0.15,
    epochs: 100,
    batchSize: 32,
    learningRate: 0.001
  });

  useEffect(() => {
    advancedMLTrainingService.initialize();
    loadExperiments();
  }, []);

  const loadExperiments = () => {
    const allExperiments = advancedMLTrainingService.getAllExperiments();
    setExperiments(allExperiments);
  };

  const handleStartTraining = async () => {
    if (!currentSession || candles.length < 100) {
      toast({
        title: "Insufficient Data",
        description: "Need active session with at least 100 candles",
        variant: "destructive"
      });
      return;
    }

    setIsTraining(true);
    try {
      const experimentName = `Training_${new Date().toISOString().slice(0, 19)}`;
      const experimentId = await advancedMLTrainingService.startExperiment(
        experimentName,
        config,
        candles
      );

      toast({
        title: "Training Started",
        description: `Experiment ${experimentId} is now running`,
      });

      // Poll for updates
      const pollInterval = setInterval(() => {
        const experiment = advancedMLTrainingService.getExperiment(experimentId);
        if (experiment && experiment.status !== 'running') {
          clearInterval(pollInterval);
          setIsTraining(false);
          loadExperiments();
          
          if (experiment.status === 'completed') {
            toast({
              title: "Training Completed",
              description: `Accuracy: ${(experiment.metrics.accuracy * 100).toFixed(2)}%, Sharpe: ${experiment.metrics.sharpeRatio.toFixed(3)}`,
            });
          }
        }
      }, 2000);

      // Auto-cleanup after 2 minutes
      setTimeout(() => {
        clearInterval(pollInterval);
        setIsTraining(false);
      }, 120000);

    } catch (error) {
      setIsTraining(false);
      toast({
        title: "Training Failed",
        description: error instanceof Error ? error.message : "Unknown error",
        variant: "destructive"
      });
    }
  };

  const handleCompareExperiments = () => {
    if (selectedExperiments.length < 2) {
      toast({
        title: "Select Experiments",
        description: "Select at least 2 experiments to compare",
        variant: "destructive"
      });
      return;
    }

    const comparisonResult = experimentTracker.compareExperiments(selectedExperiments);
    setComparison(comparisonResult);
  };

  const getStatusColor = (status: TrainingExperiment['status']) => {
    switch (status) {
      case 'running': return 'bg-blue-500';
      case 'completed': return 'bg-green-500';
      case 'failed': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const formatDuration = (start: Date, end?: Date) => {
    const endTime = end || new Date();
    const duration = endTime.getTime() - start.getTime();
    const minutes = Math.floor(duration / 60000);
    const seconds = Math.floor((duration % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  };

  const performanceData = experiments
    .filter(exp => exp.status === 'completed')
    .map(exp => ({
      name: exp.name.slice(-8),
      accuracy: exp.metrics.accuracy * 100,
      sharpeRatio: exp.metrics.sharpeRatio,
      winRate: exp.metrics.winRate * 100,
      f1Score: exp.metrics.f1Score * 100
    }));

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="w-5 h-5" />
            ML Training Dashboard
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="training" className="w-full">
            <TabsList className="grid grid-cols-4 w-full">
              <TabsTrigger value="training">Training</TabsTrigger>
              <TabsTrigger value="experiments">Experiments</TabsTrigger>
              <TabsTrigger value="comparison">Comparison</TabsTrigger>
              <TabsTrigger value="performance">Performance</TabsTrigger>
            </TabsList>

            <TabsContent value="training" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Training Configuration</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <Label>Model Type</Label>
                      <Select
                        value={config.modelType}
                        onValueChange={(value: any) => setConfig(prev => ({ ...prev, modelType: value }))}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="transformer">Transformer</SelectItem>
                          <SelectItem value="regression">Regression</SelectItem>
                          <SelectItem value="ensemble">Ensemble</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label>Lookback Period</Label>
                      <Input
                        type="number"
                        value={config.lookbackPeriod}
                        onChange={(e) => setConfig(prev => ({ 
                          ...prev, 
                          lookbackPeriod: parseInt(e.target.value) || 50 
                        }))}
                      />
                    </div>

                    <div>
                      <Label>Training Ratio</Label>
                      <Input
                        type="number"
                        step="0.1"
                        min="0.5"
                        max="0.9"
                        value={config.trainingRatio}
                        onChange={(e) => setConfig(prev => ({ 
                          ...prev, 
                          trainingRatio: parseFloat(e.target.value) || 0.7 
                        }))}
                      />
                    </div>

                    <div>
                      <Label>Learning Rate</Label>
                      <Input
                        type="number"
                        step="0.0001"
                        value={config.learningRate}
                        onChange={(e) => setConfig(prev => ({ 
                          ...prev, 
                          learningRate: parseFloat(e.target.value) || 0.001 
                        }))}
                      />
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Training Status</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {currentSession ? (
                      <div>
                        <p className="text-sm text-muted-foreground">
                          Session: {currentSession.id}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Candles: {candles.length}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Features: {config.features.join(', ')}
                        </p>
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">
                        No active session
                      </p>
                    )}

                    {isTraining && (
                      <div className="space-y-2">
                        <Progress value={undefined} className="w-full" />
                        <p className="text-sm text-muted-foreground">
                          Training in progress...
                        </p>
                      </div>
                    )}

                    <Button
                      onClick={handleStartTraining}
                      disabled={!currentSession || candles.length < 100 || isTraining}
                      className="w-full"
                    >
                      {isTraining ? (
                        <>
                          <Square className="w-4 h-4 mr-2" />
                          Training...
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4 mr-2" />
                          Start Training
                        </>
                      )}
                    </Button>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="experiments" className="space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="text-lg font-semibold">Experiment History</h3>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleCompareExperiments}
                    disabled={selectedExperiments.length < 2}
                  >
                    Compare Selected
                  </Button>
                </div>
              </div>

              <div className="grid gap-4">
                {experiments.map((experiment) => (
                  <Card key={experiment.id} className="cursor-pointer hover:bg-muted/50">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <input
                            type="checkbox"
                            checked={selectedExperiments.includes(experiment.id)}
                            onChange={(e) => {
                              if (e.target.checked) {
                                setSelectedExperiments(prev => [...prev, experiment.id]);
                              } else {
                                setSelectedExperiments(prev => prev.filter(id => id !== experiment.id));
                              }
                            }}
                            className="rounded"
                          />
                          <div>
                            <h4 className="font-medium">{experiment.name}</h4>
                            <p className="text-sm text-muted-foreground">
                              {experiment.model} • {formatDuration(experiment.startTime, experiment.endTime)}
                            </p>
                          </div>
                        </div>

                        <div className="flex items-center gap-4">
                          <Badge className={getStatusColor(experiment.status)}>
                            {experiment.status}
                          </Badge>
                          
                          {experiment.status === 'completed' && (
                            <div className="text-right text-sm">
                              <div>Acc: {(experiment.metrics.accuracy * 100).toFixed(1)}%</div>
                              <div>Sharpe: {experiment.metrics.sharpeRatio.toFixed(3)}</div>
                            </div>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="comparison" className="space-y-4">
              {comparison ? (
                <div className="space-y-6">
                  <Card>
                    <CardHeader>
                      <CardTitle>Best Performers by Metric</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {Object.entries(comparison.bestByMetric).map(([metric, experiment]) => (
                          <div key={metric} className="text-center p-3 border rounded">
                            <div className="font-medium text-sm text-muted-foreground">
                              {metric.charAt(0).toUpperCase() + metric.slice(1)}
                            </div>
                            <div className="font-semibold">
                              {experiment.name.slice(-8)}
                            </div>
                            <div className="text-sm">
                              {(experiment.metrics[metric as keyof typeof experiment.metrics] as number * 
                                (metric.includes('Rate') || metric === 'accuracy' ? 100 : 1)).toFixed(metric === 'sharpeRatio' ? 3 : 2)}
                              {metric.includes('Rate') || metric === 'accuracy' ? '%' : ''}
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Overall Rankings</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        {comparison.rankings.map((ranking) => (
                          <div key={ranking.experiment.id} className="flex items-center justify-between p-3 border rounded">
                            <div className="flex items-center gap-3">
                              <div className="flex items-center gap-2">
                                <Award className="w-4 h-4" />
                                <span className="font-medium">#{ranking.rank}</span>
                              </div>
                              <div>
                                <div className="font-medium">{ranking.experiment.name}</div>
                                <div className="text-sm text-muted-foreground">
                                  {ranking.experiment.model}
                                </div>
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="font-medium">
                                Score: {(ranking.score * 100).toFixed(1)}
                              </div>
                              <div className="text-sm text-muted-foreground">
                                Acc: {(ranking.experiment.metrics.accuracy * 100).toFixed(1)}%
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              ) : (
                <Card>
                  <CardContent className="text-center py-8">
                    <p className="text-muted-foreground">
                      Select experiments to compare them
                    </p>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="performance" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5" />
                    Performance Metrics
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {performanceData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={performanceData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="accuracy" fill="hsl(var(--primary))" name="Accuracy %" />
                        <Bar dataKey="winRate" fill="hsl(var(--secondary))" name="Win Rate %" />
                        <Bar dataKey="f1Score" fill="hsl(var(--accent))" name="F1 Score %" />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="text-center py-8">
                      <p className="text-muted-foreground">
                        No completed experiments yet
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {performanceData.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Sharpe Ratio Trend</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={performanceData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Line 
                          type="monotone" 
                          dataKey="sharpeRatio" 
                          stroke="hsl(var(--primary))" 
                          strokeWidth={2}
                          name="Sharpe Ratio"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};