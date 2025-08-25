import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { onnxInferenceService } from '@/services/ml/OnnxInferenceService';
import { Activity, Brain, Clock, TrendingUp, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

interface ModelStatus {
  name: string;
  version: string;
  loaded: boolean;
}

interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  details: any;
}

export const InferenceDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState(onnxInferenceService.getMetrics());
  const [models, setModels] = useState<ModelStatus[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const updateData = async () => {
      try {
        const currentMetrics = onnxInferenceService.getMetrics();
        const modelInfo = onnxInferenceService.getModelInfo();
        const healthStatus = await onnxInferenceService.healthCheck();
        
        setMetrics(currentMetrics);
        setModels(modelInfo);
        setHealth(healthStatus);
      } catch (error) {
        console.error('Failed to update dashboard data:', error);
      }
    };

    updateData();
    const interval = setInterval(updateData, 2000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600';
      case 'degraded': return 'text-yellow-600';
      case 'unhealthy': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'degraded': return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case 'unhealthy': return <XCircle className="h-4 w-4 text-red-600" />;
      default: return <Activity className="h-4 w-4 text-gray-600" />;
    }
  };

  const formatLatency = (latency: number) => {
    if (latency < 1) return '<1ms';
    return `${Math.round(latency)}ms`;
  };

  const formatMemory = (bytes: number) => {
    const mb = bytes / (1024 * 1024);
    return `${Math.round(mb)}MB`;
  };

  const handleLoadTestModel = async () => {
    setIsLoading(true);
    try {
      // Load a test model configuration
      const testConfig = {
        modelPath: '/models/test-model.onnx',
        name: 'test-trading-model',
        version: '1.0.0',
        inputShape: [1, 15],
        outputShape: [1, 2],
        scalerParams: {
          mean: Array(15).fill(0),
          std: Array(15).fill(1)
        }
      };
      
      await onnxInferenceService.loadModel(testConfig);
    } catch (error) {
      console.error('Failed to load test model:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">ONNX Inference Dashboard</h2>
          <p className="text-muted-foreground">
            Monitor model performance and system health
          </p>
        </div>
        
        {health && (
          <div className="flex items-center gap-2">
            {getStatusIcon(health.status)}
            <span className={`font-semibold ${getStatusColor(health.status)}`}>
              {health.status.toUpperCase()}
            </span>
          </div>
        )}
      </div>

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="models">Models</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="health">Health</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Requests</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{metrics.totalRequests.toLocaleString()}</div>
                <p className="text-xs text-muted-foreground">
                  {metrics.successfulPredictions} successful
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Avg Latency</CardTitle>
                <Clock className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{formatLatency(metrics.averageLatency)}</div>
                <p className="text-xs text-muted-foreground">
                  Target: &lt;20ms
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {((1 - metrics.errorRate) * 100).toFixed(1)}%
                </div>
                <p className="text-xs text-muted-foreground">
                  {metrics.errorRate > 0.1 ? 'Degraded' : 'Healthy'}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Throughput</CardTitle>
                <Brain className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {metrics.throughput.toFixed(1)}/s
                </div>
                <p className="text-xs text-muted-foreground">
                  Predictions per second
                </p>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Latency Performance</CardTitle>
                <CardDescription>
                  Current latency vs target performance
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Average Latency</span>
                  <span>{formatLatency(metrics.averageLatency)}</span>
                </div>
                <Progress 
                  value={Math.min((metrics.averageLatency / 20) * 100, 100)} 
                  className="h-2"
                />
                <p className="text-xs text-muted-foreground">
                  {metrics.averageLatency < 20 ? 'Within target' : 'Above target'}
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Memory Usage</CardTitle>
                <CardDescription>
                  Current memory consumption
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Used Memory</span>
                  <span>{formatMemory(metrics.memoryUsage)}</span>
                </div>
                <Progress 
                  value={Math.min((metrics.memoryUsage / (500 * 1024 * 1024)) * 100, 100)}
                  className="h-2"
                />
                <p className="text-xs text-muted-foreground">
                  Target: &lt;500MB
                </p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="models" className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Loaded Models</h3>
            <Button 
              onClick={handleLoadTestModel}
              disabled={isLoading}
              variant="outline"
            >
              {isLoading ? 'Loading...' : 'Load Test Model'}
            </Button>
          </div>

          {models.length === 0 ? (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                No models currently loaded. Load a model to start making predictions.
              </AlertDescription>
            </Alert>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {models.map((model, index) => (
                <Card key={index}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base">{model.name}</CardTitle>
                      <Badge variant={model.loaded ? "default" : "secondary"}>
                        {model.loaded ? "Loaded" : "Not Loaded"}
                      </Badge>
                    </div>
                    <CardDescription>Version {model.version}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Status:</span>
                        <span className={model.loaded ? 'text-green-600' : 'text-red-600'}>
                          {model.loaded ? 'Ready' : 'Error'}
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
                <CardDescription>
                  Detailed performance statistics
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm">Total Requests:</span>
                    <span className="text-sm font-mono">{metrics.totalRequests}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Successful Predictions:</span>
                    <span className="text-sm font-mono">{metrics.successfulPredictions}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Average Latency:</span>
                    <span className="text-sm font-mono">{formatLatency(metrics.averageLatency)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Error Rate:</span>
                    <span className="text-sm font-mono">{(metrics.errorRate * 100).toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Throughput:</span>
                    <span className="text-sm font-mono">{metrics.throughput.toFixed(1)}/s</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Batch Size:</span>
                    <span className="text-sm font-mono">{metrics.batchSize}</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>System Resources</CardTitle>
                <CardDescription>
                  Resource utilization monitoring
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Memory Usage:</span>
                    <span>{formatMemory(metrics.memoryUsage)}</span>
                  </div>
                  <Progress 
                    value={Math.min((metrics.memoryUsage / (500 * 1024 * 1024)) * 100, 100)}
                    className="h-2"
                  />
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Model Load Time:</span>
                    <span>{formatLatency(metrics.modelLoadTime)}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="health" className="space-y-4">
          {health ? (
            <div className="space-y-4">
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    {getStatusIcon(health.status)}
                    <CardTitle className={getStatusColor(health.status)}>
                      System Status: {health.status.toUpperCase()}
                    </CardTitle>
                  </div>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Models Loaded:</span>
                      <span className="ml-2">{health.details.modelsLoaded}</span>
                    </div>
                    <div>
                      <span className="font-medium">Memory Usage:</span>
                      <span className="ml-2">{health.details.memoryUsage}MB</span>
                    </div>
                    <div>
                      <span className="font-medium">Average Latency:</span>
                      <span className="ml-2">{health.details.averageLatency}ms</span>
                    </div>
                    <div>
                      <span className="font-medium">Error Rate:</span>
                      <span className="ml-2">{health.details.errorRate}%</span>
                    </div>
                    <div>
                      <span className="font-medium">Warmup Complete:</span>
                      <span className="ml-2">{health.details.warmupComplete ? 'Yes' : 'No'}</span>
                    </div>
                    <div>
                      <span className="font-medium">Total Requests:</span>
                      <span className="ml-2">{health.details.totalRequests}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {health.status !== 'healthy' && (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    {health.status === 'degraded' 
                      ? 'System is running but performance is degraded. Check latency and error rates.'
                      : 'System is unhealthy. Check model loading and system resources.'
                    }
                  </AlertDescription>
                </Alert>
              )}
            </div>
          ) : (
            <Card>
              <CardContent className="p-6">
                <p className="text-center text-muted-foreground">Loading health status...</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};