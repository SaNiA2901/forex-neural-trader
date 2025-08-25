import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useOnnxPredictions } from '@/hooks/useOnnxPredictions';
import { useTradingStore } from '@/store/TradingStore';
import { modelManager } from '@/services/ml/ModelManager';
import { Brain, Activity, Clock, TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react';

interface OnnxPredictionPanelProps {
  symbol: string;
  onPredictionGenerated?: (prediction: any) => void;
}

export const OnnxPredictionPanel: React.FC<OnnxPredictionPanelProps> = ({
  symbol,
  onPredictionGenerated
}) => {
  const { state } = useTradingStore();
  const { currentSession, candles } = state;
  const [selectedModel, setSelectedModel] = useState('binary-classifier-v1');
  const [availableModels, setAvailableModels] = useState(modelManager.getAllModelConfigurations());
  const [deploymentStatus, setDeploymentStatus] = useState<{
    deploying: boolean;
    error: string | null;
    success: boolean;
  }>({
    deploying: false,
    error: null,
    success: false
  });

  const {
    makePrediction,
    makeBatchPredictions,
    predictions,
    isLoading,
    error,
    metrics,
    getPrediction,
    getHealthStatus,
    getModelInfo
  } = useOnnxPredictions({
    modelName: selectedModel,
    batchSize: 5,
    maxConcurrency: 3
  });

  const [modelHealth, setModelHealth] = useState<any>(null);

  useEffect(() => {
    // Update model health periodically
    const updateHealth = async () => {
      try {
        const health = await getHealthStatus();
        setModelHealth(health);
      } catch (error) {
        console.error('Failed to get health status:', error);
      }
    };

    updateHealth();
    const interval = setInterval(updateHealth, 5000);
    return () => clearInterval(interval);
  }, [getHealthStatus]);

  const handleDeployModel = async (modelName: string) => {
    setDeploymentStatus({ deploying: true, error: null, success: false });
    
    try {
      const result = await modelManager.deployModel(modelName);
      
      if (result.success) {
        setSelectedModel(modelName);
        setDeploymentStatus({ deploying: false, error: null, success: true });
        
        // Clear success message after 3 seconds
        setTimeout(() => {
          setDeploymentStatus(prev => ({ ...prev, success: false }));
        }, 3000);
      } else {
        setDeploymentStatus({
          deploying: false,
          error: result.error || 'Deployment failed',
          success: false
        });
      }
    } catch (error) {
      setDeploymentStatus({
        deploying: false,
        error: error instanceof Error ? error.message : 'Unknown error',
        success: false
      });
    }
  };

  const handleMakePrediction = async () => {
    if (!currentSession || candles.length < 20) {
      return;
    }

    try {
      const result = await makePrediction(symbol, candles);
      
      if (result && onPredictionGenerated) {
        onPredictionGenerated(result.prediction);
      }
    } catch (error) {
      console.error('Prediction failed:', error);
    }
  };

  const currentPrediction = getPrediction(symbol);
  const modelInfo = getModelInfo();

  const getHealthColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600';
      case 'degraded': return 'text-yellow-600';
      case 'unhealthy': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getHealthIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'degraded': return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case 'unhealthy': return <AlertTriangle className="h-4 w-4 text-red-600" />;
      default: return <Activity className="h-4 w-4 text-gray-600" />;
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                ONNX Model Predictions
              </CardTitle>
              <CardDescription>
                High-performance neural network inference for {symbol}
              </CardDescription>
            </div>
            
            {modelHealth && (
              <div className="flex items-center gap-2">
                {getHealthIcon(modelHealth.status)}
                <span className={`text-sm font-medium ${getHealthColor(modelHealth.status)}`}>
                  {modelHealth.status.toUpperCase()}
                </span>
              </div>
            )}
          </div>
        </CardHeader>
        
        <CardContent className="space-y-4">
          {/* Model Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Active Model</label>
            <div className="flex items-center gap-2">
              <Badge variant="outline">{selectedModel}</Badge>
              {modelInfo.find(m => m.name === selectedModel)?.loaded && (
                <Badge variant="default">Loaded</Badge>
              )}
            </div>
          </div>

          {/* Available Models */}
          <div className="space-y-2">
            <label className="text-sm font-medium">Available Models</label>
            <div className="grid grid-cols-1 gap-2">
              {availableModels.map((model) => (
                <div key={model.name} className="flex items-center justify-between p-2 border rounded">
                  <div className="flex flex-col">
                    <span className="text-sm font-medium">{model.name}</span>
                    <span className="text-xs text-muted-foreground">
                      v{model.version} - {model.metadata?.description}
                    </span>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleDeployModel(model.name)}
                    disabled={deploymentStatus.deploying || selectedModel === model.name}
                  >
                    {deploymentStatus.deploying ? 'Deploying...' : 'Deploy'}
                  </Button>
                </div>
              ))}
            </div>
          </div>

          {/* Deployment Status */}
          {deploymentStatus.error && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{deploymentStatus.error}</AlertDescription>
            </Alert>
          )}

          {deploymentStatus.success && (
            <Alert>
              <CheckCircle className="h-4 w-4" />
              <AlertDescription>Model deployed successfully!</AlertDescription>
            </Alert>
          )}

          {/* Prediction Controls */}
          <div className="space-y-2">
            <Button
              onClick={handleMakePrediction}
              disabled={isLoading || !currentSession || candles.length < 20}
              className="w-full"
            >
              {isLoading ? 'Generating Prediction...' : 'Generate ONNX Prediction'}
            </Button>
            
            {!currentSession && (
              <p className="text-xs text-muted-foreground text-center">
                Start a trading session to enable predictions
              </p>
            )}
            
            {currentSession && candles.length < 20 && (
              <p className="text-xs text-muted-foreground text-center">
                Need at least 20 candles for prediction
              </p>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Current Prediction */}
      {currentPrediction && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Latest Prediction</CardTitle>
            <CardDescription>
              Generated by {currentPrediction.modelVersion}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <label className="text-sm font-medium">Direction</label>
                <Badge 
                  variant={currentPrediction.prediction.direction === 'UP' ? 'default' : 'secondary'}
                  className="text-sm"
                >
                  {currentPrediction.prediction.direction}
                </Badge>
              </div>
              
              <div className="space-y-1">
                <label className="text-sm font-medium">Confidence</label>
                <div className="space-y-1">
                  <Progress value={currentPrediction.confidence * 100} />
                  <span className="text-xs text-muted-foreground">
                    {(currentPrediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium">Probability:</span>
                <span className="ml-2">{(currentPrediction.prediction.probability * 100).toFixed(1)}%</span>
              </div>
              <div>
                <span className="font-medium">Uncertainty:</span>
                <span className="ml-2">{(currentPrediction.uncertainty * 100).toFixed(1)}%</span>
              </div>
              <div>
                <span className="font-medium">Latency:</span>
                <span className="ml-2">{currentPrediction.latency.toFixed(1)}ms</span>
              </div>
              <div>
                <span className="font-medium">Request ID:</span>
                <span className="ml-2 font-mono text-xs">{currentPrediction.requestId.slice(-8)}</span>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Recommendation</label>
              <p className="text-sm p-2 bg-muted rounded">
                {currentPrediction.prediction.recommendation}
              </p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Performance Metrics */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Activity className="h-4 w-4" />
            Performance Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <TrendingUp className="h-3 w-3" />
                <span className="font-medium">Requests</span>
              </div>
              <span>{metrics.totalRequests}</span>
            </div>
            
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                <span className="font-medium">Avg Latency</span>
              </div>
              <span>{metrics.averageLatency.toFixed(1)}ms</span>
            </div>
            
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <CheckCircle className="h-3 w-3" />
                <span className="font-medium">Success Rate</span>
              </div>
              <span>{((1 - metrics.errorRate) * 100).toFixed(1)}%</span>
            </div>
            
            <div className="space-y-1">
              <div className="flex items-center gap-1">
                <Brain className="h-3 w-3" />
                <span className="font-medium">Throughput</span>
              </div>
              <span>{metrics.throughput.toFixed(1)}/s</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};