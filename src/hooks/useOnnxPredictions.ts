import { useState, useCallback, useRef, useEffect } from 'react';
import { onnxInferenceService } from '@/services/ml/OnnxInferenceService';
import { CandleData } from '@/types/session';
import { PredictionResult } from '@/types/trading';

interface PredictionRequest {
  symbol: string;
  candles: CandleData[];
  modelName?: string;
}

interface PredictionResponse {
  symbol: string;
  prediction: PredictionResult;
  confidence: number;
  uncertainty: number;
  latency: number;
  requestId: string;
  modelVersion: string;
}

interface UseOnnxPredictionsOptions {
  batchSize?: number;
  maxConcurrency?: number;
  enableStreaming?: boolean;
  modelName?: string;
}

export const useOnnxPredictions = (options: UseOnnxPredictionsOptions = {}) => {
  const {
    batchSize = 5,
    maxConcurrency = 3,
    enableStreaming = false,
    modelName = 'binary-classifier-v1'
  } = options;

  const [predictions, setPredictions] = useState<Map<string, PredictionResponse>>(new Map());
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState(onnxInferenceService.getMetrics());

  const requestQueue = useRef<PredictionRequest[]>([]);
  const activeRequests = useRef<Set<string>>(new Set());
  const websocketRef = useRef<WebSocket | null>(null);

  // Update metrics periodically
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(onnxInferenceService.getMetrics());
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Initialize WebSocket for streaming if enabled
  useEffect(() => {
    if (enableStreaming && !websocketRef.current) {
      initializeWebSocket();
    }

    return () => {
      if (websocketRef.current) {
        websocketRef.current.close();
        websocketRef.current = null;
      }
    };
  }, [enableStreaming]);

  const initializeWebSocket = () => {
    try {
      // In a real implementation, this would connect to a WebSocket endpoint
      const ws = new WebSocket('ws://localhost:3001/predictions');
      
      ws.onopen = () => {
        console.log('WebSocket connected for streaming predictions');
      };

      ws.onmessage = (event) => {
        try {
          const response: PredictionResponse = JSON.parse(event.data);
          setPredictions(prev => new Map(prev).set(response.symbol, response));
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      ws.onclose = () => {
        console.log('WebSocket connection closed');
        websocketRef.current = null;
      };

      websocketRef.current = ws;
    } catch (error) {
      console.error('Failed to initialize WebSocket:', error);
    }
  };

  const makePrediction = useCallback(async (
    symbol: string,
    candles: CandleData[],
    modelNameOverride?: string
  ): Promise<PredictionResponse | null> => {
    if (candles.length < 20) {
      setError('Insufficient candle data for prediction');
      return null;
    }

    const useModel = modelNameOverride || modelName;

    try {
      setError(null);
      setIsLoading(true);

      // Extract features from candles
      const features = await onnxInferenceService.extractFeaturesFromCandles(candles);

      // Make prediction
      const result = await onnxInferenceService.predict(symbol, features, useModel);

      // Update predictions state
      setPredictions(prev => new Map(prev).set(symbol, result));

      return result;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Prediction failed';
      setError(errorMessage);
      console.error('Prediction error:', error);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [modelName]);

  const makeBatchPredictions = useCallback(async (
    requests: Array<{ symbol: string; candles: CandleData[] }>,
    modelNameOverride?: string
  ): Promise<PredictionResponse[]> => {
    const useModel = modelNameOverride || modelName;

    try {
      setError(null);
      setIsLoading(true);

      // Extract features for all requests
      const featuresPromises = requests.map(async (req) => ({
        symbol: req.symbol,
        features: await onnxInferenceService.extractFeaturesFromCandles(req.candles)
      }));

      const featuresRequests = await Promise.all(featuresPromises);

      // Make batch prediction
      const results = await onnxInferenceService.batchPredict(featuresRequests, useModel);

      // Update predictions state
      setPredictions(prev => {
        const newPredictions = new Map(prev);
        results.forEach(result => {
          newPredictions.set(result.symbol, result);
        });
        return newPredictions;
      });

      return results;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Batch prediction failed';
      setError(errorMessage);
      console.error('Batch prediction error:', error);
      return [];
    } finally {
      setIsLoading(false);
    }
  }, [modelName]);

  const queuePrediction = useCallback((request: PredictionRequest) => {
    requestQueue.current.push(request);
    processQueue();
  }, []);

  const processQueue = useCallback(async () => {
    if (requestQueue.current.length === 0 || activeRequests.current.size >= maxConcurrency) {
      return;
    }

    const batch = requestQueue.current.splice(0, batchSize);
    const batchId = `batch-${Date.now()}`;
    activeRequests.current.add(batchId);

    try {
      await makeBatchPredictions(batch);
    } catch (error) {
      console.error('Queue processing error:', error);
    } finally {
      activeRequests.current.delete(batchId);
      
      // Process next batch if queue has more items
      if (requestQueue.current.length > 0) {
        setTimeout(processQueue, 100);
      }
    }
  }, [makeBatchPredictions, batchSize, maxConcurrency]);

  const streamPrediction = useCallback((symbol: string, candles: CandleData[]) => {
    if (!enableStreaming || !websocketRef.current) {
      return makePrediction(symbol, candles);
    }

    try {
      const message = JSON.stringify({
        type: 'predict',
        symbol,
        features: candles.slice(-20).map(c => [c.open, c.high, c.low, c.close, c.volume]),
        modelName
      });

      websocketRef.current.send(message);
    } catch (error) {
      console.error('Streaming prediction error:', error);
      // Fallback to regular prediction
      return makePrediction(symbol, candles);
    }

    return null;
  }, [enableStreaming, modelName, makePrediction]);

  const getPrediction = useCallback((symbol: string): PredictionResponse | undefined => {
    return predictions.get(symbol);
  }, [predictions]);

  const clearPredictions = useCallback(() => {
    setPredictions(new Map());
  }, []);

  const getHealthStatus = useCallback(async () => {
    try {
      return await onnxInferenceService.healthCheck();
    } catch (error) {
      return {
        status: 'unhealthy' as const,
        details: { error: error instanceof Error ? error.message : 'Unknown error' }
      };
    }
  }, []);

  const getModelInfo = useCallback(() => {
    return onnxInferenceService.getModelInfo();
  }, []);

  return {
    // Core prediction methods
    makePrediction,
    makeBatchPredictions,
    queuePrediction,
    streamPrediction,

    // State
    predictions: Array.from(predictions.values()),
    isLoading,
    error,
    metrics,

    // Utilities
    getPrediction,
    clearPredictions,
    getHealthStatus,
    getModelInfo,

    // Queue status
    queueLength: requestQueue.current.length,
    activeRequestsCount: activeRequests.current.size,
    
    // WebSocket status
    isStreamingConnected: websocketRef.current?.readyState === WebSocket.OPEN
  };
};