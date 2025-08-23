/**
 * Cache Performance Monitoring Dashboard
 * Real-time monitoring of Feature Store performance
 */

import { useState, useEffect } from 'react';
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Activity, Database, TrendingUp, Clock, AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import { featureIntegration } from '@/services/cache/FeatureIntegration';

interface CacheMetrics {
  featureStore: {
    totalKeys: number;
    memoryUsage: string;
    metrics: {
      totalRequests: number;
      cacheHits: number;
      cacheMisses: number;
      hitRate: number;
      averageComputeTime: number;
      averageCacheTime: number;
      totalErrors: number;
      lastError?: string;
    };
    redisMetrics: {
      connectionStatus: string;
      totalCommands: number;
      successfulCommands: number;
      failedCommands: number;
      averageResponseTime: number;
      uptime: number;
    };
  };
  integration: {
    version: string;
    uptime: number;
  };
}

interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  components: {
    featureStore: {
      status: 'healthy' | 'degraded' | 'unhealthy';
      redis: boolean;
      cache: boolean;
    };
    indicators: boolean;
  };
}

export const CacheMonitoringDashboard = () => {
  const [metrics, setMetrics] = useState<CacheMetrics | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchMetrics = async () => {
    try {
      const [metricsData, healthData] = await Promise.all([
        featureIntegration.getPerformanceStats(),
        featureIntegration.healthCheck()
      ]);
      
      setMetrics(metricsData);
      setHealth(healthData);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to fetch cache metrics:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    
    // Update every 5 seconds
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  const formatUptime = (uptimeMs: number): string => {
    const seconds = Math.floor(uptimeMs / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `${days}d ${hours % 24}h`;
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'degraded':
        return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      case 'unhealthy':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Activity className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'default';
      case 'degraded':
        return 'secondary';
      case 'unhealthy':
        return 'destructive';
      default:
        return 'outline';
    }
  };

  if (isLoading) {
    return (
      <Card className="p-8 bg-card/30 border-border/50 text-center backdrop-blur-sm">
        <Activity className="h-12 w-12 mx-auto mb-4 text-muted-foreground animate-pulse" />
        <div className="text-muted-foreground">
          Загрузка метрик кэша...
        </div>
      </Card>
    );
  }

  if (!metrics || !health) {
    return (
      <Card className="p-8 bg-card/30 border-border/50 text-center backdrop-blur-sm">
        <XCircle className="h-12 w-12 mx-auto mb-4 text-red-500" />
        <div className="text-muted-foreground mb-2">
          Ошибка загрузки метрик кэша
        </div>
        <button 
          onClick={fetchMetrics}
          className="text-sm text-primary hover:underline"
        >
          Попробовать снова
        </button>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Health Overview */}
      <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            {getStatusIcon(health.status)}
            <h3 className="text-lg font-semibold text-foreground">Статус системы кэширования</h3>
          </div>
          <Badge variant={getStatusBadgeVariant(health.status)}>
            {health.status === 'healthy' ? 'Работает' : 
             health.status === 'degraded' ? 'Ограничено' : 'Неисправно'}
          </Badge>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-2 mb-2">
              {getStatusIcon(health.components.featureStore.status)}
              <span className="text-sm font-medium">Feature Store</span>
            </div>
            <div className="text-xs text-muted-foreground">
              Redis: {health.components.featureStore.redis ? '✓' : '✗'} | 
              Cache: {health.components.featureStore.cache ? '✓' : '✗'}
            </div>
          </div>

          <div className="text-center">
            <div className="flex items-center justify-center space-x-2 mb-2">
              {health.components.indicators ? 
                <CheckCircle className="h-4 w-4 text-green-500" /> :
                <XCircle className="h-4 w-4 text-red-500" />
              }
              <span className="text-sm font-medium">Индикаторы</span>
            </div>
            <div className="text-xs text-muted-foreground">
              {health.components.indicators ? 'Доступны' : 'Недоступны'}
            </div>
          </div>

          <div className="text-center">
            <div className="flex items-center justify-center space-x-2 mb-2">
              <Clock className="h-4 w-4 text-primary" />
              <span className="text-sm font-medium">Uptime</span>
            </div>
            <div className="text-xs text-muted-foreground">
              {formatUptime(metrics.integration.uptime)}
            </div>
          </div>
        </div>

        <div className="text-xs text-muted-foreground mt-4 text-center">
          Последнее обновление: {lastUpdate.toLocaleTimeString()}
        </div>
      </Card>

      {/* Cache Performance Metrics */}
      <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
        <div className="flex items-center space-x-3 mb-4">
          <Database className="h-6 w-6 text-primary" />
          <h3 className="text-lg font-semibold text-foreground">Производительность кэша</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-foreground">
              {metrics.featureStore.metrics.hitRate.toFixed(1)}%
            </div>
            <div className="text-sm text-muted-foreground">Hit Rate</div>
            <Progress 
              value={metrics.featureStore.metrics.hitRate} 
              className="mt-2 h-2"
            />
          </div>

          <div className="text-center">
            <div className="text-2xl font-bold text-foreground">
              {metrics.featureStore.totalKeys}
            </div>
            <div className="text-sm text-muted-foreground">Ключей в кэше</div>
          </div>

          <div className="text-center">
            <div className="text-2xl font-bold text-foreground">
              {metrics.featureStore.metrics.totalRequests}
            </div>
            <div className="text-sm text-muted-foreground">Всего запросов</div>
          </div>

          <div className="text-center">
            <div className="text-2xl font-bold text-foreground">
              {metrics.featureStore.memoryUsage}
            </div>
            <div className="text-sm text-muted-foreground">Память Redis</div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="font-medium mb-2 text-foreground">Производительность запросов</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Cache Hits:</span>
                <span className="text-green-600 font-medium">
                  {metrics.featureStore.metrics.cacheHits}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Cache Misses:</span>
                <span className="text-yellow-600 font-medium">
                  {metrics.featureStore.metrics.cacheMisses}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Среднее время кэша:</span>
                <span className="text-foreground font-medium">
                  {metrics.featureStore.metrics.averageCacheTime.toFixed(2)}ms
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Среднее время вычисления:</span>
                <span className="text-foreground font-medium">
                  {metrics.featureStore.metrics.averageComputeTime.toFixed(2)}ms
                </span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-medium mb-2 text-foreground">Redis метрики</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Статус подключения:</span>
                <Badge variant="outline" className="text-xs">
                  {metrics.featureStore.redisMetrics.connectionStatus}
                </Badge>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Команды выполнены:</span>
                <span className="text-foreground font-medium">
                  {metrics.featureStore.redisMetrics.successfulCommands}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Ошибки команд:</span>
                <span className={`font-medium ${
                  metrics.featureStore.redisMetrics.failedCommands > 0 ? 'text-red-500' : 'text-green-500'
                }`}>
                  {metrics.featureStore.redisMetrics.failedCommands}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Среднее время отклика:</span>
                <span className="text-foreground font-medium">
                  {metrics.featureStore.redisMetrics.averageResponseTime.toFixed(2)}ms
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Error Display */}
        {metrics.featureStore.metrics.lastError && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
            <div className="flex items-center space-x-2 mb-1">
              <AlertTriangle className="h-4 w-4 text-red-500" />
              <span className="text-sm font-medium text-red-700">Последняя ошибка:</span>
            </div>
            <div className="text-xs text-red-600 font-mono">
              {metrics.featureStore.metrics.lastError}
            </div>
          </div>
        )}
      </Card>

      {/* Performance Recommendations */}
      <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
        <div className="flex items-center space-x-3 mb-4">
          <TrendingUp className="h-6 w-6 text-primary" />
          <h3 className="text-lg font-semibold text-foreground">Рекомендации по оптимизации</h3>
        </div>

        <div className="space-y-3">
          {metrics.featureStore.metrics.hitRate < 70 && (
            <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-md">
              <div className="text-sm font-medium text-yellow-700">
                Низкий Hit Rate ({metrics.featureStore.metrics.hitRate.toFixed(1)}%)
              </div>
              <div className="text-xs text-yellow-600 mt-1">
                Рекомендуем увеличить TTL для исторических данных или оптимизировать стратегию кэширования
              </div>
            </div>
          )}

          {metrics.featureStore.metrics.averageComputeTime > 100 && (
            <div className="p-3 bg-orange-50 border border-orange-200 rounded-md">
              <div className="text-sm font-medium text-orange-700">
                Медленное вычисление индикаторов ({metrics.featureStore.metrics.averageComputeTime.toFixed(2)}ms)
              </div>
              <div className="text-xs text-orange-600 mt-1">
                Рекомендуем оптимизировать алгоритмы вычисления технических индикаторов
              </div>
            </div>
          )}

          {metrics.featureStore.redisMetrics.failedCommands > 10 && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-md">
              <div className="text-sm font-medium text-red-700">
                Высокое количество ошибок Redis ({metrics.featureStore.redisMetrics.failedCommands})
              </div>
              <div className="text-xs text-red-600 mt-1">
                Проверьте подключение к Redis и настройки сети
              </div>
            </div>
          )}

          {metrics.featureStore.metrics.hitRate >= 85 && 
           metrics.featureStore.metrics.averageComputeTime < 50 &&
           metrics.featureStore.redisMetrics.failedCommands === 0 && (
            <div className="p-3 bg-green-50 border border-green-200 rounded-md">
              <div className="text-sm font-medium text-green-700">
                Отличная производительность! 🎉
              </div>
              <div className="text-xs text-green-600 mt-1">
                Система кэширования работает оптимально
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default CacheMonitoringDashboard;