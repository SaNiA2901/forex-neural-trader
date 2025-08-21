import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Zap
} from 'lucide-react';
import { usePerformance } from '@/hooks/usePerformance';
import { useErrorHandler } from '@/hooks/useErrorHandler';
import { formatNumber } from '@/lib/utils';

interface PerformanceMonitorProps {
  isVisible?: boolean;
}

const PerformanceMonitor = ({ isVisible = false }: PerformanceMonitorProps) => {
  const { getPerformanceStats, clearCache } = usePerformance();
  const { errors, clearErrors, getErrorStats } = useErrorHandler();
  const [stats, setStats] = useState<any[]>([]);
  const [errorStats, setErrorStats] = useState<any>(null);
  const [memoryUsage, setMemoryUsage] = useState<number>(0);

  useEffect(() => {
    if (!isVisible) return;

    const updateStats = () => {
      setStats(getPerformanceStats());
      setErrorStats(getErrorStats());
      
      // Получаем приблизительное использование памяти
      if ('memory' in performance) {
        const memory = (performance as any).memory;
        setMemoryUsage(
          Math.round((memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100)
        );
      }
    };

    updateStats();
    const interval = setInterval(updateStats, 2000);

    return () => clearInterval(interval);
  }, [isVisible, getPerformanceStats, getErrorStats]);

  if (!isVisible) return null;

  const getPerformanceColor = (status: string) => {
    switch (status) {
      case 'fast': return 'text-green-500';
      case 'moderate': return 'text-yellow-500';
      case 'slow': return 'text-red-500';
      default: return 'text-muted-foreground';
    }
  };

  const getPerformanceIcon = (status: string) => {
    switch (status) {
      case 'fast': return <CheckCircle className="h-4 w-4" />;
      case 'moderate': return <Clock className="h-4 w-4" />;
      case 'slow': return <AlertTriangle className="h-4 w-4" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  return (
    <Card className="p-6 bg-card/50 border-border/50 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Activity className="h-5 w-5 text-blue-500" />
          <h3 className="text-lg font-semibold text-foreground">Мониторинг производительности</h3>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={clearCache}
            className="bg-background/50"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Очистить кэш
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={clearErrors}
            className="bg-background/50"
          >
            Очистить ошибки
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Операции */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-muted-foreground">Операции</h4>
          {stats.length === 0 ? (
            <div className="text-sm text-muted-foreground">Нет данных о производительности</div>
          ) : (
            <div className="space-y-2">
              {stats.slice(0, 5).map((stat, index) => (
                <div key={index} className="flex items-center justify-between p-2 bg-background/30 rounded-md">
                  <div className="flex items-center space-x-2">
                    <div className={getPerformanceColor(stat.status)}>
                      {getPerformanceIcon(stat.status)}
                    </div>
                    <span className="text-sm text-foreground truncate">
                      {stat.operation}
                    </span>
                  </div>
                  <Badge
                    variant="outline"
                    className={`${getPerformanceColor(stat.status)} border-current`}
                  >
                    {formatNumber(stat.time)}ms
                  </Badge>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Ошибки */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-muted-foreground">Статистика ошибок</h4>
          {errorStats && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Всего ошибок</span>
                <Badge variant={errorStats.total > 0 ? 'destructive' : 'outline'}>
                  {errorStats.total}
                </Badge>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">За последние 5 мин</span>
                <Badge variant={errorStats.recent > 0 ? 'destructive' : 'outline'}>
                  {errorStats.recent}
                </Badge>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-red-400">Критические</span>
                  <span className="text-xs text-foreground">{errorStats.byType.errors}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-yellow-400">Предупреждения</span>
                  <span className="text-xs text-foreground">{errorStats.byType.warnings}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-blue-400">Информация</span>
                  <span className="text-xs text-foreground">{errorStats.byType.info}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Системные метрики */}
        <div className="space-y-4">
          <h4 className="text-sm font-medium text-muted-foreground">Система</h4>
          <div className="space-y-3">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Использование памяти</span>
                <span className="text-sm text-foreground">{memoryUsage}%</span>
              </div>
              <Progress value={memoryUsage} className="h-2" />
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Статус приложения</span>
              <Badge 
                variant="outline"
                className={
                  errorStats?.recent === 0 && memoryUsage < 80
                    ? 'text-green-500 border-green-500'
                    : errorStats?.recent > 0 || memoryUsage > 90
                    ? 'text-red-500 border-red-500'
                    : 'text-yellow-500 border-yellow-500'
                }
              >
                {errorStats?.recent === 0 && memoryUsage < 80 ? (
                  <>
                    <TrendingUp className="h-3 w-3 mr-1" />
                    Отличное
                  </>
                ) : errorStats?.recent > 0 || memoryUsage > 90 ? (
                  <>
                    <TrendingDown className="h-3 w-3 mr-1" />
                    Проблемы
                  </>
                ) : (
                  <>
                    <Zap className="h-3 w-3 mr-1" />
                    Нормальное
                  </>
                )}
              </Badge>
            </div>
          </div>
        </div>
      </div>

      {/* Последние ошибки */}
      {errors.length > 0 && (
        <div className="mt-6 pt-4 border-t border-border/50">
          <h4 className="text-sm font-medium text-muted-foreground mb-3">Последние ошибки</h4>
          <div className="space-y-2 max-h-32 overflow-y-auto">
            {errors.slice(-3).map((error, index) => (
              <div key={error.timestamp} className="p-2 bg-destructive/10 border border-destructive/20 rounded-md">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-destructive font-medium">
                    {error.type.toUpperCase()}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {new Date(error.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <p className="text-xs text-foreground mt-1 truncate">
                  {error.message}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
};

export default PerformanceMonitor;