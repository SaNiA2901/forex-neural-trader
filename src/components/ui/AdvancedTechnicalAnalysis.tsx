import { useState, useEffect, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ComposedChart, Bar, Area, AreaChart } from 'recharts';
import { TrendingUp, BarChart3, Target, Activity, Zap, AlertTriangle } from 'lucide-react';
import { CandleData } from '@/types/session';
import { 
  TechnicalIndicator, 
  PatternDetection, 
  SupportResistanceLevel,
  VolumeProfile,
  MarketStructure
} from '@/types/trading';

interface AdvancedTechnicalAnalysisProps {
  candles: CandleData[];
  pair: string;
  timeframe: string;
}

// Продвинутые технические индикаторы
class AdvancedIndicators {
  static calculateRSI(prices: number[], period: number = 14): number[] {
    const rsi: number[] = [];
    const gains: number[] = [];
    const losses: number[] = [];

    for (let i = 1; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }

    let avgGain = gains.slice(0, period).reduce((sum, gain) => sum + gain, 0) / period;
    let avgLoss = losses.slice(0, period).reduce((sum, loss) => sum + loss, 0) / period;

    for (let i = period; i < prices.length; i++) {
      if (avgLoss === 0) {
        rsi.push(100);
      } else {
        const rs = avgGain / avgLoss;
        rsi.push(100 - (100 / (1 + rs)));
      }

      if (i < gains.length) {
        avgGain = (avgGain * (period - 1) + gains[i]) / period;
        avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
      }
    }

    return rsi;
  }

  static calculateMACD(prices: number[], fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9) {
    const emaFast = this.calculateEMA(prices, fastPeriod);
    const emaSlow = this.calculateEMA(prices, slowPeriod);
    
    const macdLine = emaFast.map((fast, i) => fast - emaSlow[i]);
    const signalLine = this.calculateEMA(macdLine, signalPeriod);
    const histogram = macdLine.map((macd, i) => macd - signalLine[i]);

    return { macdLine, signalLine, histogram };
  }

  static calculateEMA(prices: number[], period: number): number[] {
    const ema: number[] = [];
    const multiplier = 2 / (period + 1);
    
    ema[0] = prices[0];
    for (let i = 1; i < prices.length; i++) {
      ema[i] = (prices[i] * multiplier) + (ema[i - 1] * (1 - multiplier));
    }
    
    return ema;
  }

  static calculateBollingerBands(prices: number[], period: number = 20, stdDev: number = 2) {
    const sma = this.calculateSMA(prices, period);
    const upperBand: number[] = [];
    const lowerBand: number[] = [];

    for (let i = period - 1; i < prices.length; i++) {
      const slice = prices.slice(i - period + 1, i + 1);
      const mean = slice.reduce((sum, price) => sum + price, 0) / period;
      const variance = slice.reduce((sum, price) => sum + Math.pow(price - mean, 2), 0) / period;
      const standardDeviation = Math.sqrt(variance);

      upperBand.push(sma[i - period + 1] + (standardDeviation * stdDev));
      lowerBand.push(sma[i - period + 1] - (standardDeviation * stdDev));
    }

    return { upperBand, lowerBand, middleBand: sma };
  }

  static calculateSMA(prices: number[], period: number): number[] {
    const sma: number[] = [];
    for (let i = period - 1; i < prices.length; i++) {
      const slice = prices.slice(i - period + 1, i + 1);
      sma.push(slice.reduce((sum, price) => sum + price, 0) / period);
    }
    return sma;
  }

  static calculateATR(highs: number[], lows: number[], closes: number[], period: number = 14): number[] {
    const trueRanges: number[] = [];
    
    for (let i = 1; i < highs.length; i++) {
      const tr1 = highs[i] - lows[i];
      const tr2 = Math.abs(highs[i] - closes[i - 1]);
      const tr3 = Math.abs(lows[i] - closes[i - 1]);
      trueRanges.push(Math.max(tr1, tr2, tr3));
    }

    return this.calculateSMA(trueRanges, period);
  }

  static calculateStochastic(highs: number[], lows: number[], closes: number[], kPeriod: number = 14, dPeriod: number = 3) {
    const kPercent: number[] = [];
    
    for (let i = kPeriod - 1; i < closes.length; i++) {
      const highSlice = highs.slice(i - kPeriod + 1, i + 1);
      const lowSlice = lows.slice(i - kPeriod + 1, i + 1);
      const highestHigh = Math.max(...highSlice);
      const lowestLow = Math.min(...lowSlice);
      
      const k = ((closes[i] - lowestLow) / (highestHigh - lowestLow)) * 100;
      kPercent.push(k);
    }

    const dPercent = this.calculateSMA(kPercent, dPeriod);
    return { kPercent, dPercent };
  }
}

// Детекция паттернов
class PatternDetector {
  static detectCandlePatterns(candles: CandleData[]): PatternDetection[] {
    const patterns: PatternDetection[] = [];
    
    for (let i = 2; i < candles.length; i++) {
      const current = candles[i];
      const prev = candles[i - 1];
      const prev2 = candles[i - 2];

      // Doji
      if (this.isDoji(current)) {
        patterns.push({
          pattern: 'Doji',
          name: 'Doji',
          type: 'reversal',
          confidence: 70,
          timeframe: '1min',
          strength: 0.7,
          index: i,
          description: 'Потенциальный разворот тренда'
        });
      }

      // Hammer
      if (this.isHammer(current)) {
        patterns.push({
          pattern: 'Hammer',
          name: 'Hammer',
          type: 'reversal',
          confidence: 80,
          timeframe: '1min',
          strength: 0.8,
          index: i,
          description: 'Бычий разворотный паттерн'
        });
      }

      // Engulfing
      if (this.isBullishEngulfing(prev, current)) {
        patterns.push({
          pattern: 'Bullish Engulfing',
          name: 'Bullish Engulfing',
          type: 'reversal',
          confidence: 90,
          timeframe: '1min',
          strength: 0.9,
          index: i,
          description: 'Сильный бычий сигнал'
        });
      }

      if (this.isBearishEngulfing(prev, current)) {
        patterns.push({
          pattern: 'Bearish Engulfing',
          name: 'Bearish Engulfing',
          type: 'reversal',
          confidence: 90,
          timeframe: '1min',
          strength: 0.9,
          index: i,
          description: 'Сильный медвежий сигнал'
        });
      }

      // Three White Soldiers
      if (this.isThreeWhiteSoldiers(prev2, prev, current)) {
        patterns.push({
          pattern: 'Three White Soldiers',
          name: 'Three White Soldiers',
          type: 'continuation',
          confidence: 85,
          timeframe: '1min',
          strength: 0.85,
          index: i,
          description: 'Продолжение восходящего тренда'
        });
      }
    }

    return patterns;
  }

  private static isDoji(candle: CandleData): boolean {
    const bodySize = Math.abs(candle.close - candle.open);
    const totalRange = candle.high - candle.low;
    return bodySize / totalRange < 0.1;
  }

  private static isHammer(candle: CandleData): boolean {
    const bodySize = Math.abs(candle.close - candle.open);
    const lowerShadow = Math.min(candle.open, candle.close) - candle.low;
    const upperShadow = candle.high - Math.max(candle.open, candle.close);
    
    return lowerShadow > bodySize * 2 && upperShadow < bodySize * 0.5;
  }

  private static isBullishEngulfing(prev: CandleData, current: CandleData): boolean {
    return prev.close < prev.open && // Previous bearish
           current.close > current.open && // Current bullish
           current.open < prev.close && // Opens below previous close
           current.close > prev.open; // Closes above previous open
  }

  private static isBearishEngulfing(prev: CandleData, current: CandleData): boolean {
    return prev.close > prev.open && // Previous bullish
           current.close < current.open && // Current bearish
           current.open > prev.close && // Opens above previous close
           current.close < prev.open; // Closes below previous open
  }

  private static isThreeWhiteSoldiers(candle1: CandleData, candle2: CandleData, candle3: CandleData): boolean {
    return candle1.close > candle1.open &&
           candle2.close > candle2.open &&
           candle3.close > candle3.open &&
           candle2.close > candle1.close &&
           candle3.close > candle2.close;
  }
}

const AdvancedTechnicalAnalysis = ({ candles, pair, timeframe }: AdvancedTechnicalAnalysisProps) => {
  const [indicators, setIndicators] = useState<TechnicalIndicator[]>([]);
  const [patterns, setPatterns] = useState<PatternDetection[]>([]);
  const [supportResistance, setSupportResistance] = useState<SupportResistanceLevel[]>([]);
  const [chartData, setChartData] = useState<any[]>([]);

  // Вычисление всех технических индикаторов
  const technicalData = useMemo(() => {
    if (candles.length < 50) return null;

    const closes = candles.map(c => c.close);
    const highs = candles.map(c => c.high);
    const lows = candles.map(c => c.low);
    const volumes = candles.map(c => c.volume);

    // RSI
    const rsi = AdvancedIndicators.calculateRSI(closes);
    
    // MACD
    const macd = AdvancedIndicators.calculateMACD(closes);
    
    // Bollinger Bands
    const bollinger = AdvancedIndicators.calculateBollingerBands(closes);
    
    // ATR
    const atr = AdvancedIndicators.calculateATR(highs, lows, closes);
    
    // Stochastic
    const stochastic = AdvancedIndicators.calculateStochastic(highs, lows, closes);
    
    // Moving Averages
    const sma20 = AdvancedIndicators.calculateSMA(closes, 20);
    const sma50 = AdvancedIndicators.calculateSMA(closes, 50);
    const ema12 = AdvancedIndicators.calculateEMA(closes, 12);
    const ema26 = AdvancedIndicators.calculateEMA(closes, 26);

    return {
      rsi,
      macd,
      bollinger,
      atr,
      stochastic,
      sma20,
      sma50,
      ema12,
      ema26
    };
  }, [candles]);

  // Подготовка данных для графика
  useEffect(() => {
    if (!technicalData || candles.length < 50) return;

    const chartPoints = candles.map((candle, index) => ({
      index,
      timestamp: candle.timestamp,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
      volume: candle.volume,
      rsi: technicalData.rsi[index - 14] || null,
      macd: technicalData.macd.macdLine[index - 26] || null,
      signal: technicalData.macd.signalLine[index - 35] || null,
      upperBand: technicalData.bollinger.upperBand[index - 19] || null,
      lowerBand: technicalData.bollinger.lowerBand[index - 19] || null,
      sma20: technicalData.sma20[index - 19] || null,
      sma50: technicalData.sma50[index - 49] || null,
      ema12: technicalData.ema12[index] || null,
      ema26: technicalData.ema26[index] || null,
      atr: technicalData.atr[index - 14] || null,
      stochasticK: technicalData.stochastic.kPercent[index - 13] || null,
      stochasticD: technicalData.stochastic.dPercent[index - 16] || null
    }));

    setChartData(chartPoints);
  }, [technicalData, candles]);

  // Детекция паттернов
  useEffect(() => {
    if (candles.length < 10) return;
    
    const detectedPatterns = PatternDetector.detectCandlePatterns(candles);
    setPatterns(detectedPatterns.slice(-10)); // Последние 10 паттернов
  }, [candles]);

  // Создание технических индикаторов для отображения
  useEffect(() => {
    if (!technicalData) return;

    const currentRSI = technicalData.rsi[technicalData.rsi.length - 1];
    const currentMACD = technicalData.macd.macdLine[technicalData.macd.macdLine.length - 1];
    const currentSignal = technicalData.macd.signalLine[technicalData.macd.signalLine.length - 1];
    const currentStochK = technicalData.stochastic.kPercent[technicalData.stochastic.kPercent.length - 1];

    const indicatorList: TechnicalIndicator[] = [
      {
        name: 'RSI',
        value: currentRSI,
        signal: currentRSI > 70 ? 'SELL' : currentRSI < 30 ? 'BUY' : 'NEUTRAL',
        strength: Math.abs(currentRSI - 50) / 50,
        description: currentRSI > 70 ? 'Перекупленность' : currentRSI < 30 ? 'Перепроданность' : 'Нейтральная зона'
      },
      {
        name: 'MACD',
        value: currentMACD,
        signal: currentMACD > currentSignal ? 'BUY' : 'SELL',
        strength: Math.abs(currentMACD - currentSignal) / Math.max(Math.abs(currentMACD), Math.abs(currentSignal)),
        description: currentMACD > currentSignal ? 'Бычий сигнал' : 'Медвежий сигнал'
      },
      {
        name: 'Stochastic',
        value: currentStochK,
        signal: currentStochK > 80 ? 'SELL' : currentStochK < 20 ? 'BUY' : 'NEUTRAL',
        strength: Math.abs(currentStochK - 50) / 50,
        description: currentStochK > 80 ? 'Перекупленность' : currentStochK < 20 ? 'Перепроданность' : 'Нейтральная зона'
      }
    ];

    setIndicators(indicatorList);
  }, [technicalData]);

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY': return 'text-green-400';
      case 'SELL': return 'text-red-400';
      default: return 'text-yellow-400';
    }
  };

  const getSignalBadgeColor = (signal: string) => {
    switch (signal) {
      case 'BUY': return 'bg-green-600';
      case 'SELL': return 'bg-red-600';
      default: return 'bg-yellow-600';
    }
  };

  const getPatternColor = (type: string) => {
    switch (type) {
      case 'reversal': return 'bg-orange-600';
      case 'continuation': return 'bg-blue-600';
      default: return 'bg-purple-600';
    }
  };

  if (candles.length < 50) {
    return (
      <Card className="p-6 bg-slate-800/50 border-slate-700">
        <div className="text-center py-8">
          <BarChart3 className="h-12 w-12 text-slate-500 mx-auto mb-4" />
          <p className="text-slate-400">
            Недостаточно данных для технического анализа. 
            <br />
            Добавьте минимум 50 свечей для получения точных индикаторов.
          </p>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Technical Indicators Overview */}
      <Card className="p-6 bg-gradient-to-br from-slate-800/80 to-slate-900/80 border-slate-700">
        <div className="flex items-center space-x-3 mb-6">
          <BarChart3 className="h-6 w-6 text-blue-400" />
          <h3 className="text-xl font-semibold text-white">Advanced Technical Analysis</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {indicators.map((indicator, index) => (
            <div key={index} className="border border-slate-600 rounded-lg p-4 bg-slate-700/30">
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-white font-medium">{indicator.name}</h4>
                <Badge className={`${getSignalBadgeColor(indicator.signal)} text-white`}>
                  {indicator.signal}
                </Badge>
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-slate-400 text-sm">Value</span>
                  <span className={`text-sm ${getSignalColor(indicator.signal)}`}>
                    {indicator.value.toFixed(2)}
                  </span>
                </div>
                
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-slate-400 text-sm">Strength</span>
                    <span className="text-white text-sm">
                      {(indicator.strength * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={indicator.strength * 100} className="h-2" />
                </div>
                
                <p className="text-slate-300 text-xs">{indicator.description}</p>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Charts and Analysis */}
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="bg-slate-800 border-slate-600">
          <TabsTrigger value="overview" className="data-[state=active]:bg-blue-600">
            Price & Moving Averages
          </TabsTrigger>
          <TabsTrigger value="oscillators" className="data-[state=active]:bg-blue-600">
            Oscillators
          </TabsTrigger>
          <TabsTrigger value="patterns" className="data-[state=active]:bg-blue-600">
            Pattern Detection
          </TabsTrigger>
          <TabsTrigger value="volume" className="data-[state=active]:bg-blue-600">
            Volume Analysis
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <Card className="p-6 bg-slate-800/50 border-slate-700">
            <h3 className="text-lg font-semibold text-white mb-4">Price Action & Moving Averages</h3>
            
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData.slice(-100)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="index" 
                    stroke="#9CA3AF" 
                    fontSize={12} 
                  />
                  <YAxis 
                    stroke="#9CA3AF" 
                    fontSize={12}
                    domain={['dataMin - 0.001', 'dataMax + 0.001']}
                  />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#F9FAFB'
                    }}
                  />
                  
                  {/* Bollinger Bands */}
                  <Area 
                    type="monotone" 
                    dataKey="upperBand" 
                    stroke="transparent" 
                    fill="#3B82F6" 
                    fillOpacity={0.1}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="lowerBand" 
                    stroke="transparent" 
                    fill="#3B82F6" 
                    fillOpacity={0.1}
                  />
                  
                  {/* Price */}
                  <Line 
                    type="monotone" 
                    dataKey="close" 
                    stroke="#FFFFFF" 
                    strokeWidth={2} 
                    dot={false}
                    name="Close Price"
                  />
                  
                  {/* Moving Averages */}
                  <Line 
                    type="monotone" 
                    dataKey="sma20" 
                    stroke="#10B981" 
                    strokeWidth={1} 
                    dot={false}
                    name="SMA 20"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="sma50" 
                    stroke="#F59E0B" 
                    strokeWidth={1} 
                    dot={false}
                    name="SMA 50"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="ema12" 
                    stroke="#3B82F6" 
                    strokeWidth={1} 
                    dot={false}
                    name="EMA 12"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="oscillators">
          <div className="space-y-6">
            {/* RSI */}
            <Card className="p-6 bg-slate-800/50 border-slate-700">
              <h3 className="text-lg font-semibold text-white mb-4">RSI (Relative Strength Index)</h3>
              
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData.slice(-100)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="index" stroke="#9CA3AF" fontSize={12} />
                    <YAxis stroke="#9CA3AF" fontSize={12} domain={[0, 100]} />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        color: '#F9FAFB'
                      }}
                    />
                    
                    {/* RSI Levels */}
                    <Line y={70} stroke="#EF4444" strokeDasharray="5 5" strokeWidth={1} />
                    <Line y={30} stroke="#10B981" strokeDasharray="5 5" strokeWidth={1} />
                    
                    <Line 
                      type="monotone" 
                      dataKey="rsi" 
                      stroke="#8B5CF6" 
                      strokeWidth={2} 
                      dot={false}
                      name="RSI"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* MACD */}
            <Card className="p-6 bg-slate-800/50 border-slate-700">
              <h3 className="text-lg font-semibold text-white mb-4">MACD</h3>
              
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData.slice(-100)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="index" stroke="#9CA3AF" fontSize={12} />
                    <YAxis stroke="#9CA3AF" fontSize={12} />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        color: '#F9FAFB'
                      }}
                    />
                    
                    <Line 
                      type="monotone" 
                      dataKey="macd" 
                      stroke="#3B82F6" 
                      strokeWidth={2} 
                      dot={false}
                      name="MACD"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="signal" 
                      stroke="#EF4444" 
                      strokeWidth={2} 
                      dot={false}
                      name="Signal"
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="patterns">
          <Card className="p-6 bg-slate-800/50 border-slate-700">
            <h3 className="text-lg font-semibold text-white mb-4">Detected Patterns</h3>
            
            {patterns.length > 0 ? (
              <div className="space-y-4">
                {patterns.map((pattern, index) => (
                  <div key={index} className="border border-slate-600 rounded-lg p-4 bg-slate-700/30">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-3">
                        <Target className="h-4 w-4 text-blue-400" />
                        <span className="text-white font-medium">{pattern.name}</span>
                        <Badge className={`${getPatternColor(pattern.type)} text-white`}>
                          {pattern.type}
                        </Badge>
                      </div>
                      
                      <div className="text-right">
                        <div className="text-slate-400 text-sm">
                          Candle #{pattern.index}
                        </div>
                        <div className={`text-sm ${
                          pattern.strength >= 0.8 ? 'text-green-400' :
                          pattern.strength >= 0.6 ? 'text-yellow-400' : 'text-red-400'
                        }`}>
                          {pattern.strength}
                        </div>
                      </div>
                    </div>
                    
                    <p className="text-slate-300 text-sm">{pattern.description}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8">
                <AlertTriangle className="h-12 w-12 text-slate-500 mx-auto mb-4" />
                <p className="text-slate-400">No patterns detected in recent candles.</p>
              </div>
            )}
          </Card>
        </TabsContent>

        <TabsContent value="volume">
          <Card className="p-6 bg-slate-800/50 border-slate-700">
            <h3 className="text-lg font-semibold text-white mb-4">Volume Analysis</h3>
            
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={chartData.slice(-50)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="index" stroke="#9CA3AF" fontSize={12} />
                  <YAxis yAxisId="price" orientation="right" stroke="#9CA3AF" fontSize={12} />
                  <YAxis yAxisId="volume" orientation="left" stroke="#9CA3AF" fontSize={12} />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#F9FAFB'
                    }}
                  />
                  
                  <Bar yAxisId="volume" dataKey="volume" fill="#3B82F6" fillOpacity={0.6} name="Volume" />
                  <Line 
                    yAxisId="price"
                    type="monotone" 
                    dataKey="close" 
                    stroke="#FFFFFF" 
                    strokeWidth={2} 
                    dot={false}
                    name="Price"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AdvancedTechnicalAnalysis;