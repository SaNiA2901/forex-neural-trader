
import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import ErrorBoundary from "@/components/ui/ErrorBoundary";
import PriceChart from "@/components/ui/PriceChart";
import TechnicalIndicators from "@/components/ui/TechnicalIndicators";
import SessionBasedTradingSignals from "@/components/ui/SessionBasedTradingSignals";
import SessionBasedMarketOverview from "@/components/ui/SessionBasedMarketOverview";
import MLPredictions from "@/components/ui/MLPredictions";
import AdvancedAnalytics from "@/components/ui/AdvancedAnalytics";
import WeightedPriceForecast from "@/components/ui/WeightedPriceForecast";
import BinaryOptionsPredictor from "@/components/ui/BinaryOptionsPredictor";
import RealPredictionDisplay from "@/components/ui/RealPredictionDisplay";
import { InferenceDashboard } from "@/components/ui/monitoring/InferenceDashboard";
import { OnnxPredictionPanel } from "@/components/ui/predictor/OnnxPredictionPanel";
import { BarChart } from "lucide-react";

const Index = () => {
  const [selectedPair, setSelectedPair] = useState("EUR/USD");
  const [timeframe, setTimeframe] = useState("1h");

  const currencyPairs = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", 
    "AUD/USD", "USD/CAD", "NZD/USD", "EUR/GBP",
    "EUR/JPY", "GBP/JPY", "AUD/JPY", "CAD/JPY",
    "CHF/JPY", "NZD/JPY", "EUR/CHF", "GBP/CHF",
    "AUD/CHF", "CAD/CHF", "EUR/AUD", "GBP/AUD",
    "USD/NOK", "USD/SEK", "USD/DKK", "USD/PLN",
    "USD/CZK", "USD/HUF", "USD/RUB", "USD/TRY",
    "USD/ZAR", "USD/MXN", "USD/SGD", "USD/HKD",
    "EUR/NOK", "EUR/SEK", "EUR/DKK", "EUR/PLN",
    "GBP/NOK", "GBP/SEK", "AUD/CAD", "NZD/CAD",
    "BTC/USD", "ETH/USD", "LTC/USD", "XRP/USD",
    "ADA/USD", "DOT/USD", "LINK/USD", "BCH/USD"
  ];

  const timeframes = [
    { value: "1m", label: "1 минута" },
    { value: "5m", label: "5 минут" },
    { value: "15m", label: "15 минут" },
    { value: "30m", label: "30 минут" },
    { value: "1h", label: "1 час" },
    { value: "4h", label: "4 часа" },
    { value: "1d", label: "1 день" }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800">
      <div className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <BarChart className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">OptiTrend Analyzer</h1>
                <p className="text-slate-400 text-sm">Профессиональный анализ валютного рынка и криптовалют</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <Select value={selectedPair} onValueChange={setSelectedPair}>
                <SelectTrigger className="w-40 bg-slate-800 border-slate-600 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-slate-800 border-slate-600 max-h-60 overflow-y-auto">
                  {currencyPairs.map(pair => (
                    <SelectItem key={pair} value={pair} className="text-white focus:bg-slate-700">
                      {pair}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select value={timeframe} onValueChange={setTimeframe}>
                <SelectTrigger className="w-32 bg-slate-800 border-slate-600 text-white">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-slate-800 border-slate-600">
                  {timeframes.map(tf => (
                    <SelectItem key={tf.value} value={tf.value} className="text-white focus:bg-slate-700">
                      {tf.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-1">
            <ErrorBoundary>
              <SessionBasedMarketOverview pair={selectedPair} timeframe={timeframe} />
            </ErrorBoundary>
          </div>

          <div className="lg:col-span-3">
            <ErrorBoundary>
              <Tabs defaultValue="binary" className="space-y-6">
                <TabsList className="bg-slate-800 border-slate-600">
                  <TabsTrigger value="binary" className="data-[state=active]:bg-blue-600">
                    Бинарные опционы
                  </TabsTrigger>
                  <TabsTrigger value="chart" className="data-[state=active]:bg-blue-600">
                    График цены
                  </TabsTrigger>
                  <TabsTrigger value="forecast" className="data-[state=active]:bg-blue-600">
                    Прогноз цены
                  </TabsTrigger>
                  <TabsTrigger value="indicators" className="data-[state=active]:bg-blue-600">
                    Технические индикаторы
                  </TabsTrigger>
                  <TabsTrigger value="signals" className="data-[state=active]:bg-blue-600">
                    Торговые сигналы
                  </TabsTrigger>
                  <TabsTrigger value="ml" className="data-[state=active]:bg-blue-600">
                    AI Прогнозы
                  </TabsTrigger>
                  <TabsTrigger value="analytics" className="data-[state=active]:bg-blue-600">
                    Аналитика
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="binary">
                  <BinaryOptionsPredictor pair={selectedPair} timeframe={timeframe} />
                </TabsContent>

                <TabsContent value="chart">
                  <PriceChart pair={selectedPair} timeframe={timeframe} />
                </TabsContent>

                <TabsContent value="forecast">
                  <RealPredictionDisplay pair={selectedPair} timeframe={timeframe} />
                </TabsContent>

                <TabsContent value="indicators">
                  <TechnicalIndicators pair={selectedPair} timeframe={timeframe} />
                </TabsContent>

                <TabsContent value="signals">
                  <SessionBasedTradingSignals pair={selectedPair} timeframe={timeframe} />
                </TabsContent>

                <TabsContent value="ml">
                  <MLPredictions pair={selectedPair} timeframe={timeframe} />
                </TabsContent>

                <TabsContent value="analytics">
                  <AdvancedAnalytics pair={selectedPair} timeframe={timeframe} />
                </TabsContent>
              </Tabs>
            </ErrorBoundary>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
