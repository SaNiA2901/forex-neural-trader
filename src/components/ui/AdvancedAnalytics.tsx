
import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell, ScatterChart, Scatter } from "recharts";
import { TrendingUp, Activity, Target, Zap, BarChart3, PieChart as PieChartIcon } from "lucide-react";

interface AdvancedAnalyticsProps {
  pair: string;
  timeframe: string;
}

const AdvancedAnalytics = ({ pair, timeframe }: AdvancedAnalyticsProps) => {
  const [marketDepth, setMarketDepth] = useState<any[]>([]);
  const [orderFlow, setOrderFlow] = useState<any[]>([]);
  const [sentimentData, setSentimentData] = useState<any[]>([]);
  const [correlationMatrix, setCorrelationMatrix] = useState<any[]>([]);
  const [volatilityData, setVolatilityData] = useState<any[]>([]);

  useEffect(() => {
    const generateAdvancedData = () => {
      // Market Depth Data
      const depth = [];
      for (let i = 0; i < 20; i++) {
        depth.push({
          price: 1.0800 + i * 0.001,
          bidVolume: Math.random() * 1000 + 200,
          askVolume: Math.random() * 1000 + 200,
          side: i < 10 ? 'bid' : 'ask'
        });
      }
      setMarketDepth(depth);

      // Order Flow Data
      const flow = [];
      for (let i = 0; i < 24; i++) {
        flow.push({
          hour: i,
          buyOrders: Math.random() * 500 + 100,
          sellOrders: Math.random() * 500 + 100,
          netFlow: (Math.random() - 0.5) * 200
        });
      }
      setOrderFlow(flow);

      // Sentiment Data
      const sentiment = [
        { name: 'Очень бычий', value: 15 + Math.random() * 10, color: '#22C55E' },
        { name: 'Бычий', value: 25 + Math.random() * 15, color: '#84CC16' },
        { name: 'Нейтральный', value: 20 + Math.random() * 20, color: '#3B82F6' },
        { name: 'Медвежий', value: 25 + Math.random() * 15, color: '#F59E0B' },
        { name: 'Очень медвежий', value: 15 + Math.random() * 10, color: '#EF4444' }
      ];
      setSentimentData(sentiment);

      // Correlation Matrix
      const pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD'];
      const correlation = [];
      for (let i = 0; i < pairs.length; i++) {
        for (let j = 0; j < pairs.length; j++) {
          correlation.push({
            x: i,
            y: j,
            pair1: pairs[i],
            pair2: pairs[j],
            correlation: i === j ? 1 : (Math.random() - 0.5) * 2
          });
        }
      }
      setCorrelationMatrix(correlation);

      // Volatility Data
      const volatility = [];
      for (let i = 0; i < 30; i++) {
        volatility.push({
          day: i + 1,
          volatility: 0.5 + Math.random() * 2,
          impliedVol: 0.8 + Math.random() * 1.5,
          realizedVol: 0.6 + Math.random() * 1.8
        });
      }
      setVolatilityData(volatility);
    };

    generateAdvancedData();
    const interval = setInterval(generateAdvancedData, 10000);
    return () => clearInterval(interval);
  }, [pair, timeframe]);

  const RADIAN = Math.PI / 180;
  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }: any) => {
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
      <text x={x} y={y} fill="white" textAnchor={x > cx ? 'start' : 'end'} dominantBaseline="central">
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    );
  };

  return (
    <div className="space-y-6">
      {/* Analytics Overview */}
      <Card className="p-6 bg-slate-800/50 border-slate-700">
        <div className="flex items-center space-x-3 mb-6">
          <BarChart3 className="h-6 w-6 text-blue-400" />
          <h3 className="text-lg font-semibold text-white">Продвинутая аналитика</h3>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <Activity className="h-6 w-6 text-green-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-white mb-1">1.2%</div>
            <p className="text-slate-400 text-sm">Волатильность</p>
          </div>
          
          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <Target className="h-6 w-6 text-blue-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-white mb-1">78%</div>
            <p className="text-slate-400 text-sm">Точность прогнозов</p>
          </div>
          
          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <Zap className="h-6 w-6 text-yellow-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-white mb-1">142</div>
            <p className="text-slate-400 text-sm">Сигналов/день</p>
          </div>
          
          <div className="text-center p-4 bg-slate-700/50 rounded-lg">
            <TrendingUp className="h-6 w-6 text-green-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-green-400 mb-1">+23.4%</div>
            <p className="text-slate-400 text-sm">ROI месяц</p>
          </div>
        </div>
      </Card>

      {/* Advanced Charts */}
      <Tabs defaultValue="depth" className="space-y-6">
        <TabsList className="bg-slate-800 border-slate-600">
          <TabsTrigger value="depth" className="data-[state=active]:bg-blue-600">
            Глубина рынка
          </TabsTrigger>
          <TabsTrigger value="flow" className="data-[state=active]:bg-blue-600">
            Поток ордеров
          </TabsTrigger>
          <TabsTrigger value="sentiment" className="data-[state=active]:bg-blue-600">
            Настроения
          </TabsTrigger>
          <TabsTrigger value="correlation" className="data-[state=active]:bg-blue-600">
            Корреляции
          </TabsTrigger>
          <TabsTrigger value="volatility" className="data-[state=active]:bg-blue-600">
            Волатильность
          </TabsTrigger>
        </TabsList>

        <TabsContent value="depth">
          <Card className="p-6 bg-slate-800/50 border-slate-700">
            <h3 className="text-lg font-semibold text-white mb-4">Глубина рынка - {pair}</h3>
            
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={marketDepth}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="price" 
                    stroke="#9CA3AF" 
                    fontSize={12}
                    tickFormatter={(value) => value.toFixed(4)}
                  />
                  <YAxis stroke="#9CA3AF" fontSize={12} />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#F9FAFB'
                    }}
                  />
                  <Bar dataKey="bidVolume" fill="#22C55E" name="Bid Volume" />
                  <Bar dataKey="askVolume" fill="#EF4444" name="Ask Volume" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="flow">
          <Card className="p-6 bg-slate-800/50 border-slate-700">
            <h3 className="text-lg font-semibold text-white mb-4">Поток ордеров (24 часа)</h3>
            
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={orderFlow}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="hour" stroke="#9CA3AF" fontSize={12} />
                  <YAxis stroke="#9CA3AF" fontSize={12} />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#F9FAFB'
                    }}
                  />
                  <Line type="monotone" dataKey="buyOrders" stroke="#22C55E" strokeWidth={2} name="Buy Orders" />
                  <Line type="monotone" dataKey="sellOrders" stroke="#EF4444" strokeWidth={2} name="Sell Orders" />
                  <Line type="monotone" dataKey="netFlow" stroke="#3B82F6" strokeWidth={2} name="Net Flow" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="sentiment">
          <Card className="p-6 bg-slate-800/50 border-slate-700">
            <h3 className="text-lg font-semibold text-white mb-4">Настроения трейдеров</h3>
            
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={sentimentData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={renderCustomizedLabel}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {sentimentData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#F9FAFB'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              {sentimentData.map((item, index) => (
                <Badge key={index} style={{ backgroundColor: item.color }} className="text-white">
                  {item.name}: {item.value.toFixed(1)}%
                </Badge>
              ))}
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="correlation">
          <Card className="p-6 bg-slate-800/50 border-slate-700">
            <h3 className="text-lg font-semibold text-white mb-4">Матрица корреляций валютных пар</h3>
            
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis type="number" dataKey="x" domain={[0, 4]} stroke="#9CA3AF" fontSize={12} />
                  <YAxis type="number" dataKey="y" domain={[0, 4]} stroke="#9CA3AF" fontSize={12} />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#F9FAFB'
                    }}
                    formatter={(value, name, props) => [
                      `${props.payload.correlation.toFixed(2)}`,
                      `${props.payload.pair1} vs ${props.payload.pair2}`
                    ]}
                  />
                  <Scatter 
                    data={correlationMatrix} 
                    fill="#3B82F6"
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-4 text-sm text-slate-300">
              <p>Корреляция: -1 (обратная) до +1 (прямая)</p>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="volatility">
          <Card className="p-6 bg-slate-800/50 border-slate-700">
            <h3 className="text-lg font-semibold text-white mb-4">Анализ волатильности</h3>
            
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={volatilityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="day" stroke="#9CA3AF" fontSize={12} />
                  <YAxis stroke="#9CA3AF" fontSize={12} />
                  <Tooltip 
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#F9FAFB'
                    }}
                  />
                  <Line type="monotone" dataKey="volatility" stroke="#3B82F6" strokeWidth={2} name="Историческая" />
                  <Line type="monotone" dataKey="impliedVol" stroke="#F59E0B" strokeWidth={2} name="Подразумеваемая" />
                  <Line type="monotone" dataKey="realizedVol" stroke="#10B981" strokeWidth={2} name="Реализованная" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-4 grid grid-cols-3 gap-4">
              <div className="text-center p-3 bg-slate-700/50 rounded-lg">
                <div className="text-blue-400 font-medium">Историческая</div>
                <div className="text-white text-sm">Базовая волатильность</div>
              </div>
              <div className="text-center p-3 bg-slate-700/50 rounded-lg">
                <div className="text-yellow-400 font-medium">Подразумеваемая</div>
                <div className="text-white text-sm">Ожидания рынка</div>
              </div>
              <div className="text-center p-3 bg-slate-700/50 rounded-lg">
                <div className="text-green-400 font-medium">Реализованная</div>
                <div className="text-white text-sm">Фактические движения</div>
              </div>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AdvancedAnalytics;
