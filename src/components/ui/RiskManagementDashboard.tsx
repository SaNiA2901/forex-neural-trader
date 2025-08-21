import { useState, useEffect, useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { Shield, AlertTriangle, TrendingDown, Target, DollarSign, Activity } from 'lucide-react';
import { RiskManagementService } from '@/services/riskManagementService';
import { CandleData } from '@/types/session';
import { RiskMetrics, PortfolioRisk, VaRCalculation, DrawdownAnalysis } from '@/types/trading';

interface RiskManagementDashboardProps {
  candles: CandleData[];
  pair: string;
  currentBalance: number;
  positions?: any[];
}

const RiskManagementDashboard = ({ 
  candles, 
  pair, 
  currentBalance = 10000,
  positions = [] 
}: RiskManagementDashboardProps) => {
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null);
  const [portfolioRisk, setPortfolioRisk] = useState<PortfolioRisk | null>(null);
  const [varCalculation, setVarCalculation] = useState<VaRCalculation | null>(null);
  const [drawdownAnalysis, setDrawdownAnalysis] = useState<DrawdownAnalysis | null>(null);
  const [riskAlerts, setRiskAlerts] = useState<any[]>([]);
  const [positionSizing, setPositionSizing] = useState<any>(null);

  // Расчет метрик риска
  useEffect(() => {
    if (candles.length < 30) return;

    const calculateRiskMetrics = async () => {
      try {
        console.log('📊 Calculating risk metrics...');
        
        // Mock risk metrics for demo
        const metrics = {
          var95: -1500,
          var99: -2000,
          expectedShortfall: -1800,
          sharpeRatio: 1.2,
          riskLevel: 5.5,
          volatility: 18.5,
          maxLoss: -2500,
          riskAdjustedReturn: 0.15
        };
        setRiskMetrics(metrics);

        // Mock portfolio risk
        const portfolio = {
          totalExposure: currentBalance * 0.8,
          concentrationRisk: 0.3,
          correlationRisk: 0.4,
          beta: 1.1,
          alpha: 0.05,
          correlationWithMarket: 0.75,
          assetAllocation: [
            { asset: 'EUR/USD', percentage: 40 },
            { asset: 'GBP/USD', percentage: 30 },
            { asset: 'USD/JPY', percentage: 20 },
            { asset: 'Gold', percentage: 10 }
          ]
        };
        setPortfolioRisk(portfolio);

        // Mock VaR calculation
        const var95 = {
          historical: -1500,
          parametric: -1400,
          monteCarlo: -1600,
          var95: -1500,
          expectedShortfall: -1800,
          confidenceLevel: 0.95,
          historicalVaR: [-1200, -1300, -1500, -1400, -1600]
        };
        setVarCalculation(var95);

        // Mock drawdown analysis
        const drawdown = {
          currentDrawdown: -5.2,
          maxDrawdown: -12.5,
          drawdownDuration: 15,
          avgDrawdown: -7.8,
          maxDrawdownDuration: 25,
          recoveryTime: 18,
          drawdownHistory: [
            { date: '2024-01-01', value: -0.05 },
            { date: '2024-01-02', value: -0.08 },
            { date: '2024-01-03', value: -0.12 }
          ]
        };
        setDrawdownAnalysis(drawdown);

        // Mock position sizing
        const sizing = {
          optimalSize: 2500,
          kellyPercentage: 0.025,
          riskPerTrade: 2.0,
          maxPositions: 3
        };
        setPositionSizing(sizing);

        // Mock risk alerts
        const alerts = [
          {
            severity: 'medium',
            type: 'concentration',
            title: 'Концентрация риска',
            message: 'Высокая концентрация в EUR/USD'
          }
        ];
        setRiskAlerts(alerts);

        console.log('✅ Risk metrics calculated successfully');
      } catch (error) {
        console.error('❌ Error calculating risk metrics:', error);
      }
    };

    calculateRiskMetrics();
  }, [candles, currentBalance, positions]);

  // Цветовая схема для уровней риска
  const getRiskColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-400';
      case 'medium': return 'text-yellow-400';
      case 'high': return 'text-red-400';
      case 'critical': return 'text-red-600';
      default: return 'text-slate-400';
    }
  };

  const getRiskBadgeColor = (level: string) => {
    switch (level) {
      case 'low': return 'bg-green-600';
      case 'medium': return 'bg-yellow-600';
      case 'high': return 'bg-red-600';
      case 'critical': return 'bg-red-800';
      default: return 'bg-slate-600';
    }
  };

  // Данные для графика VaR
  const varChartData = useMemo(() => {
    if (!varCalculation) return [];
    
    return Array.isArray(varCalculation.historicalVaR) ? varCalculation.historicalVaR.map((value, index) => ({
      day: index + 1,
      var95: value,
      var99: value * 1.3, // Приблизительный VaR 99%
      actualLoss: Math.random() * value * 0.8 // Симуляция фактических потерь
    })) : [];
  }, [varCalculation]);

  // Данные для анализа просадок
  const drawdownChartData = useMemo(() => {
    if (!drawdownAnalysis) return [];
    
    return drawdownAnalysis.drawdownHistory.map((dd, index) => ({
      period: index + 1,
      drawdown: dd.value * 100,
      underwater: index,
      recovery: index + 5
    }));
  }, [drawdownAnalysis]);

  if (candles.length < 30) {
    return (
      <Card className="p-6 bg-slate-800/50 border-slate-700">
        <div className="text-center py-8">
          <Shield className="h-12 w-12 text-slate-500 mx-auto mb-4" />
          <p className="text-slate-400">
            Недостаточно данных для анализа рисков.
            <br />
            Добавьте минимум 30 свечей для расчета метрик риска.
          </p>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Risk Overview */}
      <Card className="p-6 bg-gradient-to-br from-slate-800/80 to-slate-900/80 border-slate-700">
        <div className="flex items-center space-x-3 mb-6">
          <Shield className="h-6 w-6 text-blue-400" />
          <h3 className="text-xl font-semibold text-white">Risk Management Dashboard</h3>
        </div>

        {/* Risk Alerts */}
        {riskAlerts.length > 0 && (
          <div className="mb-6 space-y-2">
            {riskAlerts.slice(0, 3).map((alert, index) => (
              <Alert key={index} className={`border-${alert.severity === 'high' ? 'red' : 'yellow'}-600 bg-${alert.severity === 'high' ? 'red' : 'yellow'}-600/20`}>
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription className={`text-${alert.severity === 'high' ? 'red' : 'yellow'}-200`}>
                  <strong>{alert.title}:</strong> {alert.message}
                </AlertDescription>
              </Alert>
            ))}
          </div>
        )}

        {/* Key Risk Metrics */}
        {riskMetrics && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="text-center p-4 bg-slate-700/50 rounded-lg">
                <div className={`text-2xl font-bold mb-1 text-white`}>
                  {riskMetrics.volatility.toFixed(2)}%
                </div>
                <p className="text-slate-400 text-sm">Volatility</p>
                <Badge className={`mt-2 bg-blue-600 text-white`}>
                  MODERATE
                </Badge>
              </div>
            
            <div className="text-center p-4 bg-slate-700/50 rounded-lg">
              <div className="text-2xl font-bold text-red-400 mb-1">
                ${Math.abs(riskMetrics.maxLoss).toFixed(0)}
              </div>
              <p className="text-slate-400 text-sm">Max Potential Loss</p>
            </div>
            
            <div className="text-center p-4 bg-slate-700/50 rounded-lg">
              <div className="text-2xl font-bold text-blue-400 mb-1">
                {riskMetrics.sharpeRatio.toFixed(2)}
              </div>
              <p className="text-slate-400 text-sm">Sharpe Ratio</p>
            </div>
            
            <div className="text-center p-4 bg-slate-700/50 rounded-lg">
              <div className="text-2xl font-bold text-purple-400 mb-1">
                {(riskMetrics.riskAdjustedReturn * 100).toFixed(1)}%
              </div>
              <p className="text-slate-400 text-sm">Risk-Adj. Return</p>
            </div>
          </div>
        )}

        {/* Position Sizing Recommendation */}
        {positionSizing && (
          <div className="border border-slate-600 rounded-lg p-4 bg-gradient-to-r from-blue-900/20 to-purple-900/20">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-white">Recommended Position Size</h4>
              <Badge className="bg-blue-600 text-white">Kelly Criterion</Badge>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-400 mb-1">
                  ${positionSizing.optimalSize.toFixed(0)}
                </div>
                <p className="text-slate-400 text-sm">Optimal Size</p>
              </div>
              
              <div className="text-center">
                <div className="text-2xl font-bold text-yellow-400 mb-1">
                  {(positionSizing.kellyPercentage * 100).toFixed(1)}%
                </div>
                <p className="text-slate-400 text-sm">Kelly %</p>
              </div>
              
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-400 mb-1">
                  {positionSizing.riskPerTrade.toFixed(1)}%
                </div>
                <p className="text-slate-400 text-sm">Risk per Trade</p>
              </div>
              
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-400 mb-1">
                  {positionSizing.maxPositions}
                </div>
                <p className="text-slate-400 text-sm">Max Positions</p>
              </div>
            </div>
          </div>
        )}
      </Card>

      {/* Detailed Risk Analysis */}
      <Tabs defaultValue="var" className="space-y-6">
        <TabsList className="bg-slate-800 border-slate-600">
          <TabsTrigger value="var" className="data-[state=active]:bg-blue-600">
            Value at Risk
          </TabsTrigger>
          <TabsTrigger value="drawdown" className="data-[state=active]:bg-blue-600">
            Drawdown Analysis
          </TabsTrigger>
          <TabsTrigger value="portfolio" className="data-[state=active]:bg-blue-600">
            Portfolio Risk
          </TabsTrigger>
          <TabsTrigger value="scenarios" className="data-[state=active]:bg-blue-600">
            Stress Testing
          </TabsTrigger>
        </TabsList>

        <TabsContent value="var">
          <Card className="p-6 bg-slate-800/50 border-slate-700">
            <h3 className="text-lg font-semibold text-white mb-4">Value at Risk Analysis</h3>
            
            {varCalculation && (
              <>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="text-center p-4 bg-slate-700/50 rounded-lg">
                    <div className="text-2xl font-bold text-red-400 mb-1">
                      ${Math.abs(varCalculation.var95).toFixed(0)}
                    </div>
                    <p className="text-slate-400 text-sm">VaR 95% (1 day)</p>
                  </div>
                  
                  <div className="text-center p-4 bg-slate-700/50 rounded-lg">
                    <div className="text-2xl font-bold text-red-600 mb-1">
                      ${Math.abs(varCalculation.expectedShortfall).toFixed(0)}
                    </div>
                    <p className="text-slate-400 text-sm">Expected Shortfall</p>
                  </div>
                  
                  <div className="text-center p-4 bg-slate-700/50 rounded-lg">
                    <div className="text-2xl font-bold text-yellow-400 mb-1">
                      {varCalculation.confidenceLevel * 100}%
                    </div>
                    <p className="text-slate-400 text-sm">Confidence Level</p>
                  </div>
                </div>

                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={varChartData}>
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
                      <Line type="monotone" dataKey="var95" stroke="#EF4444" strokeWidth={2} name="VaR 95%" />
                      <Line type="monotone" dataKey="var99" stroke="#DC2626" strokeWidth={2} name="VaR 99%" />
                      <Line type="monotone" dataKey="actualLoss" stroke="#3B82F6" strokeWidth={1} name="Actual Loss" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}
          </Card>
        </TabsContent>

        <TabsContent value="drawdown">
          <Card className="p-6 bg-slate-800/50 border-slate-700">
            <h3 className="text-lg font-semibold text-white mb-4">Drawdown Analysis</h3>
            
            {drawdownAnalysis && (
              <>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                  <div className="text-center p-4 bg-slate-700/50 rounded-lg">
                    <div className="text-2xl font-bold text-red-400 mb-1">
                      {drawdownAnalysis.maxDrawdown.toFixed(2)}%
                    </div>
                    <p className="text-slate-400 text-sm">Max Drawdown</p>
                  </div>
                  
                  <div className="text-center p-4 bg-slate-700/50 rounded-lg">
                    <div className="text-2xl font-bold text-yellow-400 mb-1">
                      {drawdownAnalysis.avgDrawdown.toFixed(2)}%
                    </div>
                    <p className="text-slate-400 text-sm">Avg Drawdown</p>
                  </div>
                  
                  <div className="text-center p-4 bg-slate-700/50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-400 mb-1">
                      {drawdownAnalysis.maxDrawdownDuration}
                    </div>
                    <p className="text-slate-400 text-sm">Max Duration (days)</p>
                  </div>
                  
                  <div className="text-center p-4 bg-slate-700/50 rounded-lg">
                    <div className="text-2xl font-bold text-green-400 mb-1">
                      {drawdownAnalysis.recoveryTime}
                    </div>
                    <p className="text-slate-400 text-sm">Avg Recovery (days)</p>
                  </div>
                </div>

                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={drawdownChartData.slice(-20)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="period" stroke="#9CA3AF" fontSize={12} />
                      <YAxis stroke="#9CA3AF" fontSize={12} />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: '#1F2937',
                          border: '1px solid #374151',
                          borderRadius: '8px',
                          color: '#F9FAFB'
                        }}
                      />
                      <Bar dataKey="drawdown" fill="#EF4444" name="Drawdown %" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </>
            )}
          </Card>
        </TabsContent>

        <TabsContent value="portfolio">
          <Card className="p-6 bg-slate-800/50 border-slate-700">
            <h3 className="text-lg font-semibold text-white mb-4">Portfolio Risk Metrics</h3>
            
            {portfolioRisk && (
              <>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="text-center p-4 bg-slate-700/50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-400 mb-1">
                      {portfolioRisk.beta.toFixed(2)}
                    </div>
                    <p className="text-slate-400 text-sm">Portfolio Beta</p>
                  </div>
                  
                  <div className="text-center p-4 bg-slate-700/50 rounded-lg">
                    <div className="text-2xl font-bold text-green-400 mb-1">
                      {portfolioRisk.alpha.toFixed(2)}%
                    </div>
                    <p className="text-slate-400 text-sm">Alpha</p>
                  </div>
                  
                  <div className="text-center p-4 bg-slate-700/50 rounded-lg">
                    <div className="text-2xl font-bold text-purple-400 mb-1">
                      {portfolioRisk.correlationWithMarket.toFixed(2)}
                    </div>
                    <p className="text-slate-400 text-sm">Market Correlation</p>
                  </div>
                </div>

                {/* Diversification Chart */}
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={portfolioRisk.assetAllocation}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="weight"
                      >
                        {portfolioRisk.assetAllocation.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'][index % 5]} 
                          />
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
              </>
            )}
          </Card>
        </TabsContent>

        <TabsContent value="scenarios">
          <Card className="p-6 bg-slate-800/50 border-slate-700">
            <h3 className="text-lg font-semibold text-white mb-4">Stress Testing Scenarios</h3>
            
            <div className="space-y-4">
              {/* Market Crash Scenario */}
              <div className="border border-slate-600 rounded-lg p-4 bg-slate-700/30">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-white font-medium">Market Crash (-20%)</h4>
                  <Badge className="bg-red-600 text-white">High Impact</Badge>
                </div>
                
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-xl font-bold text-red-400 mb-1">
                      -${(currentBalance * 0.15).toFixed(0)}
                    </div>
                    <p className="text-slate-400 text-sm">Estimated Loss</p>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-xl font-bold text-yellow-400 mb-1">
                      15%
                    </div>
                    <p className="text-slate-400 text-sm">Portfolio Impact</p>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-xl font-bold text-blue-400 mb-1">
                      30 days
                    </div>
                    <p className="text-slate-400 text-sm">Recovery Time</p>
                  </div>
                </div>
              </div>

              {/* Interest Rate Shock */}
              <div className="border border-slate-600 rounded-lg p-4 bg-slate-700/30">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-white font-medium">Interest Rate Shock (+200 bps)</h4>
                  <Badge className="bg-yellow-600 text-white">Medium Impact</Badge>
                </div>
                
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-xl font-bold text-red-400 mb-1">
                      -${(currentBalance * 0.08).toFixed(0)}
                    </div>
                    <p className="text-slate-400 text-sm">Estimated Loss</p>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-xl font-bold text-yellow-400 mb-1">
                      8%
                    </div>
                    <p className="text-slate-400 text-sm">Portfolio Impact</p>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-xl font-bold text-blue-400 mb-1">
                      15 days
                    </div>
                    <p className="text-slate-400 text-sm">Recovery Time</p>
                  </div>
                </div>
              </div>

              {/* Volatility Spike */}
              <div className="border border-slate-600 rounded-lg p-4 bg-slate-700/30">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-white font-medium">Volatility Spike (VIX &gt; 40)</h4>
                  <Badge className="bg-green-600 text-white">Low Impact</Badge>
                </div>
                
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center">
                    <div className="text-xl font-bold text-red-400 mb-1">
                      -${(currentBalance * 0.05).toFixed(0)}
                    </div>
                    <p className="text-slate-400 text-sm">Estimated Loss</p>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-xl font-bold text-yellow-400 mb-1">
                      5%
                    </div>
                    <p className="text-slate-400 text-sm">Portfolio Impact</p>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-xl font-bold text-blue-400 mb-1">
                      7 days
                    </div>
                    <p className="text-slate-400 text-sm">Recovery Time</p>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default RiskManagementDashboard;