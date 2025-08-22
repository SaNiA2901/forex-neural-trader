/**
 * EMA Service Usage Examples and Benchmarks
 * 
 * Demonstrates the capabilities of both batch and streaming EMA calculations
 * with real-world scenarios and performance comparisons.
 * 
 * @author Trading System Core
 * @version 1.0.0
 */

import { EMAService, EMAConfig } from './ema';
import { StreamingEMA, MultiPeriodStreamingEMA } from './streaming-ema';

/**
 * Example usage of EMA services with sample market data
 */
export class EMADemo {
  /**
   * Generates realistic market price data for demonstration
   */
  static generateSamplePriceData(length: number): number[] {
    const prices: number[] = [];
    let currentPrice = 100;
    
    for (let i = 0; i < length; i++) {
      // Add some trend and volatility
      const trend = 0.0005; // Small upward trend
      const volatility = 0.02; // 2% volatility
      const randomChange = (Math.random() - 0.5) * volatility * currentPrice;
      const trendChange = trend * currentPrice;
      
      currentPrice += randomChange + trendChange;
      prices.push(Number(currentPrice.toFixed(2)));
    }
    
    return prices;
  }

  /**
   * Demonstrates basic EMA calculation with batch processing
   */
  static demonstrateBatchEMA(): void {
    console.log('=== Batch EMA Demonstration ===\n');
    
    const prices = this.generateSamplePriceData(50);
    console.log('Sample price data (first 10 values):', prices.slice(0, 10));
    
    // Calculate 20-period EMA
    const config: EMAConfig = { period: 20 };
    const emaResult = EMAService.computeEMA(prices, config);
    
    console.log('\n20-period EMA Results:');
    console.log(`- Total values: ${emaResult.values.length}`);
    console.log(`- Valid from index: ${emaResult.validFrom}`);
    console.log(`- Smoothing factor: ${emaResult.smoothingFactor.toFixed(4)}`);
    console.log(`- Last 5 EMA values:`, 
      emaResult.values.slice(-5).map(v => v.toFixed(2)));
    
    // Multiple period calculation
    const multiResults = EMAService.computeMultipleEMA(prices, [5, 10, 20, 50]);
    console.log('\nMultiple EMA periods:');
    for (const [period, result] of multiResults) {
      const lastValue = result.values[result.values.length - 1];
      console.log(`- EMA${period}: ${lastValue.toFixed(2)}`);
    }
  }

  /**
   * Demonstrates streaming EMA for real-time processing
   */
  static demonstrateStreamingEMA(): void {
    console.log('\n=== Streaming EMA Demonstration ===\n');
    
    const streamingEMA = new StreamingEMA({ period: 20 });
    const prices = this.generateSamplePriceData(30);
    
    console.log('Processing prices one by one (streaming):');
    
    for (let i = 0; i < prices.length; i++) {
      const update = streamingEMA.update(prices[i]);
      
      // Show every 5th update
      if (i % 5 === 0 || i === prices.length - 1) {
        console.log(`Update ${i + 1}: Price=${prices[i].toFixed(2)}, ` +
          `EMA=${update.value.toFixed(2)}, Stable=${update.isStable}, ` +
          `Change=${update.change.toFixed(4)}`);
      }
    }
    
    const finalState = streamingEMA.getState();
    console.log('\nFinal state:');
    console.log(`- Current EMA: ${finalState.currentValue.toFixed(4)}`);
    console.log(`- Total updates: ${finalState.count}`);
    console.log(`- Is stable: ${finalState.isStable}`);
  }

  /**
   * Demonstrates multi-period streaming EMA
   */
  static demonstrateMultiPeriodEMA(): void {
    console.log('\n=== Multi-Period Streaming EMA ===\n');
    
    const multiEMA = new MultiPeriodStreamingEMA([5, 10, 20, 50]);
    const prices = this.generateSamplePriceData(25);
    
    console.log('Processing with multiple periods simultaneously:');
    
    // Process first few prices
    for (let i = 0; i < Math.min(prices.length, 10); i++) {
      const updates = multiEMA.updateAll(prices[i]);
      
      if (i === 4 || i === 9) { // Show progress
        console.log(`\nAfter ${i + 1} updates:`);
        for (const [period, update] of updates) {
          console.log(`  EMA${period}: ${update.value.toFixed(2)} (stable: ${update.isStable})`);
        }
      }
    }
    
    // Get final values
    const finalValues = multiEMA.getCurrentValues();
    console.log('\nFinal EMA values:');
    for (const [period, value] of finalValues) {
      console.log(`- EMA${period}: ${value.toFixed(4)}`);
    }
  }

  /**
   * Performance benchmark comparing batch vs streaming
   */
  static runPerformanceBenchmark(): void {
    console.log('\n=== Performance Benchmark ===\n');
    
    const largeDataset = this.generateSamplePriceData(10000);
    const config = { period: 20 };
    
    // Batch calculation benchmark
    console.log('Benchmarking batch calculation...');
    const batchStart = performance.now();
    const batchResult = EMAService.computeEMA(largeDataset, config);
    const batchEnd = performance.now();
    const batchTime = batchEnd - batchStart;
    
    // Streaming calculation benchmark
    console.log('Benchmarking streaming calculation...');
    const streamingStart = performance.now();
    const streamingEMA = new StreamingEMA(config);
    
    for (const price of largeDataset) {
      streamingEMA.update(price);
    }
    const streamingEnd = performance.now();
    const streamingTime = streamingEnd - streamingStart;
    
    // Results comparison
    const batchFinal = batchResult.values[batchResult.values.length - 1];
    const streamingFinal = streamingEMA.getCurrentValue();
    const difference = Math.abs(batchFinal - streamingFinal);
    
    console.log('\nBenchmark Results:');
    console.log('- Dataset size: ' + largeDataset.length.toLocaleString() + ' values');
    console.log('- Batch calculation: ' + batchTime.toFixed(2) + 'ms');
    console.log('- Streaming calculation: ' + streamingTime.toFixed(2) + 'ms');
    console.log('- Performance ratio: ' + (batchTime / streamingTime).toFixed(2) + 'x');
    console.log('- Final EMA difference: ' + difference.toExponential(3));
    console.log('- Accuracy: ' + ((1 - difference / batchFinal) * 100).toFixed(8) + '%');
    
    // Memory usage
    const metrics = streamingEMA.getPerformanceMetrics();
    console.log('- Updates per second: ' + metrics.updatesPerSecond.toFixed(0));
    console.log('- Average update time: ' + metrics.averageUpdateTime.toFixed(3) + 'μs');
    console.log('- Memory usage: ' + (metrics.memoryUsage / 1024).toFixed(2) + ' KB');
  }

  /**
   * Demonstrates advanced features and error handling
   */
  static demonstrateAdvancedFeatures(): void {
    console.log('\n=== Advanced Features Demo ===\n');
    
    const prices = this.generateSamplePriceData(100);
    
    // MACD-style convergence/divergence
    console.log('1. EMA Convergence/Divergence (MACD-style):');
    const macdLine = EMAService.calculateEMAConvergenceDivergence(prices, 12, 26);
    console.log(`   Last 5 MACD values: ${macdLine.slice(-5).map(v => v.toFixed(4)).join(', ')}`);
    
    // Custom smoothing factor
    console.log('\n2. Custom smoothing factor:');
    const customResult = EMAService.computeEMA(prices, { period: 20, smoothingFactor: 0.15 });
    const standardResult = EMAService.computeEMA(prices, { period: 20 });
    console.log(`   Custom (α=0.15): ${customResult.values[customResult.values.length - 1].toFixed(4)}`);
    console.log(`   Standard: ${standardResult.values[standardResult.values.length - 1].toFixed(4)}`);
    
    // Error handling demonstration
    console.log('\n3. Error handling:');
    try {
      EMAService.computeEMA([1, 2], { period: 5 });
    } catch (error) {
      console.log(`   Caught expected error: ${(error as Error).message}`);
    }
    
    // State management
    console.log('\n4. Streaming state management:');
    const ema = new StreamingEMA({ period: 10 });
    ema.update(100);
    ema.update(105);
    
    const stateBefore = ema.getState();
    console.log(`   State before reset: count=${stateBefore.count}, value=${stateBefore.currentValue.toFixed(2)}`);
    
    ema.reset();
    const stateAfter = ema.getState();
    console.log(`   State after reset: count=${stateAfter.count}, value=${stateAfter.currentValue.toFixed(2)}`);
  }

  /**
   * Real-world trading scenario simulation
   */
  static simulateTradingScenario(): void {
    console.log('\n=== Trading Scenario Simulation ===\n');
    
    // Simulate different market conditions
    const scenarios = [
      { name: 'Bull Market', trend: 0.002, volatility: 0.015 },
      { name: 'Bear Market', trend: -0.002, volatility: 0.015 },
      { name: 'Sideways High Vol', trend: 0, volatility: 0.03 },
      { name: 'Sideways Low Vol', trend: 0, volatility: 0.005 }
    ];
    
    for (const scenario of scenarios) {
      console.log(`\n${scenario.name}:`);
      
      const prices = this.generateMarketData(200, 100, scenario.trend, scenario.volatility);
      const multiEMA = new MultiPeriodStreamingEMA([20, 50, 200]);
      
      // Process all prices
      for (const price of prices) {
        multiEMA.updateAll(price);
      }
      
      const finalValues = multiEMA.getCurrentValues();
      const ema20 = finalValues.get(20)!;
      const ema50 = finalValues.get(50)!;
      const ema200 = finalValues.get(200)!;
      
      // Determine trend based on EMA alignment
      let trendStatus = 'Neutral';
      if (ema20 > ema50 && ema50 > ema200) {
        trendStatus = 'Strong Bullish';
      } else if (ema20 < ema50 && ema50 < ema200) {
        trendStatus = 'Strong Bearish';
      } else if (ema20 > ema50) {
        trendStatus = 'Weak Bullish';
      } else if (ema20 < ema50) {
        trendStatus = 'Weak Bearish';
      }
      
      console.log(`   EMA20: ${ema20.toFixed(2)}`);
      console.log(`   EMA50: ${ema50.toFixed(2)}`);
      console.log(`   EMA200: ${ema200.toFixed(2)}`);
      console.log(`   Trend: ${trendStatus}`);
      console.log(`   Final Price: ${prices[prices.length - 1].toFixed(2)}`);
    }
  }

  /**
   * Generates market data with specified characteristics
   */
  private static generateMarketData(
    length: number, 
    startPrice: number, 
    trend: number, 
    volatility: number
  ): number[] {
    const prices: number[] = [];
    let currentPrice = startPrice;
    
    for (let i = 0; i < length; i++) {
      const randomChange = (Math.random() - 0.5) * volatility * currentPrice;
      const trendChange = trend * currentPrice;
      currentPrice += randomChange + trendChange;
      prices.push(Number(currentPrice.toFixed(2)));
    }
    
    return prices;
  }

  /**
   * Runs all demonstrations
   */
  static runAllDemos(): void {
    console.log('🚀 EMA Service Comprehensive Demo');
    console.log('=====================================');
    
    this.demonstrateBatchEMA();
    this.demonstrateStreamingEMA();
    this.demonstrateMultiPeriodEMA();
    this.runPerformanceBenchmark();
    this.demonstrateAdvancedFeatures();
    this.simulateTradingScenario();
    
    console.log('\n✅ Demo completed successfully!');
    console.log('\nKey Features Demonstrated:');
    console.log('• Batch EMA calculation with validation');
    console.log('• O(1) streaming updates for real-time data');
    console.log('• Multi-period EMA management');
    console.log('• Performance benchmarking');
    console.log('• Error handling and edge cases');
    console.log('• Real-world trading scenarios');
    console.log('• MACD-style convergence/divergence');
    console.log('• Custom smoothing factors');
    console.log('• State management and configuration');
  }
}

// Export for easy usage
export default EMADemo;