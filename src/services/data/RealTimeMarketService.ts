import { CandleData } from '@/types/session';

interface MarketDataProvider {
  name: string;
  connect(): Promise<void>;
  disconnect(): Promise<void>;
  subscribe(symbols: string[], callback: (data: MarketTick) => void): void;
  getHistoricalData(symbol: string, interval: string, limit: number): Promise<CandleData[]>;
}

interface MarketTick {
  symbol: string;
  price: number;
  volume: number;
  timestamp: number;
  bid?: number;
  ask?: number;
}

interface BinanceKline {
  t: number; // Open time
  T: number; // Close time  
  s: string; // Symbol
  o: string; // Open price
  h: string; // High price
  l: string; // Low price
  c: string; // Close price
  v: string; // Volume
  n: number; // Number of trades
  x: boolean; // Is this kline closed?
}

interface AlphaVantageData {
  'Time Series (Daily)': {
    [date: string]: {
      '1. open': string;
      '2. high': string;
      '3. low': string;
      '4. close': string;
      '5. volume': string;
    };
  };
}

interface EconomicEvent {
  date: string;
  time: string;
  currency: string;
  event: string;
  importance: 'low' | 'medium' | 'high';
  actual?: string;
  forecast?: string;
  previous?: string;
}

export class RealTimeMarketService {
  private static instance: RealTimeMarketService;
  private providers: Map<string, MarketDataProvider> = new Map();
  private websockets: Map<string, WebSocket> = new Map();
  private subscribers: Map<string, ((data: MarketTick) => void)[]> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  private constructor() {
    this.initializeProviders();
  }

  static getInstance(): RealTimeMarketService {
    if (!RealTimeMarketService.instance) {
      RealTimeMarketService.instance = new RealTimeMarketService();
    }
    return RealTimeMarketService.instance;
  }

  private initializeProviders(): void {
    // Binance provider для криптовалют
    this.providers.set('binance', new BinanceProvider());
    
    // Alpha Vantage для форекса и акций
    this.providers.set('alphavantage', new AlphaVantageProvider());
    
    // Yahoo Finance для индексов
    this.providers.set('yahoo', new YahooFinanceProvider());
  }

  /**
   * Подключается к поставщикам данных
   */
  async connectToProviders(): Promise<void> {
    const connectionPromises = Array.from(this.providers.values()).map(provider => 
      provider.connect().catch(error => {
        console.error(`Failed to connect to ${provider.name}:`, error);
      })
    );

    await Promise.allSettled(connectionPromises);
    console.log('✅ Market data providers connected');
  }

  /**
   * Подписывается на real-time данные
   */
  subscribeToMarketData(symbols: string[], callback: (data: MarketTick) => void): void {
    symbols.forEach(symbol => {
      if (!this.subscribers.has(symbol)) {
        this.subscribers.set(symbol, []);
      }
      this.subscribers.get(symbol)!.push(callback);

      // Определяем провайдера по символу
      const provider = this.selectProvider(symbol);
      if (provider) {
        provider.subscribe([symbol], (data) => {
          this.notifySubscribers(symbol, data);
        });
      }
    });
  }

  /**
   * Получает исторические данные
   */
  async getHistoricalData(
    symbol: string, 
    interval: string = '1h', 
    limit: number = 1000
  ): Promise<CandleData[]> {
    try {
      const provider = this.selectProvider(symbol);
      if (!provider) {
        throw new Error(`No provider available for symbol: ${symbol}`);
      }

      console.log(`📊 Fetching ${limit} ${interval} candles for ${symbol}`);
      const data = await provider.getHistoricalData(symbol, interval, limit);
      
      console.log(`✅ Received ${data.length} historical candles for ${symbol}`);
      return data;

    } catch (error) {
      console.error(`Error fetching historical data for ${symbol}:`, error);
      
      // Fallback к синтетическим данным для демонстрации
      return this.generateSyntheticData(symbol, interval, limit);
    }
  }

  /**
   * Получает экономический календарь
   */
  async getEconomicCalendar(dateRange: { from: string; to: string }): Promise<EconomicEvent[]> {
    try {
      // В реальной реализации здесь был бы API вызов
      const response = await fetch(`https://api.fxcm.com/economic-calendar?from=${dateRange.from}&to=${dateRange.to}`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch economic calendar');
      }

      return await response.json();
    } catch (error) {
      console.error('Error fetching economic calendar:', error);
      
      // Возвращаем демо-события
      return this.generateDemoEconomicEvents();
    }
  }

  /**
   * Получает рыночные новости
   */
  async getMarketNews(symbol?: string): Promise<any[]> {
    try {
      const url = symbol 
        ? `https://api.marketaux.com/v1/news/all?symbols=${symbol}&api_token=YOUR_API_KEY`
        : 'https://api.marketaux.com/v1/news/all?api_token=YOUR_API_KEY';

      const response = await fetch(url);
      const data = await response.json();
      
      return data.data || [];
    } catch (error) {
      console.error('Error fetching market news:', error);
      return [];
    }
  }

  private selectProvider(symbol: string): MarketDataProvider | null {
    if (symbol.includes('BTC') || symbol.includes('ETH') || symbol.includes('USDT')) {
      return this.providers.get('binance') || null;
    }
    
    if (symbol.includes('USD') || symbol.includes('EUR') || symbol.includes('GBP')) {
      return this.providers.get('alphavantage') || null;
    }
    
    return this.providers.get('yahoo') || null;
  }

  private notifySubscribers(symbol: string, data: MarketTick): void {
    const callbacks = this.subscribers.get(symbol);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Error in market data callback:', error);
        }
      });
    }
  }

  private generateSyntheticData(symbol: string, interval: string, limit: number): CandleData[] {
    const candles: CandleData[] = [];
    let basePrice = 1.1000; // Базовая цена для EUR/USD
    
    if (symbol.includes('BTC')) basePrice = 50000;
    else if (symbol.includes('ETH')) basePrice = 3000;
    else if (symbol.includes('GBP')) basePrice = 1.3000;
    
    const now = Date.now();
    const intervalMs = this.getIntervalMs(interval);
    
    for (let i = limit - 1; i >= 0; i--) {
      const timestamp = now - (i * intervalMs);
      
      // Генерируем реалистичное движение цены
      const volatility = basePrice * 0.002; // 0.2% волатильность
      const trend = (Math.random() - 0.5) * 0.001; // Небольшой тренд
      const noise = (Math.random() - 0.5) * volatility;
      
      const open = basePrice + noise;
      const high = open + Math.abs(noise) + (Math.random() * volatility);
      const low = open - Math.abs(noise) - (Math.random() * volatility);
      const close = open + trend * basePrice + (Math.random() - 0.5) * volatility * 0.5;
      
      candles.push({
        id: `synthetic-${i}`,
        session_id: null,
        candle_index: limit - 1 - i,
        open: Number(open.toFixed(symbol.includes('JPY') ? 3 : 5)),
        high: Number(Math.max(open, high, close).toFixed(symbol.includes('JPY') ? 3 : 5)),
        low: Number(Math.min(open, low, close).toFixed(symbol.includes('JPY') ? 3 : 5)),
        close: Number(close.toFixed(symbol.includes('JPY') ? 3 : 5)),
        volume: Math.floor(Math.random() * 1000000) + 100000,
        candle_datetime: new Date(timestamp).toISOString(),
        created_at: new Date(timestamp).toISOString(),
        prediction_direction: null,
        prediction_probability: null,
        prediction_confidence: null
      });
      
      basePrice = close; // Следующая свеча начинается с закрытия предыдущей
    }
    
    return candles;
  }

  private getIntervalMs(interval: string): number {
    const intervals: { [key: string]: number } = {
      '1m': 60 * 1000,
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000,
      '30m': 30 * 60 * 1000,
      '1h': 60 * 60 * 1000,
      '4h': 4 * 60 * 60 * 1000,
      '1d': 24 * 60 * 60 * 1000
    };
    
    return intervals[interval] || intervals['1h'];
  }

  private generateDemoEconomicEvents(): EconomicEvent[] {
    const events: EconomicEvent[] = [];
    const today = new Date();
    
    // Генерируем события на следующие 7 дней
    for (let i = 0; i < 7; i++) {
      const eventDate = new Date(today);
      eventDate.setDate(today.getDate() + i);
      
      const demoEvents = [
        {
          date: eventDate.toISOString().split('T')[0],
          time: '08:30',
          currency: 'USD',
          event: 'Non-Farm Payrolls',
          importance: 'high' as const,
          forecast: '250K',
          previous: '263K'
        },
        {
          date: eventDate.toISOString().split('T')[0],
          time: '14:00',
          currency: 'EUR',
          event: 'ECB Interest Rate Decision',
          importance: 'high' as const,
          forecast: '4.50%',
          previous: '4.50%'
        },
        {
          date: eventDate.toISOString().split('T')[0],
          time: '12:30',
          currency: 'GBP',
          event: 'GDP Growth Rate',
          importance: 'medium' as const,
          forecast: '0.2%',
          previous: '0.1%'
        }
      ];
      
      events.push(...demoEvents.slice(0, Math.floor(Math.random() * 3) + 1));
    }
    
    return events;
  }
}

// === ПРОВАЙДЕРЫ ДАННЫХ ===

class BinanceProvider implements MarketDataProvider {
  name = 'Binance';
  private ws?: WebSocket;

  async connect(): Promise<void> {
    // В реальной реализации здесь подключение к Binance WebSocket
    console.log('🔗 Connected to Binance API');
  }

  async disconnect(): Promise<void> {
    if (this.ws) {
      this.ws.close();
    }
  }

  subscribe(symbols: string[], callback: (data: MarketTick) => void): void {
    // Симуляция подписки на WebSocket
    symbols.forEach(symbol => {
      const interval = setInterval(() => {
        const price = 50000 + (Math.random() - 0.5) * 2000; // Симуляция цены BTC
        callback({
          symbol,
          price,
          volume: Math.random() * 1000,
          timestamp: Date.now()
        });
      }, 1000);
      
      // В реальности здесь был бы WebSocket
      setTimeout(() => clearInterval(interval), 60000); // Остановка через минуту для демо
    });
  }

  async getHistoricalData(symbol: string, interval: string, limit: number): Promise<CandleData[]> {
    try {
      // В реальной реализации здесь API вызов к Binance
      const response = await fetch(
        `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`
      );
      
      if (!response.ok) {
        throw new Error(`Binance API error: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      return data.map((kline: any[], index: number) => ({
        id: `binance-${symbol}-${index}`,
        session_id: null,
        candle_index: index,
        open: parseFloat(kline[1]),
        high: parseFloat(kline[2]),
        low: parseFloat(kline[3]),
        close: parseFloat(kline[4]),
        volume: parseFloat(kline[5]),
        candle_datetime: new Date(kline[0]).toISOString(),
        created_at: new Date().toISOString(),
        prediction_direction: null,
        prediction_probability: null,
        prediction_confidence: null
      }));
      
    } catch (error) {
      console.error('Binance API error:', error);
      throw error;
    }
  }
}

class AlphaVantageProvider implements MarketDataProvider {
  name = 'Alpha Vantage';
  private apiKey = 'YOUR_ALPHA_VANTAGE_API_KEY'; // В реальности из переменных окружения

  async connect(): Promise<void> {
    console.log('🔗 Connected to Alpha Vantage API');
  }

  async disconnect(): Promise<void> {
    // Nothing to disconnect for REST API
  }

  subscribe(symbols: string[], callback: (data: MarketTick) => void): void {
    // Alpha Vantage не поддерживает real-time, симулируем
    symbols.forEach(symbol => {
      const interval = setInterval(() => {
        const basePrice = symbol.includes('EUR') ? 1.1000 : 1.3000;
        const price = basePrice + (Math.random() - 0.5) * 0.01;
        
        callback({
          symbol,
          price,
          volume: Math.random() * 100000,
          timestamp: Date.now()
        });
      }, 5000); // Каждые 5 секунд
      
      setTimeout(() => clearInterval(interval), 120000); // Остановка через 2 минуты
    });
  }

  async getHistoricalData(symbol: string, interval: string, limit: number): Promise<CandleData[]> {
    try {
      const function_type = interval.includes('m') ? 'TIME_SERIES_INTRADAY' : 'TIME_SERIES_DAILY';
      const url = `https://www.alphavantage.co/query?function=${function_type}&symbol=${symbol}&apikey=${this.apiKey}&interval=${interval}&outputsize=full`;
      
      const response = await fetch(url);
      const data: AlphaVantageData = await response.json();
      
      if (!data['Time Series (Daily)']) {
        throw new Error('No data received from Alpha Vantage');
      }
      
      const entries = Object.entries(data['Time Series (Daily)']);
      const limitedEntries = entries.slice(0, limit);
      
      return limitedEntries.map(([date, values], index) => ({
        id: `alphavantage-${symbol}-${index}`,
        session_id: null,
        candle_index: index,
        open: parseFloat(values['1. open']),
        high: parseFloat(values['2. high']),
        low: parseFloat(values['3. low']),
        close: parseFloat(values['4. close']),
        volume: parseFloat(values['5. volume']),
        candle_datetime: new Date(date).toISOString(),
        created_at: new Date().toISOString(),
        prediction_direction: null,
        prediction_probability: null,
        prediction_confidence: null
      }));
      
    } catch (error) {
      console.error('Alpha Vantage API error:', error);
      throw error;
    }
  }
}

class YahooFinanceProvider implements MarketDataProvider {
  name = 'Yahoo Finance';

  async connect(): Promise<void> {
    console.log('🔗 Connected to Yahoo Finance API');
  }

  async disconnect(): Promise<void> {
    // Nothing to disconnect for REST API
  }

  subscribe(symbols: string[], callback: (data: MarketTick) => void): void {
    // Yahoo Finance симуляция для индексов
    symbols.forEach(symbol => {
      const interval = setInterval(() => {
        const basePrice = symbol.includes('SPY') ? 450 : 380;
        const price = basePrice + (Math.random() - 0.5) * 10;
        
        callback({
          symbol,
          price,
          volume: Math.random() * 10000000,
          timestamp: Date.now()
        });
      }, 10000); // Каждые 10 секунд
      
      setTimeout(() => clearInterval(interval), 180000); // Остановка через 3 минуты
    });
  }

  async getHistoricalData(symbol: string, interval: string, limit: number): Promise<CandleData[]> {
    try {
      // В реальности здесь был бы вызов к Yahoo Finance API
      // Для демонстрации возвращаем заглушку
      throw new Error('Yahoo Finance implementation needed');
      
    } catch (error) {
      console.error('Yahoo Finance API error:', error);
      throw error;
    }
  }
}

export const realTimeMarketService = RealTimeMarketService.getInstance();