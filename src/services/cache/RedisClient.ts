/**
 * Redis Client Configuration and Management
 * Handles connection, health monitoring, and error recovery
 */

import { createClient, RedisClientType } from 'redis';

export interface RedisConfig {
  host: string;
  port: number;
  password?: string;
  db?: number;
  retryDelayOnFailover: number;
  maxRetries: number;
  connectTimeout: number;
  lazyConnect: boolean;
}

export interface RedisMetrics {
  connectionStatus: 'connected' | 'connecting' | 'disconnected' | 'error';
  totalCommands: number;
  successfulCommands: number;
  failedCommands: number;
  averageResponseTime: number;
  lastError?: string;
  lastErrorTime?: number;
  uptime: number;
}

export class RedisClientManager {
  private client: RedisClientType;
  private config: RedisConfig;
  private metrics: RedisMetrics;
  private startTime: number;
  private responseTimeSamples: number[] = [];
  private maxSamples = 100;

  constructor(config?: Partial<RedisConfig>) {
    this.startTime = Date.now();
    this.config = {
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379'),
      password: process.env.REDIS_PASSWORD,
      db: parseInt(process.env.REDIS_DB || '0'),
      retryDelayOnFailover: 100,
      maxRetries: 3,
      connectTimeout: 10000,
      lazyConnect: true,
      ...config
    };

    this.metrics = {
      connectionStatus: 'disconnected',
      totalCommands: 0,
      successfulCommands: 0,
      failedCommands: 0,
      averageResponseTime: 0,
      uptime: 0
    };

    this.initializeClient();
  }

  private initializeClient(): void {
    try {
      this.client = createClient({
        socket: {
          host: this.config.host,
          port: this.config.port,
          connectTimeout: this.config.connectTimeout,
          reconnectStrategy: (retries: number) => {
            if (retries > this.config.maxRetries) {
              return false;
            }
            return Math.min(retries * this.config.retryDelayOnFailover, 3000);
          }
        },
        password: this.config.password,
        database: this.config.db
      });

      this.setupEventHandlers();
    } catch (error) {
      this.handleError('Client initialization failed', error);
    }
  }

  private setupEventHandlers(): void {
    this.client.on('connect', () => {
      this.metrics.connectionStatus = 'connecting';
      console.log('Redis client connecting...');
    });

    this.client.on('ready', () => {
      this.metrics.connectionStatus = 'connected';
      console.log('Redis client connected successfully');
    });

    this.client.on('error', (error) => {
      this.handleError('Redis connection error', error);
    });

    this.client.on('end', () => {
      this.metrics.connectionStatus = 'disconnected';
      console.log('Redis client disconnected');
    });
  }

  private handleError(context: string, error: any): void {
    this.metrics.connectionStatus = 'error';
    this.metrics.lastError = `${context}: ${error.message}`;
    this.metrics.lastErrorTime = Date.now();
    this.metrics.failedCommands++;
    console.error(context, error);
  }

  private recordResponseTime(duration: number): void {
    this.responseTimeSamples.push(duration);
    if (this.responseTimeSamples.length > this.maxSamples) {
      this.responseTimeSamples.shift();
    }
    
    this.metrics.averageResponseTime = 
      this.responseTimeSamples.reduce((sum, time) => sum + time, 0) / 
      this.responseTimeSamples.length;
  }

  async connect(): Promise<void> {
    if (this.metrics.connectionStatus === 'connected') {
      return;
    }

    try {
      await this.client.connect();
    } catch (error) {
      this.handleError('Failed to connect', error);
      throw error;
    }
  }

  async disconnect(): Promise<void> {
    if (this.client) {
      await this.client.quit();
    }
  }

  async executeCommand<T>(
    operation: () => Promise<T>,
    commandName: string = 'unknown'
  ): Promise<T | null> {
    const startTime = performance.now();
    this.metrics.totalCommands++;

    try {
      if (this.metrics.connectionStatus !== 'connected') {
        await this.connect();
      }

      const result = await operation();
      
      const duration = performance.now() - startTime;
      this.recordResponseTime(duration);
      this.metrics.successfulCommands++;
      
      return result;
    } catch (error) {
      this.handleError(`Command ${commandName} failed`, error);
      return null;
    }
  }

  getClient(): RedisClientType {
    return this.client;
  }

  getMetrics(): RedisMetrics {
    return {
      ...this.metrics,
      uptime: Date.now() - this.startTime
    };
  }

  async healthCheck(): Promise<boolean> {
    try {
      const result = await this.executeCommand(
        () => this.client.ping(),
        'PING'
      );
      return result === 'PONG';
    } catch {
      return false;
    }
  }

  isConnected(): boolean {
    return this.metrics.connectionStatus === 'connected';
  }
}

// Singleton instance
export const redisClient = new RedisClientManager();