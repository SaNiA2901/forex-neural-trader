/**
 * Cache Manager for Technical Indicators
 * 
 * High-performance caching system with TTL, size limits, and automatic cleanup.
 * Optimized for technical indicator calculations with memory management.
 * 
 * @author Trading System Core
 * @version 1.0.0
 */

import { CacheEntry } from './types';

/**
 * Configuration for cache manager
 */
export interface CacheConfig {
  /** Maximum number of entries to store */
  maxSize: number;
  /** Default TTL in milliseconds */
  defaultTTL: number;
  /** Whether caching is enabled */
  enabled: boolean;
  /** Cleanup interval in milliseconds */
  cleanupInterval?: number;
  /** Memory limit in bytes */
  memoryLimit?: number;
}

/**
 * Cache statistics
 */
export interface CacheStats {
  /** Total number of entries */
  size: number;
  /** Cache hit count */
  hits: number;
  /** Cache miss count */
  misses: number;
  /** Cache hit ratio */
  hitRatio: number;
  /** Total memory usage in bytes */
  memoryUsage: number;
  /** Number of expired entries cleaned up */
  cleanupCount: number;
  /** Last cleanup timestamp */
  lastCleanup: number;
}

/**
 * High-performance cache manager with automatic cleanup and memory management
 */
export class CacheManager {
  private cache = new Map<string, CacheEntry>();
  private config: Required<CacheConfig>;
  private stats: CacheStats;
  private cleanupTimer?: NodeJS.Timeout;
  private accessOrder = new Map<string, number>(); // For LRU eviction
  private accessCounter = 0;

  constructor(config: CacheConfig) {
    this.config = {
      maxSize: config.maxSize,
      defaultTTL: config.defaultTTL,
      enabled: config.enabled,
      cleanupInterval: config.cleanupInterval || 60000, // 1 minute default
      memoryLimit: config.memoryLimit || 50 * 1024 * 1024 // 50MB default
    };

    this.stats = {
      size: 0,
      hits: 0,
      misses: 0,
      hitRatio: 0,
      memoryUsage: 0,
      cleanupCount: 0,
      lastCleanup: Date.now()
    };

    // Start automatic cleanup
    if (this.config.enabled && this.config.cleanupInterval > 0) {
      this.startCleanupTimer();
    }
  }

  /**
   * Get value from cache
   * @param key - Cache key
   * @returns Cached value or undefined
   */
  get<T>(key: string): T | undefined {
    if (!this.config.enabled) return undefined;

    const entry = this.cache.get(key);
    
    if (!entry) {
      this.stats.misses++;
      this.updateHitRatio();
      return undefined;
    }

    // Check if entry has expired
    if (Date.now() > entry.timestamp + entry.ttl) {
      this.cache.delete(key);
      this.accessOrder.delete(key);
      this.updateCacheSize();
      this.stats.misses++;
      this.updateHitRatio();
      return undefined;
    }

    // Update access order for LRU
    this.accessOrder.set(key, ++this.accessCounter);
    
    this.stats.hits++;
    this.updateHitRatio();
    
    return entry.result as T;
  }

  /**
   * Set value in cache
   * @param key - Cache key
   * @param value - Value to cache
   * @param ttl - Time to live in milliseconds (optional)
   */
  set<T>(key: string, value: T, ttl?: number): void {
    if (!this.config.enabled) return;

    const entryTTL = ttl || this.config.defaultTTL;
    const estimatedSize = this.estimateSize(value);
    
    const entry: CacheEntry<T> = {
      result: value,
      timestamp: Date.now(),
      ttl: entryTTL,
      key,
      size: estimatedSize
    };

    // Check if we need to make room
    this.ensureCapacity(estimatedSize);

    // Store the entry
    this.cache.set(key, entry);
    this.accessOrder.set(key, ++this.accessCounter);
    
    this.updateCacheSize();
  }

  /**
   * Delete value from cache
   * @param key - Cache key
   * @returns Whether the key was deleted
   */
  delete(key: string): boolean {
    const deleted = this.cache.delete(key);
    if (deleted) {
      this.accessOrder.delete(key);
      this.updateCacheSize();
    }
    return deleted;
  }

  /**
   * Check if key exists in cache
   * @param key - Cache key
   */
  has(key: string): boolean {
    if (!this.config.enabled) return false;
    
    const entry = this.cache.get(key);
    if (!entry) return false;
    
    // Check if expired
    if (Date.now() > entry.timestamp + entry.ttl) {
      this.cache.delete(key);
      this.accessOrder.delete(key);
      this.updateCacheSize();
      return false;
    }
    
    return true;
  }

  /**
   * Clear all cached entries
   */
  clear(): void {
    this.cache.clear();
    this.accessOrder.clear();
    this.updateCacheSize();
    this.accessCounter = 0;
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    return { ...this.stats };
  }

  /**
   * Manually trigger cleanup of expired entries
   * @returns Number of entries cleaned up
   */
  cleanup(): number {
    const now = Date.now();
    let cleanedCount = 0;

    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.timestamp + entry.ttl) {
        this.cache.delete(key);
        this.accessOrder.delete(key);
        cleanedCount++;
      }
    }

    this.stats.cleanupCount += cleanedCount;
    this.stats.lastCleanup = now;
    this.updateCacheSize();

    return cleanedCount;
  }

  /**
   * Get cache keys matching a pattern
   * @param pattern - Regular expression pattern
   */
  getKeys(pattern?: RegExp): string[] {
    const keys = Array.from(this.cache.keys());
    return pattern ? keys.filter(key => pattern.test(key)) : keys;
  }

  /**
   * Get cache entries matching a pattern
   * @param pattern - Regular expression pattern
   */
  getEntries<T>(pattern?: RegExp): Array<{ key: string; value: T }> {
    const entries: Array<{ key: string; value: T }> = [];
    
    for (const [key, entry] of this.cache.entries()) {
      if (!pattern || pattern.test(key)) {
        // Check if entry is still valid
        if (Date.now() <= entry.timestamp + entry.ttl) {
          entries.push({ key, value: entry.result as T });
        }
      }
    }
    
    return entries;
  }

  /**
   * Destroy cache manager and cleanup resources
   */
  destroy(): void {
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = undefined;
    }
    this.clear();
  }

  /**
   * Update cache configuration
   * @param newConfig - New configuration options
   */
  updateConfig(newConfig: Partial<CacheConfig>): void {
    this.config = { ...this.config, ...newConfig };
    
    // Restart cleanup timer if interval changed
    if (newConfig.cleanupInterval !== undefined) {
      if (this.cleanupTimer) {
        clearInterval(this.cleanupTimer);
      }
      if (this.config.enabled && this.config.cleanupInterval > 0) {
        this.startCleanupTimer();
      }
    }
    
    // If disabled, clear cache
    if (newConfig.enabled === false) {
      this.clear();
    }
  }

  // Private methods

  /**
   * Start the automatic cleanup timer
   */
  private startCleanupTimer(): void {
    this.cleanupTimer = setInterval(() => {
      this.cleanup();
    }, this.config.cleanupInterval);
  }

  /**
   * Ensure cache has capacity for new entry
   * @param newEntrySize - Size of new entry
   */
  private ensureCapacity(newEntrySize: number): void {
    // Check memory limit
    if (this.stats.memoryUsage + newEntrySize > this.config.memoryLimit) {
      this.evictByMemory(newEntrySize);
    }
    
    // Check size limit
    if (this.cache.size >= this.config.maxSize) {
      this.evictByCount();
    }
  }

  /**
   * Evict entries to free up memory
   * @param requiredSpace - Space needed for new entry
   */
  private evictByMemory(requiredSpace: number): void {
    const targetMemory = this.config.memoryLimit - requiredSpace;
    
    // Get entries sorted by access time (LRU)
    const sortedEntries = Array.from(this.cache.entries()).sort((a, b) => {
      const accessA = this.accessOrder.get(a[0]) || 0;
      const accessB = this.accessOrder.get(b[0]) || 0;
      return accessA - accessB;
    });
    
    // Evict oldest entries until we have enough space
    for (const [key, entry] of sortedEntries) {
      if (this.stats.memoryUsage <= targetMemory) break;
      
      this.cache.delete(key);
      this.accessOrder.delete(key);
    }
    
    this.updateCacheSize();
  }

  /**
   * Evict entries by count (LRU)
   */
  private evictByCount(): void {
    const targetSize = Math.floor(this.config.maxSize * 0.8); // Evict to 80% capacity
    
    // Get entries sorted by access time (LRU)
    const sortedEntries = Array.from(this.cache.entries()).sort((a, b) => {
      const accessA = this.accessOrder.get(a[0]) || 0;
      const accessB = this.accessOrder.get(b[0]) || 0;
      return accessA - accessB;
    });
    
    // Evict oldest entries
    const entriesToEvict = this.cache.size - targetSize;
    for (let i = 0; i < entriesToEvict && i < sortedEntries.length; i++) {
      const key = sortedEntries[i][0];
      this.cache.delete(key);
      this.accessOrder.delete(key);
    }
    
    this.updateCacheSize();
  }

  /**
   * Update cache size statistics
   */
  private updateCacheSize(): void {
    this.stats.size = this.cache.size;
    this.stats.memoryUsage = Array.from(this.cache.values())
      .reduce((total, entry) => total + entry.size, 0);
  }

  /**
   * Update hit ratio statistics
   */
  private updateHitRatio(): void {
    const total = this.stats.hits + this.stats.misses;
    this.stats.hitRatio = total > 0 ? this.stats.hits / total : 0;
  }

  /**
   * Estimate size of a value in bytes
   * @param value - Value to estimate
   */
  private estimateSize(value: any): number {
    if (value === null || value === undefined) return 8;
    
    if (typeof value === 'boolean') return 4;
    if (typeof value === 'number') return 8;
    if (typeof value === 'string') return value.length * 2; // UTF-16
    
    if (Array.isArray(value)) {
      return value.reduce((size, item) => size + this.estimateSize(item), 24); // Array overhead
    }
    
    if (typeof value === 'object') {
      const jsonString = JSON.stringify(value);
      return jsonString.length * 2 + 24; // Object overhead
    }
    
    return 24; // Default overhead
  }
}