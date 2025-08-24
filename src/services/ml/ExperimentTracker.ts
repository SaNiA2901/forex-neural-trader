import { TrainingExperiment, ModelMetrics } from './AdvancedMLTrainingService';

export interface ExperimentComparison {
  experiments: TrainingExperiment[];
  bestByMetric: Record<string, TrainingExperiment>;
  rankings: ExperimentRanking[];
}

export interface ExperimentRanking {
  experiment: TrainingExperiment;
  score: number;
  rank: number;
}

export interface ExperimentFilter {
  status?: TrainingExperiment['status'][];
  modelTypes?: string[];
  minAccuracy?: number;
  minSharpeRatio?: number;
  dateRange?: {
    start: Date;
    end: Date;
  };
}

export class ExperimentTracker {
  private experiments: Map<string, TrainingExperiment> = new Map();
  private experimentHistory: TrainingExperiment[] = [];

  /**
   * Add experiment to tracking
   */
  addExperiment(experiment: TrainingExperiment): void {
    this.experiments.set(experiment.id, experiment);
    this.experimentHistory.push({ ...experiment });
  }

  /**
   * Update experiment metrics
   */
  updateExperiment(experimentId: string, updates: Partial<TrainingExperiment>): void {
    const experiment = this.experiments.get(experimentId);
    if (experiment) {
      Object.assign(experiment, updates);
    }
  }

  /**
   * Get experiment by ID
   */
  getExperiment(id: string): TrainingExperiment | undefined {
    return this.experiments.get(id);
  }

  /**
   * Get all experiments with optional filtering
   */
  getExperiments(filter?: ExperimentFilter): TrainingExperiment[] {
    let experiments = Array.from(this.experiments.values());

    if (!filter) return experiments;

    if (filter.status) {
      experiments = experiments.filter(exp => filter.status!.includes(exp.status));
    }

    if (filter.modelTypes) {
      experiments = experiments.filter(exp => filter.modelTypes!.includes(exp.model));
    }

    if (filter.minAccuracy !== undefined) {
      experiments = experiments.filter(exp => exp.metrics.accuracy >= filter.minAccuracy!);
    }

    if (filter.minSharpeRatio !== undefined) {
      experiments = experiments.filter(exp => exp.metrics.sharpeRatio >= filter.minSharpeRatio!);
    }

    if (filter.dateRange) {
      experiments = experiments.filter(exp => 
        exp.startTime >= filter.dateRange!.start && exp.startTime <= filter.dateRange!.end
      );
    }

    return experiments;
  }

  /**
   * Compare multiple experiments
   */
  compareExperiments(experimentIds: string[]): ExperimentComparison {
    const experiments = experimentIds
      .map(id => this.experiments.get(id))
      .filter(exp => exp !== undefined) as TrainingExperiment[];

    if (experiments.length === 0) {
      return {
        experiments: [],
        bestByMetric: {},
        rankings: []
      };
    }

    // Find best experiment for each metric
    const bestByMetric: Record<string, TrainingExperiment> = {
      accuracy: this.getBestByMetric(experiments, 'accuracy'),
      precision: this.getBestByMetric(experiments, 'precision'),
      recall: this.getBestByMetric(experiments, 'recall'),
      f1Score: this.getBestByMetric(experiments, 'f1Score'),
      sharpeRatio: this.getBestByMetric(experiments, 'sharpeRatio'),
      winRate: this.getBestByMetric(experiments, 'winRate'),
      avgReturn: this.getBestByMetric(experiments, 'avgReturn')
    };

    // Calculate overall rankings
    const rankings = this.calculateRankings(experiments);

    return {
      experiments,
      bestByMetric,
      rankings
    };
  }

  private getBestByMetric(experiments: TrainingExperiment[], metric: keyof ModelMetrics): TrainingExperiment {
    return experiments.reduce((best, current) => {
      if (metric === 'maxDrawdown' || metric === 'volatility') {
        // Lower is better for these metrics
        return current.metrics[metric] < best.metrics[metric] ? current : best;
      } else {
        // Higher is better for other metrics
        return current.metrics[metric] > best.metrics[metric] ? current : best;
      }
    });
  }

  /**
   * Calculate composite score for experiment ranking
   */
  private calculateCompositeScore(experiment: TrainingExperiment): number {
    const metrics = experiment.metrics;
    
    // Weighted scoring system
    const weights = {
      accuracy: 0.2,
      sharpeRatio: 0.25,
      maxDrawdown: 0.15, // Negative weight (lower is better)
      winRate: 0.15,
      f1Score: 0.15,
      avgReturn: 0.1
    };

    const score = 
      metrics.accuracy * weights.accuracy +
      Math.max(0, metrics.sharpeRatio) * weights.sharpeRatio +
      (1 - Math.min(1, metrics.maxDrawdown)) * weights.maxDrawdown + // Inverted
      metrics.winRate * weights.winRate +
      metrics.f1Score * weights.f1Score +
      Math.max(0, metrics.avgReturn) * weights.avgReturn;

    return Math.max(0, Math.min(1, score)); // Normalize to 0-1
  }

  /**
   * Calculate rankings for experiments
   */
  private calculateRankings(experiments: TrainingExperiment[]): ExperimentRanking[] {
    const scoredExperiments = experiments.map(exp => ({
      experiment: exp,
      score: this.calculateCompositeScore(exp)
    }));

    // Sort by score descending
    scoredExperiments.sort((a, b) => b.score - a.score);

    // Assign ranks
    return scoredExperiments.map((item, index) => ({
      ...item,
      rank: index + 1
    }));
  }

  /**
   * Get experiment statistics
   */
  getStatistics(): {
    total: number;
    completed: number;
    running: number;
    failed: number;
    avgAccuracy: number;
    avgSharpeRatio: number;
    bestOverallScore: number;
  } {
    const experiments = Array.from(this.experiments.values());
    const completed = experiments.filter(exp => exp.status === 'completed');

    const avgAccuracy = completed.length > 0 
      ? completed.reduce((sum, exp) => sum + exp.metrics.accuracy, 0) / completed.length 
      : 0;

    const avgSharpeRatio = completed.length > 0
      ? completed.reduce((sum, exp) => sum + exp.metrics.sharpeRatio, 0) / completed.length
      : 0;

    const bestOverallScore = completed.length > 0
      ? Math.max(...completed.map(exp => this.calculateCompositeScore(exp)))
      : 0;

    return {
      total: experiments.length,
      completed: experiments.filter(exp => exp.status === 'completed').length,
      running: experiments.filter(exp => exp.status === 'running').length,
      failed: experiments.filter(exp => exp.status === 'failed').length,
      avgAccuracy,
      avgSharpeRatio,
      bestOverallScore
    };
  }

  /**
   * Export experiments to JSON
   */
  exportExperiments(experimentIds?: string[]): string {
    const experiments = experimentIds
      ? experimentIds.map(id => this.experiments.get(id)).filter(Boolean)
      : Array.from(this.experiments.values());

    return JSON.stringify(experiments, null, 2);
  }

  /**
   * Import experiments from JSON
   */
  importExperiments(jsonData: string): number {
    try {
      const experiments = JSON.parse(jsonData) as TrainingExperiment[];
      let imported = 0;

      for (const exp of experiments) {
        if (exp.id && !this.experiments.has(exp.id)) {
          this.experiments.set(exp.id, exp);
          imported++;
        }
      }

      return imported;
    } catch (error) {
      throw new Error(`Failed to import experiments: ${error}`);
    }
  }

  /**
   * Clean up old experiments (keep only recent ones)
   */
  cleanup(keepDays: number = 30): number {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - keepDays);

    const toDelete: string[] = [];
    
    for (const [id, experiment] of this.experiments.entries()) {
      if (experiment.startTime < cutoffDate && experiment.status !== 'running') {
        toDelete.push(id);
      }
    }

    toDelete.forEach(id => this.experiments.delete(id));
    
    return toDelete.length;
  }

  /**
   * Get performance trends over time
   */
  getPerformanceTrends(): {
    date: string;
    accuracy: number;
    sharpeRatio: number;
    experiments: number;
  }[] {
    const completed = Array.from(this.experiments.values())
      .filter(exp => exp.status === 'completed')
      .sort((a, b) => a.startTime.getTime() - b.startTime.getTime());

    const trends: Map<string, {
      accuracy: number[];
      sharpeRatio: number[];
      count: number;
    }> = new Map();

    for (const exp of completed) {
      const dateKey = exp.startTime.toISOString().split('T')[0];
      
      if (!trends.has(dateKey)) {
        trends.set(dateKey, { accuracy: [], sharpeRatio: [], count: 0 });
      }
      
      const trend = trends.get(dateKey)!;
      trend.accuracy.push(exp.metrics.accuracy);
      trend.sharpeRatio.push(exp.metrics.sharpeRatio);
      trend.count++;
    }

    return Array.from(trends.entries()).map(([date, data]) => ({
      date,
      accuracy: data.accuracy.reduce((a, b) => a + b, 0) / data.accuracy.length,
      sharpeRatio: data.sharpeRatio.reduce((a, b) => a + b, 0) / data.sharpeRatio.length,
      experiments: data.count
    }));
  }
}

export const experimentTracker = new ExperimentTracker();