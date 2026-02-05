import { MatrixFactorization } from '../ml/matrixFactorization';
import { TrackingService } from './trackingService';
import * as Redis from 'ioredis';

const DEFAULT_SEQUENCE_WINDOW = 20;
const DEFAULT_SESSION_TIMEOUT = 30 * 60 * 1000;
const RECENCY_HALF_LIFE_HOURS = 24;

interface Movie {
  id: number;
  title: string;
  genres: string[];
  directors: string[];
  actors: string[];
  releaseYear: number;
  runtime: number;
  averageRating: number;
  ratingCount: number;
  popularity: number;
  metadata?: any;
}

interface UserProfile {
  userId: string;
  ratingCount: number;
  avgRating: number;
  ratingVariance?: number;
  timeActive: number;
  engagement: number;
  genres: { [genre: string]: number };
  directors: { [director: string]: number };
  actors: { [actor: string]: number };
  lastActive: number | null;
  sessionDepth: number;
  recencyScore: number;
  recentActions: any[];
  preferences: {
    genreWeights: { [genre: string]: number };
    directorWeights: { [director: string]: number };
    actorWeights: { [actor: string]: number };
    runtimePreference: { min: number; max: number; ideal: number };
    yearPreference: { min: number; max: number };
    ratingThreshold: number;
  };
}

interface RecommendationItem {
  movieId: number;
  movie: Movie;
  score: number;
  contentScore: number;
  collaborativeScore: number;
  sequenceScore: number;
  ruleScore: number;
  source: string;
  weights: {
    content: number;
    collaborative: number;
    sequence: number;
    rule: number;
  };
  explanation?: string[];
}

interface RecommendationOptions {
  count?: number;
  excludeRated?: boolean;
  excludeWatchlist?: boolean;
  minScore?: number;
  includeExplanations?: boolean;
  diversityFactor?: number;
}

class HybridRecommendationEngine {
  private matrixFactorization: MatrixFactorization;
  private trackingService: TrackingService;
  private redis: Redis.Redis;
  private cacheTimeout: number = 300; // 5 minutes

  constructor(redis: Redis.Redis, mongodb: any) {
    this.matrixFactorization = new MatrixFactorization({
      factors: 50,
      learningRate: 0.001,
      regularization: 1e-6,
      epochs: 100,
      batchSize: 1024
    });
    this.trackingService = new TrackingService(redis, mongodb);
    this.redis = redis;
  }

  async generateRecommendations(userId: string, options: RecommendationOptions = {}): Promise<RecommendationItem[]> {
    try {
      const {
        count = 25,
        excludeRated = true,
        excludeWatchlist = true,
        minScore = 0.5,
        includeExplanations = false,
        diversityFactor = 0.3
      } = options;

      const cacheKey = `recommendations:${userId}:${JSON.stringify(options)}`;
      const cached = await this.redis.get(cacheKey);
      if (cached) {
        return JSON.parse(cached);
      }

      const userProfile = await this.getUserProfile(userId);
      const weights = this.calculateWeights(userProfile);

      const availableMovies = await this.getAvailableMovies(userId, {
        excludeRated,
        excludeWatchlist
      });

      if (availableMovies.length === 0) {
        return [];
      }

      const [contentScores, collaborativeScores, sequenceScores, ruleScores] = await Promise.all([
        this.contentBasedRecommendation(availableMovies, userProfile),
        this.collaborativeFiltering(userId, availableMovies),
        this.sequenceBasedRecommendation(availableMovies, userProfile),
        this.ruleBasedRecommendation(availableMovies, userProfile)
      ]);

      const hybridScores = this.combineScores(
        contentScores,
        collaborativeScores,
        sequenceScores,
        ruleScores,
        weights,
        includeExplanations
      );

      const diverseRecommendations = this.applyDiversityFilter(
        hybridScores,
        diversityFactor,
        userProfile
      );

      const recommendations = diverseRecommendations
        .filter(item => item.score >= minScore)
        .sort((a, b) => b.score - a.score)
        .slice(0, count);

      await this.redis.setex(cacheKey, this.cacheTimeout, JSON.stringify(recommendations));

      await this.updateRecommendationMetrics(userId, recommendations);

      return recommendations;
    } catch (error) {
      console.error('Error generating recommendations:', error);
      throw error;
    }
  }

  async contentBasedRecommendation(
    availableMovies: Movie[],
    userProfile: UserProfile
  ): Promise<Partial<RecommendationItem>[]> {
    try {
      if (userProfile.ratingCount === 0) {
        return this.popularityFallback(availableMovies, 'content-cold');
      }

      const contentScores = availableMovies.map(movie => {
        const genreScore = this.calculateGenreScore(movie, userProfile.preferences.genreWeights);
        const directorScore = this.calculateDirectorScore(movie, userProfile.preferences.directorWeights);
        const actorScore = this.calculateActorScore(movie, userProfile.preferences.actorWeights);
        const runtimeScore = this.calculateRuntimeScore(movie, userProfile.preferences.runtimePreference);
        const yearScore = this.calculateYearScore(movie, userProfile.preferences.yearPreference);

        const score = (
          genreScore * 0.4 +
          directorScore * 0.2 +
          actorScore * 0.2 +
          runtimeScore * 0.1 +
          yearScore * 0.1
        );

        return {
          movieId: movie.id,
          movie,
          score: this.normalizeScore(score * 10),
          source: 'content-based'
        };
      });

      return contentScores;
    } catch (error) {
      console.error('Error in content-based recommendation:', error);
      return [];
    }
  }

  async collaborativeFiltering(userId: string, availableMovies: Movie[]): Promise<Partial<RecommendationItem>[]> {
    try {
      const movieIds = availableMovies.map(movie => movie.id);
      const predictions = await this.matrixFactorization.predict(parseInt(userId, 10), movieIds);

      if (predictions.length > 0) {
        return predictions.map(pred => {
          const movie = availableMovies.find(m => m.id === pred.movieId);
          return {
            movieId: pred.movieId,
            movie,
            score: this.normalizeScore(pred.score),
            source: 'collaborative-matrix'
          };
        });
      }

      return this.userBasedCollaborativeFiltering(userId, availableMovies);
    } catch (error) {
      console.error('Error in collaborative filtering:', error);
      return this.userBasedCollaborativeFiltering(userId, availableMovies);
    }
  }

  async userBasedCollaborativeFiltering(userId: string, availableMovies: Movie[]): Promise<Partial<RecommendationItem>[]> {
    try {
      const similarUsers = await this.findSimilarUsers(userId);

      if (similarUsers.length === 0) {
        return this.popularityFallback(availableMovies, 'collaborative-cold');
      }

      const collaborativeScores = await Promise.all(
        availableMovies.map(async movie => {
          const score = await this.calculateCollaborativeScore(movie.id, similarUsers);
          return {
            movieId: movie.id,
            movie,
            score,
            source: 'collaborative-user'
          };
        })
      );

      return collaborativeScores;
    } catch (error) {
      console.error('Error in user-based collaborative filtering:', error);
      return [];
    }
  }

  async sequenceBasedRecommendation(availableMovies: Movie[], userProfile: UserProfile): Promise<Partial<RecommendationItem>[]> {
    try {
      const recentActions = userProfile.recentActions || [];
      if (recentActions.length === 0) {
        return this.popularityFallback(availableMovies, 'sequence-cold');
      }

      const sessionSignals = this.buildSessionSignals(recentActions);
      return availableMovies.map(movie => ({
        movieId: movie.id,
        movie,
        score: this.calculateSessionSimilarity(movie, sessionSignals),
        source: 'sequence'
      }));
    } catch (error) {
      console.error('Error in sequence recommendation:', error);
      return [];
    }
  }

  async ruleBasedRecommendation(availableMovies: Movie[], userProfile: UserProfile): Promise<Partial<RecommendationItem>[]> {
    try {
      const rulePreferences = this.extractRulePreferences(userProfile);
      if (!rulePreferences) {
        return this.popularityFallback(availableMovies, 'rule-cold');
      }

      return availableMovies.map(movie => ({
        movieId: movie.id,
        movie,
        score: this.calculateRuleScore(movie, rulePreferences),
        source: 'rule-based'
      }));
    } catch (error) {
      console.error('Error in rule-based recommendation:', error);
      return [];
    }
  }

  calculateWeights(userProfile: UserProfile): { content: number; collaborative: number; sequence: number; rule: number } {
    const ratingCount = userProfile.ratingCount || 0;
    const sessionDepth = userProfile.sessionDepth || 0;
    const recencyScore = userProfile.recencyScore || 0;

    if (ratingCount < 5) {
      return this.normalizeWeights({
        content: 0.4,
        collaborative: 0.1,
        sequence: 0.2 + recencyScore * 0.1,
        rule: 0.3
      });
    }

    if (ratingCount < 25) {
      return this.normalizeWeights({
        content: 0.35,
        collaborative: 0.25,
        sequence: 0.25 + sessionDepth * 0.05,
        rule: 0.15
      });
    }

    return this.normalizeWeights({
      content: 0.25,
      collaborative: 0.45,
      sequence: 0.2 + recencyScore * 0.1,
      rule: 0.1
    });
  }

  normalizeWeights(weights: { [key: string]: number }): { [key: string]: number } {
    const total = Object.values(weights).reduce((sum, value) => sum + value, 0) || 1;
    return Object.entries(weights).reduce((acc: { [key: string]: number }, [key, value]) => {
      acc[key] = Math.max(0, value / total);
      return acc;
    }, {});
  }

  combineScores(
    contentScores: Partial<RecommendationItem>[],
    collaborativeScores: Partial<RecommendationItem>[],
    sequenceScores: Partial<RecommendationItem>[],
    ruleScores: Partial<RecommendationItem>[],
    weights: { content: number; collaborative: number; sequence: number; rule: number },
    includeExplanations: boolean = false
  ): RecommendationItem[] {
    const movieScoreMap = new Map<number, RecommendationItem>();

    const ingestScores = (
      scores: Partial<RecommendationItem>[],
      scoreKey: 'contentScore' | 'collaborativeScore' | 'sequenceScore' | 'ruleScore'
    ) => {
      scores.forEach(item => {
        if (!item.movie) {
          return;
        }

        if (!movieScoreMap.has(item.movieId)) {
          movieScoreMap.set(item.movieId, {
            movieId: item.movieId,
            movie: item.movie,
            score: 0,
            contentScore: 0,
            collaborativeScore: 0,
            sequenceScore: 0,
            ruleScore: 0,
            source: 'hybrid',
            weights,
            explanation: includeExplanations ? [] : undefined
          });
        }

        movieScoreMap.get(item.movieId)![scoreKey] = item.score || 0;
      });
    };

    ingestScores(contentScores, 'contentScore');
    ingestScores(collaborativeScores, 'collaborativeScore');
    ingestScores(sequenceScores, 'sequenceScore');
    ingestScores(ruleScores, 'ruleScore');

    return Array.from(movieScoreMap.values()).map(item => {
      const hybridScore = (
        (item.contentScore * weights.content) +
        (item.collaborativeScore * weights.collaborative) +
        (item.sequenceScore * weights.sequence) +
        (item.ruleScore * weights.rule)
      );

      item.score = hybridScore;

      if (includeExplanations && item.explanation) {
        item.explanation = this.generateExplanations(item, weights);
      }

      return item;
    });
  }

  private generateExplanations(item: RecommendationItem, weights: any): string[] {
    const explanations: string[] = [];

    if (item.contentScore > 0.7 && weights.content > 0.3) {
      explanations.push(`Strong match with your preferences (${(item.contentScore * 100).toFixed(0)}% content similarity)`);
    }

    if (item.collaborativeScore > 0.7 && weights.collaborative > 0.3) {
      explanations.push(`Users with similar taste enjoyed this movie (${(item.collaborativeScore * 100).toFixed(0)}% collaborative score)`);
    }

    if (item.sequenceScore > 0.7 && weights.sequence > 0.2) {
      explanations.push('Trending in your recent session activity');
    }

    if (item.ruleScore > 0.6 && weights.rule > 0.1) {
      explanations.push('Matches your onboarding preferences');
    }

    return explanations;
  }

  private applyDiversityFilter(
    recommendations: RecommendationItem[],
    diversityFactor: number,
    userProfile: UserProfile
  ): RecommendationItem[] {
    if (diversityFactor === 0) {
      return recommendations;
    }

    const diverseRecommendations: RecommendationItem[] = [];
    const selectedGenres = new Set<string>();
    const selectedDirectors = new Set<string>();

    const sortedRecommendations = [...recommendations].sort((a, b) => b.score - a.score);

    for (const recommendation of sortedRecommendations) {
      const movie = recommendation.movie;
      const genreOverlap = movie.genres.some(genre => selectedGenres.has(genre));
      const directorOverlap = movie.directors.some(director => selectedDirectors.has(director));

      let diversityPenalty = 0;
      if (genreOverlap) diversityPenalty += 0.3;
      if (directorOverlap) diversityPenalty += 0.2;

      const adjustedScore = recommendation.score * (1 - (diversityPenalty * diversityFactor));
      recommendation.score = adjustedScore;

      diverseRecommendations.push(recommendation);

      movie.genres.forEach(genre => selectedGenres.add(genre));
      movie.directors.forEach(director => selectedDirectors.add(director));
    }

    return diverseRecommendations;
  }

  private calculateGenreScore(movie: Movie, genreWeights: { [genre: string]: number }): number {
    if (Object.keys(genreWeights).length === 0) return 0.5;

    const movieGenreScores = movie.genres.map(genre => genreWeights[genre] || 0);
    return movieGenreScores.length > 0
      ? movieGenreScores.reduce((sum, score) => sum + score, 0) / movieGenreScores.length
      : 0;
  }

  private calculateDirectorScore(movie: Movie, directorWeights: { [director: string]: number }): number {
    if (Object.keys(directorWeights).length === 0) return 0.5;

    const movieDirectorScores = movie.directors.map(director => directorWeights[director] || 0);
    return movieDirectorScores.length > 0
      ? Math.max(...movieDirectorScores)
      : 0;
  }

  private calculateActorScore(movie: Movie, actorWeights: { [actor: string]: number }): number {
    if (Object.keys(actorWeights).length === 0) return 0.5;

    const movieActorScores = movie.actors.map(actor => actorWeights[actor] || 0);
    return movieActorScores.length > 0
      ? movieActorScores.slice(0, 3).reduce((sum, score) => sum + score, 0) / Math.min(3, movieActorScores.length)
      : 0;
  }

  private calculateRuntimeScore(movie: Movie, runtimePref: { min: number; max: number; ideal: number }): number {
    if (movie.runtime < runtimePref.min || movie.runtime > runtimePref.max) {
      return 0.2;
    }

    const distance = Math.abs(movie.runtime - runtimePref.ideal);
    const maxDistance = Math.max(runtimePref.ideal - runtimePref.min, runtimePref.max - runtimePref.ideal);

    return 1 - (distance / maxDistance);
  }

  private calculateYearScore(movie: Movie, yearPref: { min: number; max: number }): number {
    if (movie.releaseYear < yearPref.min || movie.releaseYear > yearPref.max) {
      return 0.3;
    }
    return 1;
  }

  private calculatePopularityScore(movie: Movie): number {
    const popularityScore = movie.popularity / 100;
    const ratingScore = movie.averageRating / 10;
    const ratingCountScore = Math.log(movie.ratingCount + 1) / Math.log(10000);

    return (popularityScore * 0.4 + ratingScore * 0.4 + ratingCountScore * 0.2);
  }

  private calculateRatingVariance(ratings: Array<{ value: number }>): number {
    if (ratings.length < 2) return 0;

    const mean = ratings.reduce((sum, r) => sum + r.value, 0) / ratings.length;
    const variance = ratings.reduce((sum, r) => sum + Math.pow(r.value - mean, 2), 0) / ratings.length;

    return variance;
  }

  private normalizeScore(score: number): number {
    return Math.max(0, Math.min(1, score));
  }

  private buildSessionSignals(actions: any[]) {
    const signals = {
      genres: {} as { [key: string]: number },
      directors: {} as { [key: string]: number },
      actors: {} as { [key: string]: number },
      totalWeight: 0
    };

    actions.forEach((action, index) => {
      const metadata = action.metadata || {};
      const recencyWeight = this.calculateRecencyWeight(action.timestamp, index);
      const genres = Array.isArray(metadata.genres) ? metadata.genres : [];
      const directors = Array.isArray(metadata.directors) ? metadata.directors : [];
      const actors = Array.isArray(metadata.actors) ? metadata.actors : [];

      const actionWeight = recencyWeight * this.actionTypeBoost(action.actionType, action.value);
      signals.totalWeight += actionWeight;

      genres.forEach(genre => {
        signals.genres[genre] = (signals.genres[genre] || 0) + actionWeight;
      });

      directors.forEach(director => {
        signals.directors[director] = (signals.directors[director] || 0) + actionWeight;
      });

      actors.forEach(actor => {
        signals.actors[actor] = (signals.actors[actor] || 0) + actionWeight;
      });
    });

    return signals;
  }

  private calculateSessionSimilarity(movie: Movie, sessionSignals: any): number {
    if (!sessionSignals || sessionSignals.totalWeight === 0) {
      return 0.4;
    }

    const genreScore = this.calculatePreferenceScore(movie.genres, sessionSignals.genres);
    const directorScore = this.calculatePreferenceScore(movie.directors, sessionSignals.directors, true);
    const actorScore = this.calculatePreferenceScore(movie.actors, sessionSignals.actors);

    const score = (
      genreScore * 0.5 +
      directorScore * 0.3 +
      actorScore * 0.2
    );

    return this.normalizeScore(score * 10);
  }

  private extractRulePreferences(userProfile: UserProfile) {
    const preferences = userProfile?.preferences;
    if (!preferences || !preferences.genreWeights) {
      return null;
    }

    const sortedGenres = Object.entries(preferences.genreWeights)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 3)
      .map(([genre]) => genre);

    return {
      preferredGenres: sortedGenres,
      minRating: preferences.ratingThreshold || 6.5,
      yearRange: preferences.yearPreference,
      runtimeRange: preferences.runtimePreference
    };
  }

  private calculateRuleScore(movie: Movie, rulePreferences: any): number {
    let score = 0.4;

    if (rulePreferences.preferredGenres?.length) {
      const genreMatches = (movie.genres || []).filter((genre: string) => rulePreferences.preferredGenres.includes(genre));
      score += Math.min(0.4, genreMatches.length * 0.15);
    }

    if (rulePreferences.yearRange) {
      if (movie.releaseYear >= rulePreferences.yearRange.min && movie.releaseYear <= rulePreferences.yearRange.max) {
        score += 0.1;
      }
    }

    if (rulePreferences.runtimeRange && movie.runtime) {
      if (movie.runtime >= rulePreferences.runtimeRange.min && movie.runtime <= rulePreferences.runtimeRange.max) {
        score += 0.1;
      }
    }

    if (movie.averageRating >= rulePreferences.minRating) {
      score += 0.1;
    }

    return this.normalizeScore(score * 10);
  }

  private popularityFallback(movies: Movie[], source: string): Partial<RecommendationItem>[] {
    return movies.map(movie => ({
      movieId: movie.id,
      movie,
      score: this.calculatePopularityScore(movie),
      source
    }));
  }

  async getUserProfile(userId: string): Promise<UserProfile> {
    const ratings = await this.trackingService.getUserActions(userId, 1000, 'rate');
    const recentActions = await this.trackingService.getRecentActions(userId);

    const ratingCount = ratings.length;
    const avgRating = ratings.length > 0
      ? ratings.reduce((sum, r) => sum + r.value, 0) / ratings.length
      : 0;

    const firstRating = ratings.length > 0
      ? Math.min(...ratings.map(r => new Date(r.timestamp).getTime()))
      : Date.now();

    const timeActive = Math.floor((Date.now() - firstRating) / (1000 * 60 * 60 * 24));

    const allActions = await this.trackingService.getUserActions(userId, 1000);
    const sessions = this.groupActionsBySessions(allActions);
    const engagement = sessions.length > 0
      ? allActions.length / sessions.length
      : 0;
    const sessionDepth = sessions.length > 0
      ? Math.min(1, sessions[sessions.length - 1].length / 10)
      : 0;

    const preferences = this.calculateUserPreferences(ratings);
    const recencyScore = this.calculateRecencyScore(recentActions);

    return {
      userId,
      ratingCount,
      avgRating,
      ratingVariance: preferences.ratingVariance,
      timeActive,
      engagement,
      genres: {},
      directors: {},
      actors: {},
      lastActive: ratings.length > 0
        ? Math.max(...ratings.map(r => new Date(r.timestamp).getTime()))
        : null,
      sessionDepth,
      recencyScore,
      recentActions: recentActions.slice(0, DEFAULT_SEQUENCE_WINDOW),
      preferences: {
        genreWeights: preferences.genres,
        directorWeights: preferences.directors,
        actorWeights: preferences.actors,
        runtimePreference: preferences.runtimePreference,
        yearPreference: preferences.yearPreference,
        ratingThreshold: preferences.ratingThreshold
      }
    } as UserProfile;
  }

  async getAvailableMovies(
    userId: string,
    options: {
      excludeRated?: boolean;
      excludeWatchlist?: boolean;
    }
  ): Promise<Movie[]> {
    const {
      excludeRated = true,
      excludeWatchlist = true
    } = options;

    const allMovies: Movie[] = [];

    if (!excludeRated && !excludeWatchlist) {
      return allMovies;
    }

    let excludedMovieIds = new Set<number>();

    if (excludeRated) {
      const ratings = await this.trackingService.getUserActions(userId, 1000, 'rate');
      ratings.forEach(rating => excludedMovieIds.add(rating.movieId));
    }

    if (excludeWatchlist) {
      const watchlistActions = await this.trackingService.getUserActions(userId, 1000, 'add_watchlist');
      watchlistActions.forEach(action => excludedMovieIds.add(action.movieId));
    }

    return allMovies.filter(movie => !excludedMovieIds.has(movie.id));
  }

  async findSimilarUsers(userId: string): Promise<any[]> {
    return [];
  }

  async calculateCollaborativeScore(movieId: number, similarUsers: any[]): Promise<number> {
    return 0.5;
  }

  private calculateUserPreferences(ratings: any[]) {
    const genrePreferences: { [key: string]: number } = {};
    const directorPreferences: { [key: string]: number } = {};
    const actorPreferences: { [key: string]: number } = {};
    const genreCounts: { [key: string]: number } = {};
    const directorCounts: { [key: string]: number } = {};
    const actorCounts: { [key: string]: number } = {};
    const runtimeSignals: { value: number; weight: number }[] = [];
    const yearSignals: { value: number; weight: number }[] = [];

    ratings.forEach(rating => {
      const ratingSignal = this.normalizeRatingSignal(rating.value);
      const metadata = rating.metadata || {};
      const genres = Array.isArray(metadata.genres) ? metadata.genres : [];
      const directors = Array.isArray(metadata.directors) ? metadata.directors : [];
      const actors = Array.isArray(metadata.actors) ? metadata.actors : [];

      genres.forEach((genre: string) => {
        genrePreferences[genre] = (genrePreferences[genre] || 0) + ratingSignal;
        genreCounts[genre] = (genreCounts[genre] || 0) + 1;
      });

      directors.forEach((director: string) => {
        directorPreferences[director] = (directorPreferences[director] || 0) + ratingSignal;
        directorCounts[director] = (directorCounts[director] || 0) + 1;
      });

      actors.forEach((actor: string) => {
        actorPreferences[actor] = (actorPreferences[actor] || 0) + ratingSignal;
        actorCounts[actor] = (actorCounts[actor] || 0) + 1;
      });

      if (metadata.runtime && ratingSignal > 0) {
        runtimeSignals.push({ value: metadata.runtime, weight: ratingSignal });
      }

      if (metadata.releaseYear && ratingSignal > 0) {
        yearSignals.push({ value: metadata.releaseYear, weight: ratingSignal });
      }
    });

    const normalizePreferenceMap = (preferences: { [key: string]: number }, counts: { [key: string]: number }) => {
      return Object.keys(preferences).reduce((acc: { [key: string]: number }, key) => {
        acc[key] = preferences[key] / (counts[key] || 1);
        return acc;
      }, {});
    };

    return {
      genres: normalizePreferenceMap(genrePreferences, genreCounts),
      directors: normalizePreferenceMap(directorPreferences, directorCounts),
      actors: normalizePreferenceMap(actorPreferences, actorCounts),
      runtimePreference: this.calculateRuntimePreference(runtimeSignals),
      yearPreference: this.calculateYearPreference(yearSignals),
      avgRating: ratings.reduce((sum, r) => sum + r.value, 0) / (ratings.length || 1),
      ratingVariance: this.calculateRatingVariance(ratings),
      ratingThreshold: 6.5
    };
  }

  private calculatePreferenceScore(items: string[], preferenceMap: { [key: string]: number }, useMax = false): number {
    if (!items || items.length === 0 || !preferenceMap || Object.keys(preferenceMap).length === 0) {
      return 0.5;
    }

    const scores = items
      .map(item => preferenceMap[item])
      .filter(score => score !== undefined);

    if (scores.length === 0) {
      return 0.45;
    }

    const adjustedScores = scores.map(score => (score + 1) / 2);
    return useMax
      ? Math.max(...adjustedScores)
      : adjustedScores.reduce((sum, value) => sum + value, 0) / adjustedScores.length;
  }

  private calculateRuntimePreference(runtimeSignals: any[]) {
    if (!runtimeSignals || runtimeSignals.length === 0) {
      return { min: 70, max: 190, ideal: 120 };
    }

    const weightedSum = runtimeSignals.reduce((sum, item) => sum + (item.value * item.weight), 0);
    const totalWeight = runtimeSignals.reduce((sum, item) => sum + item.weight, 0) || 1;
    const ideal = weightedSum / totalWeight;
    return { min: Math.max(50, ideal - 40), max: ideal + 50, ideal };
  }

  private calculateYearPreference(yearSignals: any[]) {
    if (!yearSignals || yearSignals.length === 0) {
      return { min: 1980, max: new Date().getFullYear() };
    }

    const weightedSum = yearSignals.reduce((sum, item) => sum + (item.value * item.weight), 0);
    const totalWeight = yearSignals.reduce((sum, item) => sum + item.weight, 0) || 1;
    const ideal = weightedSum / totalWeight;
    return { min: Math.max(1950, Math.floor(ideal - 15)), max: Math.min(new Date().getFullYear(), Math.ceil(ideal + 15)) };
  }

  private normalizeRatingSignal(value: number) {
    const normalized = (value - 5.5) / 4.5;
    return Math.max(-1, Math.min(1, normalized));
  }

  private calculateRecencyScore(actions: any[]) {
    if (!actions || actions.length === 0) {
      return 0;
    }

    const mostRecent = Math.max(...actions.map(a => new Date(a.timestamp).getTime()));
    const hoursSince = (Date.now() - mostRecent) / (1000 * 60 * 60);
    const score = Math.exp(-Math.log(2) * (hoursSince / RECENCY_HALF_LIFE_HOURS));

    return Math.min(1, Math.max(0, score));
  }

  private calculateRecencyWeight(timestamp: string, index: number) {
    const hoursSince = (Date.now() - new Date(timestamp).getTime()) / (1000 * 60 * 60);
    const decay = Math.exp(-Math.log(2) * (hoursSince / RECENCY_HALF_LIFE_HOURS));
    const positionBoost = 1 - Math.min(0.3, index / (DEFAULT_SEQUENCE_WINDOW * 2));
    return decay * positionBoost;
  }

  private actionTypeBoost(actionType: string, value: number) {
    switch (actionType) {
      case 'watchTime':
        return Math.min(1.2, (value || 0) / 60);
      case 'rate':
        return (value || 5) / 10;
      case 'add_watchlist':
        return 0.7;
      case 'view':
        return 0.5;
      case 'click':
      default:
        return 0.4;
    }
  }

  private async updateRecommendationMetrics(userId: string, recommendations: RecommendationItem[]): Promise<void> {
    try {
      await this.redis.hincrby('metrics:recommendations', 'total_generated', 1);
      await this.redis.hincrby('metrics:recommendations', 'total_items', recommendations.length);

      const avgScore = recommendations.reduce((sum, rec) => sum + rec.score, 0) / recommendations.length;
      await this.redis.hset('metrics:recommendations', 'last_avg_score', avgScore.toString());
    } catch (error) {
      console.error('Error updating recommendation metrics:', error);
    }
  }
}

export { HybridRecommendationEngine, Movie, UserProfile, RecommendationItem, RecommendationOptions };
