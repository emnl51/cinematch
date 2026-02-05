const MatrixFactorization = require('../ml/matrixFactorization');
const { TrackingService } = require('./trackingService');
const Redis = require('ioredis');

const DEFAULT_SEQUENCE_WINDOW = 20;
const DEFAULT_SESSION_TIMEOUT = 30 * 60 * 1000;
const RECENCY_HALF_LIFE_HOURS = 24;

class HybridRecommendationEngine {
  constructor() {
    this.matrixFactorization = new MatrixFactorization();
    this.trackingService = new TrackingService();
    this.redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379');
    this.cacheTimeout = 300; // 5 minutes
  }

  async generateRecommendations(userId, options = {}) {
    try {
      const {
        count = 25,
        excludeRated = true,
        excludeWatchlist = true,
        minScore = 0.5,
        diversityFactor = 0.25,
        includeExplanations = false
      } = options;

      const cacheKey = `recommendations:${userId}:${JSON.stringify(options)}`;
      const cached = await this.redis.get(cacheKey);
      if (cached) {
        return JSON.parse(cached);
      }

      const userProfile = await this.getUserProfile(userId);
      const weights = this.calculateWeights(userProfile);

      const availableMovies = await this.getAvailableMovies(userId, excludeRated, excludeWatchlist);
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

      const diversifiedScores = this.applyDiversityFilter(hybridScores, diversityFactor, userProfile);

      const recommendations = diversifiedScores
        .filter(item => item.score >= minScore)
        .sort((a, b) => b.score - a.score)
        .slice(0, count);

      await this.redis.setex(cacheKey, this.cacheTimeout, JSON.stringify(recommendations));

      return recommendations;
    } catch (error) {
      console.error('Error generating recommendations:', error);
      throw error;
    }
  }

  async contentBasedRecommendation(availableMovies, userProfile) {
    try {
      const preferences = userProfile?.preferences;
      if (!preferences || userProfile.ratingCount === 0) {
        return this.popularityFallback(availableMovies, 'content-cold');
      }

      return availableMovies.map(movie => {
        const score = this.calculateContentSimilarity(movie, preferences);
        return {
          movieId: movie.id,
          movie,
          score,
          source: 'content-based'
        };
      });
    } catch (error) {
      console.error('Error in content-based recommendation:', error);
      return [];
    }
  }

  async collaborativeFiltering(userId, availableMovies) {
    try {
      const movieIds = availableMovies.map(movie => movie.id);
      const predictions = await this.matrixFactorization.predict(userId, movieIds);

      if (predictions.length === 0) {
        return this.userBasedCollaborativeFiltering(userId, availableMovies);
      }

      return predictions.map(pred => {
        const movie = availableMovies.find(m => m.id === pred.movieId);
        return {
          movieId: pred.movieId,
          movie,
          score: this.normalizeScore(pred.score),
          source: 'collaborative-matrix'
        };
      });
    } catch (error) {
      console.error('Error in collaborative filtering:', error);
      return this.userBasedCollaborativeFiltering(userId, availableMovies);
    }
  }

  async userBasedCollaborativeFiltering(userId, availableMovies) {
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

  async sequenceBasedRecommendation(availableMovies, userProfile) {
    try {
      const recentActions = userProfile.recentActions || [];
      if (recentActions.length === 0) {
        return this.popularityFallback(availableMovies, 'sequence-cold');
      }

      const recentSignals = this.buildSessionSignals(recentActions);
      return availableMovies.map(movie => {
        const score = this.calculateSessionSimilarity(movie, recentSignals);
        return {
          movieId: movie.id,
          movie,
          score,
          source: 'sequence'
        };
      });
    } catch (error) {
      console.error('Error in sequence recommendation:', error);
      return [];
    }
  }

  async ruleBasedRecommendation(availableMovies, userProfile) {
    try {
      const rulePreferences = this.extractRulePreferences(userProfile);
      if (!rulePreferences) {
        return this.popularityFallback(availableMovies, 'rule-cold');
      }

      return availableMovies.map(movie => {
        const score = this.calculateRuleScore(movie, rulePreferences);
        return {
          movieId: movie.id,
          movie,
          score,
          source: 'rule-based'
        };
      });
    } catch (error) {
      console.error('Error in rule-based recommendation:', error);
      return [];
    }
  }

  calculateWeights(userProfile) {
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

  normalizeWeights(weights) {
    const total = Object.values(weights).reduce((sum, value) => sum + value, 0) || 1;
    return Object.entries(weights).reduce((acc, [key, value]) => {
      acc[key] = Math.max(0, value / total);
      return acc;
    }, {});
  }

  combineScores(contentScores, collaborativeScores, sequenceScores, ruleScores, weights, includeExplanations = false) {
    const movieScoreMap = new Map();

    const ingestScores = (scores, scoreKey) => {
      scores.forEach(item => {
        if (!item || !item.movie) {
          return;
        }

        if (!movieScoreMap.has(item.movieId)) {
          movieScoreMap.set(item.movieId, {
            movieId: item.movieId,
            movie: item.movie,
            contentScore: 0,
            collaborativeScore: 0,
            sequenceScore: 0,
            ruleScore: 0,
            explanation: includeExplanations ? [] : undefined
          });
        }

        movieScoreMap.get(item.movieId)[scoreKey] = item.score || 0;
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

      if (includeExplanations && item.explanation) {
        item.explanation = this.generateExplanations(item, weights);
      }

      return {
        ...item,
        score: hybridScore,
        weights,
        source: 'hybrid'
      };
    });
  }

  async getUserProfile(userId) {
    try {
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
      const engagement = sessions.length > 0 ? allActions.length / sessions.length : 0;
      const sessionDepth = sessions.length > 0
        ? Math.min(1, sessions[sessions.length - 1].length / 10)
        : 0;

      const recencyScore = this.calculateRecencyScore(recentActions);
      const preferences = this.calculateUserPreferences(ratings);

      return {
        userId,
        ratingCount,
        avgRating,
        ratingVariance: preferences.ratingVariance,
        timeActive,
        engagement,
        lastActive: ratings.length > 0
          ? Math.max(...ratings.map(r => new Date(r.timestamp).getTime()))
          : null,
        sessionDepth,
        recencyScore,
        recentActions: recentActions.slice(0, DEFAULT_SEQUENCE_WINDOW),
        preferences
      };
    } catch (error) {
      console.error('Error getting user profile:', error);
      return { userId, ratingCount: 0, recentActions: [] };
    }
  }

  calculateUserPreferences(ratings) {
    const genrePreferences = {};
    const directorPreferences = {};
    const actorPreferences = {};
    const genreCounts = {};
    const directorCounts = {};
    const actorCounts = {};
    const runtimeSignals = [];
    const yearSignals = [];

    ratings.forEach(rating => {
      const ratingSignal = this.normalizeRatingSignal(rating.value);
      const metadata = rating.metadata || {};
      const genres = Array.isArray(metadata.genres) ? metadata.genres : [];
      const directors = Array.isArray(metadata.directors) ? metadata.directors : [];
      const actors = Array.isArray(metadata.actors) ? metadata.actors : [];

      genres.forEach(genre => {
        genrePreferences[genre] = (genrePreferences[genre] || 0) + ratingSignal;
        genreCounts[genre] = (genreCounts[genre] || 0) + 1;
      });

      directors.forEach(director => {
        directorPreferences[director] = (directorPreferences[director] || 0) + ratingSignal;
        directorCounts[director] = (directorCounts[director] || 0) + 1;
      });

      actors.forEach(actor => {
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

    const normalizePreferenceMap = (preferences, counts) => {
      return Object.keys(preferences).reduce((acc, key) => {
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

  calculateContentSimilarity(movie, userPreferences) {
    if (!userPreferences) {
      return Math.random() * 0.6 + 0.3;
    }

    const genreScore = this.calculatePreferenceScore(movie.genres, userPreferences.genres);
    const directorScore = this.calculatePreferenceScore(movie.directors, userPreferences.directors, true);
    const actorScore = this.calculatePreferenceScore(movie.actors, userPreferences.actors);
    const runtimeScore = this.calculateRuntimeScore(movie, userPreferences.runtimePreference);
    const yearScore = this.calculateYearScore(movie, userPreferences.yearPreference);

    const score = (
      genreScore * 0.4 +
      directorScore * 0.2 +
      actorScore * 0.2 +
      runtimeScore * 0.1 +
      yearScore * 0.1
    );

    return this.normalizeScore(score * 10);
  }

  buildSessionSignals(actions) {
    const signals = {
      genres: {},
      directors: {},
      actors: {},
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

  calculateSessionSimilarity(movie, sessionSignals) {
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

  extractRulePreferences(userProfile) {
    const preferences = userProfile?.preferences;
    if (!preferences || !preferences.genres) {
      return null;
    }

    const sortedGenres = Object.entries(preferences.genres)
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

  calculateRuleScore(movie, rulePreferences) {
    let score = 0.4;

    if (rulePreferences.preferredGenres?.length) {
      const genreMatches = (movie.genres || []).filter(genre => rulePreferences.preferredGenres.includes(genre));
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

    if ((movie.averageRating || 0) >= rulePreferences.minRating) {
      score += 0.1;
    }

    return this.normalizeScore(score * 10);
  }

  async findSimilarUsers(userId) {
    try {
      const userRatings = await this.trackingService.getUserActions(userId, 1000, 'rate');
      if (userRatings.length === 0) {
        return [];
      }

      return [];
    } catch (error) {
      console.error('Error finding similar users:', error);
      return [];
    }
  }

  async calculateCollaborativeScore(movieId, similarUsers) {
    if (similarUsers.length === 0) {
      return 0;
    }

    let totalWeight = 0;
    let weightedSum = 0;

    for (const user of similarUsers) {
      const userActions = await this.trackingService.getUserActions(user.userId, 1000, 'rate');
      const movieRating = userActions.find(a => a.movieId === movieId);

      if (movieRating) {
        weightedSum += movieRating.value * user.similarity;
        totalWeight += user.similarity;
      }
    }

    return totalWeight > 0 ? this.normalizeScore(weightedSum / totalWeight) : 0;
  }

  popularityFallback(movies, source) {
    return movies.map(movie => ({
      movieId: movie.id,
      movie,
      score: this.calculatePopularityScore(movie),
      source
    }));
  }

  async getAvailableMovies(userId, excludeRated = true, excludeWatchlist = true) {
    try {
      const allMovies = [];

      if (!excludeRated && !excludeWatchlist) {
        return allMovies;
      }

      let excludedMovieIds = new Set();

      if (excludeRated) {
        const ratings = await this.trackingService.getUserActions(userId, 1000, 'rate');
        ratings.forEach(rating => excludedMovieIds.add(rating.movieId));
      }

      if (excludeWatchlist) {
        const watchlistActions = await this.trackingService.getUserActions(userId, 1000, 'add_watchlist');
        watchlistActions.forEach(action => excludedMovieIds.add(action.movieId));
      }

      return allMovies.filter(movie => !excludedMovieIds.has(movie.id));
    } catch (error) {
      console.error('Error getting available movies:', error);
      return [];
    }
  }

  normalizeScore(score) {
    if (score < 1) return 0;
    if (score > 10) return 1;
    return (score - 1) / 9;
  }

  normalizeRatingSignal(value) {
    const normalized = (value - 5.5) / 4.5;
    return Math.max(-1, Math.min(1, normalized));
  }

  calculatePreferenceScore(items, preferenceMap, useMax = false) {
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

  calculateRuntimePreference(runtimeSignals) {
    if (!runtimeSignals || runtimeSignals.length === 0) {
      return { min: 70, max: 190, ideal: 120 };
    }

    const weightedSum = runtimeSignals.reduce((sum, item) => sum + (item.value * item.weight), 0);
    const totalWeight = runtimeSignals.reduce((sum, item) => sum + item.weight, 0) || 1;
    const ideal = weightedSum / totalWeight;
    return { min: Math.max(50, ideal - 40), max: ideal + 50, ideal };
  }

  calculateYearPreference(yearSignals) {
    if (!yearSignals || yearSignals.length === 0) {
      return { min: 1980, max: new Date().getFullYear() };
    }

    const weightedSum = yearSignals.reduce((sum, item) => sum + (item.value * item.weight), 0);
    const totalWeight = yearSignals.reduce((sum, item) => sum + item.weight, 0) || 1;
    const ideal = weightedSum / totalWeight;
    return {
      min: Math.max(1950, Math.floor(ideal - 15)),
      max: Math.min(new Date().getFullYear(), Math.ceil(ideal + 15))
    };
  }

  calculateRuntimeScore(movie, runtimePreference) {
    if (!runtimePreference || !movie.runtime) {
      return 0.5;
    }

    if (movie.runtime < runtimePreference.min || movie.runtime > runtimePreference.max) {
      return 0.2;
    }

    const distance = Math.abs(movie.runtime - runtimePreference.ideal);
    const maxDistance = Math.max(runtimePreference.ideal - runtimePreference.min, runtimePreference.max - runtimePreference.ideal);

    return 1 - (distance / (maxDistance || 1));
  }

  calculateYearScore(movie, yearPreference) {
    if (!yearPreference || !movie.releaseYear) {
      return 0.5;
    }

    if (movie.releaseYear < yearPreference.min || movie.releaseYear > yearPreference.max) {
      return 0.3;
    }

    return 1;
  }

  calculatePopularityScore(movie) {
    const popularityScore = (movie.popularity || 0) / 100;
    const ratingScore = (movie.averageRating || 0) / 10;
    const ratingCountScore = Math.log((movie.ratingCount || 0) + 1) / Math.log(10000);
    return (popularityScore * 0.4) + (ratingScore * 0.4) + (ratingCountScore * 0.2);
  }

  calculateRatingVariance(ratings) {
    if (ratings.length < 2) return 0;

    const mean = ratings.reduce((sum, r) => sum + r.value, 0) / ratings.length;
    const variance = ratings.reduce((sum, r) => sum + Math.pow(r.value - mean, 2), 0) / ratings.length;

    return variance;
  }

  groupActionsBySessions(actions, sessionTimeout = DEFAULT_SESSION_TIMEOUT) {
    const sessions = [];
    let currentSession = [];

    actions.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

    for (const action of actions) {
      const actionTime = new Date(action.timestamp).getTime();

      if (currentSession.length === 0) {
        currentSession.push(action);
      } else {
        const lastActionTime = new Date(currentSession[currentSession.length - 1].timestamp).getTime();

        if (actionTime - lastActionTime <= sessionTimeout) {
          currentSession.push(action);
        } else {
          sessions.push(currentSession);
          currentSession = [action];
        }
      }
    }

    if (currentSession.length > 0) {
      sessions.push(currentSession);
    }

    return sessions;
  }

  calculateRecencyScore(actions) {
    if (!actions || actions.length === 0) {
      return 0;
    }

    const mostRecent = Math.max(...actions.map(a => new Date(a.timestamp).getTime()));
    const hoursSince = (Date.now() - mostRecent) / (1000 * 60 * 60);
    const score = Math.exp(-Math.log(2) * (hoursSince / RECENCY_HALF_LIFE_HOURS));

    return Math.min(1, Math.max(0, score));
  }

  calculateRecencyWeight(timestamp, index) {
    const hoursSince = (Date.now() - new Date(timestamp).getTime()) / (1000 * 60 * 60);
    const decay = Math.exp(-Math.log(2) * (hoursSince / RECENCY_HALF_LIFE_HOURS));
    const positionBoost = 1 - Math.min(0.3, index / (DEFAULT_SEQUENCE_WINDOW * 2));
    return decay * positionBoost;
  }

  actionTypeBoost(actionType, value) {
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

  applyDiversityFilter(recommendations, diversityFactor, userProfile) {
    if (!diversityFactor || diversityFactor <= 0) {
      return recommendations;
    }

    const diverseRecommendations = [];
    const selectedGenres = new Set();
    const selectedDirectors = new Set();

    const sortedRecommendations = [...recommendations].sort((a, b) => b.score - a.score);

    for (const recommendation of sortedRecommendations) {
      const movie = recommendation.movie || {};
      const genres = movie.genres || [];
      const directors = movie.directors || [];
      const genreOverlap = genres.some(genre => selectedGenres.has(genre));
      const directorOverlap = directors.some(director => selectedDirectors.has(director));

      let diversityPenalty = 0;
      if (genreOverlap) diversityPenalty += 0.3;
      if (directorOverlap) diversityPenalty += 0.2;

      recommendation.score *= (1 - (diversityPenalty * diversityFactor));
      diverseRecommendations.push(recommendation);

      genres.forEach(genre => selectedGenres.add(genre));
      directors.forEach(director => selectedDirectors.add(director));
    }

    return diverseRecommendations;
  }

  generateExplanations(item, weights) {
    const explanations = [];

    if (item.contentScore > 0.7 && weights.content > 0.2) {
      explanations.push('Güçlü içerik uyumu');
    }

    if (item.collaborativeScore > 0.7 && weights.collaborative > 0.2) {
      explanations.push('Benzer zevke sahip kullanıcıların tercihi');
    }

    if (item.sequenceScore > 0.7 && weights.sequence > 0.2) {
      explanations.push('Son izleme akışına göre önerildi');
    }

    if (item.ruleScore > 0.6 && weights.rule > 0.1) {
      explanations.push('Onboarding tercihlerinize uygun');
    }

    return explanations;
  }

  async getRecommendationStats() {
    try {
      const stats = await this.redis.hgetall('recommendation_stats');
      return {
        totalRecommendations: parseInt(stats.total_recommendations || 0),
        cacheHitRate: parseFloat(stats.cache_hit_rate || 0),
        avgResponseTime: parseFloat(stats.avg_response_time || 0),
        modelAccuracy: parseFloat(stats.model_accuracy || 0)
      };
    } catch (error) {
      console.error('Error getting recommendation stats:', error);
      return {};
    }
  }
}

module.exports = HybridRecommendationEngine;
