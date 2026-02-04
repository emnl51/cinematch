const MatrixFactorization = require('../ml/matrixFactorization');
const { TrackingService } = require('./trackingService');
const Redis = require('ioredis');

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

      // Check cache first
      const cacheKey = `recommendations:${userId}:${count}`;
      const cached = await this.redis.get(cacheKey);
      if (cached) {
        return JSON.parse(cached);
      }

      // Get user profile to determine weights
      const userProfile = await this.getUserProfile(userId);
      const weights = this.calculateWeights(userProfile);

      // Get available movies
      const availableMovies = await this.getAvailableMovies(userId, excludeRated, excludeWatchlist);
      
      if (availableMovies.length === 0) {
        return [];
      }

      // Generate content-based scores
      const contentBasedScores = await this.contentBasedRecommendation(userId, availableMovies, userProfile);
      
      // Generate collaborative filtering scores
      const collaborativeScores = await this.collaborativeFiltering(userId, availableMovies);

      // Popularity-based scores (cold start + fallback)
      const popularityScores = this.getPopularityBasedScores(availableMovies);

      // Cross recommendation scores (exploration across genres/directors/actors)
      const crossScores = this.crossRecommendation(availableMovies, userProfile);

      // Combine scores using adaptive weighting
      const hybridScores = this.combineScores(
        contentBasedScores,
        collaborativeScores,
        popularityScores,
        crossScores,
        weights,
        includeExplanations
      );

      const diversifiedScores = this.applyDiversityFilter(hybridScores, diversityFactor, userProfile);

      // Sort and filter recommendations
      const recommendations = diversifiedScores
        .filter(item => item.score >= minScore)
        .sort((a, b) => b.score - a.score)
        .slice(0, count);

      // Cache results
      await this.redis.setex(cacheKey, this.cacheTimeout, JSON.stringify(recommendations));

      return recommendations;
    } catch (error) {
      console.error('Error generating recommendations:', error);
      throw error;
    }
  }

  async contentBasedRecommendation(userId, availableMovies, userProfile) {
    try {
      // Get user's rating history
      const userRatings = await this.trackingService.getUserActions(userId, 1000, 'rate');
      
      if (userRatings.length === 0) {
        // Cold start: return popularity-based recommendations
        return this.getPopularityBasedScores(availableMovies);
      }

      // Calculate user preferences
      const userPreferences = userProfile?.preferences || this.calculateUserPreferences(userRatings);
      
      // Score movies based on content similarity
      const contentScores = availableMovies.map(movie => {
        const score = this.calculateContentSimilarity(movie, userPreferences);
        return {
          movieId: movie.id,
          movie,
          score,
          source: 'content-based'
        };
      });

      return contentScores;
    } catch (error) {
      console.error('Error in content-based recommendation:', error);
      return [];
    }
  }

  async collaborativeFiltering(userId, availableMovies) {
    try {
      // Try to get predictions from matrix factorization model
      const movieIds = availableMovies.map(movie => movie.id);
      const predictions = await this.matrixFactorization.predict(userId, movieIds);
      
      if (predictions.length === 0) {
        // Fallback to user-based collaborative filtering
        return this.userBasedCollaborativeFiltering(userId, availableMovies);
      }

      // Convert predictions to recommendation format
      const collaborativeScores = predictions.map(pred => {
        const movie = availableMovies.find(m => m.id === pred.movieId);
        return {
          movieId: pred.movieId,
          movie,
          score: this.normalizeScore(pred.score), // Normalize to 0-1 range
          source: 'collaborative-matrix'
        };
      });

      return collaborativeScores;
    } catch (error) {
      console.error('Error in collaborative filtering:', error);
      return this.userBasedCollaborativeFiltering(userId, availableMovies);
    }
  }

  async userBasedCollaborativeFiltering(userId, availableMovies) {
    try {
      // Find similar users
      const similarUsers = await this.findSimilarUsers(userId);
      
      if (similarUsers.length === 0) {
        return this.getPopularityBasedScores(availableMovies);
      }

      // Calculate scores based on similar users' ratings
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

  calculateWeights(userProfile) {
    const ratingCount = userProfile.ratingCount || 0;
    const timeActive = userProfile.timeActive || 0; // days since first rating
    const engagement = userProfile.engagement || 0; // average actions per session
    const ratingVariance = userProfile.ratingVariance || 0;
    const avgRating = userProfile.avgRating || 0;

    const consistency = 1 - Math.min(1, ratingVariance / 6);
    const engagementFactor = Math.min(1, engagement / 15);
    const longevityBoost = Math.min(0.1, timeActive / 3650); // up to 10% for long-term users
    const biasFactor = Math.max(-0.1, Math.min(0.1, (avgRating - 6) / 20));

    let baseWeights;
    if (ratingCount < 10) {
      baseWeights = { content: 0.55, collaborative: 0.15, popularity: 0.2, cross: 0.1 };
    } else if (ratingCount < 50) {
      baseWeights = { content: 0.45, collaborative: 0.35, popularity: 0.1, cross: 0.1 };
    } else if (ratingCount < 200) {
      baseWeights = { content: 0.35, collaborative: 0.45, popularity: 0.05, cross: 0.15 };
    } else {
      baseWeights = { content: 0.3, collaborative: 0.5, popularity: 0.05, cross: 0.15 };
    }

    baseWeights.collaborative += (consistency * 0.1) + (engagementFactor * 0.05) + longevityBoost;
    baseWeights.content -= (consistency * 0.05);
    baseWeights.cross += (1 - consistency) * 0.05;
    baseWeights.popularity += (ratingCount < 10 ? 0.05 : 0) - biasFactor;

    return this.normalizeWeights(baseWeights);
  }

  normalizeWeights(weights) {
    const total = Object.values(weights).reduce((sum, value) => sum + value, 0) || 1;
    return Object.entries(weights).reduce((acc, [key, value]) => {
      acc[key] = Math.max(0, value / total);
      return acc;
    }, {});
  }

  combineScores(contentScores, collaborativeScores, popularityScores, crossScores, weights, includeExplanations = false) {
    const movieScoreMap = new Map();

    // Add content-based scores
    contentScores.forEach(item => {
      movieScoreMap.set(item.movieId, {
        ...item,
        contentScore: item.score,
        collaborativeScore: 0,
        popularityScore: 0,
        crossScore: 0,
        explanation: includeExplanations ? [] : undefined
      });
    });

    // Add collaborative scores
    collaborativeScores.forEach(item => {
      if (movieScoreMap.has(item.movieId)) {
        const existing = movieScoreMap.get(item.movieId);
        existing.collaborativeScore = item.score;
      } else {
        movieScoreMap.set(item.movieId, {
          ...item,
          contentScore: 0,
          collaborativeScore: item.score,
          popularityScore: 0,
          crossScore: 0,
          explanation: includeExplanations ? [] : undefined
        });
      }
    });

    // Add popularity scores
    popularityScores.forEach(item => {
      if (movieScoreMap.has(item.movieId)) {
        const existing = movieScoreMap.get(item.movieId);
        existing.popularityScore = item.score;
      } else {
        movieScoreMap.set(item.movieId, {
          ...item,
          contentScore: 0,
          collaborativeScore: 0,
          popularityScore: item.score,
          crossScore: 0,
          explanation: includeExplanations ? [] : undefined
        });
      }
    });

    // Add cross recommendation scores
    crossScores.forEach(item => {
      if (movieScoreMap.has(item.movieId)) {
        const existing = movieScoreMap.get(item.movieId);
        existing.crossScore = item.score;
      } else {
        movieScoreMap.set(item.movieId, {
          ...item,
          contentScore: 0,
          collaborativeScore: 0,
          popularityScore: 0,
          crossScore: item.score,
          explanation: includeExplanations ? [] : undefined
        });
      }
    });

    // Calculate hybrid scores
    const hybridScores = Array.from(movieScoreMap.values()).map(item => {
      const hybridScore = (
        (item.contentScore * weights.content) +
        (item.collaborativeScore * weights.collaborative) +
        (item.popularityScore * (weights.popularity || 0)) +
        (item.crossScore * (weights.cross || 0))
      );

      if (includeExplanations && item.explanation) {
        item.explanation = this.generateExplanations(item, weights);
      }

      return {
        ...item,
        score: hybridScore,
        weights: weights,
        source: 'hybrid'
      };
    });

    return hybridScores;
  }

  async getUserProfile(userId) {
    try {
      // Get user rating history
      const ratings = await this.trackingService.getUserActions(userId, 1000, 'rate');
      
      // Calculate profile metrics
      const ratingCount = ratings.length;
      const avgRating = ratings.length > 0 
        ? ratings.reduce((sum, r) => sum + r.value, 0) / ratings.length 
        : 0;
      
      const firstRating = ratings.length > 0 
        ? Math.min(...ratings.map(r => new Date(r.timestamp).getTime()))
        : Date.now();
      
      const timeActive = Math.floor((Date.now() - firstRating) / (1000 * 60 * 60 * 24));
      
      // Get all user actions for engagement calculation
      const allActions = await this.trackingService.getUserActions(userId, 1000);
      const sessions = this.groupActionsBySessions(allActions);
      const engagement = sessions.length > 0 
        ? allActions.length / sessions.length 
        : 0;

      const preferences = this.calculateUserPreferences(ratings);

      return {
        userId,
        ratingCount,
        avgRating,
        ratingVariance: preferences.ratingVariance,
        timeActive,
        engagement,
        genres: this.calculateGenrePreferences(ratings),
        lastActive: ratings.length > 0 
          ? Math.max(...ratings.map(r => new Date(r.timestamp).getTime()))
          : null,
        preferences
      };
    } catch (error) {
      console.error('Error getting user profile:', error);
      return { userId, ratingCount: 0 };
    }
  }

  calculateUserPreferences(ratings) {
    // This would integrate with movie metadata to calculate preferences
    // For now, return a simplified version
    const genrePreferences = {};
    const directorPreferences = {};
    const actorPreferences = {};
    const genreCounts = {};
    const directorCounts = {};
    const actorCounts = {};
    const runtimeSignals = [];
    const yearSignals = [];
    
    // This would be expanded with actual movie metadata
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

    const normalizedGenres = normalizePreferenceMap(genrePreferences, genreCounts);
    const normalizedDirectors = normalizePreferenceMap(directorPreferences, directorCounts);
    const normalizedActors = normalizePreferenceMap(actorPreferences, actorCounts);

    return {
      genres: normalizedGenres,
      directors: normalizedDirectors,
      actors: normalizedActors,
      runtimePreference: this.calculateRuntimePreference(runtimeSignals),
      yearPreference: this.calculateYearPreference(yearSignals),
      avgRating: ratings.reduce((sum, r) => sum + r.value, 0) / ratings.length,
      ratingVariance: this.calculateRatingVariance(ratings)
    };
  }

  calculateContentSimilarity(movie, userPreferences) {
    // Placeholder for content-based similarity calculation
    // Would require movie metadata (genres, directors, actors, etc.)
    
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

  async findSimilarUsers(userId, limit = 10) {
    try {
      // Get user's ratings
      const userRatings = await this.trackingService.getUserActions(userId, 1000, 'rate');
      
      if (userRatings.length === 0) {
        return [];
      }

      const userMovieRatings = new Map(
        userRatings.map(r => [r.movieId, r.value])
      );

      // This would be optimized with proper indexing in a real implementation
      // For now, return a simplified similar users list
      const similarUsers = [];
      
      // Placeholder implementation
      // In real system, would use cosine similarity or Pearson correlation
      
      return similarUsers;
    } catch (error) {
      console.error('Error finding similar users:', error);
      return [];
    }
  }

  async calculateCollaborativeScore(movieId, similarUsers) {
    if (similarUsers.length === 0) {
      return 0;
    }

    // Calculate weighted average rating from similar users
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

  getPopularityBasedScores(movies) {
    // Return scores based on movie popularity
    return movies.map(movie => ({
      movieId: movie.id,
      movie,
      score: this.calculatePopularityScore(movie),
      source: 'popularity'
    }));
  }

  async getAvailableMovies(userId, excludeRated = true, excludeWatchlist = true) {
    try {
      // This would fetch from your movie database
      // For now, return a placeholder list
      const allMovies = []; // Would fetch from database
      
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
    // Normalize score to 0-1 range
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
    return { min: Math.max(1950, Math.floor(ideal - 15)), max: Math.min(new Date().getFullYear(), Math.ceil(ideal + 15)) };
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

  crossRecommendation(movies, userProfile) {
    const preferences = userProfile?.preferences;
    if (!preferences || !preferences.genres || Object.keys(preferences.genres).length === 0) {
      return movies.map(movie => ({
        movieId: movie.id,
        movie,
        score: 0.4,
        source: 'cross'
      }));
    }

    const sortedGenres = Object.entries(preferences.genres)
      .sort(([, a], [, b]) => b - a);
    const topGenres = sortedGenres.slice(0, 3).map(([genre]) => genre);
    const dislikedGenres = sortedGenres.filter(([, score]) => score < -0.2).map(([genre]) => genre);

    return movies.map(movie => {
      const genres = movie.genres || [];
      const hasTop = genres.some(genre => topGenres.includes(genre));
      const hasDisliked = genres.some(genre => dislikedGenres.includes(genre));
      const newGenreCount = genres.filter(genre => !topGenres.includes(genre)).length;
      const noveltyRatio = genres.length > 0 ? newGenreCount / genres.length : 0;

      let score = hasTop ? 0.5 + (0.4 * noveltyRatio) : 0.35 + (0.3 * noveltyRatio);
      if (hasDisliked) {
        score *= 0.6;
      }

      return {
        movieId: movie.id,
        movie,
        score: this.normalizeScore(score * 10),
        source: 'cross'
      };
    });
  }

  calculateRatingVariance(ratings) {
    if (ratings.length < 2) return 0;
    
    const mean = ratings.reduce((sum, r) => sum + r.value, 0) / ratings.length;
    const variance = ratings.reduce((sum, r) => sum + Math.pow(r.value - mean, 2), 0) / ratings.length;
    
    return variance;
  }

  calculateGenrePreferences(ratings) {
    // Placeholder for genre preference calculation
    // Would require movie metadata
    return {};
  }

  groupActionsBySessions(actions, sessionTimeout = 30 * 60 * 1000) {
    // Group actions by sessions (30 minute timeout)
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

    if (item.popularityScore > 0.7 && (weights.popularity || 0) > 0.05) {
      explanations.push('Genel popülerlik trendleriyle uyumlu');
    }

    if (item.crossScore > 0.6 && (weights.cross || 0) > 0.05) {
      explanations.push('Keşif ve çapraz tür dengesi');
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
