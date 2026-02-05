/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import type { Movie, TVShow, Genre, UserRating, UserProfile, Recommendation } from '../../content/types';
import { tmdbService } from '../../content/services/tmdb';

import { StorageService } from '../../../shared/services/storage';
import { ProfileService } from '../../profile/services/profileService';
import { RecommendationService } from '../services/recommendationService';
import { CuratedMovieService } from '../services/curatedMovieService';
import { LearningService } from '../../learning/services/learningService';
import { RealTimeLearningService } from '../../learning/services/realTimeLearningService';
import type { CuratedContentFilters } from '../components/CuratedContentFilters';
import type { AppSettings } from '../../profile/components/SettingsModal';

export const useMovieData = (settings?: AppSettings) => {
  const [watchlist, setWatchlist] = useState<any[]>(StorageService.getWatchlist());
  const [ratings, setRatings] = useState<any[]>(StorageService.getRatings());
  const [profile, setProfile] = useState<any>(StorageService.getProfile());
  const [movies, setMoviesLocal] = useState<(Movie | TVShow)[]>([]);
  const [allMovies, setAllMoviesLocal] = useState<(Movie | TVShow)[]>([]);
  const [genres, setGenresLocal] = useState<Genre[]>([]);
  const [tvGenres, setTVGenresLocal] = useState<Genre[]>([]);
  const [recommendations, setRecommendationsLocal] = useState<Recommendation[]>([]);
  const [filteredRecommendations, setFilteredRecommendationsLocal] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState({ current: 0, total: 0, message: '' });
  const [searchQuery, setSearchQueryLocal] = useState(() => {
    return localStorage.getItem('searchQuery') || '';
  });
  const [showingCuratedMovies, setShowingCuratedMovies] = useState(() => {
    const stored = localStorage.getItem('showingCuratedMovies');
    return stored ? JSON.parse(stored) : true;
  });
  const [error, setError] = useState<string | null>(null);

  // Separate loading states for independent operation
  const [recommendationsLoading, setRecommendationsLoading] = useState(false);
  const [curatedContentLoading, setCuratedContentLoading] = useState(false);
  const [recommendationsLoadingProgress, setRecommendationsLoadingProgress] = useState({ current: 0, total: 0, message: '' });
  const [curatedContentLoadingProgress, setCuratedContentLoadingProgress] = useState({ current: 0, total: 0, message: '' });

  // Recommendation filters - GELİŞTİRİLDİ
  const [recommendationFilters, setRecommendationFiltersLocal] = useState(() => {
    const stored = localStorage.getItem('recommendationFilters');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        return {
          genres: parsed.genres || [],
          minYear: parsed.minYear || settings?.defaultFilters.minYear || 1950,
          maxYear: parsed.maxYear || settings?.defaultFilters.maxYear || new Date().getFullYear(),
          minRating: parsed.minRating || settings?.defaultFilters.minRating || 0,
          maxRating: parsed.maxRating || settings?.defaultFilters.maxRating || 10,
          mediaType: parsed.mediaType || 'all',
          sortBy: parsed.sortBy || 'match_score',
          minMatchScore: parsed.minMatchScore || settings?.defaultFilters.minMatchScore || 0,
          languages: parsed.languages || [],
          showKidsContent: typeof parsed.showKidsContent === 'boolean' ? parsed.showKidsContent : (settings?.showKidsContent !== undefined ? settings.showKidsContent : false),
          showAnimationContent: typeof parsed.showAnimationContent === 'boolean' ? parsed.showAnimationContent : (settings?.showAnimationContent !== undefined ? settings.showAnimationContent : true),
          showAnimeContent: typeof parsed.showAnimeContent === 'boolean' ? parsed.showAnimeContent : (settings?.showAnimeContent !== undefined ? settings.showAnimeContent : true)
        };
      } catch (error) {
        console.warn('Failed to parse stored recommendation filters:', error);
      }
    }
    return {
      genres: [] as number[],
      minYear: settings?.defaultFilters.minYear || 1950,
      maxYear: settings?.defaultFilters.maxYear || new Date().getFullYear(),
      minRating: settings?.defaultFilters.minRating || 0,
      maxRating: settings?.defaultFilters.maxRating || 10,
      mediaType: 'all' as 'all' | 'movie' | 'tv',
      sortBy: 'match_score' as 'match_score' | 'rating' | 'year' | 'title',
      minMatchScore: settings?.defaultFilters.minMatchScore || 0,
      languages: [] as string[],
      showKidsContent: settings?.showKidsContent || false,
      showAnimationContent: settings?.showAnimationContent || true,
      showAnimeContent: settings?.showAnimeContent || true
    };
  });

  // Curated content filters
  const [curatedContentFilters, setCuratedContentFiltersLocal] = useState<CuratedContentFilters>(() => {
    const stored = localStorage.getItem('curatedContentFilters');
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        return {
          mediaType: parsed.mediaType || 'all',
          minRating: parsed.minRating || settings?.defaultFilters.minRating || 0,
          maxRating: parsed.maxRating || settings?.defaultFilters.maxRating || 10,
          minYear: parsed.minYear || settings?.defaultFilters.minYear || 1900,
          maxYear: parsed.maxYear || settings?.defaultFilters.maxYear || new Date().getFullYear(),
          genres: parsed.genres || [],
          sortBy: parsed.sortBy || 'rating',
          sortOrder: parsed.sortOrder || 'desc',
          minVoteCount: parsed.minVoteCount || 0,
          languages: parsed.languages || []
        };
      } catch (error) {
        console.warn('Failed to parse stored curated content filters:', error);
      }
    }
    return {
      mediaType: 'all',
      minRating: settings?.defaultFilters.minRating || 0,
      maxRating: settings?.defaultFilters.maxRating || 10,
      minYear: settings?.defaultFilters.minYear || 1900,
      maxYear: settings?.defaultFilters.maxYear || new Date().getFullYear(),
      genres: [],
      sortBy: 'rating',
      sortOrder: 'desc',
      minVoteCount: 0,
      languages: []
    };
  });

  const [showCuratedFilters, setShowCuratedFiltersLocal] = useState(false);

  // Refs to prevent unnecessary re-renders and infinite loops
  const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastSearchQueryRef = useRef<string>('');
  const isInitialLoadRef = useRef(true);
  const loadingRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Error handling helper
  const handleError = useCallback((error: any, context: string) => {
    console.error(`Error in ${context}:`, error);
    setError(`${context} sırasında hata oluştu: ${error.message || 'Bilinmeyen hata'}`);
    setLoading(false);
    loadingRef.current = false;
  }, []);

  // Safe state updates with error boundaries
  const safeSetState = useCallback(<T>(setter: React.Dispatch<React.SetStateAction<T>>, value: T, fallback: T) => {
    try {
      setter(value);
    } catch (error) {
      console.error('State update error:', error);
      setter(fallback);
    }
  }, []);

  // Save recommendation filters to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem('recommendationFilters', JSON.stringify(recommendationFilters));
    } catch (error) {
      console.warn('Failed to save recommendation filters to localStorage:', error);
    }
  }, [recommendationFilters]);

  // Save curated content filters to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem('curatedContentFilters', JSON.stringify(curatedContentFilters));
    } catch (error) {
      console.warn('Failed to save curated content filters to localStorage:', error);
    }
  }, [curatedContentFilters]);

  // Save search query to localStorage whenever it changes
  useEffect(() => {
    try {
      localStorage.setItem('searchQuery', searchQuery);
    } catch (error) {
      console.warn('Failed to save search query to localStorage:', error);
    }
  }, [searchQuery]);

  // Save showingCuratedMovies to localStorage whenever it changes
  useEffect(() => {
    try {
      localStorage.setItem('showingCuratedMovies', JSON.stringify(showingCuratedMovies));
    } catch (error) {
      console.warn('Failed to save showingCuratedMovies to localStorage:', error);
    }
  }, [showingCuratedMovies]);

  // Update filters when settings change
  useEffect(() => {
    if (settings?.defaultFilters) {
      try {
        setRecommendationFiltersLocal(prev => ({
          ...prev,
          minYear: settings.defaultFilters.minYear,
          maxYear: settings.defaultFilters.maxYear,
          minRating: settings.defaultFilters.minRating,
          maxRating: settings.defaultFilters.maxRating,
          minMatchScore: settings.defaultFilters.minMatchScore
        }));

        setCuratedContentFiltersLocal(prev => ({
          ...prev,
          minRating: settings.defaultFilters.minRating,
          maxRating: settings.defaultFilters.maxRating,
          minYear: settings.defaultFilters.minYear,
          maxYear: settings.defaultFilters.maxYear
        }));
      } catch (error) {
        handleError(error, 'Ayarlar güncellenirken');
      }
    }
  }, [settings, handleError]);

  // Refresh recommendations when recommendation count changes
  useEffect(() => {
    if (profile && !isInitialLoadRef.current && settings?.recommendationCount) {
      // Debounce the refresh to avoid too many API calls
      const timeoutId = setTimeout(() => {
        refreshRecommendations();
      }, 500);
      
      return () => clearTimeout(timeoutId);
    }
    return undefined;
  }, [settings?.recommendationCount, profile]);

  // Refresh curated content when discovery content count changes
  useEffect(() => {
    if (!isInitialLoadRef.current && settings?.discoveryContentCount) {
      // Debounce the refresh to avoid too many API calls
      const timeoutId = setTimeout(() => {
        refreshCuratedContent();
      }, 500);
      
      return () => clearTimeout(timeoutId);
    }
    return undefined;
  }, [settings?.discoveryContentCount]);

  // Enhanced search function with abort controller
  const searchMovies = useCallback(async (query: string, onSearchComplete?: (results: { totalResults: number; searchType: 'content' | 'person' | 'mixed' }) => void) => {
    if (loadingRef.current) return;
    
    // Create new abort controller
    abortControllerRef.current = new AbortController();
    
    if (!query || typeof query !== 'string' || !query.trim()) {
      setLoading(true);
      loadingRef.current = true;
      try {
        // Arama yoksa boş liste göster
        setAllMoviesLocal([]);
        setMoviesLocal([]);
        setShowingCuratedMovies(false);
        onSearchComplete?.({ totalResults: 0, searchType: 'content' });
      } catch (error) {
        if (error instanceof Error) {
          handleError(error, 'Arama temizleme');
        } else {
          handleError(new Error(String(error)), 'Arama temizleme');
        }
      } finally {
        setLoading(false);
        loadingRef.current = false;
      }
      return;
    }

    setLoading(true);
    loadingRef.current = true;
    setShowingCuratedMovies(false);
    setError(null);
    
    try {
      const searchResults = await tmdbService.enhancedSearch(query);
      
      if (abortControllerRef.current?.signal.aborted) {
        return; // Request was aborted
      }

      const allResults = [...(searchResults.movies || []), ...(searchResults.tvShows || [])];
      
      // Filter out rated content and watchlist items
      const excludedContentIds = new Set([
        ...(ratings || [])
          .filter(r => r.rating === 'not_interested' || r.rating === 'skip' || (typeof r.rating === 'number'))
          .map(r => r.movieId),
        ...(watchlist || []).map(item => item.id)
      ]);
      
      let unratedResults = allResults.filter(item => item && !excludedContentIds.has(item.id));

      // Apply settings filters to search results
      if (settings?.minContentRating && typeof settings.minContentRating === 'number') {
        unratedResults = unratedResults.filter(item => 
          item && typeof item.vote_average === 'number' &&
          item.vote_average >= settings.minContentRating
        );
      }

      // Apply TMDB score filter
      if (settings?.minTmdbScore && typeof settings.minTmdbScore === 'number') {
        unratedResults = unratedResults.filter(item => 
          item && typeof item.vote_average === 'number' &&
          item.vote_average >= settings.minTmdbScore
        );
      }

      // Apply TMDB vote count filter
      if (settings?.minTmdbVoteCount && typeof settings.minTmdbVoteCount === 'number') {
        unratedResults = unratedResults.filter(item => 
          item && typeof item.vote_count === 'number' &&
          item.vote_count >= settings.minTmdbVoteCount
        );
      }

      if (settings && !settings.showAdultContent) {
        unratedResults = unratedResults.filter(item => !('adult' in item && !!item.adult));
      }
      
      setAllMoviesLocal(unratedResults);
      setMoviesLocal(unratedResults);
      
      // Call callback with search result info
      onSearchComplete?.({
        totalResults: searchResults.totalResults,
        searchType: searchResults.searchType
      });
    
    } catch (error: unknown) {
      if (error instanceof Error && error.name === 'AbortError') {
        return; // Request was aborted, don't handle as error
      }
      
      if (error instanceof Error) {
        handleError(error, 'Arama');
      } else {
        handleError(new Error(String(error)), 'Arama');
      }
      
      // Fallback search
      try {
        const results = await tmdbService.searchMulti(query);
        const excludedContentIds = new Set([
          ...(ratings || [])
            .filter(r => r.rating === 'not_interested' || r.rating === 'skip' || (typeof r.rating === 'number'))
            .map(r => r.movieId),
          ...(watchlist || []).map(item => item.id)
        ]);
        let unratedResults = (results.results || []).filter(item => item && !excludedContentIds.has(item.id));

        // Apply settings filters
        if (settings?.minContentRating) {
          unratedResults = unratedResults.filter(item => 
            item && typeof item.vote_average === 'number' &&
            item.vote_average >= settings.minContentRating
          );
        }

        // Apply TMDB score filter
        if (settings?.minTmdbScore && typeof settings.minTmdbScore === 'number') {
          unratedResults = unratedResults.filter(item => 
            item && typeof item.vote_average === 'number' &&
            item.vote_average >= settings.minTmdbScore
          );
        }

        // Apply TMDB vote count filter
        if (settings?.minTmdbVoteCount && typeof settings.minTmdbVoteCount === 'number') {
          unratedResults = unratedResults.filter(item => 
            item && typeof item.vote_count === 'number' &&
            item.vote_count >= settings.minTmdbVoteCount
          );
        }

        if (settings && !settings.showAdultContent) {
          unratedResults = unratedResults.filter(item => !('adult' in item && !!item.adult));
        }

        setAllMoviesLocal(unratedResults);
        setMoviesLocal(unratedResults);
        
        // Call callback with fallback search result info
        onSearchComplete?.({
          totalResults: results.total_results || 0,
          searchType: 'content'
        });
      } catch (fallbackError) {
        if (fallbackError instanceof Error) {
          handleError(fallbackError, 'Yedek arama');
        } else {
          handleError(new Error(String(fallbackError)), 'Yedek arama');
        }
        setAllMoviesLocal([]);
        setMoviesLocal([]);
        onSearchComplete?.({ totalResults: 0, searchType: 'content' });
      }
    } finally {
      setLoading(false);
      loadingRef.current = false;
    }
  }, [profile, ratings, genres, tvGenres, settings, watchlist, handleError, safeSetState]);

  // Memoized functions to prevent recreation on every render
  const stableSetSearchQuery = useCallback((query: string) => {
    if (typeof query === 'string') {
      setSearchQueryLocal(query);
    }
  }, []);

  const stableSetRecommendationFilters = useCallback((filters: typeof recommendationFilters) => {
    if (filters && typeof filters === 'object') {
      setRecommendationFiltersLocal(filters);
    }
  }, []);

  const stableSetCuratedContentFilters = useCallback((filters: CuratedContentFilters) => {
    if (filters && typeof filters === 'object') {
      setCuratedContentFiltersLocal(filters);
    }
  }, []);

  const stableSetShowCuratedFilters = useCallback((show: boolean) => {
    setShowCuratedFiltersLocal(Boolean(show));
  }, []);

  // Load initial data only once with proper error handling
  useEffect(() => {
    if (!isInitialLoadRef.current) return;
    
    const loadInitialData = async () => {
      if (loadingRef.current) return;
      loadingRef.current = true;
      
      try {
        setLoading(true);
        setError(null);
        setLoadingProgress({ current: 0, total: 4, message: 'Film türleri yükleniyor...' });
        
        const [genresData, tvGenresData] = await Promise.all([
          tmdbService.fetchGenres().catch(err => {
            console.warn('Movie genres fetch failed:', err);
            return [];
          }),
          tmdbService.fetchTVGenres().catch(err => {
            console.warn('TV genres fetch failed:', err);
            return [];
          })
        ]);
        
        safeSetState(setGenresLocal, genresData, []);
        safeSetState(setTVGenresLocal, tvGenresData, []);
        setLoadingProgress({ current: 2, total: 4, message: 'Veriler yükleniyor...' });
        
        // Load stored data with validation
        const storedRatings = StorageService.getRatings();
        const storedWatchlist = StorageService.getWatchlist();
        
        // Validate stored data
        const validRatings = Array.isArray(storedRatings) ? storedRatings.filter(r => 
          r && typeof r === 'object' && 
          typeof r.movieId === 'number' && 
          r.movieId > 0 &&
          (typeof r.rating === 'number' || ['not_watched', 'not_interested', 'skip'].includes(r.rating as string))
        ) : [];
        
        const validWatchlist = Array.isArray(storedWatchlist) ? storedWatchlist.filter(w => 
          w && typeof w === 'object' && 
          typeof w.id === 'number' && 
          w.content
        ) : [];

        // Cache existing user content (rated and watchlisted)
        const contentToCache: Array<{ id: number; mediaType: 'movie' | 'tv' }> = [];
        
        // Add rated content to cache list
        validRatings.forEach(rating => {
          if (rating.rating !== 'skip') {
            contentToCache.push({
              id: rating.movieId,
              mediaType: rating.mediaType || 'movie'
            });
          }
        });
        
        // Add watchlisted content to cache list
        validWatchlist.forEach(item => {
          if (item.content && !contentToCache.some(c => c.id === item.id)) {
            const mediaType = 'media_type' in item.content && item.content.media_type === 'tv' ? 'tv' : 'movie';
            contentToCache.push({
              id: item.id,
              mediaType
            });
          }
        });
        
        // Cache all user content in background
        if (contentToCache.length > 0) {
          tmdbService.cacheMultipleUserContent(contentToCache);
        }
        
        // safeSetState(setWatchlist, validWatchlist, []); // Local storage kullanıldığı için gerek yok
        setLoadingProgress({ current: 3, total: 4, message: 'İçerikler hazırlanıyor...' });
        
        // Determine what movies to show based on learning phase
        await loadMoviesBasedOnPhase(profile, validRatings, genresData, tvGenresData);
        
        setLoadingProgress({ current: 4, total: 4, message: 'Öneriler hazırlanıyor...' });
        
        // Generate recommendations if profile exists and user has rated at least 10 items
        const validRatingCount = validRatings.filter(r => 
          typeof r.rating === 'number' && r.rating >= 1 && r.rating <= 10
        ).length;
        
        if (profile && validRatingCount >= 10) {
          try {
            setLoadingProgress({ current: 4, total: 5, message: 'AI önerileri hazırlanıyor...' });
            const watchlistIds = validWatchlist.map(w => w.id);
            const recs = await RecommendationService.generateRecommendations(
              profile, 
              genresData, 
              tvGenresData,
              validRatings,
              { ...recommendationFilters, showKidsContent: settings?.showKidsContent ?? false, showAnimationContent: settings?.showAnimationContent ?? true, showAnimeContent: settings?.showAnimeContent ?? true },
              settings?.recommendationCount !== undefined ? { recommendationCount: settings.recommendationCount } : {},
              watchlistIds
            );
            safeSetState(setRecommendationsLocal, recs, []);
            safeSetState(setFilteredRecommendationsLocal, recs, []);
            setLoadingProgress({ current: 5, total: 5, message: 'Öneriler hazırlandı!' });
          } catch (error) {
            console.warn('Recommendation generation failed:', error);
          }
        } else if (validRatingCount < 10) {
          // Clear any existing recommendations if user hasn't rated enough items
          safeSetState(setRecommendationsLocal, [] as any[], []);
          safeSetState(setFilteredRecommendationsLocal, [] as any[], []);
        }
      } catch (error) {
        if (error instanceof Error) {
          handleError(error, 'İlk veri yükleme');
        } else {
          handleError(new Error(String(error)), 'İlk veri yükleme');
        }
      } finally {
        setLoading(false);
        setLoadingProgress({ current: 0, total: 0, message: '' });
        loadingRef.current = false;
        isInitialLoadRef.current = false;
      }
    };

    loadInitialData();
  }, [handleError, safeSetState]);

  // Handle search with debouncing and abort controller
  useEffect(() => {
    // Abort previous request if exists
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Clear existing timeout
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    // Only proceed if search query actually changed
    if (lastSearchQueryRef.current === searchQuery) {
      return;
    }

    lastSearchQueryRef.current = searchQuery;

    // Set new timeout
    searchTimeoutRef.current = setTimeout(() => {
      if (!loadingRef.current) {
        searchMovies(searchQuery);
      }
    }, 500);

    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [searchQuery]);

  // Filter recommendations when filters change with error handling
  useEffect(() => {
    if (!Array.isArray(recommendations) || recommendations.length === 0) {
      safeSetState(setFilteredRecommendationsLocal, [], []);
      return;
    }
    
    try {
      const resolveMediaType = (item: Movie | TVShow) => {
        if (item.media_type) {
          return item.media_type;
        }
        return 'title' in item ? 'movie' : 'tv';
      };

      const resolveYear = (item: Movie | TVShow) => {
        const date = 'release_date' in item ? item.release_date : item.first_air_date;
        if (!date) {
          return 0;
        }
        const year = Number(date.slice(0, 4));
        return Number.isNaN(year) ? 0 : year;
      };

      const filterRecommendations = () => {
        let filtered = [...recommendations];

        // Çocuk içeriklerini her zaman filtrele (TMDb filtreleri ile uyumlu)
        filtered = filtered.filter(rec => {
          if (!rec?.movie?.genre_ids) return true;
          const isKidsGenre = rec.movie.genre_ids.includes(16) || rec.movie.genre_ids.includes(10751);
          return !isKidsGenre;
        });

        if (recommendationFilters.mediaType !== 'all') {
          filtered = filtered.filter(rec => resolveMediaType(rec.movie) === recommendationFilters.mediaType);
        }

        if (Array.isArray(recommendationFilters.genres) && recommendationFilters.genres.length > 0) {
          filtered = filtered.filter(rec => {
            if (!rec.movie?.genre_ids || !Array.isArray(rec.movie.genre_ids)) return false;
            return recommendationFilters.genres.some(genreId => rec.movie.genre_ids.includes(genreId));
          });
        }

        filtered = filtered.filter(rec => {
          if (!rec?.movie || typeof rec.movie.vote_average !== 'number' || typeof rec.movie.vote_count !== 'number') {
            return false;
          }
          const minRating = Math.max(6.0, recommendationFilters.minRating);
          return rec.movie.vote_average >= minRating && 
            rec.movie.vote_average <= recommendationFilters.maxRating &&
            rec.movie.vote_count >= 100;
        });

        filtered = filtered.filter(rec => {
          const year = resolveYear(rec.movie);
          if (!year) return true;
          return year >= recommendationFilters.minYear && year <= recommendationFilters.maxYear;
        });

        filtered = filtered.filter(rec => (rec.matchScore || 0) >= recommendationFilters.minMatchScore);

        if (Array.isArray(recommendationFilters.languages) && recommendationFilters.languages.length > 0) {
          filtered = filtered.filter(rec => {
            const language = rec.movie?.original_language;
            return language ? recommendationFilters.languages?.includes(language) : false;
          });
        }

        if (recommendationFilters.showAnimationContent === false) {
          filtered = filtered.filter(rec => {
            if (!rec?.movie?.genre_ids) return true;
            return !rec.movie.genre_ids.includes(16);
          });
        }

        if (recommendationFilters.showAnimeContent === false) {
          filtered = filtered.filter(rec => {
            if (!rec?.movie?.genre_ids) return true;
            const isAnime = rec.movie.genre_ids.includes(16) && rec.movie.original_language === 'ja';
            return !isAnime;
          });
        }

        // Sort with error handling
        filtered.sort((a, b) => {
          try {
            switch (recommendationFilters.sortBy) {
              case 'match_score':
                return (b.matchScore || 0) - (a.matchScore || 0);
              case 'rating':
                return (b.movie?.vote_average || 0) - (a.movie?.vote_average || 0);
              case 'year':
                return resolveYear(b.movie) - resolveYear(a.movie);
              case 'title':
                const titleA = 'title' in (a.movie || {}) ? (a.movie as any).title : (a.movie as any).name;
                const titleB = 'title' in (b.movie || {}) ? (b.movie as any).title : (b.movie as any).name;
                return (titleA || '').localeCompare(titleB || '', 'tr');
              default:
                return 0;
            }
          } catch (error) {
            console.warn('Sort error:', error);
            return 0;
          }
        });

        safeSetState(setFilteredRecommendationsLocal, filtered, []);
      };

      filterRecommendations();
    } catch (error) {
      if (error instanceof Error) {
        handleError(error, 'Öneri filtreleme');
      } else {
        handleError(new Error(String(error)), 'Öneri filtreleme');
      }
    }
  }, [
    recommendations,
    recommendationFilters,
    handleError,
    safeSetState
  ]);

  // Filter curated content when filters change with error handling
  useEffect(() => {
    if (!showingCuratedMovies || !Array.isArray(allMovies) || allMovies.length === 0) return;
    
    try {
      const filterCuratedContent = () => {
        let filtered = [...allMovies];

        // Apply content quality filter from settings
        if (settings?.minContentRating && typeof settings.minContentRating === 'number') {
          filtered = filtered.filter(content => 
            content && typeof content.vote_average === 'number' &&
            content.vote_average >= settings.minContentRating
          );
        }

        // Apply adult content filter from settings
        if (settings && !settings.showAdultContent) {
          filtered = filtered.filter(content => !('adult' in content && !!content.adult));
        }

        // Media type filter
        if (curatedContentFilters.mediaType !== 'all') {
          filtered = filtered.filter(content => {
            if (!content) return false;
            if (curatedContentFilters.mediaType === 'movie') {
              return content.media_type === 'movie' || 'title' in content;
            } else {
              return content.media_type === 'tv' || 'name' in content;
            }
          });
        }

        // Rating filter with null checks
        filtered = filtered.filter(content => {
          if (!content || typeof content.vote_average !== 'number') return false;
          return content.vote_average >= curatedContentFilters.minRating && 
                 content.vote_average <= curatedContentFilters.maxRating;
        });

        // Year filter with null checks
        filtered = filtered.filter(content => {
          if (!content) return false;
          let year: number;
          if ('release_date' in content && content.release_date) {
            year = new Date(content.release_date).getFullYear();
          } else if ('first_air_date' in content && content.first_air_date) {
            year = new Date(content.first_air_date).getFullYear();
          } else {
            return true;
          }
          
          return year >= curatedContentFilters.minYear && year <= curatedContentFilters.maxYear;
        });

        // Vote count filter
        filtered = filtered.filter(content => {
          if (!content || typeof content.vote_count !== 'number') return false;
          return content.vote_count >= curatedContentFilters.minVoteCount;
        });

        // Genre filter
        if (Array.isArray(curatedContentFilters.genres) && curatedContentFilters.genres.length > 0) {
          filtered = filtered.filter(content => {
            if (!content?.genre_ids || !Array.isArray(content.genre_ids)) return false;
            return curatedContentFilters.genres.some(genreId => 
              content.genre_ids.includes(genreId)
            );
          });
        }

        // Language filter
        if (curatedContentFilters.languages && Array.isArray(curatedContentFilters.languages) && curatedContentFilters.languages.length > 0) {
          filtered = filtered.filter(content => {
            if (!content?.original_language) return false;
            return curatedContentFilters.languages!.includes(content.original_language);
          });
        }

        // Sort with error handling
        filtered.sort((a, b) => {
          try {
            let valueA: any, valueB: any;
            
            switch (curatedContentFilters.sortBy) {
              case 'rating':
                valueA = a?.vote_average || 0;
                valueB = b?.vote_average || 0;
                break;
              case 'year':
                const yearA = 'release_date' in (a || {}) ? 
                  new Date((a as any).release_date || '').getFullYear() : 
                  new Date((a as any).first_air_date || '').getFullYear();
                const yearB = 'release_date' in (b || {}) ? 
                  new Date((b as any).release_date || '').getFullYear() : 
                  new Date((b as any).first_air_date || '').getFullYear();
                valueA = yearA || 0;
                valueB = yearB || 0;
                break;
              case 'title':
                valueA = 'title' in (a || {}) ? (a as any).title : (a as any).name;
                valueB = 'title' in (b || {}) ? (b as any).title : (b as any).name;
                return curatedContentFilters.sortOrder === 'asc' 
                  ? (valueA || '').localeCompare(valueB || '', 'tr')
                  : (valueB || '').localeCompare(valueA || '', 'tr');
              case 'popularity':
                valueA = a?.vote_count || 0;
                valueB = b?.vote_count || 0;
                break;
              default:
                return 0;
            }
            
            if (curatedContentFilters.sortOrder === 'asc') {
              return valueA - valueB;
            } else {
              return valueB - valueA;
            }
          } catch (error) {
            console.warn('Sort error:', error);
            return 0;
          }
        });

        safeSetState(setMoviesLocal, filtered, []);
      };

      filterCuratedContent();
    } catch (error) {
      if (error instanceof Error) {
        handleError(error, 'İçerik filtreleme');
      } else {
        handleError(new Error(String(error)), 'İçerik filtreleme');
      }
    }
  }, [allMovies, curatedContentFilters, showingCuratedMovies, settings, handleError, safeSetState]);

  const loadMoviesBasedOnPhase = async (
    _profile: UserProfile | null, 
    _ratings: UserRating[], 
    _movieGenres: Genre[], 
    _tvGenres: Genre[]
  ) => {
    if (!Array.isArray(ratings)) {
      console.warn('Invalid ratings array provided');
      return;
    }

    // Arama ekranında sadece arama yapılabilsin, arama yoksa boş liste göster
    setAllMoviesLocal([]);
    setMoviesLocal([]);
    setShowingCuratedMovies(false);
  };

  // AI öğrenme içeriklerini yükle (sadece AI önerileri ekranında kullanılır)
  const loadAILearningContent = useCallback(async () => {
    if (curatedContentLoading) return;
    
    setCuratedContentLoading(true);
    setCuratedContentLoadingProgress({ current: 0, total: 10, message: 'AI öğrenme içerikleri yükleniyor...' });
    
    try {
      const validRatings = Array.isArray(ratings) ? ratings : [];
      const ratedIds = new Set(validRatings.map(r => r.movieId));
      const watchlistIds = new Set(watchlist.map(w => w.id));
      
      // CuratedMovieService'den AI öğrenme içeriklerini al
      const learningContent = await CuratedMovieService.getInitialRatingContent();
      
      // Puanlanmamış ve izleme listesinde olmayan içerikleri filtrele
      const filteredContent = learningContent.filter(content => 
        !ratedIds.has(content.id) && !watchlistIds.has(content.id)
      );
      
      setCuratedContentLoadingProgress({ current: 10, total: 10, message: 'Tamamlandı!' });
      
      setAllMoviesLocal(filteredContent);
      setMoviesLocal(filteredContent);
      setShowingCuratedMovies(true);
      
    } catch (error) {
      console.error('AI learning content loading failed:', error);
      setError('AI öğrenme içerikleri yüklenirken hata oluştu');
    } finally {
      setCuratedContentLoading(false);
      setCuratedContentLoadingProgress({ current: 0, total: 0, message: '' });
    }
  }, [ratings, watchlist, curatedContentLoading]);

  // Watchlist management with validation
  const removeFromWatchlist = useCallback((itemId: number) => {
    if (typeof itemId !== 'number' || itemId <= 0) {
      console.warn('Invalid itemId provided to removeFromWatchlist');
      return;
    }

    try {
      setWatchlist(prev => {
        const validPrev = Array.isArray(prev) ? prev : [];
        const updated = validPrev.filter(item => item.id !== itemId);
        StorageService.saveWatchlist(updated);
        
        // Note: Content will be available again in recommendations and discovery
        // when the lists are refreshed or regenerated
        return updated;
      });
    } catch (error) {
      if (error instanceof Error) {
        handleError(error, 'İzleme listesinden çıkarma');
      } else {
        handleError(new Error(String(error)), 'İzleme listesinden çıkarma');
      }
    }
  }, [handleError]);

  const addToWatchlist = useCallback((content: Movie | TVShow) => {
    if (!content || typeof content.id !== 'number' || content.id <= 0) {
      console.warn('Invalid content provided to addToWatchlist');
      return;
    }

    try {
      setWatchlist(prev => {
        const validPrev = Array.isArray(prev) ? prev : [];
        const exists = validPrev.some(item => item.id === content.id);
        if (!exists) {
          const updated = [
            ...validPrev,
            { id: content.id, content, addedAt: Date.now() }
          ];
          StorageService.saveWatchlist(updated);
          
          // Cache the watchlisted content
          const mediaType = 'media_type' in content && content.media_type === 'tv' ? 'tv' : 'movie';
          tmdbService.cacheUserContent(content.id, mediaType);
          
          return updated;
        }
        return validPrev;
      });
    } catch (error) {
      if (error instanceof Error) {
        handleError(error, 'İzleme listesine ekleme');
      } else {
        handleError(new Error(String(error)), 'İzleme listesine ekleme');
      }
    }
  }, [handleError]);

  const isInWatchlist = useCallback((itemId: number): boolean => {
    if (typeof itemId !== 'number' || itemId <= 0) return false;
    return Array.isArray(watchlist) && watchlist.some(item => item.id === itemId);
  }, [watchlist]);

  // Enhanced rate function with validation
  const rateMovie = useCallback(async (itemId: number, rating: number | 'not_watched' | 'not_interested' | 'skip', mediaType: 'movie' | 'tv' = 'movie') => {
    if (loadingRef.current) return;
    
    // Validation
    if (typeof itemId !== 'number' || itemId <= 0) {
      console.warn('Invalid itemId provided to rateMovie');
      return;
    }

    if (typeof rating === 'number' && (rating < 1 || rating > 10)) {
      console.warn('Invalid rating provided to rateMovie');
      return;
    }

    if (!['movie', 'tv'].includes(mediaType)) {
      console.warn('Invalid mediaType provided to rateMovie');
      return;
    }

    try {
      const validRatings = Array.isArray(ratings) ? ratings : [];
      
      // Check if rating already exists
      const existingRatingIndex = validRatings.findIndex(r => r.movieId === itemId);
      const existingRating = existingRatingIndex >= 0 ? validRatings[existingRatingIndex] : null;
      
      const newRatings = validRatings.filter(r => r.movieId !== itemId);
      newRatings.push({
        movieId: itemId,
        rating,
        timestamp: Date.now(),
        mediaType
      });

      safeSetState(setRatings, newRatings, []);
      StorageService.saveRatings(newRatings);

      // Cache the rated content
      if (rating !== 'skip') {
        tmdbService.cacheUserContent(itemId, mediaType);
      }

      // Remove from watchlist if rated (except for skip/not_interested)
      if (rating !== 'not_watched' && rating !== 'skip') {
        removeFromWatchlist(itemId);
      }

      let currentProfile = profile;

      // Update profile if enough valid ratings
      const validNewRatings = newRatings.filter(r => 
        r.rating !== 'not_watched' && 
        r.rating !== 'not_interested' && 
        r.rating !== 'skip'
      );
      
      // Anında UI güncellemesi - puanlanan filmi listeden kaldır
      setAllMoviesLocal((prev: any) => Array.isArray(prev) ? prev.filter((movie: any) => movie.id !== itemId) : []);
      setMoviesLocal((prev: any) => Array.isArray(prev) ? prev.filter((movie: any) => movie.id !== itemId) : []);
      setRecommendationsLocal((prev: any) => Array.isArray(prev) ? prev.filter((rec: any) => rec.id !== itemId) : []);
      setFilteredRecommendationsLocal((prev: any) => Array.isArray(prev) ? prev.filter((rec: any) => rec.id !== itemId) : []);
      
      // 10 puanlama sonrası AI önerilerini başlat
      if (validNewRatings.length === 10 && (!Array.isArray(recommendations) || recommendations.length === 0) && currentProfile) {
        try {
          console.log('10 puanlama tamamlandı! AI önerileri başlatılıyor...');
          const watchlistIds = Array.isArray(watchlist) ? watchlist.map(w => w.id) : [];
          const recs = await RecommendationService.generateRecommendations(
            currentProfile, 
            genres, 
            tvGenres,
            newRatings,
            { ...recommendationFilters, showKidsContent: settings?.showKidsContent ?? false, showAnimationContent: settings?.showAnimationContent ?? true, showAnimeContent: settings?.showAnimeContent ?? true },
            settings?.recommendationCount !== undefined ? { recommendationCount: settings.recommendationCount } : {},
            watchlistIds
          );
          safeSetState(setRecommendationsLocal, recs, []);
          safeSetState(setFilteredRecommendationsLocal, recs, []);
        } catch (error) {
          console.warn('Initial AI recommendations failed:', error);
        }
      }

      // 20 puanlama sonrası neural network eğitimini başlat
      if (validNewRatings.length === 20 && currentProfile) {
        try {
          console.log('20 puanlama tamamlandı! Neural network eğitimi başlatılıyor...');
          const { NeuralRecommendationService } = await import('../services/neuralRecommendationService');
          await NeuralRecommendationService.trainModel(newRatings, currentProfile);
          console.log('Neural network eğitimi tamamlandı!');
        } catch (error) {
          console.warn('Neural network training failed:', error);
        }
      }

      // Her 10 puanlama sonrası neural network eğitimini güncelle (20'den sonra)
      if (validNewRatings.length >= 20 && validNewRatings.length % 10 === 0 && currentProfile) {
        try {
          console.log(`${validNewRatings.length} puanlama tamamlandı! Neural network güncelleniyor...`);
          const { NeuralRecommendationService } = await import('../services/neuralRecommendationService');
          await NeuralRecommendationService.trainModel(newRatings, currentProfile);
          console.log('Neural network güncelleme tamamlandı!');
        } catch (error) {
          console.warn('Neural network update failed:', error);
        }
      }

      // Gerçek zamanlı öğrenme eşiğini 3'e düşür
      if (validNewRatings.length >= 3) {
        try {
          // Process real-time learning event
          if (currentProfile) {
            const learningEvent = {
              type: (existingRatingIndex >= 0 ? 'rating_updated' : 'rating_added') as 'rating_updated' | 'rating_added',
              contentId: itemId,
              oldRating: existingRating?.rating,
              newRating: rating,
              mediaType
            };

            try {
              const learningResult = await RealTimeLearningService.processRatingEvent(
                learningEvent,
                currentProfile,
                newRatings
              );

              // Show learning insights if available
              if (learningResult.learningInsights.length > 0) {
                console.log('Learning insights:', learningResult.learningInsights);
              }

              // Retrain neural network if needed
              if (learningResult.shouldRetrainNeural) {
                await RealTimeLearningService.retrainNeuralNetworkIfNeeded(
                  true,
                  newRatings,
                  learningResult.updatedProfile
                );
              }

            } catch (learningError) {
              console.warn('Real-time learning failed, falling back to standard update:', learningError);
              
              // Fallback to standard profile update
              const newProfile = await ProfileService.generateProfile(newRatings);
              if (newProfile) {
                currentProfile = newProfile;
                safeSetState(setProfile, newProfile, null);
              }
            }
          } else {
            // No profile yet, use standard update
            const newProfile = await ProfileService.generateProfile(newRatings);
            if (newProfile) {
              currentProfile = newProfile;
              safeSetState(setProfile, newProfile, null);
            }
          }
        } catch (error) {
          console.warn('Profile generation failed:', error);
        }
      } else if (validNewRatings.length >= 1) {
        // 1-2 puanlama için basit profil güncellemesi
        try {
          const newProfile = await ProfileService.generateProfile(newRatings);
          if (newProfile) {
            currentProfile = newProfile;
            safeSetState(setProfile, newProfile, null);
          }
        } catch (error) {
          console.warn('Simple profile generation failed:', error);
        }
      }

      // Update movie list based on context - sadece puanlanan filmi kaldır, listeyi yeniden yükleme
      if (!searchQuery) {
        const currentPhase = currentProfile?.learningPhase || LearningService.determineLearningPhase(newRatings);
        
        if (currentPhase === 'testing') {
          // Test aşamasında sadece 2 film kaldıysa yeni içerik yükle
          const updatedMovies = Array.isArray(movies) ? movies.filter(movie => movie.id !== itemId) : [];
          if (updatedMovies.length <= 2 && currentProfile) {
            setLoading(true);
            loadingRef.current = true;
            try {
              const newTestContent = await LearningService.generateTestRecommendations(
                currentProfile, 
                genres,
                tvGenres,
                newRatings, 
                setLoadingProgress
              );
              safeSetState(setAllMoviesLocal, newTestContent, []);
              safeSetState(setMoviesLocal, newTestContent, []);
            } catch (error) {
              if (error instanceof Error) {
                handleError(error, 'Yeni test içeriği yükleme');
              } else {
                handleError(new Error(String(error)), 'Yeni test içeriği yükleme');
              }
            } finally {
              setLoading(false);
              loadingRef.current = false;
            }
          }
        }
        // Diğer aşamalarda sadece puanlanan filmi kaldır, listeyi yeniden yükleme
      } else {
        // Search mode - just remove rated item
        setAllMoviesLocal((prev: any) => Array.isArray(prev) ? prev.filter((movie: any) => movie.id !== itemId) : []);
        setMoviesLocal((prev: any) => Array.isArray(prev) ? prev.filter((movie: any) => movie.id !== itemId) : []);
      }
    } catch (error) {
      if (error instanceof Error) {
        handleError(error, 'Puanlama');
      } else {
        handleError(new Error(String(error)), 'Puanlama');
      }
    }
  }, [ratings, profile, genres, tvGenres, searchQuery, allMovies, movies, removeFromWatchlist, handleError, safeSetState]);

  // Refresh recommendations with error handling
  const refreshRecommendations = useCallback(async () => {
    if (!profile || recommendationsLoading) return;
    
    setRecommendationsLoading(true);
    setRecommendationsLoadingProgress({ current: 0, total: 3, message: 'Filtrelere göre yeni öneriler hazırlanıyor...' });
    setError(null);
    
    try {
      setRecommendationsLoadingProgress({ current: 1, total: 3, message: 'Filtrelere göre öneriler hesaplanıyor...' });
      
      // Filtreleri kullanarak yeni öneriler oluştur
      const newRecs = await RecommendationService.generateRecommendations(
        profile, 
        genres, 
        tvGenres,
        Array.isArray(ratings) ? ratings : [],
        recommendationFilters, // Filtreleri geçir
        settings?.recommendationCount !== undefined ? { recommendationCount: settings.recommendationCount } : {} // Ayarları geçir
      );
      
      // Filtrelenmiş önerileri doğrudan set et
      setRecommendationsLocal(newRecs);
      setFilteredRecommendationsLocal(newRecs);
      
      setRecommendationsLoadingProgress({ current: 3, total: 3, message: 'Tamamlandı!' });
      
    } catch (error) {
      if (error instanceof Error) {
        handleError(error, 'Öneri yenileme');
      } else {
        handleError(new Error(String(error)), 'Öneri yenileme');
      }
    } finally {
      setRecommendationsLoading(false);
      setRecommendationsLoadingProgress({ current: 0, total: 0, message: '' });
    }
  }, [profile, genres, tvGenres, ratings, recommendationFilters, handleError, safeSetState, recommendationsLoading]);

  // Refresh curated content with filters
  const refreshCuratedContent = useCallback(async () => {
    if (curatedContentLoading) return;
    
    setCuratedContentLoading(true);
    setCuratedContentLoadingProgress({ current: 0, total: 3, message: 'Filtrelere göre yeni keşif içerikleri hazırlanıyor...' });
    setError(null);
    
    try {
      setCuratedContentLoadingProgress({ current: 1, total: 3, message: 'Filtrelere göre içerikler aranıyor...' });
      
      // Filtreleri kullanarak yeni keşif içerikleri al
      const newContent = await CuratedMovieService.getCuratedContentWithFilters(
        ratings,
        curatedContentFilters
      );
      
      safeSetState(setAllMoviesLocal, newContent, []);
      safeSetState(setMoviesLocal, newContent, []);
      setShowingCuratedMovies(true);
      
      setCuratedContentLoadingProgress({ current: 3, total: 3, message: 'Tamamlandı!' });
      
    } catch (error) {
      if (error instanceof Error) {
        handleError(error, 'Keşif içerikleri yenileme');
      } else {
        handleError(new Error(String(error)), 'Keşif içerikleri yenileme');
      }
    } finally {
      setCuratedContentLoading(false);
      setCuratedContentLoadingProgress({ current: 0, total: 0, message: '' });
    }
  }, [ratings, curatedContentFilters, genres, tvGenres, handleError, safeSetState, curatedContentLoading]);

  // Get user rating with validation
  const getUserRating = useCallback((itemId: number): number | 'not_watched' | 'not_interested' | 'skip' | null => {
    if (typeof itemId !== 'number' || itemId <= 0) return null;
    if (!Array.isArray(ratings)) return null;
    
    const rating = ratings.find(r => r && r.movieId === itemId);
    return rating ? rating.rating : null;
  }, [ratings]);

  // Data management functions with validation
  const exportData = useCallback((): string => {
    try {
      return StorageService.exportData();
    } catch (error) {
      if (error instanceof Error) {
        handleError(error, 'Veri dışa aktarma');
      } else {
        handleError(new Error(String(error)), 'Veri dışa aktarma');
      }
      return '{}';
    }
  }, [handleError]);

  const importData = useCallback(async (jsonData: string): Promise<boolean> => {
    try {
      const success = await StorageService.importData(jsonData);
      if (success) {
        setRatings(StorageService.getRatings());
        setWatchlist(StorageService.getWatchlist());
        setProfile(StorageService.getProfile());
        // Optionally refresh recommendations, movies, etc.
        // You may want to call refreshRecommendations() or similar here
      }
      return success;
    } catch (error) {
      if (error instanceof Error) {
        handleError(error, 'Veri içe aktarma');
      } else {
        handleError(new Error(String(error)), 'Veri içe aktarma');
      }
      return false;
    }
  }, [handleError]);

  const clearData = useCallback(async () => {
    try {
      // Clear all storage data
      StorageService.clearAllData();
      
      // Clear real-time learning history
      RealTimeLearningService.clearLearningHistory();
      
      // Reset all state variables immediately
      setRatings([]);
      setWatchlist([]);
      setProfile(null);
      setRecommendationsLocal([]);
      setFilteredRecommendationsLocal([]);
      setAllMoviesLocal([]);
      setMoviesLocal([]);
      setSearchQueryLocal('');
      setShowingCuratedMovies(true);
      
      // Reset filters to defaults
      setRecommendationFiltersLocal({
        genres: [],
        minYear: 1950,
        maxYear: new Date().getFullYear(),
        minRating: 0,
        maxRating: 10,
        mediaType: 'all',
        sortBy: 'match_score',
        minMatchScore: 0,
        languages: [],
        showKidsContent: false,
        showAnimationContent: true,
        showAnimeContent: true
      });
      setCuratedContentFiltersLocal({
        mediaType: 'all',
        minRating: 0,
        maxRating: 10,
        minYear: 1900,
        maxYear: new Date().getFullYear(),
        genres: [],
        sortBy: 'rating',
        sortOrder: 'desc',
        minVoteCount: 0,
        languages: []
      });
      setShowCuratedFiltersLocal(false);
      
      // Reset loading states
      setLoading(false);
      loadingRef.current = false;
      setRecommendationsLoading(false);
      setCuratedContentLoading(false);
      setError(null);
      
      // Clear localStorage items
      localStorage.removeItem('searchQuery');
      localStorage.removeItem('showingCuratedMovies');
      localStorage.removeItem('recommendationFilters');
      localStorage.removeItem('curatedContentFilters');
      localStorage.removeItem('onboardingCompleted');
      localStorage.removeItem('onboardingState');
      
      // Load initial curated content
      setLoading(true);
      loadingRef.current = true;
      try {
        const curatedContent = await CuratedMovieService.getCuratedInitialContent([], undefined, [], 
          {}
        );
        safeSetState(setAllMoviesLocal, curatedContent, []);
        safeSetState(setMoviesLocal, curatedContent, []);
        setShowingCuratedMovies(true);
      } catch (error) {
        if (error instanceof Error) {
          handleError(error, 'Varsayılan içerik yükleme');
        } else {
          handleError(new Error(String(error)), 'Varsayılan içerik yükleme');
        }
      } finally {
        setLoading(false);
        loadingRef.current = false;
      }
      
      console.log('All data cleared and system reset to initial state');
    } catch (error) {
      if (error instanceof Error) {
        handleError(error, 'Veri temizleme');
      } else {
        handleError(new Error(String(error)), 'Veri temizleme');
      }
    }
  }, [handleError, safeSetState]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const mergedGenres = useMemo(() => {
    const uniqueGenres = new Map<number, Genre>();
    (Array.isArray(genres) ? genres : []).forEach(genre => uniqueGenres.set(genre.id, genre));
    (Array.isArray(tvGenres) ? tvGenres : []).forEach(genre => {
      if (!uniqueGenres.has(genre.id)) {
        uniqueGenres.set(genre.id, genre);
      }
    });
    return Array.from(uniqueGenres.values());
  }, [genres, tvGenres]);

  return {
    user: null,
    profile,
    ratings,
    watchlist,
    addToWatchlist,
    removeFromWatchlist,
    rateMovie,
    removeRating: undefined,
    updateProfile: undefined,
    movies: Array.isArray(movies) ? movies : [],
    allMovies: Array.isArray(allMovies) ? allMovies : [],
    genres: mergedGenres,
    recommendations: Array.isArray(recommendations) ? recommendations : [],
    filteredRecommendations: Array.isArray(filteredRecommendations) ? filteredRecommendations : [],
    loading,
    _loadingProgress: loadingProgress,
    recommendationsLoading,
    recommendationsLoadingProgress,
    curatedContentLoading,
    curatedContentLoadingProgress,
    searchQuery: searchQuery || '',
    showingCuratedMovies,
    recommendationFilters,
    curatedContentFilters,
    showCuratedFilters,
    error,
    setSearchQuery: stableSetSearchQuery,
    setRecommendationFilters: stableSetRecommendationFilters,
    setCuratedContentFilters: stableSetCuratedContentFilters,
    setShowCuratedFilters: stableSetShowCuratedFilters,
    searchMovies,
    getUserRating,
    isInWatchlist,
    exportData,
    importData,
    clearData,
    refreshRecommendations,
    refreshCuratedContent,
    loadAILearningContent
  };
};
