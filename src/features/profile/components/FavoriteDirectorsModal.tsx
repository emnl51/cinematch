import React, { useState, useEffect } from 'react';
import { X, Film, Star, Calendar, AlertCircle, Loader, Bookmark, BookMarked, EyeOff } from 'lucide-react';
import { tmdbService } from '../../content/services/tmdb';
import { ContentFiltersComponent, type ContentFilters } from '../../recommendation/components/ContentFilters';
import type { UserProfile, Movie, TVShow, Genre } from '../types';

interface FavoriteDirectorsModalProps {
  isOpen: boolean;
  onClose: () => void;
  profile: UserProfile;
  genres: Genre[];
  getUserRating: (itemId: number) => number | 'not_watched' | 'not_interested' | 'skip' | null;
  onRateContent: (itemId: number, rating: number | 'not_watched', mediaType?: 'movie' | 'tv') => void;
  isInWatchlist: (itemId: number) => boolean;
  onAddToWatchlist?: (content: Movie | TVShow) => void;
  onRemoveFromWatchlist: (itemId: number) => void;
}

interface Credit {
  id: number;
  title?: string;
  name?: string;
  media_type: 'movie' | 'tv';
  poster_path: string | null;
  vote_average?: number;
  vote_count?: number;
  release_date?: string;
  first_air_date?: string;
  genre_ids?: number[];
  overview?: string;
  adult?: boolean;
  job?: string;
}

interface DirectorContent {
  director: { id: number; name: string; count: number };
  movies: Credit[];
  loading: boolean;
  error: string | undefined;
}

export const FavoriteDirectorsModal: React.FC<FavoriteDirectorsModalProps> = ({
  isOpen,
  onClose,
  profile,
  genres,
  getUserRating,
  onRateContent,
  isInWatchlist,
  onAddToWatchlist,
  onRemoveFromWatchlist
}) => {
  const [directorContents, setDirectorContents] = useState<DirectorContent[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedDirectorIndex, setSelectedDirectorIndex] = useState<number | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [showRatingSlider, setShowRatingSlider] = useState<number | null>(null);
  const [tempRating, setTempRating] = useState<number>(5);
  
  const [filters, setFilters] = useState<ContentFilters>({
    mediaType: 'all',
    minRating: 0,
    maxRating: 10,
    minYear: 1900,
    maxYear: new Date().getFullYear(),
    genres: [],
    sortBy: 'rating',
    sortOrder: 'desc',
    minVoteCount: 0
  });

  useEffect(() => {
    if (isOpen) {
      loadDirectorContents();
    }
  }, [isOpen, profile]);

  const loadDirectorContents = async (): Promise<void> => {
    setLoading(true);
    
    const topDirectors = Object.entries(profile.favoriteDirectors)
      .map(([id, directorData]) => ({
        id: parseInt(id, 10),
        name: directorData.name,
        count: directorData.count
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);

    const directorData: DirectorContent[] = topDirectors.map(director => ({
      director,
      movies: [],
      loading: true,
      error: undefined
    }));

    setDirectorContents(directorData);

    const loadPromises = directorData.map(async (directorData, index) => {
      try {
        console.log(`Loading content for director: ${directorData.director.name} (ID: ${directorData.director.id})`);
        
        // Yönetmen için film ve dizi kredilerini al
        const creditsData = await tmdbService.getPersonCombinedCredits(directorData.director.id);
        
        console.log(`${directorData.director.name} API response:`, {
          totalCrew: creditsData.crew?.length || 0
        });

        // Sadece yönetmenlik yaptığı içerikleri filtrele
        const directorCredits = creditsData.crew.filter(credit => 
          credit.job === 'Director'
        );

        console.log(`${directorData.director.name} - Director credits:`, directorCredits.length);

        // Puanlanmamış içerikleri filtrele
        const filteredContent = directorCredits.filter((item: any) => {
          const hasId = item.id && item.id > 0;
          const hasTitle = (item.title && item.title.trim()) || (item.name && item.name.trim());
          const notAdult = !item.adult;
          
          const userRating = getUserRating(item.id);
          const notRated = !userRating || userRating === 'not_watched';
          
          return hasId && hasTitle && notAdult && notRated;
        });

        console.log(`${directorData.director.name} - After filtering (unrated only):`, filteredContent.length);

        // Kalite ve popülerliğe göre sırala
        const sortedContent = filteredContent.sort((a: any, b: any) => {
          const getScore = (item: any) => {
            const rating = Math.max(0, item.vote_average || 0);
            const popularity = Math.max(1, item.vote_count || 1);
            const logPopularity = Math.log10(popularity);
            return (rating * 0.6) + (logPopularity * 0.4);
          };
          
          return getScore(b) - getScore(a);
        });

        const topContent = sortedContent.slice(0, 30);

        console.log(`${directorData.director.name} - Final content:`, topContent.length);
        
        return {
          index,
          movies: topContent,
          error: topContent.length === 0 ? 'Bu yönetmen için puanlanmamış içerik bulunamadı' : undefined
        };
      } catch (error) {
        console.error(`Error loading content for director ${directorData.director.name}:`, error);
        return {
          index,
          movies: [],
          error: 'API hatası oluştu'
        };
      }
    });

    try {
      const results = await Promise.all(loadPromises);
      
      results.forEach(result => {
        setDirectorContents(prev => prev.map((item, index) => 
          index === result.index ? { 
            ...item, 
            movies: result.movies, 
            loading: false,
            error: result.error
          } : item
        ));
      });
    } catch (error) {
      console.error('Error in loadPromises:', error);
      setDirectorContents(prev => prev.map(item => ({
        ...item,
        loading: false,
        error: 'Genel API hatası'
      })));
    }

    setLoading(false);
  };

  // Enhanced rating handler that preserves filtering
  const handleRateContent = (itemId: number, rating: number | 'not_watched', mediaType?: 'movie' | 'tv') => {
    // Call the original rating function
    onRateContent(itemId, rating, mediaType);
    
    // Update the director contents to remove the rated item while preserving the current filter state
    setDirectorContents(prev => prev.map(directorContent => ({
      ...directorContent,
      movies: directorContent.movies.filter(movie => movie.id !== itemId)
    })));
  };

  const handleSliderRating = (contentId: number, rating: number, mediaType?: 'movie' | 'tv') => {
    handleRateContent(contentId, rating, mediaType);
    setShowRatingSlider(null);
  };

  const openRatingSlider = (contentId: number) => {
    const currentRating = getUserRating(contentId);
    setTempRating(typeof currentRating === 'number' ? currentRating : 5);
    setShowRatingSlider(contentId);
  };

  const getFilteredContent = (content: Credit[]): Credit[] => {
    let filtered = [...content];

    if (filters.mediaType !== 'all') {
      filtered = filtered.filter(item => {
        if (filters.mediaType === 'movie') {
          return item.media_type === 'movie' || 'title' in item;
        } else {
          return item.media_type === 'tv' || 'name' in item;
        }
      });
    }

    filtered = filtered.filter(item => {
      const rating = item.vote_average || 0;
      return rating >= filters.minRating && rating <= filters.maxRating;
    });

    filtered = filtered.filter(item => {
      let year: number;
      if ('release_date' in item && item.release_date) {
        year = new Date(item.release_date).getFullYear();
      } else if ('first_air_date' in item && item.first_air_date) {
        year = new Date(item.first_air_date).getFullYear();
      } else {
        return true;
      }
      
      return year >= filters.minYear && year <= filters.maxYear;
    });

    if (filters.genres.length > 0) {
      filtered = filtered.filter(item => {
        const itemGenres = item.genre_ids || [];
        return filters.genres.some(genreId => itemGenres.includes(genreId));
      });
    }

    filtered.sort((a, b) => {
      let valueA: any, valueB: any;
      
      switch (filters.sortBy) {
        case 'rating':
          valueA = a.vote_average || 0;
          valueB = b.vote_average || 0;
          break;
        case 'year':
          const yearA = 'release_date' in a ? 
            new Date(a.release_date || '').getFullYear() : 
            new Date(a.first_air_date || '').getFullYear();
          const yearB = 'release_date' in b ? 
            new Date(b.release_date || '').getFullYear() : 
            new Date(b.first_air_date || '').getFullYear();
          valueA = yearA;
          valueB = yearB;
          break;
        case 'title':
          valueA = 'title' in a ? a.title : a.name;
          valueB = 'title' in b ? b.title : b.name;
          return filters.sortOrder === 'asc' 
            ? valueA.localeCompare(valueB, 'tr')
            : valueB.localeCompare(valueA, 'tr');
        case 'popularity':
          valueA = a.vote_count || 0;
          valueB = b.vote_count || 0;
          break;
        default:
          return 0;
      }
      
      if (filters.sortOrder === 'asc') {
        return valueA - valueB;
      } else {
        return valueB - valueA;
      }
    });

    return filtered;
  };

  const getGenreNames = (genreIds: number[]): string => {
    if (!genreIds || genreIds.length === 0) return '';
    
    return genreIds
      .map(id => genres.find(g => g.id === id)?.name)
      .filter(Boolean)
      .slice(0, 2)
      .join(', ') || '';
  };

  const getYear = (content: Credit): string => {
    if ('release_date' in content && content.release_date) {
      return new Date(content.release_date).getFullYear().toString();
    } else if ('first_air_date' in content && content.first_air_date) {
      return new Date(content.first_air_date).getFullYear().toString();
    }
    return 'Bilinmiyor';
  };

  const getTitle = (content: Credit): string => {
    return 'title' in content ? content.title || '' : content.name || '';
  };

  const getRatingText = (rating: number): string => {
    if (rating >= 9) return 'Mükemmel';
    if (rating >= 8) return 'Çok İyi';
    if (rating >= 7) return 'İyi';
    if (rating >= 6) return 'Fena Değil';
    if (rating >= 5) return 'Orta';
    if (rating >= 4) return 'Zayıf';
    if (rating >= 3) return 'Kötü';
    return 'Çok Kötü';
  };

  const getRatingColor = (rating: number): string => {
    if (rating >= 9) return 'text-green-400';
    if (rating >= 7) return 'text-amber-400';
    if (rating >= 5) return 'text-orange-400';
    return 'text-red-400';
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      <div className="bg-slate-800 rounded-xl max-w-7xl w-full max-h-[90vh] overflow-hidden">
        <div className="flex items-center justify-between p-6 border-b border-slate-700">
          <div className="flex items-center space-x-3">
            <Film className="h-6 w-6 text-blue-400" />
            <h2 className="text-xl font-bold text-white">Favori Yönetmenler ve İçerikleri</h2>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        <div className="p-6 overflow-y-auto max-h-[calc(90vh-120px)]">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader className="animate-spin h-8 w-8 text-blue-500" />
              <span className="ml-3 text-slate-300">Yönetmen içerikleri yükleniyor...</span>
            </div>
          ) : (
            <div className="space-y-8">
              <div className="flex flex-wrap gap-2 mb-6">
                <button
                  onClick={() => setSelectedDirectorIndex(null)}
                  className={`px-4 py-2 rounded-lg text-sm transition-colors ${
                    selectedDirectorIndex === null
                      ? 'bg-blue-500 text-white'
                      : 'bg-slate-600 text-slate-300 hover:bg-slate-500'
                  }`}
                >
                  Tüm Yönetmenler
                </button>
                {directorContents.map((directorContent, index) => (
                  <button
                    key={directorContent.director.id}
                    onClick={() => setSelectedDirectorIndex(index)}
                    className={`px-4 py-2 rounded-lg text-sm transition-colors ${
                      selectedDirectorIndex === index
                        ? 'bg-blue-500 text-white'
                        : 'bg-slate-600 text-slate-300 hover:bg-slate-500'
                    }`}
                  >
                    {directorContent.director.name}
                  </button>
                ))}
              </div>

              {selectedDirectorIndex !== null ? (
                <div>
                  {(() => {
                    const directorContent = directorContents[selectedDirectorIndex];
                    const filteredMovies = getFilteredContent(directorContent.movies);
                    
                    return (
                      <div>
                        <div className="flex items-center justify-between mb-6">
                          <div>
                            <h3 className="text-xl font-semibold text-white mb-1">
                              {directorContent.director.name}
                            </h3>
                            <p className="text-slate-400 text-sm">
                              Ortalama puanın: {directorContent.director.count.toFixed(1)}/10
                            </p>
                          </div>
                          <div className="text-right">
                            <span className="bg-blue-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                              #{selectedDirectorIndex + 1} Favori
                            </span>
                          </div>
                        </div>

                        <div className="mb-6">
                          <ContentFiltersComponent
                            filters={filters}
                            onFiltersChange={setFilters}
                            genres={genres}
                            isOpen={showFilters}
                            onToggle={() => setShowFilters(!showFilters)}
                            totalCount={directorContent.movies.length}
                            filteredCount={filteredMovies.length}
                          />
                        </div>

                        {directorContent.loading ? (
                          <div className="flex items-center justify-center py-12">
                            <Loader className="animate-spin h-6 w-6 text-blue-500" />
                            <span className="ml-2 text-slate-400">Filmografi yükleniyor...</span>
                          </div>
                        ) : directorContent.error ? (
                          <div className="text-center py-12">
                            <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
                            <p className="text-red-400 text-lg mb-2">
                              {directorContent.error}
                            </p>
                          </div>
                        ) : filteredMovies.length > 0 ? (
                          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
                            {filteredMovies.map((content) => (
                              <div key={`${content.id}-${content.media_type}`} className="bg-slate-600 rounded-lg overflow-hidden hover:bg-slate-500 transition-all duration-300 hover:scale-105">
                                <div className="aspect-[2/3] relative">
                                  <img
                                    src={tmdbService.getImageUrl(content.poster_path || null, 'w342')}
                                    alt={getTitle(content)}
                                    className="w-full h-full object-cover"
                                    onError={(e) => {
                                      (e.target as HTMLImageElement).src = 'https://placehold.co/300x450/374151/f8fafc?text=Poster+Yok';
                                    }}
                                  />
                                  
                                  <div className="absolute top-2 right-2 bg-black/80 backdrop-blur-sm rounded px-2 py-1">
                                    <div className="flex items-center space-x-1 text-yellow-400">
                                      <Star className="h-3 w-3 fill-current" />
                                      <span className="text-xs font-medium">
                                        {content.vote_average ? content.vote_average.toFixed(1) : 'N/A'}
                                      </span>
                                    </div>
                                  </div>
                                  
                                  <div className="absolute top-2 left-2 bg-black/80 backdrop-blur-sm rounded px-2 py-1">
                                    <span className="text-xs text-white font-medium">
                                      {content.media_type === 'tv' ? 'Dizi' : 'Film'}
                                    </span>
                                  </div>

                                  <div className="absolute bottom-2 left-2 bg-black/80 backdrop-blur-sm rounded px-2 py-1">
                                    <div className="flex items-center space-x-1">
                                      <Calendar className="h-3 w-3 text-slate-300" />
                                      <span className="text-xs text-white">{getYear(content)}</span>
                                    </div>
                                  </div>
                                </div>
                                
                                <div className="p-3">
                                  <h4 className="font-medium text-white text-sm mb-2 line-clamp-2 leading-tight">
                                    {getTitle(content)}
                                  </h4>
                                  
                                  {content.genre_ids && content.genre_ids.length > 0 && (
                                    <p className="text-slate-300 text-xs mb-2">
                                      {getGenreNames(content.genre_ids)}
                                    </p>
                                  )}
                                  
                                  {content.overview && (
                                    <p className="text-slate-400 text-xs line-clamp-2 mb-3">
                                      {content.overview}
                                    </p>
                                  )}

                                  <div className="space-y-2">
                                    {/* Current Rating Display */}
                                    {(() => {
                                      const userRating = getUserRating(content.id);
                                      return userRating && typeof userRating === 'number' && (
                                        <div className="flex items-center justify-between bg-gradient-to-r from-slate-600/50 to-slate-500/50 rounded-lg p-2 border border-slate-500/50">
                                          <span className="text-white text-xs font-medium">Puanın:</span>
                                          <div className="flex items-center space-x-1">
                                            <span className={`text-sm font-bold ${getRatingColor(userRating)}`}>
                                              {userRating}/10
                                            </span>
                                            <span className="text-slate-400 text-xs">
                                              ({getRatingText(userRating)})
                                            </span>
                                          </div>
                                        </div>
                                      );
                                    })()}

                                    {/* Rating Slider */}
                                    {showRatingSlider === content.id ? (
                                      <div className="bg-slate-600/50 rounded-lg p-3 border border-slate-500/50">
                                        <div className="mb-3">
                                          <div className="flex items-center justify-between mb-2">
                                            <span className="text-white text-sm font-medium">Puan: {tempRating}/10</span>
                                            <span className={`text-xs ${getRatingColor(tempRating)}`}>
                                              {getRatingText(tempRating)}
                                            </span>
                                          </div>
                                          <input
                                            type="range"
                                            min="1"
                                            max="10"
                                            step="1"
                                            value={tempRating}
                                            onChange={(e) => setTempRating(parseInt(e.target.value))}
                                            className="w-full h-2 bg-slate-600 rounded-lg appearance-none cursor-pointer slider"
                                          />
                                          <div className="flex justify-between text-xs text-slate-400 mt-1">
                                            <span>1</span>
                                            <span>5</span>
                                            <span>10</span>
                                          </div>
                                        </div>
                                        
                                        <div className="flex space-x-2">
                                          <button
                                            onClick={() => handleSliderRating(content.id, tempRating, content.media_type === 'tv' ? 'tv' : 'movie')}
                                            className="flex-1 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200"
                                          >
                                            Puanla
                                          </button>
                                          <button
                                            onClick={() => setShowRatingSlider(null)}
                                            className="px-3 py-2 bg-slate-600 hover:bg-slate-500 text-white rounded-lg text-xs transition-colors"
                                          >
                                            İptal
                                          </button>
                                        </div>
                                      </div>
                                    ) : (
                                      <button
                                        onClick={() => openRatingSlider(content.id)}
                                        className={`w-full flex items-center justify-center space-x-2 px-3 py-2 rounded-lg transition-all duration-200 text-sm ${
                                          getUserRating(content.id) && typeof getUserRating(content.id) === 'number'
                                            ? 'bg-gradient-to-r from-green-500/20 to-emerald-500/20 border border-green-500/30 text-green-300 hover:bg-green-500/30'
                                            : 'bg-gradient-to-r from-blue-500/20 to-cyan-500/20 border border-blue-500/30 text-blue-300 hover:bg-blue-500/30'
                                        }`}
                                      >
                                        <Star className="h-3 w-3" />
                                        <span>
                                          {getUserRating(content.id) && typeof getUserRating(content.id) === 'number' 
                                            ? 'Puanı Değiştir' 
                                            : 'Puan Ver (1-10)'
                                          }
                                        </span>
                                      </button>
                                    )}

                                    <div className="flex items-center justify-between">
                                      <button
                                        onClick={() => handleRateContent(content.id, 'not_watched', content.media_type === 'tv' ? 'tv' : 'movie')}
                                        className={`flex items-center space-x-1 px-2 py-1 rounded text-xs transition-colors ${
                                          getUserRating(content.id) === 'not_watched'
                                            ? 'bg-slate-600 text-white'
                                            : 'bg-slate-700 text-slate-400 hover:bg-slate-600 hover:text-white'
                                        }`}
                                      >
                                        <EyeOff className="h-3 w-3" />
                                        <span>İzlemedim</span>
                                      </button>

                                      <button
                                        onClick={() => {
                                          if (isInWatchlist(content.id)) {
                                            onRemoveFromWatchlist(content.id);
                                          } else {
                                            onAddToWatchlist?.(content as Movie | TVShow);
                                          }
                                        }}
                                        className={`flex items-center space-x-1 px-2 py-1 rounded text-xs transition-colors ${
                                          isInWatchlist(content.id)
                                            ? 'bg-green-500 text-white hover:bg-green-600'
                                            : 'bg-slate-700 text-slate-400 hover:bg-slate-600 hover:text-white'
                                        }`}
                                      >
                                        {isInWatchlist(content.id) ? (
                                          <>
                                            <BookMarked className="h-3 w-3" />
                                            <span>Listede</span>
                                          </>
                                        ) : (
                                          <>
                                            <Bookmark className="h-3 w-3" />
                                            <span>Listeye Ekle</span>
                                          </>
                                        )}
                                      </button>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="text-center py-12">
                            <Film className="h-12 w-12 text-slate-500 mx-auto mb-4" />
                            <p className="text-slate-400 text-lg mb-2">
                              Filtrelere uygun içerik bulunamadı
                            </p>
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>
              ) : (
                directorContents.map((directorContent, index) => (
                  <div key={directorContent.director.id} className="bg-slate-700 rounded-lg p-6">
                    <div className="flex items-center justify-between mb-6">
                      <div>
                        <h3 className="text-xl font-semibold text-white mb-1">
                          {directorContent.director.name}
                        </h3>
                        <p className="text-slate-400 text-sm">
                          Ortalama puanın: {directorContent.director.count.toFixed(1)}/10
                        </p>
                      </div>
                      <div className="flex items-center space-x-3">
                        <button
                          onClick={() => setSelectedDirectorIndex(index)}
                          className="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded-lg text-sm transition-colors"
                        >
                          Filtrele
                        </button>
                        <span className="bg-blue-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                          #{index + 1} Favori
                        </span>
                      </div>
                    </div>

                    {directorContent.loading ? (
                      <div className="flex items-center justify-center py-12">
                        <Loader className="animate-spin h-6 w-6 text-blue-500" />
                        <span className="ml-2 text-slate-400">Filmografi yükleniyor...</span>
                      </div>
                    ) : directorContent.error ? (
                      <div className="text-center py-12">
                        <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
                        <p className="text-red-400 text-lg mb-2">
                          {directorContent.error}
                        </p>
                      </div>
                    ) : directorContent.movies.length > 0 ? (
                      <>
                        <div className="mb-4">
                          <p className="text-slate-300 text-sm">
                            {directorContent.movies.length} puanlanmamış içerik bulundu
                          </p>
                        </div>
                        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-4">
                          {directorContent.movies.slice(0, 12).map((content) => (
                            <div key={`${content.id}-${content.media_type}`} className="bg-slate-600 rounded-lg overflow-hidden hover:bg-slate-500 transition-all duration-300 hover:scale-105">
                              <div className="aspect-[2/3] relative">
                                <img
                                  src={tmdbService.getImageUrl(content.poster_path || null, 'w342')}
                                  alt={getTitle(content)}
                                  className="w-full h-full object-cover"
                                  onError={(e) => {
                                    (e.target as HTMLImageElement).src = 'https://placehold.co/300x450/374151/f8fafc?text=Poster+Yok';
                                  }}
                                />
                                
                                <div className="absolute top-2 right-2 bg-black/80 backdrop-blur-sm rounded px-2 py-1">
                                  <div className="flex items-center space-x-1 text-yellow-400">
                                    <Star className="h-3 w-3 fill-current" />
                                    <span className="text-xs font-medium">
                                      {content.vote_average ? content.vote_average.toFixed(1) : 'N/A'}
                                    </span>
                                  </div>
                                </div>
                                
                                <div className="absolute top-2 left-2 bg-black/80 backdrop-blur-sm rounded px-2 py-1">
                                  <span className="text-xs text-white font-medium">
                                    {content.media_type === 'tv' ? 'Dizi' : 'Film'}
                                  </span>
                                </div>

                                <div className="absolute bottom-2 left-2 bg-black/80 backdrop-blur-sm rounded px-2 py-1">
                                  <div className="flex items-center space-x-1">
                                    <Calendar className="h-3 w-3 text-slate-300" />
                                    <span className="text-xs text-white">{getYear(content)}</span>
                                  </div>
                                </div>
                              </div>
                              
                              <div className="p-3">
                                <h4 className="font-medium text-white text-sm mb-2 line-clamp-2 leading-tight">
                                  {getTitle(content)}
                                </h4>
                                
                                {content.genre_ids && content.genre_ids.length > 0 && (
                                  <p className="text-slate-300 text-xs mb-2">
                                    {getGenreNames(content.genre_ids)}
                                  </p>
                                )}

                                <div className="flex items-center justify-between mb-2">
                                  <button
                                    onClick={() => {
                                      if (isInWatchlist(content.id)) {
                                        onRemoveFromWatchlist(content.id);
                                      } else {
                                        onAddToWatchlist?.(content as Movie | TVShow);
                                      }
                                    }}
                                    className={`flex items-center space-x-1 px-2 py-1 rounded text-xs transition-colors ${
                                      isInWatchlist(content.id)
                                        ? 'bg-green-500 text-white'
                                        : 'bg-slate-700 text-slate-400 hover:bg-slate-600 hover:text-white'
                                    }`}
                                  >
                                    {isInWatchlist(content.id) ? (
                                                                             <BookMarked className="h-3 w-3" />
                                    ) : (
                                      <Bookmark className="h-3 w-3" />
                                    )}
                                  </button>

                                  <button
                                    onClick={() => openRatingSlider(content.id)}
                                    className="bg-blue-500 hover:bg-blue-600 text-white px-2 py-1 rounded text-xs transition-colors"
                                  >
                                    Puan Ver
                                  </button>
                                </div>

                                {/* Rating Slider for Quick View */}
                                {showRatingSlider === content.id && (
                                  <div className="bg-slate-600/50 rounded-lg p-2 border border-slate-500/50 mt-2">
                                    <div className="mb-2">
                                      <div className="flex items-center justify-between mb-1">
                                        <span className="text-white text-xs font-medium">{tempRating}/10</span>
                                        <span className={`text-xs ${getRatingColor(tempRating)}`}>
                                          {getRatingText(tempRating)}
                                        </span>
                                      </div>
                                      <input
                                        type="range"
                                        min="1"
                                        max="10"
                                        step="1"
                                        value={tempRating}
                                        onChange={(e) => setTempRating(parseInt(e.target.value))}
                                        className="w-full h-1 bg-slate-600 rounded-lg appearance-none cursor-pointer slider"
                                      />
                                    </div>
                                    
                                    <div className="flex space-x-1">
                                      <button
                                        onClick={() => handleSliderRating(content.id, tempRating, content.media_type === 'tv' ? 'tv' : 'movie')}
                                        className="flex-1 bg-blue-500 hover:bg-blue-600 text-white px-2 py-1 rounded text-xs transition-colors"
                                      >
                                        Puanla
                                      </button>
                                      <button
                                        onClick={() => setShowRatingSlider(null)}
                                        className="px-2 py-1 bg-slate-600 hover:bg-slate-500 text-white rounded text-xs transition-colors"
                                      >
                                        İptal
                                      </button>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                        {directorContent.movies.length > 12 && (
                          <div className="mt-4 text-center">
                            <button
                              onClick={() => setSelectedDirectorIndex(index)}
                              className="text-blue-400 hover:text-blue-300 text-sm"
                            >
                              +{directorContent.movies.length - 12} içerik daha göster
                            </button>
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="text-center py-12">
                        <Film className="h-12 w-12 text-slate-500 mx-auto mb-4" />
                        <p className="text-slate-400 text-lg mb-2">
                          Bu yönetmen için puanlanmamış içerik bulunamadı
                        </p>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          )}
        </div>

        <div className="border-t border-slate-700 p-4 bg-slate-750">
          <p className="text-slate-400 text-sm text-center">
            Sadece puanlanmamış içerikler gösteriliyor • Puan verdiğiniz içerikler otomatik olarak listeden kaldırılır
          </p>
        </div>
      </div>
    </div>
  );
};