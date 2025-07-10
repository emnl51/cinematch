import React, { useState, useEffect } from 'react';
import { 
  Settings, 
  Palette, 
  Brain, 
  Eye, 
  Save, 
  RotateCcw, 
  Monitor, 
  Sun, 
  Moon, 
  Zap,
  Target,
  Info,
  AlertTriangle
} from 'lucide-react';

export interface AppSettings {
  // Görünüm Ayarları
  theme: 'light' | 'dark' | 'auto';
  compactMode: boolean;
  animationsEnabled: boolean;
  
  // İçerik Ayarları
  recommendationCount: number;
  discoveryContentCount: number;
  showAdultContent: boolean;
  minContentRating: number;
  minTmdbScore: number;
  minTmdbVoteCount: number;
  showKidsContent: boolean;
  showAnimationContent: boolean;
  showAnimeContent: boolean;
  
  // Öneri Sistemi Ayarları
  recommendationAlgorithm: {
    profileWeight: number;        // Profil ağırlığı (0-100)
    surpriseWeight: number;       // Sürpriz faktörü (0-100)
    diversityWeight: number;      // Çeşitlilik faktörü (0-100)
    qualityWeight: number;        // Kalite faktörü (0-100)
    recencyWeight: number;        // Güncellik faktörü (0-100)
  };
  
  // Filtre Ayarları
  defaultFilters: {
    minYear: number;
    maxYear: number;
    minRating: number;
    maxRating: number;
    minMatchScore: number;
  };
  
  // Performans Ayarları
  cacheEnabled: boolean;
  preloadImages: boolean;
  autoRefreshRecommendations: boolean;
  refreshInterval: number; // dakika
  
  // Gizlilik Ayarları
  analyticsEnabled: boolean;
  crashReportingEnabled: boolean;
  
  // Deneysel Özellikler
  experimentalFeatures: {
    advancedFiltering: boolean;
    aiInsights: boolean;
    personalizedHomepage: boolean;
    smartNotifications: boolean;
  };
}

interface SettingsPageProps {
  settings: AppSettings;
  onSettingsChange: (settings: AppSettings) => void;
  isInitialSetup?: boolean;
  onInitialSetupComplete?: () => void;
}

const defaultSettings: AppSettings = {
  theme: 'dark',
  compactMode: false,
  animationsEnabled: true,
  recommendationCount: 25,
  discoveryContentCount: 20,
  showAdultContent: false,
  minContentRating: 6.0,
  minTmdbScore: 6.0,
  minTmdbVoteCount: 100,
  showKidsContent: false,
  showAnimationContent: true,
  showAnimeContent: true,
  recommendationAlgorithm: {
    profileWeight: 70,
    surpriseWeight: 15,
    diversityWeight: 10,
    qualityWeight: 80,
    recencyWeight: 20
  },
  defaultFilters: {
    minYear: 1950,
    maxYear: new Date().getFullYear(),
    minRating: 0,
    maxRating: 10,
    minMatchScore: 0
  },
  cacheEnabled: true,
  preloadImages: true,
  autoRefreshRecommendations: false,
  refreshInterval: 30,
  analyticsEnabled: true,
  crashReportingEnabled: true,
  experimentalFeatures: {
    advancedFiltering: true,
    aiInsights: true,
    personalizedHomepage: false,
    smartNotifications: false
  }
};

export const SettingsPage: React.FC<SettingsPageProps> = ({
  settings,
  onSettingsChange,
  isInitialSetup = false,
  onInitialSetupComplete
}) => {
  const [activeTab, setActiveTab] = useState<'appearance' | 'content' | 'algorithm' | 'performance' | 'privacy' | 'experimental'>('appearance');
  const [localSettings, setLocalSettings] = useState<AppSettings>(settings);
  const [hasChanges, setHasChanges] = useState(false);
  const [setupStep, setSetupStep] = useState(1);
  const [, setCompletedSteps] = useState<Set<number>>(new Set());

  useEffect(() => {
    setLocalSettings(settings);
    setHasChanges(false);
  }, [settings]);

  const updateSetting = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    const newSettings = { ...localSettings, [key]: value };
    setLocalSettings(newSettings);
    setHasChanges(true);
    
    // Auto-save during initial setup for immediate effect
    if (isInitialSetup) {
      onSettingsChange(newSettings);
    }
  };

  const updateNestedSetting = <T extends keyof AppSettings>(
    category: T,
    key: keyof AppSettings[T],
    value: any
  ) => {
    const newSettings = {
      ...localSettings,
      [category]: {
        ...(localSettings[category] as object),
        [key]: value
      }
    };
    setLocalSettings(newSettings);
    setHasChanges(true);
    
    // Auto-save during initial setup for immediate effect
    if (isInitialSetup) {
      onSettingsChange(newSettings);
    }
  };

  const handleSave = () => {
    onSettingsChange(localSettings);
    setHasChanges(false);
  };

  const handleReset = () => {
    setLocalSettings(defaultSettings);
    setHasChanges(true);
  };

  const handleNextStep = () => {
    if (setupStep === 1) {
      setCompletedSteps((prev: Set<number>) => new Set([...prev, 1]));
      setActiveTab('content');
      setSetupStep(2);
    } else if (setupStep === 2) {
      setCompletedSteps((prev: Set<number>) => new Set([...prev, 2]));
      setActiveTab('algorithm');
      setSetupStep(3);
    } else if (setupStep === 3) {
      setCompletedSteps((prev: Set<number>) => new Set([...prev, 3]));
      // Save settings when completing initial setup
      onSettingsChange(localSettings);
      setHasChanges(false);
      if (onInitialSetupComplete) {
        onInitialSetupComplete();
      }
    }
  };

  const canProceedToNext = () => {
    if (setupStep === 1) {
      // At least theme selection is required
      return true;
    } else if (setupStep === 2) {
      // At least recommendation count is set
      return localSettings.recommendationCount > 0;
    } else if (setupStep === 3) {
      return true;
    }
    return false;
  };

  const tabs = [
    { id: 'appearance', label: 'Görünüm', icon: Palette },
    { id: 'content', label: 'İçerik', icon: Eye },
    { id: 'algorithm', label: 'AI Algoritması', icon: Brain },
    { id: 'performance', label: 'Performans', icon: Zap },
    { id: 'privacy', label: 'Gizlilik', icon: Settings },
    { id: 'experimental', label: 'Deneysel', icon: Target }
  ];

  return (
    <div className="w-full min-h-screen bg-theme-primary flex flex-col lg:flex-row">
      {/* Sidebar Navigation for larger screens */}
      <div className="lg:w-72 bg-theme-card border-r border-theme-primary">
        <div className="p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="bg-gradient-to-r from-amber-400 to-orange-500 p-3 rounded-xl shadow-lg">
              <Settings className="h-6 w-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-theme-primary">
                {isInitialSetup ? 'İlk Kurulum' : 'Ayarlar'}
              </h2>
              <p className="text-theme-secondary text-sm">
                {isInitialSetup ? 'Temel ayarları yapalım' : 'Kişiselleştirin'}
              </p>
            </div>
          </div>

          {/* Initial Setup Progress */}
          {isInitialSetup && (
            <div className="mb-6 bg-gradient-to-r from-indigo-500/10 to-purple-500/10 border border-indigo-500/30 rounded-xl p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-theme-primary font-medium">Kurulum İlerlemesi</span>
                <span className="text-indigo-400 text-sm">Adım {setupStep}/3</span>
              </div>
              <div className="w-full bg-theme-tertiary rounded-full h-2 mb-3">
                <div 
                  className="bg-gradient-to-r from-indigo-500 to-purple-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${(setupStep / 3) * 100}%` }}
                ></div>
              </div>
              <div className="text-sm text-theme-secondary">
                {setupStep === 1 && "İlk olarak görünüm tercihlerinizi ayarlayalım"}
                {setupStep === 2 && "Şimdi içerik tercihlerinizi belirleyelim"}
                {setupStep === 3 && "Son olarak AI algoritması ayarlarını yapalım"}
              </div>
            </div>
          )}

          {/* Navigation Tabs */}
          <nav className="space-y-2">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200
                  ${activeTab === tab.id
                    ? 'bg-gradient-to-r from-amber-500/20 to-orange-500/20 border border-amber-500/30 text-amber-400'
                    : 'text-theme-secondary hover:text-theme-primary hover:bg-theme-secondary'}
                `}
              >
                <tab.icon className="h-5 w-5" />
                <span>{tab.label}</span>
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Mobile Header */}
        <div className="lg:hidden bg-theme-card border-b border-theme-primary p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-r from-amber-400 to-orange-500 p-2 rounded-lg">
                <Settings className="h-5 w-5 text-white" />
              </div>
              <h2 className="text-lg font-bold text-theme-primary">
                {isInitialSetup ? 'İlk Kurulum' : 'Ayarlar'}
              </h2>
            </div>
            <div className="flex gap-2">
              {!isInitialSetup && (
                <button
                  onClick={handleReset}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg bg-theme-tertiary text-theme-primary hover:bg-theme-secondary border border-theme-primary transition-all text-sm font-medium"
                >
                  <RotateCcw className="h-4 w-4" /> Reset
                </button>
              )}
              {isInitialSetup ? (
                <button
                  onClick={handleNextStep}
                  disabled={!canProceedToNext()}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-green-500 to-emerald-500 text-white font-bold shadow-lg transition-all text-sm ${!canProceedToNext() ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  {setupStep === 3 ? 'Tamamla' : 'İleri'}
                </button>
              ) : (
                <button
                  onClick={handleSave}
                  disabled={!hasChanges}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-amber-500 to-orange-500 text-white font-bold shadow-lg disabled:opacity-60 disabled:cursor-not-allowed transition-all text-sm`}
                >
                  <Save className="h-4 w-4" /> Kaydet
                </button>
              )}
            </div>
          </div>

          {/* Mobile Tab Selector */}
          <div className="mt-4">
            <select 
              value={activeTab} 
              onChange={(e) => setActiveTab(e.target.value as any)}
              className="w-full bg-theme-tertiary border border-theme-primary rounded-lg px-3 py-2 text-theme-primary focus:outline-none focus:ring-2 focus:ring-amber-500"
            >
              {tabs.map((tab) => (
                <option key={tab.id} value={tab.id}>{tab.label}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 p-6 overflow-y-auto">
          <div className="max-w-4xl mx-auto">
            {/* Desktop Header */}
            <div className="hidden lg:flex items-center justify-between mb-6">
              <div>
                <h3 className="text-2xl font-bold text-theme-primary">
                  {tabs.find(t => t.id === activeTab)?.label}
                </h3>
                <p className="text-theme-secondary mt-1">
                  {activeTab === 'appearance' && 'Görünüm ve kullanıcı arayüzü ayarları'}
                  {activeTab === 'content' && 'İçerik filtreleme ve görüntüleme tercihleri'}
                  {activeTab === 'algorithm' && 'AI öneri algoritması ince ayarları'}
                  {activeTab === 'performance' && 'Uygulama performansı ve optimizasyon'}
                  {activeTab === 'privacy' && 'Gizlilik ve veri koruma ayarları'}
                  {activeTab === 'experimental' && 'Beta özellikler ve deneysel fonksiyonlar'}
                </p>
              </div>
              {!isInitialSetup && (
                <div className="flex gap-3">
                  <button
                    onClick={handleReset}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-theme-tertiary text-theme-primary hover:bg-theme-secondary border border-theme-primary transition-all font-medium"
                  >
                    <RotateCcw className="h-4 w-4" /> Varsayılanlar
                  </button>
                  <button
                    onClick={handleSave}
                    disabled={!hasChanges}
                    className={`flex items-center gap-2 px-5 py-2 rounded-lg bg-gradient-to-r from-amber-500 to-orange-500 text-white font-bold shadow-lg disabled:opacity-60 disabled:cursor-not-allowed transition-all`}
                  >
                    <Save className="h-4 w-4" /> Kaydet
                  </button>
                </div>
              )}
              {isInitialSetup && (
                <button
                  onClick={handleNextStep}
                  disabled={!canProceedToNext()}
                  className={`flex items-center gap-2 px-5 py-2 rounded-lg bg-gradient-to-r from-green-500 to-emerald-500 text-white font-bold shadow-lg transition-all ${!canProceedToNext() ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  {setupStep === 3 ? 'Kurulumu Tamamla' : 'Sonraki Adım'}
                </button>
              )}
            </div>

            <div className="space-y-8">
              {/* Appearance Tab */}
              {activeTab === 'appearance' && (
                <div className="space-y-6">
                  {/* Theme Selection */}
                  <div className="bg-theme-card rounded-xl p-6 border border-theme-primary">
                    <h4 className="text-lg font-semibold text-theme-primary mb-4 flex items-center space-x-2">
                      <Palette className="h-5 w-5 text-amber-400" />
                      <span>Tema Seçimi</span>
                    </h4>
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                      {[
                        { value: 'light', label: 'Açık Tema', icon: Sun, desc: 'Gündüz kullanım için ideal' },
                        { value: 'dark', label: 'Koyu Tema', icon: Moon, desc: 'Gece kullanım için ideal' },
                        { value: 'auto', label: 'Otomatik', icon: Monitor, desc: 'Sistem temasını takip eder' }
                      ].map((theme) => (
                        <button
                          key={theme.value}
                          onClick={() => updateSetting('theme', theme.value as any)}
                          className={`flex flex-col items-center space-y-3 p-4 rounded-xl border-2 transition-all hover:scale-105 ${
                            localSettings.theme === theme.value
                              ? 'border-amber-500 bg-amber-500/10 shadow-lg'
                              : 'border-theme-primary hover:border-amber-300 bg-theme-secondary'
                          }`}
                        >
                          <theme.icon className={`h-8 w-8 ${localSettings.theme === theme.value ? 'text-amber-400' : 'text-theme-secondary'}`} />
                          <div className="text-center">
                            <div className={`font-medium ${localSettings.theme === theme.value ? 'text-amber-400' : 'text-theme-primary'}`}>
                              {theme.label}
                            </div>
                            <div className="text-xs text-theme-tertiary mt-1">{theme.desc}</div>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Display Options */}
                  <div className="bg-theme-card rounded-xl p-6 border border-theme-primary">
                    <h4 className="text-lg font-semibold text-theme-primary mb-4">Görüntüleme Seçenekleri</h4>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-4 bg-theme-secondary rounded-lg">
                        <div>
                          <span className="text-theme-primary font-medium">Kompakt Mod</span>
                          <p className="text-sm text-theme-tertiary">Daha az boşluk, daha fazla içerik görün</p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            checked={localSettings.compactMode}
                            onChange={(e) => updateSetting('compactMode', e.target.checked)}
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-theme-tertiary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-amber-500"></div>
                        </label>
                      </div>

                      <div className="flex items-center justify-between p-4 bg-theme-secondary rounded-lg">
                        <div>
                          <span className="text-theme-primary font-medium">Animasyonlar</span>
                          <p className="text-sm text-theme-tertiary">Geçiş efektleri ve animasyonlar</p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            checked={localSettings.animationsEnabled}
                            onChange={(e) => updateSetting('animationsEnabled', e.target.checked)}
                            className="sr-only peer"
                          />
                          <div className="w-11 h-6 bg-theme-tertiary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-amber-500"></div>
                        </label>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Content Tab */}
              {activeTab === 'content' && (
                <div className="space-y-6">
                  {/* Content Counts */}
                  <div className="bg-theme-card rounded-xl p-6 border border-theme-primary">
                    <h4 className="text-lg font-semibold text-theme-primary mb-4 flex items-center space-x-2">
                      <Eye className="h-5 w-5 text-blue-400" />
                      <span>İçerik Sayıları</span>
                    </h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-sm font-medium text-theme-primary mb-3">
                          AI Önerileri: {localSettings.recommendationCount} adet
                        </label>
                        <input
                          type="range"
                          min="10"
                          max="50"
                          step="5"
                          value={localSettings.recommendationCount}
                          onChange={(e) => updateSetting('recommendationCount', parseInt(e.target.value))}
                          className="w-full h-2 bg-theme-tertiary rounded-lg appearance-none cursor-pointer slider"
                        />
                        <div className="flex justify-between text-xs text-theme-tertiary mt-2">
                          <span>10</span>
                          <span>50</span>
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-theme-primary mb-3">
                          Keşif İçeriği: {localSettings.discoveryContentCount} adet
                        </label>
                        <input
                          type="range"
                          min="10"
                          max="40"
                          step="5"
                          value={localSettings.discoveryContentCount}
                          onChange={(e) => updateSetting('discoveryContentCount', parseInt(e.target.value))}
                          className="w-full h-2 bg-theme-tertiary rounded-lg appearance-none cursor-pointer slider"
                        />
                        <div className="flex justify-between text-xs text-theme-tertiary mt-2">
                          <span>10</span>
                          <span>40</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Content Filtering */}
                  <div className="bg-theme-card rounded-xl p-6 border border-theme-primary">
                    <h4 className="text-lg font-semibold text-theme-primary mb-4">Kalite Filtreleri</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <label className="block text-sm font-medium text-theme-primary mb-3">
                          Minimum İçerik Puanı: {localSettings.minContentRating.toFixed(1)}
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="10"
                          step="0.5"
                          value={localSettings.minContentRating}
                          onChange={(e) => updateSetting('minContentRating', parseFloat(e.target.value))}
                          className="w-full h-2 bg-theme-tertiary rounded-lg appearance-none cursor-pointer slider"
                        />
                        <div className="flex justify-between text-xs text-theme-tertiary mt-2">
                          <span>0.0</span>
                          <span>10.0</span>
                        </div>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-theme-primary mb-3">
                          Minimum TMDB Puanı: {localSettings.minTmdbScore.toFixed(1)}
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="10"
                          step="0.1"
                          value={localSettings.minTmdbScore}
                          onChange={(e) => updateSetting('minTmdbScore', parseFloat(e.target.value))}
                          className="w-full h-2 bg-theme-tertiary rounded-lg appearance-none cursor-pointer slider"
                        />
                        <div className="flex justify-between text-xs text-theme-tertiary mt-2">
                          <span>0.0</span>
                          <span>10.0</span>
                        </div>
                      </div>
                    </div>

                    <div className="mt-6">
                      <label className="block text-sm font-medium text-theme-primary mb-3">
                        Minimum Oy Sayısı: {localSettings.minTmdbVoteCount}
                      </label>
                      <p className="text-xs text-theme-tertiary mb-3">TMDB'de minimum kaç kişi puanlamış olması gerekir</p>
                      <input
                        type="range"
                        min="10"
                        max="5000"
                        step="50"
                        value={localSettings.minTmdbVoteCount}
                        onChange={(e) => updateSetting('minTmdbVoteCount', parseInt(e.target.value))}
                        className="w-full h-2 bg-theme-tertiary rounded-lg appearance-none cursor-pointer slider"
                      />
                      <div className="flex justify-between text-xs text-theme-tertiary mt-2">
                        <span>10</span>
                        <span>5000</span>
                      </div>
                    </div>
                  </div>

                  {/* Content Categories */}
                  <div className="bg-theme-card rounded-xl p-6 border border-theme-primary">
                    <h4 className="text-lg font-semibold text-theme-primary mb-4">İçerik Kategorileri</h4>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {[
                        { key: 'showAdultContent', label: 'Yetişkin İçeriği', desc: '18+ içerikleri dahil et' },
                        { key: 'showKidsContent', label: 'Çocuk İçeriği', desc: '13 yaşa kadar uygun içerikler' },
                        { key: 'showAnimationContent', label: 'Çizgi Film', desc: 'Animasyon içerikleri dahil et' },
                        { key: 'showAnimeContent', label: 'Anime', desc: 'Anime içerikleri dahil et' }
                      ].map(({ key, label, desc }) => (
                        <div key={key} className="flex items-center justify-between p-4 bg-theme-secondary rounded-lg">
                          <div>
                            <span className="text-theme-primary font-medium">{label}</span>
                            <p className="text-sm text-theme-tertiary">{desc}</p>
                          </div>
                          <label className="relative inline-flex items-center cursor-pointer">
                            <input
                              type="checkbox"
                              checked={localSettings[key as keyof AppSettings] as boolean}
                              onChange={(e) => updateSetting(key as keyof AppSettings, e.target.checked as any)}
                              className="sr-only peer"
                            />
                            <div className="w-11 h-6 bg-theme-tertiary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-amber-500"></div>
                          </label>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Algorithm Tab */}
              {activeTab === 'algorithm' && (
                <div className="bg-theme-card rounded-xl p-6 border border-theme-primary space-y-6">
                  <h4 className="text-lg font-semibold text-theme-primary mb-4 flex items-center space-x-2">
                    <Brain className="h-5 w-5 text-purple-400" />
                    <span>AI Algoritması İnce Ayarları</span>
                  </h4>
                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3 mb-4">
                    <div className="flex items-start space-x-2">
                      <Info className="h-4 w-4 text-purple-400 mt-0.5 flex-shrink-0" />
                      <p className="text-purple-200 text-sm">
                        Bu ayarlar öneri algoritmasının nasıl çalıştığını etkiler. Değişiklikler yeni öneriler oluşturulduğunda geçerli olur.
                      </p>
                    </div>
                  </div>

                  <div className="bg-theme-card rounded-lg p-4 space-y-6">
                    <h4 className="font-medium text-theme-primary">Algoritma Ağırlıkları</h4>
                    
                    <div>
                      <label className="block text-sm font-medium text-theme-primary mb-2">
                        Profil Uyumu: %{localSettings.recommendationAlgorithm.profileWeight}
                      </label>
                      <p className="text-xs text-theme-tertiary mb-2">Geçmiş puanlarınıza ne kadar uygun olacağı</p>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        step="5"
                        value={localSettings.recommendationAlgorithm.profileWeight}
                        onChange={(e) => updateNestedSetting('recommendationAlgorithm', 'profileWeight', parseInt(e.target.value))}
                        className="w-full h-2 bg-theme-tertiary rounded-lg appearance-none cursor-pointer slider"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-theme-primary mb-2">
                        Sürpriz Faktörü: %{localSettings.recommendationAlgorithm.surpriseWeight}
                      </label>
                      <p className="text-xs text-theme-tertiary mb-2">Beklenmedik ama ilginç önerilerin oranı</p>
                      <input
                        type="range"
                        min="0"
                        max="50"
                        step="5"
                        value={localSettings.recommendationAlgorithm.surpriseWeight}
                        onChange={(e) => updateNestedSetting('recommendationAlgorithm', 'surpriseWeight', parseInt(e.target.value))}
                        className="w-full h-2 bg-theme-tertiary rounded-lg appearance-none cursor-pointer slider"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-theme-primary mb-2">
                        Çeşitlilik: %{localSettings.recommendationAlgorithm.diversityWeight}
                      </label>
                      <p className="text-xs text-theme-tertiary mb-2">Farklı türlerden önerilerin oranı</p>
                      <input
                        type="range"
                        min="0"
                        max="50"
                        step="5"
                        value={localSettings.recommendationAlgorithm.diversityWeight}
                        onChange={(e) => updateNestedSetting('recommendationAlgorithm', 'diversityWeight', parseInt(e.target.value))}
                        className="w-full h-2 bg-theme-tertiary rounded-lg appearance-none cursor-pointer slider"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-theme-primary mb-2">
                        Kalite Önceliği: %{localSettings.recommendationAlgorithm.qualityWeight}
                      </label>
                      <p className="text-xs text-theme-tertiary mb-2">Yüksek puanlı içeriklerin önceliği</p>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        step="5"
                        value={localSettings.recommendationAlgorithm.qualityWeight}
                        onChange={(e) => updateNestedSetting('recommendationAlgorithm', 'qualityWeight', parseInt(e.target.value))}
                        className="w-full h-2 bg-theme-tertiary rounded-lg appearance-none cursor-pointer slider"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-theme-primary mb-2">
                        Güncellik: %{localSettings.recommendationAlgorithm.recencyWeight}
                      </label>
                      <p className="text-xs text-theme-tertiary mb-2">Yeni çıkan içeriklerin önceliği</p>
                      <input
                        type="range"
                        min="0"
                        max="50"
                        step="5"
                        value={localSettings.recommendationAlgorithm.recencyWeight}
                        onChange={(e) => updateNestedSetting('recommendationAlgorithm', 'recencyWeight', parseInt(e.target.value))}
                        className="w-full h-2 bg-theme-tertiary rounded-lg appearance-none cursor-pointer slider"
                      />
                    </div>

                    {/* Preset Buttons */}
                    <div className="border-t border-theme-tertiary pt-4">
                      <p className="text-sm text-theme-tertiary mb-3">Hazır Ayarlar:</p>
                      <div className="grid grid-cols-3 gap-2">
                        <button
                          onClick={() => updateSetting('recommendationAlgorithm', {
                            profileWeight: 80,
                            surpriseWeight: 10,
                            diversityWeight: 5,
                            qualityWeight: 90,
                            recencyWeight: 15
                          })}
                          className="px-3 py-2 bg-theme-tertiary hover:bg-theme-secondary text-white rounded-lg text-sm transition-colors"
                        >
                          Güvenli
                        </button>
                        <button
                          onClick={() => updateSetting('recommendationAlgorithm', {
                            profileWeight: 60,
                            surpriseWeight: 25,
                            diversityWeight: 20,
                            qualityWeight: 70,
                            recencyWeight: 30
                          })}
                          className="px-3 py-2 bg-theme-tertiary hover:bg-theme-secondary text-white rounded-lg text-sm transition-colors"
                        >
                          Dengeli
                        </button>
                        <button
                          onClick={() => updateSetting('recommendationAlgorithm', {
                            profileWeight: 40,
                            surpriseWeight: 35,
                            diversityWeight: 30,
                            qualityWeight: 60,
                            recencyWeight: 40
                          })}
                          className="px-3 py-2 bg-theme-tertiary hover:bg-theme-secondary text-white rounded-lg text-sm transition-colors"
                        >
                          Maceracı
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Performance Tab */}
              {activeTab === 'performance' && (
                <div className="bg-theme-card rounded-xl p-6 border border-theme-primary space-y-6">
                  <h4 className="text-lg font-semibold text-theme-primary mb-4 flex items-center space-x-2">
                    <Zap className="h-5 w-5 text-green-400" />
                    <span>Performans Ayarları</span>
                  </h4>

                  <div className="bg-theme-card rounded-lg p-4 space-y-4">
                    <h4 className="font-medium text-theme-primary">Önbellekleme</h4>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-theme-primary font-medium">Önbellek Etkin</span>
                        <p className="text-sm text-theme-tertiary">Daha hızlı yükleme için verileri sakla</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={localSettings.cacheEnabled}
                          onChange={(e) => updateSetting('cacheEnabled', e.target.checked)}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-theme-tertiary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-amber-500"></div>
                      </label>
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-theme-primary font-medium">Görselleri Önceden Yükle</span>
                        <p className="text-sm text-theme-tertiary">Posterleri arka planda yükle</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={localSettings.preloadImages}
                          onChange={(e) => updateSetting('preloadImages', e.target.checked)}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-theme-tertiary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-amber-500"></div>
                      </label>
                    </div>
                  </div>

                  <div className="bg-theme-card rounded-lg p-4 space-y-4">
                    <h4 className="font-medium text-theme-primary">Otomatik Yenileme</h4>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-theme-primary font-medium">Önerileri Otomatik Yenile</span>
                        <p className="text-sm text-theme-tertiary">Belirli aralıklarla yeni öneriler getir</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={localSettings.autoRefreshRecommendations}
                          onChange={(e) => updateSetting('autoRefreshRecommendations', e.target.checked)}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-theme-tertiary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-amber-500"></div>
                      </label>
                    </div>

                    {localSettings.autoRefreshRecommendations && (
                      <div>
                        <label className="block text-sm font-medium text-theme-primary mb-2">
                          Yenileme Aralığı: {localSettings.refreshInterval} dakika
                        </label>
                        <input
                          type="range"
                          min="15"
                          max="120"
                          step="15"
                          value={localSettings.refreshInterval}
                          onChange={(e) => updateSetting('refreshInterval', parseInt(e.target.value))}
                          className="w-full h-2 bg-theme-tertiary rounded-lg appearance-none cursor-pointer slider"
                        />
                        <div className="flex justify-between text-xs text-theme-tertiary mt-2">
                          <span>15 dk</span>
                          <span>120 dk</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Privacy Tab */}
              {activeTab === 'privacy' && (
                <div className="bg-theme-card rounded-xl p-6 border border-theme-primary space-y-6">
                  <h4 className="text-lg font-semibold text-theme-primary mb-4 flex items-center space-x-2">
                    <Settings className="h-5 w-5 text-red-400" />
                    <span>Gizlilik Ayarları</span>
                  </h4>

                  <div className="bg-theme-card rounded-lg p-4 space-y-4">
                    <h4 className="font-medium text-theme-primary">Veri Toplama</h4>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-theme-primary font-medium">Analitik Veriler</span>
                        <p className="text-sm text-theme-tertiary">Uygulamayı geliştirmek için kullanım verilerini topla</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={localSettings.analyticsEnabled}
                          onChange={(e) => updateSetting('analyticsEnabled', e.target.checked)}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-theme-tertiary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-amber-500"></div>
                      </label>
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-theme-primary font-medium">Hata Raporlama</span>
                        <p className="text-sm text-theme-tertiary">Hataları otomatik olarak rapor et</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={localSettings.crashReportingEnabled}
                          onChange={(e) => updateSetting('crashReportingEnabled', e.target.checked)}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-theme-tertiary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-amber-500"></div>
                      </label>
                    </div>
                  </div>
                </div>
              )}

              {/* Experimental Tab */}
              {activeTab === 'experimental' && (
                <div className="bg-theme-card rounded-xl p-6 border border-theme-primary space-y-6">
                  <h4 className="text-lg font-semibold text-theme-primary mb-4 flex items-center space-x-2">
                    <Target className="h-5 w-5 text-orange-400" />
                    <span>Deneysel Özellikler</span>
                  </h4>
                  <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-3 mb-4">
                    <div className="flex items-start space-x-2">
                      <AlertTriangle className="h-4 w-4 text-orange-400 mt-0.5 flex-shrink-0" />
                      <p className="text-orange-200 text-sm">
                        Bu özellikler deneysel aşamadadır ve beklenmedik davranışlara neden olabilir.
                      </p>
                    </div>
                  </div>

                  <div className="bg-theme-card rounded-lg p-4 space-y-4">
                    <h4 className="font-medium text-theme-primary">Beta Özellikler</h4>
                    
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-theme-primary font-medium">Gelişmiş Filtreleme</span>
                        <p className="text-sm text-theme-tertiary">Daha detaylı filtre seçenekleri</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={localSettings.experimentalFeatures.advancedFiltering}
                          onChange={(e) => updateNestedSetting('experimentalFeatures', 'advancedFiltering', e.target.checked)}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-theme-tertiary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-amber-500"></div>
                      </label>
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-theme-primary font-medium">AI İçgörüleri</span>
                        <p className="text-sm text-theme-tertiary">Detaylı profil analizi ve öneriler</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={localSettings.experimentalFeatures.aiInsights}
                          onChange={(e) => updateNestedSetting('experimentalFeatures', 'aiInsights', e.target.checked)}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-theme-tertiary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-amber-500"></div>
                      </label>
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-theme-primary font-medium">Kişiselleştirilmiş Ana Sayfa</span>
                        <p className="text-sm text-theme-tertiary">Dinamik ana sayfa düzeni</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={localSettings.experimentalFeatures.personalizedHomepage}
                          onChange={(e) => updateNestedSetting('experimentalFeatures', 'personalizedHomepage', e.target.checked)}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-theme-tertiary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-amber-500"></div>
                      </label>
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-theme-primary font-medium">Akıllı Bildirimler</span>
                        <p className="text-sm text-theme-tertiary">Yeni öneriler için bildirimler</p>
                      </div>
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={localSettings.experimentalFeatures.smartNotifications}
                          onChange={(e) => updateNestedSetting('experimentalFeatures', 'smartNotifications', e.target.checked)}
                          className="sr-only peer"
                        />
                        <div className="w-11 h-6 bg-theme-tertiary peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-amber-500"></div>
                      </label>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};