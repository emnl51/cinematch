import { useState, useEffect } from 'react';
import type { AppSettings } from '../components/SettingsModal';

const SETTINGS_STORAGE_KEY = 'cinematch_settings';

const defaultSettings: AppSettings = {
  theme: 'dark',
  compactMode: false,
  animationsEnabled: true,
  recommendationCount: 50, // Increased from 25
  discoveryContentCount: 40, // Increased from 20
  showAdultContent: false,
  minContentRating: 6.0,
  minTmdbScore: 6.0,
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

export const useSettings = () => {
  const [settings, setSettings] = useState<AppSettings>(defaultSettings);
  const [isLoading, setIsLoading] = useState(true);

  // Load settings from localStorage on mount
  useEffect(() => {
    try {
      const savedSettings = localStorage.getItem(SETTINGS_STORAGE_KEY);
      if (savedSettings) {
        const parsed = JSON.parse(savedSettings);
        // Merge with defaults to ensure all properties exist
        setSettings({ ...defaultSettings, ...parsed });
      }
    } catch (error) {
      console.error('Error loading settings:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Apply theme changes to document
  useEffect(() => {
    const applyTheme = () => {
      const root = document.documentElement;
      
      if (settings.theme === 'auto') {
        // Use system preference
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        root.classList.toggle('dark', prefersDark);
      } else {
        root.classList.toggle('dark', settings.theme === 'dark');
      }
    };

    applyTheme();

    // Listen for system theme changes if auto mode is enabled
    if (settings.theme === 'auto') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      mediaQuery.addEventListener('change', applyTheme);
      
      return () => {
        mediaQuery.removeEventListener('change', applyTheme);
      };
    }
    
    return undefined;
  }, [settings.theme]);

  // Apply animation preferences
  useEffect(() => {
    const root = document.documentElement;
    const body = document.body;
    
    if (settings.animationsEnabled) {
      root.style.removeProperty('--animation-duration');
      body.classList.remove('no-animations');
    } else {
      root.style.setProperty('--animation-duration', '0s');
      body.classList.add('no-animations');
    }
  }, [settings.animationsEnabled]);

  // Apply compact mode
  useEffect(() => {
    const root = document.documentElement;
    root.classList.toggle('compact-mode', settings.compactMode);
  }, [settings.compactMode]);

  const updateSettings = (newSettings: AppSettings) => {
    setSettings(newSettings);
    
    try {
      localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(newSettings));
    } catch (error) {
      console.error('Error saving settings:', error);
    }
  };

  const resetSettings = () => {
    setSettings(defaultSettings);
    
    try {
      localStorage.removeItem(SETTINGS_STORAGE_KEY);
    } catch (error) {
      console.error('Error resetting settings:', error);
    }
  };

  const getSetting = <K extends keyof AppSettings>(key: K): AppSettings[K] => {
    return settings[key];
  };

  const updateSetting = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    const newSettings = { ...settings, [key]: value };
    updateSettings(newSettings);
  };

  return {
    settings,
    isLoading,
    updateSettings,
    resetSettings,
    getSetting,
    updateSetting
  };
};
