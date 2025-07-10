import React from 'react';
import LoadingSpinner from '../LoadingSpinner';

const RecommendationsSkeleton: React.FC = () => {
  return (
    <div className="min-h-screen bg-brand-dark p-8">
      <div className="max-w-6xl mx-auto">
        <LoadingSpinner />
        <div className="text-center mt-4">
          <p className="text-theme-secondary">Öneriler yükleniyor...</p>
        </div>
      </div>
    </div>
  );
};

export default RecommendationsSkeleton;