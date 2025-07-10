import React from 'react';

const NotFound: React.FC = () => {
  return (
    <div className="min-h-screen bg-brand-dark p-8">
      <div className="max-w-6xl mx-auto text-center">
        <h1 className="text-6xl font-bold text-theme-primary mb-4">404</h1>
        <h2 className="text-3xl font-bold text-theme-primary mb-8">Sayfa Bulunamadı</h2>
        <p className="text-theme-secondary">Aradığınız sayfa mevcut değil.</p>
      </div>
    </div>
  );
};

export default NotFound;