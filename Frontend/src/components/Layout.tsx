import { Link, useLocation } from 'react-router-dom';
import { ReactNode, useEffect, useState } from 'react';

interface LayoutProps {
  children: ReactNode;
}

interface Star {
  id: number;
  x: number;
  y: number;
  size: 'small' | 'medium' | 'large';
  animationDelay: number;
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const [stars, setStars] = useState<Star[]>([]);

  useEffect(() => {
    // Generate random stars for the background
    const generateStars = () => {
      const starArray: Star[] = [];
      const starCount = 200;

      for (let i = 0; i < starCount; i++) {
        starArray.push({
          id: i,
          x: Math.random() * 100,
          y: Math.random() * 100,
          size: Math.random() > 0.8 ? 'large' : Math.random() > 0.6 ? 'medium' : 'small',
          animationDelay: Math.random() * 5
        });
      }
      setStars(starArray);
    };

    generateStars();
  }, []);

  return (
    <div className="min-h-screen nasa-gradient text-white relative overflow-hidden">
      {/* Animated Starfield */}
      <div className="starfield">
        {stars.map((star) => (
          <div
            key={star.id}
            className={`star star-${star.size}`}
            style={{
              left: `${star.x}%`,
              top: `${star.y}%`,
              animationDelay: `${star.animationDelay}s`
            }}
          />
        ))}
      </div>

      {/* Navigation Header */}
      <nav className="relative z-10 border-b border-white/20 nasa-panel">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link to="/" className="flex items-center space-x-3">
              <div className="text-3xl">🚀</div>
              <div>
                <div className="text-2xl font-bold text-white">
                  Wednesday Wings
                </div>
                <div className="text-xs text-red-400 font-semibold tracking-wider">
                  NASA SPACE APPS 2024
                </div>
              </div>
            </Link>
            <div className="flex space-x-6">
              <Link
                to="/"
                className={`px-6 py-2 rounded-lg transition-all font-medium ${
                  location.pathname === '/' 
                    ? 'nasa-button text-white' 
                    : 'hover:bg-white/10 text-gray-300 border border-transparent hover:border-white/20'
                }`}
              >
                Mission Control
              </Link>
              <Link
                to="/library"
                className={`px-6 py-2 rounded-lg transition-all font-medium ${
                  location.pathname.startsWith('/library') 
                    ? 'nasa-button text-white' 
                    : 'hover:bg-white/10 text-gray-300 border border-transparent hover:border-white/20'
                }`}
              >
                Data Archive
              </Link>
              <Link
                to="/upload"
                className={`px-6 py-2 rounded-lg transition-all font-medium ${
                  location.pathname === '/upload' 
                    ? 'nasa-button text-white' 
                    : 'hover:bg-white/10 text-gray-300 border border-transparent hover:border-white/20'
                }`}
              >
                Upload Module
              </Link>
              <Link
                to="/model"
                className={`px-6 py-2 rounded-lg transition-all font-medium ${
                  location.pathname === '/model' 
                    ? 'nasa-button text-white' 
                    : 'hover:bg-white/10 text-gray-300 border border-transparent hover:border-white/20'
                }`}
              >
                AI Models
              </Link>
              <Link
                to="/data-preview"
                className={`px-6 py-2 rounded-lg transition-all font-medium ${
                  location.pathname === '/data-preview' 
                    ? 'nasa-button text-white' 
                    : 'hover:bg-white/10 text-gray-300 border border-transparent hover:border-white/20'
                }`}
              >
                Data Explorer
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="relative z-10 container mx-auto px-4 py-8">
        {children}
      </main>
    </div>
  );
}
