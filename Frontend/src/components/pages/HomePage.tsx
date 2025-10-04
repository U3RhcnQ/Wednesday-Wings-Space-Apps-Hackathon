import { Link } from 'react-router-dom';

export default function HomePage() {
  return (
    <div className="py-8">
      <div className="text-center mb-16">
        <h1 className="text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-red-400 via-orange-400 to-yellow-400">
          Exoplanet Mission
        </h1>
        <div className="text-2xl font-light text-blue-300 mb-4">
          NASA Space Apps Challenge 2024
        </div>
        <p className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed">
          Embark on a journey to discover worlds beyond our solar system. Access NASA's exoplanet databases,
          analyze cutting-edge astronomical data, and contribute to humanity's quest to find life among the stars.
        </p>
      </div>

      <div className="flex justify-center">
        <Link
          to="/library"
          className="nasa-panel rounded-xl p-12 hover:bg-white/10 transition-all transform hover:scale-105 cursor-pointer max-w-lg group"
        >
          <div className="text-center">
            <div className="w-20 h-20 mx-auto mb-8 nasa-button rounded-xl flex items-center justify-center group-hover:animate-pulse">
              <span className="text-4xl">🌌</span>
            </div>
            <h2 className="text-4xl font-bold mb-6 text-white">Mission Archive</h2>
            <p className="text-gray-300 text-lg leading-relaxed mb-4">
              Access comprehensive exoplanet datasets from Kepler, K2, and TESS missions
            </p>
            <div className="text-red-400 font-semibold tracking-wider text-sm">
              ▶ INITIATE DATA EXPLORATION
            </div>
          </div>
        </Link>
      </div>

      {/* Mission Stats */}
      <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
        <div className="nasa-panel rounded-lg p-6 text-center">
          <div className="text-3xl font-bold text-red-400">5,000+</div>
          <div className="text-gray-300">Confirmed Exoplanets</div>
        </div>
        <div className="nasa-panel rounded-lg p-6 text-center">
          <div className="text-3xl font-bold text-orange-400">10,000+</div>
          <div className="text-gray-300">Candidate Objects</div>
        </div>
        <div className="nasa-panel rounded-lg p-6 text-center">
          <div className="text-3xl font-bold text-yellow-400">3</div>
          <div className="text-gray-300">Active Missions</div>
        </div>
      </div>
    </div>
  );
}
