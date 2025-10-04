import { Link } from 'react-router-dom';

export default function LibraryPage() {
  return (
    <div className="py-8">
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-red-400 via-orange-400 to-yellow-400">
          Mission Data Archive
        </h1>
        <p className="text-xl text-gray-300 mb-4">
          Access exoplanet datasets from NASA's premier space telescopes and missions
        </p>
        <div className="text-red-400 font-semibold tracking-wider text-sm">
          CLASSIFIED • AUTHORIZED PERSONNEL ONLY
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
        <Link
          to="/library/koi"
          className="nasa-panel rounded-xl p-8 hover:bg-white/10 transition-all transform hover:scale-105 cursor-pointer group"
        >
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-6 nasa-button rounded-xl flex items-center justify-center group-hover:animate-pulse">
              <span className="text-3xl">🔭</span>
            </div>
            <h2 className="text-2xl font-bold mb-4 text-white">KOI Database</h2>
            <div className="text-red-400 font-mono text-sm mb-3">KEPLER MISSION</div>
            <p className="text-gray-300 text-sm leading-relaxed mb-4">
              Kepler Objects of Interest - Confirmed and candidate exoplanets discovered by the revolutionary Kepler Space Telescope
            </p>
            <div className="text-blue-400 font-semibold text-xs tracking-wider">
              STATUS: OPERATIONAL
            </div>
          </div>
        </Link>

        <Link
          to="/library/k2"
          className="nasa-panel rounded-xl p-8 hover:bg-white/10 transition-all transform hover:scale-105 cursor-pointer group"
        >
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-6 nasa-button rounded-xl flex items-center justify-center group-hover:animate-pulse">
              <span className="text-3xl">🛰️</span>
            </div>
            <h2 className="text-2xl font-bold mb-4 text-white">K2 Archive</h2>
            <div className="text-orange-400 font-mono text-sm mb-3">EXTENDED MISSION</div>
            <p className="text-gray-300 text-sm leading-relaxed mb-4">
              K2 Mission data - Extended Kepler observations across the ecliptic plane revealing new exoplanet candidates
            </p>
            <div className="text-green-400 font-semibold text-xs tracking-wider">
              STATUS: COMPLETE
            </div>
          </div>
        </Link>

        <Link
          to="/library/toi"
          className="nasa-panel rounded-xl p-8 hover:bg-white/10 transition-all transform hover:scale-105 cursor-pointer group"
        >
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-6 nasa-button rounded-xl flex items-center justify-center group-hover:animate-pulse">
              <span className="text-3xl">🌍</span>
            </div>
            <h2 className="text-2xl font-bold mb-4 text-white">TOI Registry</h2>
            <div className="text-yellow-400 font-mono text-sm mb-3">TESS MISSION</div>
            <p className="text-gray-300 text-sm leading-relaxed mb-4">
              TESS Objects of Interest - Latest exoplanet discoveries from the Transiting Exoplanet Survey Satellite
            </p>
            <div className="text-blue-400 font-semibold text-xs tracking-wider">
              STATUS: ACTIVE
            </div>
          </div>
        </Link>
      </div>

      {/* Mission Control Panel */}
      <div className="mt-16 nasa-panel rounded-xl p-8 max-w-4xl mx-auto">
        <h3 className="text-2xl font-bold mb-6 text-center text-white">Mission Control Dashboard</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-red-400 mb-2">3</div>
            <div className="text-gray-300 text-sm">Active Telescopes</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-orange-400 mb-2">∞</div>
            <div className="text-gray-300 text-sm">Data Points</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-yellow-400 mb-2">24/7</div>
            <div className="text-gray-300 text-sm">Monitoring</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-green-400 mb-2">LIVE</div>
            <div className="text-gray-300 text-sm">Data Stream</div>
          </div>
        </div>
      </div>
    </div>
  );
}
