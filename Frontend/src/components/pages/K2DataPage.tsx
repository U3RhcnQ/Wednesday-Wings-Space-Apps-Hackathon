export default function K2DataPage() {
  return (
    <div className="py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-400">
          K2 Mission Data Visualization
        </h1>
        <p className="text-gray-300 text-lg">
          Extended Kepler Mission - Exoplanet discoveries from the K2 mission's ecliptic plane observations
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart Placeholder 1 */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4 text-purple-300">Campaign Distribution</h3>
          <div className="bg-white/20 rounded-lg h-64 flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-2">🗓️</div>
              <p className="text-gray-300">Campaign Chart Placeholder</p>
              <p className="text-sm text-gray-400">Visualization coming soon</p>
            </div>
          </div>
        </div>

        {/* Chart Placeholder 2 */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4 text-purple-300">Planet Detection Methods</h3>
          <div className="bg-white/20 rounded-lg h-64 flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-2">🔍</div>
              <p className="text-gray-300">Methods Chart Placeholder</p>
              <p className="text-sm text-gray-400">Visualization coming soon</p>
            </div>
          </div>
        </div>

        {/* Chart Placeholder 3 */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4 text-purple-300">Stellar Magnitude Distribution</h3>
          <div className="bg-white/20 rounded-lg h-64 flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-2">✨</div>
              <p className="text-gray-300">Magnitude Histogram Placeholder</p>
              <p className="text-sm text-gray-400">Visualization coming soon</p>
            </div>
          </div>
        </div>

        {/* Chart Placeholder 4 */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4 text-purple-300">Planet Equilibrium Temperature</h3>
          <div className="bg-white/20 rounded-lg h-64 flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-2">🌡️</div>
              <p className="text-gray-300">Temperature Chart Placeholder</p>
              <p className="text-sm text-gray-400">Visualization coming soon</p>
            </div>
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-purple-300">---</div>
          <div className="text-sm text-gray-300">K2 Candidates</div>
        </div>
        <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-purple-300">---</div>
          <div className="text-sm text-gray-300">Campaigns</div>
        </div>
        <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-purple-300">---</div>
          <div className="text-sm text-gray-300">Confirmed Planets</div>
        </div>
      </div>
    </div>
  );
}
