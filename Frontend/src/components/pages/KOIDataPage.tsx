export default function KOIDataPage() {
  return (
    <div className="py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-400">
          KOI Data Visualization
        </h1>
        <p className="text-gray-300 text-lg">
          Kepler Objects of Interest - Exoplanet candidates and confirmed planets from the Kepler Space Telescope
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart Placeholder 1 */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4 text-blue-300">Planet Size Distribution</h3>
          <div className="bg-white/20 rounded-lg h-64 flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-2">📊</div>
              <p className="text-gray-300">Chart Placeholder</p>
              <p className="text-sm text-gray-400">Visualization coming soon</p>
            </div>
          </div>
        </div>

        {/* Chart Placeholder 2 */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4 text-blue-300">Orbital Period Analysis</h3>
          <div className="bg-white/20 rounded-lg h-64 flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-2">📈</div>
              <p className="text-gray-300">Chart Placeholder</p>
              <p className="text-sm text-gray-400">Visualization coming soon</p>
            </div>
          </div>
        </div>

        {/* Chart Placeholder 3 */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4 text-blue-300">Star Temperature vs Planet Radius</h3>
          <div className="bg-white/20 rounded-lg h-64 flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-2">🌟</div>
              <p className="text-gray-300">Scatter Plot Placeholder</p>
              <p className="text-sm text-gray-400">Visualization coming soon</p>
            </div>
          </div>
        </div>

        {/* Chart Placeholder 4 */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4 text-blue-300">Habitability Zone Analysis</h3>
          <div className="bg-white/20 rounded-lg h-64 flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-2">🌍</div>
              <p className="text-gray-300">Zone Chart Placeholder</p>
              <p className="text-sm text-gray-400">Visualization coming soon</p>
            </div>
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-blue-300">---</div>
          <div className="text-sm text-gray-300">Total Objects</div>
        </div>
        <div className="bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-blue-300">---</div>
          <div className="text-sm text-gray-300">Confirmed Planets</div>
        </div>
        <div className="bg-gradient-to-r from-blue-500/20 to-cyan-500/20 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-blue-300">---</div>
          <div className="text-sm text-gray-300">Candidates</div>
        </div>
      </div>
    </div>
  );
}
