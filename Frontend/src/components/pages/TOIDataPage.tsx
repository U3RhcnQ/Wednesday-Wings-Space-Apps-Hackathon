export default function TOIDataPage() {
  return (
    <div className="py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-teal-400">
          TOI Data Visualization
        </h1>
        <p className="text-gray-300 text-lg">
          TESS Objects of Interest - Latest exoplanet discoveries from NASA's Transiting Exoplanet Survey Satellite
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart Placeholder 1 */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4 text-green-300">Transit Depth Analysis</h3>
          <div className="bg-white/20 rounded-lg h-64 flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-2">📉</div>
              <p className="text-gray-300">Transit Chart Placeholder</p>
              <p className="text-sm text-gray-400">Visualization coming soon</p>
            </div>
          </div>
        </div>

        {/* Chart Placeholder 2 */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4 text-green-300">Sector Coverage</h3>
          <div className="bg-white/20 rounded-lg h-64 flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-2">🗺️</div>
              <p className="text-gray-300">Sky Map Placeholder</p>
              <p className="text-sm text-gray-400">Visualization coming soon</p>
            </div>
          </div>
        </div>

        {/* Chart Placeholder 3 */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4 text-green-300">Planet Radius vs Period</h3>
          <div className="bg-white/20 rounded-lg h-64 flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-2">🪐</div>
              <p className="text-gray-300">Radius-Period Scatter Placeholder</p>
              <p className="text-sm text-gray-400">Visualization coming soon</p>
            </div>
          </div>
        </div>

        {/* Chart Placeholder 4 */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6">
          <h3 className="text-xl font-semibold mb-4 text-green-300">Discovery Timeline</h3>
          <div className="bg-white/20 rounded-lg h-64 flex items-center justify-center">
            <div className="text-center">
              <div className="text-4xl mb-2">⏰</div>
              <p className="text-gray-300">Timeline Chart Placeholder</p>
              <p className="text-sm text-gray-400">Visualization coming soon</p>
            </div>
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gradient-to-r from-green-500/20 to-teal-500/20 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-green-300">---</div>
          <div className="text-sm text-gray-300">Total TOIs</div>
        </div>
        <div className="bg-gradient-to-r from-green-500/20 to-teal-500/20 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-green-300">---</div>
          <div className="text-sm text-gray-300">Confirmed Planets</div>
        </div>
        <div className="bg-gradient-to-r from-green-500/20 to-teal-500/20 rounded-lg p-4 text-center">
          <div className="text-2xl font-bold text-green-300">---</div>
          <div className="text-sm text-gray-300">TESS Sectors</div>
        </div>
      </div>
    </div>
  );
}
