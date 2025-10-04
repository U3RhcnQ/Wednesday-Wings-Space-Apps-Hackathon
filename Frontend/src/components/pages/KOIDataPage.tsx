import { useState, useEffect } from 'react';

interface PlotInfo {
  filename: string;
  size_bytes: number;
  created: string;
  modified: string;
  url: string;
}

export default function KOIDataPage() {
  const [plots, setPlots] = useState<PlotInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPlots();
  }, []);

  const fetchPlots = async () => {
    try {
      const response = await fetch('http://localhost:8000/plots');
      const data = await response.json();
      setPlots(data.datasets.koi || []);
    } catch (error) {
      console.error('Error fetching plots:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatPlotName = (filename: string): string => {
    const name = filename
      .replace(/^koi_/, '')
      .replace(/\.png$/, '')
      .replace(/_/g, ' ');
    return name
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

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

      {loading ? (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-400 mx-auto mb-4"></div>
          <p className="text-gray-300">Loading visualizations...</p>
        </div>
      ) : plots.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-gray-400 text-xl">No visualizations available</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {plots.map((plot) => (
            <div key={plot.filename} className="bg-white/10 backdrop-blur-lg rounded-lg overflow-hidden border border-white/10 hover:border-blue-400/50 transition-all">
              <div className="p-4 border-b border-white/10">
                <h3 className="text-xl font-semibold text-blue-300">{formatPlotName(plot.filename)}</h3>
              </div>
              <div className="p-4 bg-white/5">
                <img
                  src={`http://localhost:8000${plot.url}`}
                  alt={formatPlotName(plot.filename)}
                  className="w-full h-auto rounded"
                  loading="lazy"
                />
              </div>
            </div>
          ))}
        </div>
      )}

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
