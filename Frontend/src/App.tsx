function App() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-purple-900 to-black text-white">
      <div className="container mx-auto px-4 py-16">
        <h1 className="text-5xl font-bold text-center mb-8 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
          Wednesday Wings
        </h1>
        <p className="text-xl text-center text-gray-300 mb-12">
          Space Apps Hackathon - Exoplanet Explorer
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 hover:bg-white/20 transition-all">
            <h2 className="text-2xl font-semibold mb-4">KOI Data</h2>
            <p className="text-gray-300">Explore Kepler Objects of Interest</p>
          </div>
          <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 hover:bg-white/20 transition-all">
            <h2 className="text-2xl font-semibold mb-4">K2 Mission</h2>
            <p className="text-gray-300">Discover K2 exoplanet candidates</p>
          </div>
          <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 hover:bg-white/20 transition-all">
            <h2 className="text-2xl font-semibold mb-4">Analysis</h2>
            <p className="text-gray-300">View data visualizations</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

