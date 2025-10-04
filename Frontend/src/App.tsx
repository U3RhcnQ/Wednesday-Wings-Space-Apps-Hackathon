import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import HomePage from './components/pages/HomePage';
import LibraryPage from './components/pages/LibraryPage';
import KOIDataPage from './components/pages/KOIDataPage';
import K2DataPage from './components/pages/K2DataPage';
import TOIDataPage from './components/pages/TOIDataPage';
import UploadPage from './components/pages/UploadPage';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/library" element={<LibraryPage />} />
          <Route path="/library/koi" element={<KOIDataPage />} />
          <Route path="/library/k2" element={<K2DataPage />} />
          <Route path="/library/toi" element={<TOIDataPage />} />
          <Route path="/upload" element={<UploadPage />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
