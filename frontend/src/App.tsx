import { useState } from "react"
import SearchBar from "./components/SearchBar.tsx"
import ResultsList from "./components/ResultsList.tsx"
import type { RecommendationItem } from "./api/semantica"
import "./App.css"
import { motion } from "framer-motion"

function App() {
  const [results, setResults] = useState<RecommendationItem[]>([]) // App owns results

  return (
    <motion.div
      className="app-container"
      animate={{ y: results.length > 0 ? -120 : 0 }}
      transition={{ duration: 0.6, ease: "easeInOut" }}>
      <div className="app-header">
        <h1 className="center-heading">Semantica</h1>
      </div>

      <div className="app-main">
        <SearchBar setResults={setResults} /> {/* pass setter down */}

        {results.length > 0 && (
          <div className="results-container">
            {results.length > 0 && <ResultsList results={results} />}
          </div>)} 
        
      </div>
    </motion.div>
  )
}

export default App