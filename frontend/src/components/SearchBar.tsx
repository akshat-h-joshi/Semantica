import { useState } from "react"
import { recommend } from "../api/semantica"
import type { RecommendationItem } from "../api/semantica"
import ModelSelector from "./ModelSelector"
import "../App.css"

interface Props {
  setResults: (results: RecommendationItem[]) => void
}

export default function SearchBar({ setResults }: Props) {
  const [query, setQuery] = useState("")
  const [model, setModel] = useState("mpnet")
  // const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSearch = async () => {
    if (!query.trim()) return

    // setLoading(true)
    setError(null)

    try {
      const res = await recommend(query, model, 5)
      setResults(res.results) // call the setter from props
    } catch (err) {
      setError("Failed to fetch recommendations")
    } finally {
      // setLoading(false)
    }
  }

  return (
    <div className="search-container">
      <input
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="Search for research papers..."
        className="search-input"
      />

      <div className="controls">
        <ModelSelector model={model} setModel={setModel} />
        <button onClick={handleSearch}>Search</button>
      </div>

      {/* {loading && <p>Loading...</p>} */}
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  )
}